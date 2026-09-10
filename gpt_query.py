import os
import tiktoken

# for hugging face inference endpoints for codellama
import requests

from typing import Optional

from openai import (
    OpenAI,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    BadRequestError,
)

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# fallback max_tokens for the legacy hugging_face_request/replicate_request paths, which
# aren't threaded through BaselineConfig
DEFAULT_MAX_TOKENS = 10

_clients: dict[tuple[Optional[str], Optional[str]], OpenAI] = {}


def _get_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """Lazily construct and cache an OpenAI-SDK client, keyed by (api_key, base_url) so
    a run can talk to the real OpenAI API and/or any OpenAI-compatible endpoint (e.g.
    Gemini's compatibility layer) without the two colliding.

    If `api_key` isn't given, falls back to the OPENAI_API_KEY env var, then
    gpt_inputs/api_key.txt, to preserve the existing setup flow. `base_url` is passed
    straight through to the OpenAI SDK; leave it unset to talk to api.openai.com.
    """
    cache_key = (api_key, base_url)
    if cache_key in _clients:
        return _clients[cache_key]

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_key:
        with open("gpt_inputs/api_key.txt", "r") as f:
            resolved_key = f.read().strip()

    client_kwargs = {"api_key": resolved_key}
    if base_url is not None:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    _clients[cache_key] = client
    return client


_unsupported_sampling_params_warned: set[str] = set()


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
    ),
)
def get_chat_response(
    messages: list[dict],
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    system_message: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Send a real multi-turn chat history to an OpenAI-compatible Chat Completions
    API and return the assistant's reply text. Defaults to the real OpenAI API;
    pass `base_url`/`api_key` to target any OpenAI-compatible endpoint instead (e.g.
    Gemini's `https://generativelanguage.googleapis.com/v1beta/openai/`).

    `messages` is never mutated -- a system message, if given, is prepended to a copy.
    """
    client = _get_client(api_key=api_key, base_url=base_url)

    full_messages = list(messages)
    if system_message is not None:
        full_messages = [{"role": "system", "content": system_message}] + full_messages

    record_messages(full_messages, model)

    kwargs = {
        "model": model,
        "messages": full_messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    try:
        completion = client.chat.completions.create(**kwargs)
    except BadRequestError as e:
        # Some newer models reject temperature/top_p entirely. Warn once per model and
        # retry without them rather than failing the whole run -- but this is a fidelity
        # concern worth knowing about, so it's printed, not swallowed.
        if model not in _unsupported_sampling_params_warned:
            print(
                f"Warning: model '{model}' rejected sampling parameters ({e}); "
                "retrying without temperature/top_p."
            )
            _unsupported_sampling_params_warned.add(model)
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)
        completion = client.chat.completions.create(**kwargs)

    response = completion.choices[0].message.content

    record_messages(
        full_messages + [{"role": "assistant", "content": response}], model
    )

    return response


def hugging_face_request(
    model: str, messages: list[dict], temperature: float, max_tokens: int = DEFAULT_MAX_TOKENS
) -> str:
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    messages = translate_to_string_input(messages)
    API_URL = "https://xxxxxxxx.us-east-1.aws.endpoints.huggingface.cloud"
    headers = {
        "Authorization": "Bearer xxxxx",
        "Content-Type": "application/json",
    }

    if temperature == 0:
        temperature = 0.001

    output = query(
        {
            "inputs": messages,
            "parameters": {"temperature": temperature, "max_new_tokens": max_tokens},
        }
    )

    return output[0]["generated_text"]


def replicate_request(
    model: str, messages: list[dict], temperature: float, max_tokens: int = DEFAULT_MAX_TOKENS
) -> str:
    import replicate

    if temperature == 0:
        temperature = 0.001

    with open("gpt_inputs/replicate_api_key.txt", "r") as f:
        api_key = f.read().strip()
    os.environ["REPLICATE_API_TOKEN"] = api_key

    model = model.replace("replicate/", "")

    prompt = translate_to_string_input(messages)

    output = replicate.run(
        model,
        input={
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        },
    )

    response = ""
    for item in output:
        response += item

    return response


def translate_to_string_input(
    openai_messages: list[dict], roles_included: bool = False
):
    # Translate from OpenAI's dict to a single string input
    messages = []
    for message in openai_messages:
        if roles_included:
            messages.append(message["role"] + ": ")
        messages.append(message["content"])
    if roles_included:
        messages.append("assistant: ")
    return "\n".join(messages)


def count_all_tokens(model: str, messages: list[dict[str, str]]) -> int:
    total_tokens = 0
    for message in messages:
        total_tokens += count_tokens(model, message["content"])
    return total_tokens


def count_tokens(model: str, prompt: str) -> int:
    if "gpt" not in model:
        model = "gpt-4"

    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(prompt))
    return num_tokens


def get_prompt_cost(model: str, num_tokens: int) -> float:
    # good enough for quick evals
    if model not in pricing_dict:
        return num_tokens * 0.001 * pricing_dict["gpt-4"]
    return num_tokens * 0.001 * pricing_dict[model]


def get_response_cost(model: str, num_tokens: int) -> float:
    # good enough for quick evals
    if model not in pricing_dict:
        return num_tokens * 0.001 * pricing_dict["gpt-4"]

    cost = num_tokens * 0.001 * pricing_dict[model]

    if model == "gpt-4":
        cost *= 2

    return cost


def record_messages(messages: list[dict], model: str):
    """Overwrite the debug transcript with the current, role-tagged multi-turn
    conversation. This is a live debug aid only (gitignored), not an accumulated log."""
    with open("gpt_outputs/transcript.txt", "w") as f:
        f.write(model + "\n\n")
        for message in messages:
            f.write(f"[{message['role']}]\n{message['content']}\n\n")
