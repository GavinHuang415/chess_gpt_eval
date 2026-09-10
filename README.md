**Overview**

There has recently been some buzz about the ability of GPT-3.5-turbo-instruct's chess playing ability. I wanted to take a more rigourous look and created this repo. With it, you can play two models against each other, whether that's LLM vs Stockfish, LLM vs LLM, or Stockfish vs Stockfish. The primary goal is to test and record the performance of these players against one another in various configurations. Illegal moves, resignations, and game states are all tracked and recorded for later analysis.

**Baseline replication**

`main.py`'s `__main__` block runs the baseline procedure from Kuo, Hsueh & Tsai (2023), "Large Language Models on the Chessboard": the LLM plays Black against Stockfish (White), White's opening move is randomly chosen from `{e4, d4, Nf3, e3}`, moves are exchanged as a real multi-turn chat conversation in SAN, Stockfish samples its move from its top 3 candidates each turn (recording its centipawn evaluation of White's advantage along the way), and the LLM gets up to 10 consecutive illegal-move retries (resampled against the same conversation history, not shown its own rejected moves) before the game is terminated.

- Set up:
  - `pip install -r requirements.txt`.
  - Put your OpenAI key in `gpt_inputs/api_key.txt`, or set the `OPENAI_API_KEY` environment variable.
  - Install Stockfish and either put it on your system `PATH`, or point at it explicitly with the `STOCKFISH_PATH` environment variable (or `BaselineConfig.stockfish_path`). On Mac: `brew install stockfish`. On Linux: `apt install stockfish`. On Windows: download a build from [stockfishchess.org](https://stockfishchess.org/download/) and set `STOCKFISH_PATH` to the `.exe` path.
- Configure and run: edit the `BaselineConfig(...)` at the bottom of `main.py` -- `model` has no default and must be set explicitly (the paper's `gpt-3.5-turbo-0301` is retired). All other run parameters (temperature, top_p, illegal-retry cap, engine search depth, number of games, etc.) live on that same config object. Then run `python main.py`.
- Output: each run writes three files to `logs/baseline/`, named by `run_id` (defaults to `<model>_<timestamp>`):
  - `<run_id>_moves.csv` -- one row per ply (move text, illegal-attempt counts, Stockfish's evaluation before that ply, Stockfish's top-3 candidates on its own turns, etc.), appended live so a crashed run keeps its data.
  - `<run_id>_games.csv` -- one row per game, including the paper's metrics computed at game end: `imr_final`, `rblm_final`, `gl_plies`/`gl_fullmoves`, `be_move20`, `be_final`, `be_mean`.
  - `<run_id>_config.json` -- full run configuration, prompt text, Stockfish version, and git commit, for provenance.
- Analysis: `utils/paper_metrics.py` implements the paper's IMR/RBLM/GL/BE formulas against these CSVs; `utils/baseline_analysis.ipynb` is a thin notebook that loads a run and plots the per-move IMR/RBLM curves and the game-length distribution.

**Legacy results (pre-refactor)**

The following results were gathered with the older, non-paper-methodology version of this codebase (LLM as White, best-move-only Stockfish, 5-illegal-move cap, flat-transcript prompting) and are kept here for reference; they are not comparable to `logs/baseline/` output.

GPT-3.5-turbo instruct does well in 150 games against various Stockfish levels and 30 games against GPT-4. Most of gpt-4's losses were due to illegal moves, so it may be possible to come up with a prompt to have gpt-4 correct illegal moves and improve its score.

![](./gpt-3.5-turbo-instruct-win-rate.png)

gpt-3.5-turbo-instruct's illegal move rate is under 0.1% over 8205 moves (possibly 0%, I had insufficient recording and state validation going on during my run... TODO), and the longest game had 147 moves.

`analysis.ipynb` results:
```
total moves: 8205, total illegal moves: 5 or less
Ratio of Player One's Illegal Moves to Total Moves: 0.0006 or less
Minimum Moves: 15
Maximum Moves: 147
Median Moves: 45.0
Standard Deviation of Moves: 21.90
```

All results were gathered on Stockfish 16 with 0.1 seconds per move on a 2023 M1 Mac. I ran a Stockfish benchmark using `% stockfish bench 1024 16 26 default depth nnue 1>/dev/null 2>stockfish_M1Mac.bench` and stored the output in `logs/stockfish_M1Mac.bench`.

**Game Recording (legacy path)**

The legacy `record_results()`/`logs/games.csv` path described in earlier versions of this README has been superseded by the baseline-replication logging described above (`logs/baseline/*_games.csv`, `*_moves.csv`); `logs/games.csv` itself is left untouched since its columns have opposite color semantics (LLM was White there).

**Other Capabilities**

There is the ability to run other models using Hugging Face inference endpoints or Replicate (see `gpt_query.py`'s `hugging_face_request`/`replicate_request`). However, I've found that other models, like Llama2-70b chat won't provide formatted moves, and Llama2-70b base will hallucinate illegal moves.

You can use Llama and NanoGPT models. For more information, see the READMEs contained in `local_llama/` and `nanogpt/`.

In the dataset_generation branch, I played a distribution of Stockfish vs Stockfish levels against each other to create a dataset to train a NanoGPT model on chess games, which worked to create a ~1200 ELO chess LLM.

**Stockfish to ELO**

Stockfish provides this estimate of [Stockfish level to ELO](https://github.com/official-stockfish/Stockfish/commit/a08b8d4):

```
   # PLAYER             :  RATING  ERROR  POINTS  PLAYED   (%)
   1 master-skill-19    :  3191.1   40.4   940.0    1707    55
   2 master-skill-18    :  3170.3   39.3  1343.0    2519    53
   3 master-skill-17    :  3141.3   37.8  2282.0    4422    52
   4 master-skill-16    :  3111.2   37.1  2773.0    5423    51
   5 master-skill-15    :  3069.5   37.2  2728.5    5386    51
   6 master-skill-14    :  3024.8   36.1  2702.0    5339    51
   7 master-skill-13    :  2972.9   35.4  2645.5    5263    50
   8 master-skill-12    :  2923.1   35.0  2653.5    5165    51
   9 master-skill-11    :  2855.5   33.6  2524.0    5081    50
  10 master-skill-10    :  2788.3   32.0  2724.5    5511    49
  11 stash-bot-v25.0    :  2744.0   31.5  1952.5    3840    51
  12 master-skill-9     :  2702.8   30.5  2670.0    5018    53
  13 master-skill-8     :  2596.2   28.5  2669.5    4975    54
  14 stash-bot-v21.0    :  2561.2   30.0  1338.0    3366    40
  15 master-skill-7     :  2499.5   28.5  1934.0    4178    46
  16 stash-bot-v20.0    :  2452.6   27.7  1606.5    3378    48
  17 stash-bot-v19.0    :  2425.3   26.7  1787.0    3365    53
  18 master-skill-6     :  2363.2   26.4  2510.5    4379    57
  19 stash-bot-v17.0    :  2280.7   25.4  2209.0    4378    50
  20 master-skill-5     :  2203.7   25.3  2859.5    5422    53
  21 stash-bot-v15.3    :  2200.0   25.4  1757.0    4383    40
  22 stash-bot-v14      :  2145.9   25.5  2890.0    5167    56
  23 stash-bot-v13      :  2042.7   25.8  2263.5    4363    52
  24 stash-bot-v12      :  1963.4   25.8  1769.5    4210    42
  25 master-skill-4     :  1922.9   25.9  2690.0    5399    50
  26 stash-bot-v11      :  1873.0   26.3  2203.5    4335    51
  27 stash-bot-v10      :  1783.8   27.8  2568.5    4301    60
  28 master-skill-3     :  1742.3   27.8  1909.5    4439    43
  29 master-skill-2     :  1608.4   29.4  2064.5    4389    47
  30 stash-bot-v9       :  1582.6   30.2  2130.0    4230    50
  31 master-skill-1     :  1467.6   31.3  2015.5    4244    47
  32 stash-bot-v8       :  1452.8   31.5  1953.5    3780    52
  33 master-skill-0     :  1320.1   32.9   651.5    2083    31
```

It's difficult to make an exact comparison with my results due to differences in hardware and run time, but this can give a ballpark estimate. Stockfish ELO estimates were obtained with 120 seconds of run time, while I used 0.1 seconds. From my testing at lower Stockfish skill levels, 0.01 seconds vs 1 second makes relatively little difference.
