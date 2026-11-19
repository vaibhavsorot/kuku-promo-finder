# Kuku Promo Finder

This project helps an editor find short promo clips in a show's screenplays. It reads the timed scripts, suggests moments with a strong opening and an exciting ending, checks the timestamps, then ranks the clips.

It was built for **Problem 1: Find the Promos** in Kuku's AI Generalist assignment.

## Open the app

In Terminal, go to this folder and run:

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

Open the local web address printed in Terminal, usually `http://localhost:8501`.

To see the saved results, choose **replay** as the provider in the app. Replay uses saved AI responses and does not make new AI calls. It only works when the matching scripts and cache are available. The input scripts, cache and saved results are kept out of Git because they contain Kuku's content. If you cloned this repo, ask for the private assignment files to view or replay those results.

## Run it on scripts

The input can be a folder of episode `.txt` files or a `.zip` containing the scripts. In the app, upload a zip or enter a folder path, choose an AI provider, then select **Run pipeline**.

To run from Terminal:

```bash
python run_pipeline.py --inspect --input "data/input/<show folder>"  # read scripts and check parsing
python run_pipeline.py --input "data/input/<show folder>"             # find and rank promos
```

To process new scripts, choose a provider and add its API key to your local `.env` file. The project supports Gemini, Claude, OpenAI-compatible services, and the Claude Code CLI. Use `.env.example` to see the settings. **Do not commit or share your API key.**

Useful options:

- `--episodes 1-5,9` runs only selected episodes.
- `--no-cross` skips montages made from multiple episodes.
- `--blind-top 0` skips the new-viewer check.
- `--provider replay` uses cached answers only and stops if an answer is missing.

## What the tool does

1. **Reads the scripts.** It splits each episode into timed dialogue, action, camera and sound blocks.
2. **Finds possible promos.** An AI model suggests up to three clips per episode using script block IDs. It can also suggest montages using moments from different episodes.
3. **Builds the clips.** Code turns the chosen blocks into timestamps and avoids cutting a dialogue line in half.
4. **Checks the suggestions.** Code checks that clips are 20–60 seconds, segments are in order, the hook starts near the beginning, the cliffhanger is near the cut, and quoted lines appear in the script. If a check fails, the AI gets one chance to fix the suggestion.
5. **Scores and ranks the clips.** Scores cover the hook, cliffhanger, clarity, emotion, curiosity, flow and how well the clip sells the show's premise. Similar clips are marked as duplicates.
6. **Checks how a new viewer might react.** A separate AI call sees only the clip transcript and estimates what a first-time viewer would understand. This is a useful check, not a real audience test.

## Where to find the results

The app shows the saved run. The files for each show are in `outputs/<show name>/`:

- `final_promos/promos.json` and `promos.csv`: all suggested clips and their scores.
- `final_promos/ranked_promos.md`: a readable ranked list with clip transcripts.
- `raw_outputs/`: unedited AI responses, including retries.
- `evaluation/`: automatic checks and a review sheet for an editor.
- `processed/`: the parsed scripts and an input summary.
- `run_manifest.json`: which model ran, how long it took and how many clips it found.

## What the full run found

On the provided 50 episodes, the tool suggested 147 promos. Of these, 145 were ranked, 2 were marked as duplicates, and every episode had at least one valid promo. The code verified the hook timing and cliffhanger timing for all 147 suggestions and found no unsupported quotes.

These checks confirm facts such as timestamps and quoted lines. They cannot prove that a promo is exciting or clear to real viewers. The manual review and quality report describe examples where the tool made weak choices.

## Re-running and limits

AI suggestions can change between runs. The cache saves responses so the same prompts can be replayed without another AI call. A new show or changed prompt needs a working provider and may incur usage costs. Costs and run time depend on the provider and the amount of script text.

The parser and scoring rules have automated tests. Run them with:

```bash
python -m pytest -q
```

## Assignment files and privacy

The solution PDF is `docs/solution_document.pdf`. Run steps are above, and a short optional demo plan is in `docs/walkthrough_plan.md`.

The assignment asks for unedited run outputs. They are in `outputs/i-married-a-broken-soldier-he-was-a-duke/raw_outputs/`. `data/input/`, `data/cache/` and `outputs/` are excluded from Git because they contain Kuku screenplay text. Share the required raw outputs privately with the assignment reviewers; do not publish them in a public repo.
