# Kuku Promo Finder — Problem 1: Find the Promos

## The problem and my approach

Editors watch every episode to find short clips that can bring new viewers to a show. I built a tool that reads the timed screenplays and suggests promos that open with a hook, end on an unresolved moment, and make sense without knowing the series.

I use AI for creative choices and code for checks that can be measured. The AI suggests moments using script block IDs; code turns those IDs into timestamps, checks the clips, and ranks them. This keeps the model from inventing cut times.

## Assumptions

- The screenplay timestamps match the final video, and clips use the existing footage without re-editing.
- A hook should begin within 3 seconds of the clip starting. The cliffhanger should end within 8 seconds of the clip ending.
- Clips should play in story order. One continuous clip is best, but a short montage can join moments from different episodes.
- Speaker names in the scripts can differ from character names. The AI creates a small guide to those labels.
- Spoilers are allowed. This is one English-language show per run.

## How it works

| Step | What happens |
|---|---|
| Read | Code splits each screenplay into timed dialogue, action, camera and sound blocks. It handles format changes, missing timestamps, duplicate files and zip uploads. |
| Find | An AI call per episode suggests up to three promos by block ID, plus the hook, cliffhanger, quotes and scores. A separate call can join standout moments across episodes. |
| Build and check | Code turns block IDs into timestamps, avoids cutting dialogue in half, and checks length, order, hook timing, cliffhanger timing and quotes. If a suggestion fails, it goes back to the AI once for a fix. |
| Rank | Code combines scores for hook, cliffhanger, clarity, emotion, curiosity, flow and premise appeal. It marks clips that overlap too much with a higher-ranked clip. |
| Get a new-viewer check | A separate AI call sees only the clip transcript and estimates what a first-time viewer understands. It is a model opinion, not real audience research. |

The AI provider can be changed in `.env`. The project supports Gemini, Claude, compatible chat APIs, the Claude Code CLI and replay from cache.

## Results on the 50-episode input pack

The run found **147 promos**: 141 from one episode and 6 montages. It ranked 145, marked 2 as duplicates, and found at least one valid promo in every episode. Code checked all 147 suggestions: every hook and cliffhanger was in the expected place, and no quoted line was missing from the script.

The best scored clips are a shortlist, not a guaranteed posting order. For example, the highest-ranked clip opens with a poison accusation and cuts off as the Duchess collapses. The second-ranked clip uses a ring's family crest to hint that a poor soldier may be a lost duke.

The original generation run took about 9 minutes with four workers. Across the recorded generation and viewer-check runs, reported model usage was about **$7 total**, or roughly **$0.14 per episode**. Replaying cached answers does not make new model calls. Actual cost and speed will vary by model and provider.

## What it gets wrong

There are no human-labeled examples, so I do not claim precision or recall. I checked the measurable claims in code, ran a separate AI new-viewer check on the promos, and reviewed a sample against the scripts. The review is AI-assisted; an editor should repeat it.

- A clip can pass the timing check but still start weakly. E26-P1 opens with a quiet line; the stronger visual moment comes about 10 seconds later.
- The model does not judge silent visual moments as well as dialogue. It misses the Duke's entrance through smoke in E42.
- The best premise montage is ranked too low because several cuts receive penalties. Resolved reveals can also score lower than unresolved endings.
- Late-series clips can confuse new viewers who do not know the characters.
- The AI's explanations can be wrong even when its timestamps and quotes are correct. For E45-P3 it misstated Catherine's relationship.
- Scores are close together. Treat roughly the top 30 as an editor's shortlist, not a precise order.

## Scale and success measures

The work grows with the number of episodes: each new episode gets an AI call, and cached episodes cost no additional calls. The run can use more workers, save progress and continue after an interruption. In production, an editor should approve or adjust the top clips.

I would measure how many top-ten clips editors approve without changing, how many seconds they spend fixing timestamps, and how long it takes to review a show. Then I would compare promos with editor-picked promos using 3-second viewing, completion and starts of episode one.

## AI use

Claude Code helped build the code, tests and first-pass review. Claude Sonnet selected promos and ran the new-viewer checks. It returned malformed JSON once; the parser caught it and the model corrected it on retry. It also gave one wrong relationship explanation, which a script check could not catch. More examples are in `docs/ai_usage_log.md`.
