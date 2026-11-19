# Optional 3-minute demo

The assignment recommends a short screen recording, but it is optional. Use your own voice. Keep the app open and show the saved run so the demo does not depend on making new AI calls.

## Before you record

1. In Terminal, open the project folder and run `streamlit run app.py`.
2. Open the local link Streamlit prints.
3. In the sidebar, load the saved show. If you run the pipeline, choose **replay** to use cached answers.
4. Keep the top-ranked promos and Quality tab ready.

## What to show and say

| Time | Show | Say |
|---|---|---|
| 0:00–0:25 | README overview or app Overview | “Editors have to watch every episode to find ads. This tool reads the timed scripts and suggests short promo clips.” |
| 0:25–0:50 | Overview numbers | “This run covers all 50 episodes. It found 147 promos, ranked 145, and marked 2 as duplicates. Every episode has at least one valid suggestion.” |
| 0:50–1:20 | Ranked promos, then promo #1 | “Here are the clips in score order. This one opens with an accusation and cuts off as the Duchess collapses. The app shows the episode and exact timestamps.” |
| 1:20–1:50 | Promo detail | “The AI suggests the moment. Code checks the timing and confirms the quoted words appear in the script. Another AI reads only the clip transcript and checks whether a new viewer can follow it.” |
| 1:50–2:30 | Quality tab | “These checks do not prove that real viewers will like a clip. The model sometimes misses visual moments, and it can misunderstand character relationships. Here are examples where it picks badly.” |
| 2:30–3:00 | Export tab or README run steps | “An editor can download the ranked results. Replay shows the saved run without an API. To analyze new scripts, choose an AI provider and configure its key.” |

## If something goes wrong

If a fresh run fails because there is no API key or provider, stop and load the saved results instead. Do not claim the saved run is a new live analysis. The README explains how to configure a provider for new scripts.
