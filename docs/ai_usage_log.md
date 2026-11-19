# How AI was used

The assignment asks us to name the AI tools we used and give an example of where one got something wrong.

| Tool | How it was used |
|---|---|
| Claude Code (Opus 5.5) | Helped write the code, tests and documents. It also made an AI-assisted first pass through a small sample of promos; this was not a human editor review. |
| Claude Sonnet 5.5 | Read the screenplays and suggested promo moments. It also read each suggested clip transcript as if it were a new viewer. |

## Mistakes caught and fixes made

- **It returned invalid JSON.** One response had an extra quote, so the program could not read it. The parser caught the problem and sent it back to the model for correction. Both attempts were saved in the raw outputs.
- **It left out a required question.** The model omitted the open question that should keep a viewer curious. I added a check and made the prompt clearer; the model then had to fill the missing field.
- **It gave a wrong character relationship.** For E45-P3, the explanation described Catherine as a bride Alexander had abandoned. The screenplay shows she was marrying Rupert. The code checks quotes and timestamps, but it cannot verify every explanation, so an editor still needs to review the wording.
- **It sometimes chose weak or confusing clips.** The model rated its own clips more highly than the separate new-viewer check did. I use both scores in the ranking and show the viewer's confusion notes in the quality report.
- **The command-line model service failed under load.** Some viewer-check calls failed with a model-name error. The program logged failures instead of inventing answers; I added retries and ran the checks again.

The raw responses and retry notes are kept in `outputs/<show>/raw_outputs/`. They can include screenplay text.
