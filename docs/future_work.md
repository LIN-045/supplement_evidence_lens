# Future Work

This document records improvements intentionally deferred until after the
current project version is complete. None of these sources or test cases are
part of the current index or reported evaluation results.

## Candidate data sources

### 1. NIH ODS Consumer FAQ

- Source: https://ods.od.nih.gov/HealthInformation/ODS_Frequently_Asked_Questions.aspx
- Priority: High
- Value: Adds consumer-facing guidance about supplement safety, regulation,
  nutrient deficiencies, medication replacement, children, and when to consult
  a health care provider.
- Ingestion idea: Parse each question-and-answer pair as one source document,
  preserving the question as the title and the answer as the content.
- Evaluation caution: Do not use an ingested FAQ question verbatim to evaluate
  query rewriting. Use independently collected or naturally paraphrased
  questions to avoid exact-match leakage.

### 2. NCCIH Herbs at a Glance

- Source: https://www.nccih.nih.gov/health/herbsataglance
- Priority: High
- Value: Expands evidence, side-effect, caution, and drug-interaction coverage
  for herbs and botanicals.

### 3. FDA Dietary Supplement Ingredient Directory

- Source: https://www.fda.gov/food/dietary-supplements/dietary-supplement-ingredient-directory
- Priority: Medium
- Value: Adds United States regulatory actions and communications about
  selected supplement ingredients.

## Real-world regression case

### ADHD, children, and replacing medication

Question:

> My son has attention deficit hyperactivity disorder (ADHD) and I want to
> avoid using medications. Are there any dietary supplements that might help?

Origin:

- NIH ODS Consumer FAQ
- https://ods.od.nih.gov/HealthInformation/ODS_Frequently_Asked_Questions.aspx

Official FAQ guidance:

- Most supplements have not been proven to help people with ADHD, and some may
  be dangerous.
- A health care provider may recommend dietary changes or a specific supplement
  when a child has an identified vitamin or mineral deficiency.
- Taking additional vitamins or minerals without such a deficiency does not
  necessarily help, and high doses can be harmful.
- Fish oil and other supplements are still being studied, but the available
  research does not establish that they are effective for ADHD.
- Parents considering supplements should use reliable information and discuss
  the decision with the child's health care provider.

Observed V1 behavior:

- Retrieved relevant NIH omega-3 evidence and Health Canada reference sections.
- Correctly described the omega-3 evidence as mixed and stated that the excerpts
  did not establish fish oil as a replacement for ADHD medication.
- Opened with "Yes," which could overstate the evidence in a question about a
  child and avoiding medication.
- Included a wide research dose range that could be mistaken for a practical
  pediatric recommendation.
- Relied on several reference-list chunks that established that studies existed
  but did not provide clear effectiveness conclusions.
- Did not foreground the broader guidance that most supplements have not been
  proven to help ADHD, that high doses can be harmful, and that a child's health
  care provider should be involved.

Desired V2 behavior:

- Lead with whether the evidence supports supplements as a replacement for
  established treatment.
- Preserve the distinction between "studied" and "shown to be effective."
- Treat children, pregnancy, medication replacement, and similarly high-risk
  contexts with additional caution.
- Do not present research dose ranges as recommendations.
- Prefer conclusion-bearing evidence over reference-list sections.
- Clearly state when the indexed sources do not support a definitive answer.

Failure tags:

- `pediatric_context`
- `medication_replacement`
- `overpositive_opening`
- `research_dose_vs_recommendation`
- `reference_list_retrieval`
- `missing_consumer_safety_context`

## V2 evaluation plan

- Collect real-world questions from application feedback and a small,
  privacy-preserving sample of public consumer questions.
- Remove usernames and identifying health details; paraphrase when appropriate
  and retain only the source URL and origin type.
- Keep the current source-generated retrieval set for reproducibility.
- Add a separate real-world challenge set for Fixed RAG versus Agentic RAG.
- After adding new sources, rebuild chunks and the Elasticsearch index, then
  rerun retrieval pooling, relevance judgments, retrieval metrics, answer
  generation, and answer judgments.
