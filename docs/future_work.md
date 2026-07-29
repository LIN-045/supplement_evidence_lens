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
- Initial use: Keep the FAQ outside the index and use its questions and official
  answers as a real-world answer-evaluation set.
- Evaluation value: The official answers provide reference answers for testing
  answer correctness, which the current source-generated evaluation does not
  have.
- Ingestion decision: First use the FAQ to identify knowledge gaps. Prefer
  adding the underlying NIH, NCCIH, FDA, or other authoritative source when it
  contains the missing evidence. Add FAQ content to the index only when the
  relevant guidance is not available in a more direct source.
- Leakage rule: A FAQ question-and-answer pair used as indexed content must not
  also be used as a holdout evaluation example for the same project version.

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

### 4. FAQ-linked official resources

The NIH ODS Consumer FAQ links to several official resources that may fill
different evidence gaps. Do not ingest all linked pages by default. First run
the FAQ holdout evaluation against the current index, then use the observed
coverage gaps to decide which resources to add.

#### NIH ODS Dietary Supplement Fact Sheets directory

- Source: https://ods.od.nih.gov/factsheets/list-all/
- Ingestion decision: Do not index the directory page as evidence. Use it as an
  official source registry for discovering relevant ODS, NCCIH, MedlinePlus,
  FDA, OPSS, and other federal resources.
- Existing coverage: The current index already includes the ODS health
  professional fact sheets.
- Consumer versions: Avoid indexing all consumer and health professional
  versions when their content substantially overlaps. Add a consumer version
  selectively when it contains practical guidance missing from the health
  professional version, and label it `consumer_guidance`.

#### NIH ODS Nutrient Recommendations and Databases

- Source:
  https://ods.od.nih.gov/HealthInformation/nutrientrecommendations.aspx
- Priority: High
- Value: Provides authoritative explanations of DRI, RDA, AI, EAR, UL, and DV,
  together with links to official tables and the DRI calculator.
- Initial ingestion: Index the explanatory page as `nutrient_reference`.
- Structured expansion: Add age- and sex-specific DRI tables only if the FAQ
  coverage audit shows that the current nutrient fact sheets do not support the
  required answers. Do not treat calculator output as a personalised
  recommendation.

#### FDA supplement safety alerts

- Source:
  https://www.fda.gov/food/recalls-outbreaks-emergencies/alerts-advisories-safety-information
- Priority: Medium
- Value: Adds current warnings about contaminated, adulterated, substituted,
  or otherwise unsafe supplement products and ingredients.
- Ingestion decision: Do not index the mixed food-safety listing wholesale.
  Select only dietary-supplement-related alerts and ingest each underlying
  alert as a separate `safety_alert` document.
- Required metadata: Preserve the product or ingredient, publication date,
  update date, alert status, and source URL.
- Maintenance implication: Because alerts change over time, this source needs
  an explicit refresh policy and should not be treated as a static fact-sheet
  snapshot.

#### FTC consumer and advertising guidance

- General homepage: https://consumer.ftc.gov/
- Supplement-specific consumer resource:
  https://consumer.ftc.gov/media/79912
- Health-products compliance guidance:
  https://www.ftc.gov/business-guidance/resources/health-products-compliance-guidance
- Ingestion decision: Do not index the general FTC consumer homepage. Consider
  the supplement-specific consumer resource as `consumer_guidance` for
  evaluating marketing claims, product reliability, side effects,
  interactions, and purchasing questions.
- Scope decision: Add the broader health-products compliance guidance only if
  the application is expected to answer advertising-substantiation or business
  compliance questions. Keep this role distinct from clinical evidence and
  personalised purchasing advice.

## Evidence roles and purchasing questions

The current index combines documents that serve different purposes. A Health
Canada monograph can define acceptable product-licence and label claims, doses,
and warnings, but it is not by itself a consumer purchasing recommendation or a
clinical guideline. This distinction becomes important for questions such as
"Does this product work?" and "Should I buy this for my parents?"

V2 should:

- Add an `evidence_role` field during document construction, with values such
  as:
  - `regulatory_claim`
  - `regulatory_monograph`
  - `clinical_evidence_summary`
  - `consumer_guidance`
  - `safety_reference`
- Preserve the difference between an accepted regulatory label claim,
  traditional use, evidence of clinical effectiveness, and a clinical
  recommendation.
- For effectiveness and purchasing questions, prioritize clinical evidence
  summaries and guidelines over regulatory monographs.
- Use regulatory monographs to explain permitted uses, product conditions,
  doses, and warnings without treating them as proof that a product is worth
  buying.
- Avoid inferring that a branded product is effective from evidence about one
  ingredient, especially when the formulation, ingredient form, or dose is
  unknown.
- State explicitly when the index contains regulatory support but lacks enough
  clinical evidence to make an effectiveness comparison.

This may require adding broader NCCIH evidence summaries or other authoritative
clinical guidelines, not only changing the answer prompt. Retrieval evaluation
should include questions that test whether the system distinguishes regulatory
status from clinical effectiveness and consumer recommendations.

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

### Move Free, knee pain, and joint protection

Question:

> Is Move Free a good choice for a man in his 60s to relieve knee pain and
> protect his knees?

Relevant clinical evidence source:

- NCCIH, *Glucosamine and Chondroitin for Osteoarthritis: What You Need to
  Know*
- https://www.nccih.nih.gov/health/glucosamine-and-chondroitin-for-osteoarthritis-what-you-need-to-know

Observed current behavior:

- Correctly avoided claiming that Move Free was proven to relieve pain or
  prevent future knee damage.
- Distinguished osteoarthritis-related knee pain from knee pain with an
  unknown cause.
- Still opened by calling Move Free a "reasonable joint-health supplement,"
  even though the indexed excerpts did not evaluate the brand or confirm its
  exact formulation and dose.
- Treated Health Canada monograph claims as relatively positive effectiveness
  evidence instead of clearly identifying them as regulatory product-licence
  and labelling information.
- Suggested that glucosamine sulfate and chondroitin sulfate may help knee
  osteoarthritis without retrieving the broader clinical conclusion that
  studies and professional guidelines have reached inconsistent conclusions.

Desired V2 behavior:

- Do not recommend or positively characterize a branded product when its exact
  formula, ingredient forms, and doses are unknown.
- Clearly separate Health Canada permitted-use or label information from
  evidence of clinical effectiveness.
- For pain-relief and purchasing questions, retrieve an authoritative clinical
  evidence summary such as NCCIH in addition to regulatory monographs.
- State that evidence for glucosamine and chondroitin in knee osteoarthritis is
  inconsistent and that guidelines disagree.
- Do not translate cartilage-maintenance wording into proof that a supplement
  prevents future knee deterioration.
- Explain that evidence concerning knee osteoarthritis does not establish
  effectiveness for knee pain from an unknown cause.

Ingestion implication:

- Expand the NCCIH candidate source beyond botanical fact sheets to include
  condition- and ingredient-level clinical evidence summaries.
- Use this case to test whether source-role-aware retrieval prioritizes clinical
  evidence for effectiveness questions while retaining regulatory sources for
  permitted claims, doses, and warnings.

Failure tags:

- `brand_without_formula`
- `regulatory_claim_vs_effectiveness`
- `purchasing_recommendation`
- `mixed_clinical_evidence`
- `joint_protection_overstatement`
- `missing_clinical_evidence_summary`

## V2 evaluation plan

- Build a structured NIH Consumer FAQ dataset with at least the question,
  official reference answer, source URL, and topic.
- Keep this dataset separate from the Elasticsearch index while it is used as
  holdout evaluation data.
- Run a coverage audit against the current evidence index and label each FAQ:
  - `fully_answerable`: the indexed evidence covers the important conclusions
    in the official answer.
  - `partially_answerable`: some useful evidence is present, but one or more
    important conclusions or safety qualifications are missing.
  - `not_answerable`: the current index does not contain evidence supporting the
    important conclusions.
- Compare system answers with the official FAQ answers using answer correctness
  in addition to answer relevance, faithfulness, and citation correctness.
- Use the coverage audit to decide which underlying authoritative sources
  should be added to ingestion.
- If some FAQ entries are eventually ingested, create a clean split between
  indexed development examples and untouched holdout evaluation examples.
- Collect real-world questions from application feedback and a small,
  privacy-preserving sample of public consumer questions.
- Remove usernames and identifying health details; paraphrase when appropriate
  and retain only the source URL and origin type.
- Keep the current source-generated retrieval set for reproducibility.
- Add a separate real-world challenge set for Fixed RAG versus Agentic RAG.
- After adding new sources, rebuild chunks and the Elasticsearch index, then
  rerun retrieval pooling, relevance judgments, retrieval metrics, answer
  generation, and answer judgments.
