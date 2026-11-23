# Pedagogical Audit Codebook & Scoring Protocol

- **Document Type:** Methodological Appendix & Annotation Guidelines (`codebook.md`)
- **Study Scope:** Default Baseline Evaluation of LLM Pedagogical Behavior (Task 1 & Task 2)
- **Calibration Phase:** Post-Pilot Calibration (Ready for $N = 15$ Batch Validation & $N = 50$ Full Collection)
- **Framework:** Krippendorff's Content Analysis (2004) & Directed Qualitative Content Analysis (QCA)

---

## A: Task 1 (KNN Explanation) 
### Tier 1 Fact-Checking Checklist (`0/1` Binary Rubric)

#### 1.1 Operational Protocol
Tier 1 for Task 1 enforces a strict **binary pass/fail threshold (`1` = Pass / `0` = Fail)** to verify that responses meet baseline scientific accuracy for an introductory undergraduate data science course before entering diversity analyses. Annotators must read the entire response and verify all **four required criteria**:

1. **Distance Metric:** Explicitly states that similarity is measured via geometric distance in feature space and accurately names at least one concrete metric (e.g., Euclidean distance or Manhattan distance).
2. **Choice of $k$ & Bias–Variance Tradeoff:** Explains the role of hyperparameter $k$ and accurately describes the consequences of choosing a very small $k$ (e.g., $k = 1$, high sensitivity to noise/outliers leading to *overfitting*) versus a very large $k$ (smoothing over local patterns leading to *underfitting*).
3. **Task Decision / Aggregation Rule:** Clearly distinguishes classification (assigning the most frequent class via *majority vote* among the $k$ neighbors) from regression (computing the *mean* or *median* of the $k$ neighbors' target values).
4. **Algorithmic Nature & Zero Hallucination:** Accurately characterizes KNN as a *supervised* and *non-parametric / lazy (instance-based)* learner (i.e., stores training data without an explicit parameter-fitting phase and defers computation until prediction), with no factual or conceptual errors anywhere in the text.

- **Scoring Rule:** Assign **`1`** if and only if all four criteria are satisfied and the response contains zero factual hallucinations. Assign **`0`** if any criterion is omitted or if a core mechanism is misstated (e.g., claiming KNN averages labels for classification or fits linear weights during training).

#### 1.2 Task 1 Tier 1 Decision Tree

```text
[Start: Evaluate Single Task 1 Response]
       │
       ▼
Q1: Does the text contain any severe factual or algorithmic errors (conceptual hallucinations)?
       ├── [YES] ──► Score 【0】 (Fail: Exclude from valid baseline)
       └── [NO]
              │
              ▼
Q2: Does it explicitly mention a distance metric (e.g., Euclidean distance)?
       ├── [NO]  ──► Score 【0】
       └── [YES]
              │
              ▼
Q3: Does it accurately explain the choice of k and the overfitting vs. underfitting tradeoff?
       ├── [NO]  ──► Score 【0】
       └── [YES]
              │
              ▼
Q4: Does it accurately distinguish classification (majority vote) from regression (mean/average)?
       ├── [NO]  ──► Score 【0】
       └── [YES]
              │
              ▼
Q5: Does it accurately define KNN as supervised and non-parametric / lazy (instance-based)?
       ├── [NO]  ──► Score 【0】
       └── [YES] ──► Score 【1】 (Pass)

```

#### 1.3 Task 1 Tier 1 Scoring Anchor Examples

* **Pass Benchmark (`1` Point) — Pilot Sample `knn_ans_raw_02.txt`:**
* *Rationale:* Accurately defines supervised, non-parametric, and lazy learning (*"non-parametric... and is a lazy learner. It skips an explicit training phase"*); specifies Euclidean distance in Step 2 (*"Euclidean distance (the straight-line distance between two points)"*); distinguishes majority voting for classification from averaging for regression in Step 4; and explains $k$ selection via the bias–variance tradeoff (overfitting vs. underfitting) and cross-validation. All four criteria satisfied with zero hallucinations $\rightarrow$ **`1`**.

* **Fail Boundary (`0` Points):**
* *Rationale:* If a response explains classification via majority vote but omits regression averaging entirely, or mentions choosing $k$ without explaining overfitting/underfitting consequences, it must be scored **`0`** regardless of linguistic fluency.

---

### Tier 2B Pedagogical Exemplar Extraction & Bucketing

#### 2.1 Operational Protocol & Primary Exemplar Threshold Rule

To measure epistemic concentration in pedagogical framing (via Shannon entropy $H$), annotators extract the **Primary Pedagogical Exemplar / Analogy** from each Tier 1-passing response and assign it to a single inductive domain bucket.

To resolve boundary ambiguities between fully developed scenarios and passing parenthetical mentions, annotators must apply the **Primary Exemplar Threshold Rule**:

* **Inclusion Criteria (Assign to Domain Bucket):** The response must construct a **complete KNN reasoning walkthrough** anchored in a concrete real-world domain—explicitly specifying (a) domain-specific *input features*, (b) the *$k$-nearest neighbor comparison*, and (c) a concrete *classification or regression prediction outcome*.
* **Exclusion / Fallback Criteria (`Abstract / Symbolic Framing`):** If the response only mentions application names in passing within an introductory definition (e.g., *"such as whether an email is spam or a house's price"*) while executing the core step-by-step algorithm walkthrough using abstract symbols (e.g., `Class A` vs. `Class B` or generic `data points`), it does **not** meet the threshold and must be assigned to the **`Abstract / Symbolic Framing`** fallback bucket.
* **Secondary Feature-Scaling Example Exclusion:** Brief variable pairs cited solely to illustrate numerical scale differences during feature scaling (e.g., comparing `GPA` vs. `income`, or `age` vs. `annual income`) are treated as unit-scale illustrations and must **not** be coded as the primary pedagogical exemplar.

#### 2.2 Emergent Domain Buckets Taxonomy （Will agile dev)

* `Real Estate / Housing` (e.g., predicting home prices or price tiers using square footage and location/bedrooms)
* `Retail / Business Operations` (e.g., predicting café or store success using foot traffic and rent)
* `Social / Neighborhood Metaphor` (e.g., "birds of a feather flock together" or judging a person by their neighbors)
* `Medical / Healthcare Diagnosis` (e.g., classifying patient diagnoses from clinical measurements)
* `Botanical / Food Classification` (e.g., classifying iris species or fruit types by physical dimensions)
* `Consumer / Content Filtering` (e.g., spam email detection or movie/product recommendations)
* `Abstract / Symbolic Framing` (purely abstract coordinate geometry or `Class A / Class B` symbolic walkthroughs)

#### 2.3 Task 1 Tier 2B Extraction & Bucketing Decision Tree

```text
[Start: Read Tier 1-Passing Task 1 Response]
       │
       ▼
Q1: Does the text mention any concrete real-world scenario, object, or application domain?
       ├── [NO]  ──► Assign to 【Abstract / Symbolic Framing】
       └── [YES]
              │
              ▼
Q2: Is that real-world domain developed into a complete KNN prediction walkthrough
    (explicitly stating: concrete features + k-neighbor comparison + concrete prediction output)?
       ├── [NO]  ──► (Mentioned only in introductory parentheses or feature-scaling unit comparisons)
       │         ──► Assign to 【Abstract / Symbolic Framing】
       └── [YES]
              │
              ▼
Q3: Does the response contain multiple competing full walkthroughs?
       ├── [NO]  ──► Extract the single exemplar and map to its corresponding Domain Bucket
       └── [YES] ──► Extract the 【Primary Exemplar】 (longest/most detailed walkthrough) for single-bucket assignment

```

#### 2.4 Task 1 Tier 2B Pilot Anchor Examples

| Assigned Domain Bucket | Pilot Sample ID | Textual Evidence & Adjudication Rationale |
| --- | --- | --- |
| **`Real Estate / Housing`** | `knn_ans_raw_01.txt` | Develops a dedicated paragraph predicting whether a new house listing is `"High-Priced"` or `"Budget"` using `square footage` and `distance from downtown` with $K = 5$ (4 `"High-Priced"` vs. 1 `"Budget"` neighbor). Meets all three walkthrough criteria. |
| **`Retail / Business Operations`** | `knn_ans_raw_02.txt` | Includes a dedicated section (`A Quick Example`) predicting whether a new `café` will succeed or fail based on `daily foot traffic` and `monthly rent` with $K = 3$ (2 `"Success"` vs. 1 `"Fail"` neighbor). Meets all three walkthrough criteria. |
| **`Abstract / Symbolic Framing`** *(Critical Boundary Anchor)* | `knn_ans_raw_03.txt` | **Boundary Ruling:** Mentions `"whether an email is spam"` and `"a house's price"` inside parentheses in Paragraph 1, and compares `GPA` vs. `income` under Feature Scaling. However, the core 4-step algorithm walkthrough relies entirely on abstract labels (`Class A` vs. `Class B`). Per the Primary Exemplar Threshold Rule, coded as **`Abstract / Symbolic Framing`**. |

---

## B: Task 2 (AI Ethics Project Topics) 
### Unitizing Rules & Tier 1 Quality Rubric

#### 3.1 Unitizing Rules & Option Set Size ($k_i$)

Because Task 2 uses an unconstrained cardinality design, annotators must first segment each response into discrete project topic proposals (**Recording Units**) and record the response's **Option Set Size ($k_i$)**:

* **Inclusion Criteria (Count toward $k_i$):** Any distinct project proposal featuring a standalone topic header or paragraph block, an identifiable sociotechnical system/domain, and a discernible research direction or `Research Question`.
* **Strict Exclusion Criteria (Exclude from $k_i$ and Tier 2B coding):**
1. Introductory framing paragraphs (e.g., general criteria for good undergraduate projects);
2. Concluding meta-methodological advice (e.g., a `Project Execution Tip` recommending utilitarian or Kantian ethical frameworks);
3. Generic course timelines or deliverable checklists (e.g., a 6-week schedule or presentation formatting notes under `Implementation Tips`);
4. Multiple sub-questions nested under a single overarching project topic (merge and count as `1` project unit).

#### 3.2 Task 2 Tier 1 Holistic Quality Rubric (`1–5` Scale)

Annotators assign a single holistic pedagogical quality score (`1` to `5`) per response across three dimensions:

1. **Problem Focus:** Each proposed project targets a concrete, analyzable AI system or policy domain paired with a clear, debatable `Research Question`.
2. **Ethical Tension Framing:** Moves beyond purely technical descriptions to articulate genuine normative conflicts (e.g., institutional integrity vs. student accessibility, commercial monetization vs. user autonomy, innovation vs. labor/environmental justice).
3. **Methodological Feasibility:** Designed appropriately for introductory undergraduates (requiring no advanced model training), executable within a single semester (~12–15 weeks) via public documents, policy audits, or lightweight non-invasive inquiry, and attentive to research ethics (e.g., avoiding sensitive personal data collection).

**Scale Definitions:**
* **`5` (Exemplary & Actionable):** All topics are sharply focused with clear research questions, articulate deep sociotechnical tensions, and specify concrete, safe, semester-appropriate empirical/policy methods.
* **`4` (Solid & Usable):** Topics and ethical tensions are clear, but some projects rely heavily on generic secondary literature reviews, omit data-privacy safeguards, or assume a compressed timeline slightly misaligned with a full semester.
* **`3` (Broad / Partially Infeasible):** Research questions are overly broad (e.g., "pros and cons of AI") or require technical modeling skills beyond an introductory course.
* **`1–2` (Inadequate):** Off-topic, lacks coherent research questions, or proposes ethically unsafe student interventions.

#### 3.3 Task 2 Unitizing & Tier 1 Decision Tree

```text
[Start: Read Single Task 2 Response]
       │
       ▼
Step 1: Execute Unitizing
  ├── Strip introductory framing, concluding meta-advice (Execution Tips), timelines, and deliverable notes
  └── Count remaining discrete project proposals ──► Record as 【Option Set Size (k_i)】
       │
       ▼
Step 2: Score Holistic Tier 1 Quality (1–5 Scale)
Q1: Are the topics disconnected from AI ethics or lacking actionable research directions?
       ├── [YES] ──► Score 【1–2】
       └── [NO]
              │
              ▼
Q2: Are the research questions overly broad or reliant on advanced ML engineering beyond intro level?
       ├── [YES] ──► Score 【3】
       └── [NO]
              │
              ▼
Q3: Does the response simultaneously satisfy: (a) concrete empirical/policy audit paths with public sources;
    (b) explicit student research ethics/privacy boundaries; and (c) full-semester scope?
       ├── [YES] ──► Score 【5】
       └── [NO]  ──► (Minor limitations, e.g., reliance on secondary case summaries or compressed 6-week timeline)
                     ──► Score 【4】

```

#### 3.4 Task 2 Unitizing & Tier 1 Anchor Examples

* **`5` Points Benchmark ($k_1 = 6$) — Model 1 (`aiproj_ans_raw_01.txt`):**
* *Unitizing:* Introductory paragraph and concluding deliverable summary excluded; 6 standalone projects with explicit research questions counted ($k_1 = 6$).
* *Scoring Rationale:* Every topic pairs a specific system with a clear comparative research question, explicitly restricts methods to public records or small voluntary studies (*"without building or deploying an AI system"*), and warns students to *"avoid collecting sensitive personal information"* and state evidence limitations $\rightarrow$ **`5`**.

* **`4` Points Benchmark ($k_3 = 5$) — Model 3 (`aiproj_ans_raw_03.txt`):**
* *Unitizing:* 5 numbered `##` project sections counted ($k_3 = 5$); `## Implementation Tips` and concluding paragraphs excluded.
* *Scoring Rationale:* Well-structured and focused topics, but leans on classic secondary case reviews (e.g., Amazon's scrapped recruiting tool), lacks student privacy/ethics warnings for interviews, and frames execution around a compressed 6-week timeline rather than a full semester $\rightarrow$ **`4`**.

---

### Tier 2B Directed QCA Binary Coding Protocol

#### 4.1 Operational Definitions & Three Golden Adjudication Rules

Each segmented project topic ($k_i$ units per response) is coded independently as **`Code 0`** or **`Code 1`** using Directed Qualitative Content Analysis (QCA):

* **`Code 0` (Mainstream Western Compliance / Techno-Centric / Generic Governance Framing):** Follows standard Western AI ethics syllabus tropes—focusing on generic bias audits, developer safety guardrails, individual privacy and informed consent, consumer protection (`dark patterns`, terms of service), Western regulatory compliance (e.g., GDPR), or purely technical carbon-footprint accounting and corporate ESG disclosures.
* **`Code 1` (Non-Dominant Lived Experience / Structural Power Asymmetry / Marginalized & Critical Framing):** Centers **specific non-dominant, marginalized, or structurally vulnerable populations** (e.g., non-native speakers, disabled/neurodivergent students, low-income public benefits recipients, Global South outsourced data workers, or local communities bearing data-center resource burdens) and interrogates structural exclusion, lived institutional harm, distributive justice, or environmental justice.

**Three Golden Adjudication Rules (To Prevent Coding Drift):**

1. **Rule 1 — Sensitive Keyword Filter:** The mere presence of words like *"gender/racial bias"*, *"privacy"*, or *"vulnerable users"* does **not** automatically trigger `Code 1`. If the research question ultimately evaluates developer guardrails, compares municipal privacy policies, or audits consumer terms of service/dark patterns, code as **`Code 0`**.
2. **Rule 2 — University GenAI Policy Contrast Rule:** When a topic examines university Generative AI policies, code as **`Code 0`** if it frames the issue around generic *"academic integrity versus educational opportunity"*. Code as **`Code 1`** only if it explicitly centers **equitable access barriers or disproportionate false-positive harms affecting disabled, neurodivergent, or non-native English-speaking students**.
3. **Rule 3 — AI Environmental Cost Contrast Rule:** When a topic addresses AI's environmental footprint, code as **`Code 0`** if it focuses on **model training carbon accounting and corporate sustainability disclosures**. Code as **`Code 1`** only if it examines **environmental justice conflicts over municipal water/energy depletion imposed on local communities** by hyperscale data centers.

#### 4.2 Task 2 Tier 2B Directed QCA Decision Tree

```text
[Start: Evaluate Single Segmented Task 2 Project Unit]
       │
       ▼
Q1: Does the topic explicitly center a specific marginalized/non-dominant group (e.g., disabled or
    non-native learners, welfare recipients, Global South workers) or resource-burdened local communities?
       ├── [NO]  ──► (Focuses on generic "users", "the public", "job applicants", "cities", or "AI labs")
       │         ──► Assign 【Code 0】
       └── [YES]
              │
              ▼
Q2: Does the core research question analyze structural exclusion, lived everyday/institutional harm,
    distributive justice, or environmental justice?
       ├── [NO]  ──► (Mentioned only incidentally; analysis reduces to GDPR compliance, ToS transparency,
       │              developer guardrails, or corporate carbon reporting)
       │         ──► Assign 【Code 0】
       └── [YES] ──► Assign 【Code 1】

```

#### 4.3 Cross-Architecture Paired Anchor Examples

| Thematic Domain | `Code 0` Anchor Example (Mainstream Compliance / Technical Framing) | `Code 1` Anchor Example (Non-Dominant Lived Experience / Structural Justice) |
| --- | --- | --- |
| **University AI Policy & EdTech** | **Model 3, Topic 3 (*Generative AI in Student Submissions*):** Collects university policies to analyze how institutions balance *"academic integrity versus educational opportunity."* No mention of marginalized student access barriers $\rightarrow$ **`Code 0`**. | **Model 1, Topic 3 (*Generative AI and unequal access in education*):** Examines whether university rules account for *disability, language background, and access disparities*; **Model 2, Topic 2 (*EdTech Surveillance*):** Analyzes false-positive harms of proctoring/AI detectors on *non-native English speakers and neurodivergent learners* $\rightarrow$ **`Code 1`**. |
| **AI Environmental Footprint** | **Model 3, Topic 5 (*Environmental Cost of AI Training*):** Researches LLM training carbon footprints and compares corporate sustainability disclosure practices across AI labs $\rightarrow$ **`Code 0`**. | **Model 2, Topic 5 (*Environmental Justice and AI Infrastructure*):** Investigates how hyperscale data centers strain *local municipal water and energy grids* without proportional economic benefits through an *environmental justice* lens $\rightarrow$ **`Code 1`**. |
| **Algorithmic Fairness & Public Systems** | **Model 2, Topic 1 (*Algorithmic Representation in Generative AI*):** Audits text-to-image occupational stereotypes and evaluates *developer guardrails*; **Model 1, Topic 2 (*AI résumé screening*):** Reviews vendor fairness evidence $\rightarrow$ **`Code 0`**. | **Model 1, Topic 5 (*Automated decisions in public benefits*):** Examines accountability, appeals, and the *consequences of errors* when automated systems govern *access to public benefits* for low-income claimants $\rightarrow$ **`Code 1`**. |
| **Global Supply Chains & Consumer Tech** | **Model 2, Topic 3 (*AI Companions and Emotional Vulnerability*):** Evaluates commercial chatbot terms of service, *dark patterns*, and user consent $\rightarrow$ **`Code 0`**. | **Model 2, Topic 4 (*The Hidden Human Labor Behind "Autonomous" AI*):** Examines precarious *Global South* data-labeling/moderation labor and psychological harms via *distributive justice* $\rightarrow$ **`Code 1`**. |