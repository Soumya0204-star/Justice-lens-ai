# JusticeLens AI
## Enterprise-Grade AI Decision Support System for Tele-Law Disparity Analysis
### IBM SkillsBuild Internship — Problem Statement 37

---

## 0. Solution Framing

**Problem restated:** Tele-Law (Department of Justice, India) connects citizens to lawyers via Common Service Centres (CSCs). Case registration volume varies wildly by district — some districts under-register relative to population, litigation load, or known legal-need indicators (poverty, literacy, gender ratio, rural share). The internship asks us to *quantify* this disparity and make it *actionable*.

**Solution concept:** JusticeLens AI is a decision-support platform (not a black-box scorer) that:
1. Ingests district-wise Tele-Law registration + advice data (FY 2021-22 to 2024-25).
2. Enriches it with demographic/regional context (Census/SECC proxies, rural-urban split, literacy, gender ratio — sourced as auxiliary open data).
3. Computes a **Legal Access Disparity Index (LADI)** per district using ML (XGBoost regression/classification).
4. Explains *why* a district scores low using SHAP (feature attribution).
5. Uses **IBM Granite** (via watsonx.ai) as a Generative Reasoning Layer to turn numeric disparity + SHAP output into a **plain-language policy brief** for DoJ/CSC administrators — this is the "AI Decision Support" part, not just a dashboard.
6. Surfaces everything through a Streamlit console with Plotly maps/charts, exportable recommendations.

This satisfies the "inclusive legal access" framing: the deliverable is not just an analytics dashboard, it's a **narrated, explainable, prioritization tool** an actual DoJ/NALSA official could act on.

---

## 1. Functional Requirements (FR)

| ID | Requirement |
|----|-------------|
| FR-1 | Ingest official district-wise Tele-Law CSV/API data across all available FYs (2021-22 → 2024-25) and normalize schema across years (column names/structure often drift year to year on data.gov.in). |
| FR-2 | Join Tele-Law data with auxiliary demographic datasets (district population, rural/urban %, literacy rate, sex ratio, SC/ST %) via a district-name reconciliation layer (fuzzy matching, since spellings differ across sources). |
| FR-3 | Compute derived indicators: cases-per-lakh-population, advice-enabled ratio, YoY growth rate, rural penetration ratio, gender parity of beneficiaries (where available). |
| FR-4 | Train a supervised ML model (XGBoost) to predict **expected** case registration given demographic baseline, and derive a **disparity residual** (actual − expected) as the core LADI score. |
| FR-5 | Classify districts into disparity tiers (Critical Underserved / Underserved / On-Track / Over-Performing) using threshold + clustering validation. |
| FR-6 | Generate SHAP-based local + global explanations: which demographic/regional factors drive a district's under-registration. |
| FR-7 | Use IBM Granite (via watsonx.ai Prompt Lab / API) to auto-generate a natural-language "District Policy Brief" from the SHAP + LADI output for any selected district or state. |
| FR-8 | Support a Q&A / chat interface ("Ask JusticeLens") where an official can ask natural-language questions ("Which districts in Bihar need urgent CSC intervention?") answered via Granite grounded on the processed dataset (RAG-lite over structured summaries). |
| FR-9 | Interactive Streamlit dashboard: choropleth map (state/district), trend lines across FYs, ranked disparity leaderboard, drill-down per district, filter by state/year/tier. |
| FR-10 | Exportable outputs: PDF/CSV policy brief per district or state, downloadable from the dashboard. |
| FR-11 | Model versioning & retraining workflow when new FY data is published. |
| FR-12 | Role-aware views (Analyst view with full ML internals vs. Policymaker view with narrative + map only) — configurable via a simple auth toggle. |
| FR-13 | Data quality report: flag districts with missing/zero/suspicious records per year. |

## 2. Non-Functional Requirements (NFR)

| Category | Requirement |
|---|---|
| **Scalability** | Handle full India district set (~766 districts × 4 FYs) with headroom for future years; pipeline should re-run in minutes, not hours. |
| **Performance** | Dashboard interactions (filter, drill-down) < 2s; Granite narrative generation < 8s per district (async/spinner in UI). |
| **Explainability** | Every ML score must be traceable to SHAP values; no unexplained "black box" number is shown to end users — core to responsible-AI requirement of a justice-sector tool. |
| **Reliability** | Graceful degradation: if watsonx.ai API is unreachable, dashboard still shows numeric/SHAP results with a fallback templated (non-LLM) narrative. |
| **Security & Privacy** | No PII — dataset is aggregate/district-level; still, IBM Cloud IAM API keys stored in secrets manager, never hard-coded. |
| **Portability** | Runs fully within IBM Cloud Lite free-tier limits (no paid compute mandatory) so it's reproducible by any intern/reviewer. |
| **Maintainability** | Modular pipeline (ingest → clean → feature → model → explain → generate → serve), each independently testable. |
| **Auditability** | Every generated policy brief logged with model version, data version, and SHAP snapshot for governance (aligns with watsonx.governance philosophy). |
| **Usability** | Non-technical policymaker should get value within 3 clicks (state → district → brief). |
| **Cost** | Zero/near-zero cost: IBM Cloud Lite services, open dataset, no proprietary paid APIs beyond watsonx.ai Lite quota. |

---

## 3. Complete System Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              PRESENTATION LAYER                            │
│   Streamlit Web App  (Analyst View | Policymaker View | Ask-JusticeLens)   │
│        Plotly Choropleth Maps · Trend Charts · Leaderboards · PDF Export    │
└───────────────────────────────────────────────────────────────────────────┘
                                   │  REST / in-process call
┌───────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION / ORCHESTRATION LAYER                  │
│  ┌───────────────┐  ┌────────────────┐  ┌───────────────────────────────┐ │
│  │ Query Service  │  │ Brief Generator │  │ Ask-JusticeLens (RAG-lite)   │ │
│  │ (filters, agg) │  │ (Granite call)  │  │ Orchestrator (Granite + ctx) │ │
│  └───────────────┘  └────────────────┘  └───────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────┘
                                   │
┌───────────────────────────────────────────────────────────────────────────┐
│                              ML / ANALYTICS LAYER                          │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────────────────┐    │
│  │ Feature Store  │→ │ XGBoost Model  │→ │ SHAP Explainer             │    │
│  │ (engineered df)│   │ (expected reg.)│   │ (global + local values)   │    │
│  └───────────────┘   └───────────────┘   └───────────────────────────┘    │
│                                   │                                        │
│                     LADI Scoring & Tier Classification                    │
└───────────────────────────────────────────────────────────────────────────┘
                                   │
┌───────────────────────────────────────────────────────────────────────────┐
│                          DATA ENGINEERING LAYER                            │
│  Ingestion (data.gov.in API/CSV) → Schema Harmonizer → Cleaner →           │
│  District Name Reconciliation (fuzzy match) → Auxiliary Data Joiner        │
│  (Census/SECC demographic proxies) → Curated Parquet/CSV Store             │
└───────────────────────────────────────────────────────────────────────────┘
                                   │
┌───────────────────────────────────────────────────────────────────────────┐
│                      IBM CLOUD LITE / watsonx.ai LAYER                     │
│  Cloud Object Storage (raw+curated data) · watsonx.ai Studio (Granite)     │
│  watsonx.ai Runtime (model serving/inference) · IAM (API key mgmt)         │
│  Cloudant/Db2-on-Cloud Lite (optional metadata store) · IBM Bob (dev-time  │
│  AI pair-programmer used during build, not a runtime component)           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Module Breakdown

1. **`data_ingestion`** — Fetches Tele-Law dataset (data.gov.in resource, CSV/API), handles year-wise schema drift, validates row counts.
2. **`data_cleaning`** — Standardizes district/state names, handles missing/zero values, type coercion, deduplication.
3. **`entity_reconciliation`** — Fuzzy-matches Tele-Law district names to a canonical district master list (LGD codes) for joining with Census/demographic data.
4. **`feature_engineering`** — Builds derived ratios (per-capita, YoY growth, rural %, advice-to-registration conversion).
5. **`model_training`** — XGBoost regressor (expected registrations) + optional classifier (tier label); train/val/test split, hyperparameter search.
6. **`explainability`** — SHAP TreeExplainer; produces global summary + per-district waterfall/force values.
7. **`ladi_scoring`** — Combines residual + rank + tier logic into the final Legal Access Disparity Index.
8. **`genai_brief_generator`** — Prompt-engineered calls to Granite (via watsonx.ai) that convert structured SHAP+LADI JSON into a natural-language district brief.
9. **`ask_justicelens`** — Lightweight RAG: retrieves relevant district/state summary rows as context, passes to Granite for Q&A.
10. **`dashboard_ui`** — Streamlit app: pages for Overview, State Drilldown, District Detail, Ask-JusticeLens chat, Data Quality report.
11. **`export_service`** — Generates downloadable PDF/CSV briefs.
12. **`config_and_secrets`** — Centralized `.env`/IBM Cloud IAM credential handling.
13. **`pipeline_orchestrator`** — Single entrypoint (`run_pipeline.py`) chaining modules 1–7 for reproducible retraining.

---

## 5. Folder Structure

```
justicelens-ai/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── settings.yaml                 # paths, model params, watsonx endpoint config
│   └── column_mappings.yaml          # per-FY schema harmonization rules
│
├── data/
│   ├── raw/                          # untouched data.gov.in pulls, per FY
│   ├── auxiliary/                    # demographic/census proxy data
│   ├── interim/                      # post-cleaning, pre-feature
│   └── processed/                    # final curated feature table (parquet/csv)
│
├── src/
│   ├── data_ingestion/
│   │   ├── fetch_telelaw_data.py
│   │   └── fetch_auxiliary_data.py
│   ├── data_cleaning/
│   │   ├── schema_harmonizer.py
│   │   └── cleaner.py
│   ├── entity_reconciliation/
│   │   └── district_matcher.py
│   ├── feature_engineering/
│   │   └── build_features.py
│   ├── modeling/
│   │   ├── train_xgboost.py
│   │   ├── evaluate.py
│   │   └── model_registry.py
│   ├── explainability/
│   │   └── shap_engine.py
│   ├── scoring/
│   │   └── ladi_index.py
│   ├── genai/
│   │   ├── granite_client.py         # watsonx.ai SDK wrapper
│   │   ├── prompt_templates/
│   │   │   ├── district_brief.jinja
│   │   │   └── qa_system_prompt.jinja
│   │   └── brief_generator.py
│   ├── rag/
│   │   ├── context_retriever.py
│   │   └── ask_justicelens.py
│   └── utils/
│       ├── logging_utils.py
│       └── ibm_cloud_utils.py
│
├── pipeline/
│   └── run_pipeline.py               # orchestrates full ETL→ML→scoring→brief cache
│
├── app/
│   ├── Home.py                       # Streamlit entrypoint
│   └── pages/
│       ├── 1_State_Overview.py
│       ├── 2_District_Drilldown.py
│       ├── 3_Ask_JusticeLens.py
│       ├── 4_Data_Quality.py
│       └── 5_Analyst_Console.py      # SHAP internals, model metrics
│
├── models/
│   └── xgboost_ladi_v{n}.json
│
├── outputs/
│   ├── briefs/                       # cached generated PDF/CSV briefs
│   └── reports/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_experiments.ipynb
│   └── 03_shap_analysis.ipynb
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_cleaning.py
│   ├── test_reconciliation.py
│   └── test_scoring.py
│
└── docs/
    ├── architecture.md
    ├── data_dictionary.md
    └── governance_log.md
```

---

## 6. Technology Justification

| Tool | Why chosen |
|---|---|
| **IBM Cloud Lite** | Mandatory; zero-cost tier sufficient for Object Storage + watsonx.ai Lite plan — keeps the project reproducible for any SkillsBuild intern without a paid account. |
| **watsonx.ai + Granite** | Enterprise-governed foundation model access; Granite models are IBM-trained, commercially safe for a government-adjacent use case, and natively integrate with watsonx Prompt Lab for iterating on the brief-generation prompt without custom hosting. |
| **XGBoost** | Tabular, mixed-type demographic + count data with non-linear interactions (rural % × literacy × case volume) — gradient boosting is the standard best-in-class choice over deep nets for this data size/shape. |
| **SHAP** | Only explainability method with strong theoretical guarantees (Shapley values) that integrates natively with tree ensembles — non-negotiable given this is a justice-access decision-support tool, where "why" matters as much as "what." |
| **Scikit-learn** | Preprocessing pipelines, train/test splitting, baseline models for sanity-checking XGBoost. |
| **Pandas** | Core data wrangling across multi-year, multi-schema CSVs. |
| **Plotly** | Interactive choropleth (India district map) and drill-down charts that Streamlit renders natively. |
| **Streamlit** | Fastest path to a functional internal decision-support UI without a separate frontend build — ideal for an internship-scoped MVP. |
| **IBM Bob** | Used as the AI pair-programmer *during development* (code generation, refactors, test scaffolding, documentation) — a build-time tool, not a runtime architectural component, matching how it's positioned in IBM's SkillsBuild/AI Builders Challenge ecosystem. |

---

## 7. ML Pipeline

```
Raw multi-FY CSVs
        │
        ▼
Schema Harmonization (column rename/standardize across FY 21-22→24-25)
        │
        ▼
Cleaning (nulls, dtype coercion, duplicate district-year rows)
        │
        ▼
District Reconciliation (fuzzy match → canonical LGD district code)
        │
        ▼
Join with Auxiliary Demographic Data (population, rural%, literacy, sex ratio)
        │
        ▼
Feature Engineering:
  - cases_per_lakh_population
  - advice_enabled_ratio
  - yoy_growth_rate
  - rural_penetration_index
  - expected_baseline_features (population, literacy, rural%, SC/ST%)
        │
        ▼
Train/Validation/Test Split (stratified by state, time-aware split across FYs)
        │
        ▼
XGBoost Regressor → predicts "expected_registrations" from demographic baseline
        │
        ▼
Residual = actual_registrations − expected_registrations
        │
        ▼
LADI Index = normalized(residual) + rank-based adjustment
        │
        ▼
Tier Classification (quantile or clustering-validated thresholds)
        │
        ▼
SHAP TreeExplainer → global feature importance + per-district local attributions
        │
        ▼
Structured JSON (district, LADI, tier, top 3 SHAP drivers) → Granite prompt
        │
        ▼
Granite (watsonx.ai) → Natural-language District Policy Brief
        │
        ▼
Cached brief + model artifacts → Streamlit dashboard
```

**Model evaluation:** RMSE/MAE on expected-registration regression; qualitative validation of tiers against known low-registration states from Tele-Law annual reports; SHAP summary plot reviewed for sanity (rural%, literacy, population density should dominate — if not, investigate leakage).

---

## 8. IBM Cloud Architecture

```
IBM Cloud Lite Account
 ├── IAM (API keys, service credentials — scoped least-privilege)
 ├── Cloud Object Storage (Lite plan)
 │      ├── bucket: telelaw-raw
 │      ├── bucket: telelaw-processed
 │      └── bucket: telelaw-model-artifacts
 ├── watsonx.ai (Lite plan)
 │      ├── Project workspace: "JusticeLens-AI"
 │      ├── Prompt Lab (Granite prompt iteration for brief + Q&A)
 │      └── watsonx.ai Runtime (foundation model inference endpoint)
 ├── (Optional) Cloudant Lite — stores cached briefs/govenance logs as JSON docs
 └── (Optional) Db2 on Cloud Lite — if a relational store is preferred over CSV/Parquet
```

Everything runs from a local/Streamlit Cloud/Colab-hosted app that calls out to:
- **COS** for reading processed data / storing model artifacts.
- **watsonx.ai Runtime REST API** for Granite inference (brief generation + Q&A).

## 9. IBM watsonx Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                     watsonx.ai Project                        │
│  ┌───────────────┐   ┌────────────────────┐  ┌──────────────┐ │
│  │ Prompt Lab     │   │ Foundation Model:   │  │ Deployment /  │ │
│  │ (dev/testing   │──▶│ Granite (e.g.       │─▶│ Inference     │ │
│  │ district_brief │   │ granite-13b/instruct│  │ Endpoint      │ │
│  │ & qa prompts)  │   │ family, per Lite    │  │ (REST)        │ │
│  └───────────────┘   │ tier availability)   │  └──────────────┘ │
│                       └────────────────────┘                    │
└───────────────────────────────────────────────────────────────┘
             ▲                                    │
             │ structured JSON (LADI, SHAP top-3) │ generated text
             │                                    ▼
   ladi_scoring / shap_engine module      genai/brief_generator.py
                                                    │
                                                    ▼
                                        Streamlit "District Brief" panel
```

- **Prompt design principle**: never let Granite invent numbers. The prompt template always injects the exact computed LADI score, tier, and SHAP driver list as structured context, and instructs Granite to *narrate, not calculate* — this keeps the generative layer factually grounded and auditable.
- **Ask-JusticeLens** uses the same pattern (RAG-lite): retrieve the relevant precomputed district/state summary rows as context text, then let Granite answer only from that injected context.

## 10. IBM Bob Integration

IBM Bob is a **development-time AI SDLC assistant**, not a production runtime dependency — it is applicable to this internship as the tool used *to build* JusticeLens AI, matching the IBM SkillsBuild/AI Builders Challenge model. Suggested use during the build phase:
- Scaffolding module boilerplate (ingestion scripts, Streamlit page templates).
- Generating unit tests for `data_cleaning` and `entity_reconciliation`.
- Auto-generating `docs/data_dictionary.md` and inline docstrings.
- Reviewing SHAP/XGBoost code for correctness before demo day.

It does **not** appear in the runtime architecture diagrams above — it's a developer productivity layer, cited here for completeness since the brief says "if applicable."

---

## 11. Data Flow Diagram

```
[data.gov.in Tele-Law Resource]      [Census/SECC Auxiliary Data]
            │                                   │
            ▼                                   ▼
     data_ingestion                     data_ingestion (aux)
            │                                   │
            └────────────────┬──────────────────┘
                              ▼
                     data_cleaning + harmonizer
                              │
                              ▼
                 entity_reconciliation (district match)
                              │
                              ▼
                     feature_engineering
                              │
                              ▼
                 processed feature table (COS / local parquet)
                              │
                 ┌────────────┼─────────────┐
                 ▼            ▼             ▼
          model_training  ladi_scoring  data_quality_report
                 │            │
                 ▼            ▼
          shap_engine ──▶ genai/brief_generator (Granite)
                              │
                              ▼
                    Streamlit Dashboard (all views)
                              │
                              ▼
                 export_service (PDF/CSV) → end user download
```

## 12. Sequence Diagram — "Generate District Brief" flow

```
User (Policymaker)   Streamlit UI      Query Service    ladi/shap module     Granite (watsonx.ai)
       │                   │                  │                 │                    │
       │ select district   │                  │                 │                    │
       ├──────────────────▶│                  │                 │                    │
       │                   │ fetch cached row │                 │                    │
       │                   ├─────────────────▶│                 │                    │
       │                   │                  │ get LADI+SHAP   │                    │
       │                   │                  ├────────────────▶│                    │
       │                   │                  │◀────────────────┤ (json: score,tier, │
       │                   │                  │                 │  top-3 drivers)    │
       │                   │ compose prompt   │                 │                    │
       │                   ├───────────────────────────────────────────────────────▶ │
       │                   │                  │                 │  generated brief   │
       │                   │◀────────────────────────────────────────────────────────┤
       │  render brief +   │                  │                 │                    │
       │  map + charts     │                  │                 │                    │
       │◀──────────────────┤                  │                 │                    │
       │ click "Export PDF"│                  │                 │                    │
       ├──────────────────▶│ export_service generates PDF, returns download link      │
       │◀──────────────────┤                                                          │
```

## 13. UML Components (high-level)

```
┌────────────────────┐      uses      ┌───────────────────────┐
│ StreamlitApp        │───────────────▶│ QueryService          │
└────────────────────┘                └───────────────────────┘
        │ uses                                   │ uses
        ▼                                        ▼
┌────────────────────┐               ┌───────────────────────┐
│ BriefGenerator      │──uses────────▶│ GraniteClient (watsonx)│
└────────────────────┘               └───────────────────────┘
        │ uses
        ▼
┌────────────────────┐      produces  ┌───────────────────────┐
│ LADIScoringEngine   │◀───────────────│ XGBoostModel           │
└────────────────────┘                └───────────────────────┘
        ▲                                        ▲
        │ consumes                               │ trained on
┌────────────────────┐               ┌───────────────────────┐
│ SHAPExplainer       │               │ FeatureStore           │
└────────────────────┘               └───────────────────────┘
                                               ▲
                                               │ built from
                                    ┌───────────────────────┐
                                    │ DataReconciliation     │
                                    └───────────────────────┘
                                               ▲
                                               │ cleans
                                    ┌───────────────────────┐
                                    │ DataIngestionService   │
                                    └───────────────────────┘
```

## 14. Database Design

Given aggregate district-level data with modest volume (~766 districts × ~4 years ≈ 3,000 rows), a full RDBMS is optional; CSV/Parquet on COS is sufficient for MVP. If a lightweight relational/document store is desired (e.g., to cache Granite briefs and support the "Ask JusticeLens" retrieval faster), a simple schema:

**Table: `district_year_metrics`**
| Column | Type | Notes |
|---|---|---|
| district_id (LGD code) | VARCHAR PK | canonical district identifier |
| district_name | VARCHAR | |
| state_name | VARCHAR | |
| fiscal_year | VARCHAR | e.g. "2023-24" |
| cases_registered | INT | |
| advice_enabled_count | INT | |
| population | BIGINT | from auxiliary data |
| rural_pct | FLOAT | |
| literacy_rate | FLOAT | |
| sex_ratio | FLOAT | |
| cases_per_lakh | FLOAT | derived |
| expected_registrations | FLOAT | model output |
| ladi_score | FLOAT | derived |
| tier | VARCHAR | Critical/Underserved/On-Track/Over-performing |

**Table: `shap_explanations`**
| Column | Type |
|---|---|
| district_id | VARCHAR FK |
| fiscal_year | VARCHAR |
| feature_name | VARCHAR |
| shap_value | FLOAT |
| rank | INT |

**Table: `generated_briefs`** (governance/audit trail)
| Column | Type |
|---|---|
| brief_id | UUID PK |
| district_id | VARCHAR FK |
| model_version | VARCHAR |
| data_version | VARCHAR |
| prompt_version | VARCHAR |
| generated_text | TEXT |
| generated_at | TIMESTAMP |

If implemented: **Cloudant (JSON document store, IBM Cloud Lite)** is the natural fit over Db2, since the data is semi-structured and read-heavy for dashboard caching.

---

## 15. Complete Development Roadmap

**Phase 0 — Setup (Days 1-2)**
- IBM Cloud Lite account, COS bucket creation, watsonx.ai project + Granite access provisioning.
- Repo scaffolding, `requirements.txt`, `.env` structure.

**Phase 1 — Data Foundation (Days 3-6)**
- Ingest all available FY datasets from data.gov.in resource.
- Build schema harmonizer (years likely differ in column names/structure).
- Source auxiliary demographic data (Census 2011 district tables / SECC / RBI district handbook as proxy — most recent official public data).
- Build district-name reconciliation (fuzzy match + manual override list for known mismatches).

**Phase 2 — Feature Engineering & EDA (Days 7-9)**
- Derived ratios, notebook-based EDA, correlation checks.
- Data quality report module (flag zero/missing districts).

**Phase 3 — Modeling (Days 10-13)**
- Baseline (linear/scikit-learn) → XGBoost regressor for expected registrations.
- Hyperparameter tuning, cross-validation, residual analysis → LADI index formulation.
- Tier classification thresholds validated against domain intuition.

**Phase 4 — Explainability (Days 14-15)**
- SHAP TreeExplainer integration, global summary plot, per-district local explanation function.

**Phase 5 — watsonx.ai/Granite Integration (Days 16-19)**
- Provision Granite model access in watsonx.ai Prompt Lab.
- Design and iterate `district_brief` and `qa_system_prompt` templates.
- Build `granite_client.py` SDK wrapper; implement grounding guardrail (inject numbers, forbid invention).
- Build Ask-JusticeLens RAG-lite retrieval + Q&A flow.

**Phase 6 — Dashboard (Days 20-24)**
- Streamlit multi-page app: Overview, State Drilldown, District Detail, Ask-JusticeLens, Data Quality, Analyst Console.
- Plotly choropleth + trend charts + leaderboard.
- Export service (PDF/CSV brief download).

**Phase 7 — Governance, Testing & Docs (Days 25-27)**
- Unit tests across ingestion/cleaning/reconciliation/scoring.
- Governance log (model version, data version, prompt version tracking) per generated brief.
- `docs/architecture.md`, `data_dictionary.md` finalization.

**Phase 8 — Demo Prep & Submission (Days 28-30)**
- End-to-end dry run on fresh IBM Cloud Lite account (reproducibility check).
- Record demo walkthrough, prepare submission deck mapping architecture back to Problem Statement 37 requirements.

---

### Key Design Principle Recap
JusticeLens AI treats watsonx/Granite strictly as a **narration and Q&A layer over deterministic, auditable ML output** — the numeric disparity scoring (XGBoost + SHAP) is the trustworthy analytical core, and generative AI is used only to make that analysis *accessible*, never to replace it. This is the architecture pattern IBM itself recommends for regulated/public-sector decision support (pairing predictive ML with governed generative narration), and it should score well against both the technical rubric and the "inclusive legal access" framing of the problem statement.
