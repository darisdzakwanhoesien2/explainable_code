# Explainable Code

## Project Overview
`explainable_code` is a Streamlit-based Explainable AI application for tabular CSV data. It lets users upload a dataset, select a target column, train a model (classification or regression), inspect performance metrics, and generate SHAP-based global and local explanations.

The core problem it solves is: model outputs alone are not enough for decision-making. This project adds transparent, feature-level reasoning and portable experiment artifacts (metrics/config/report) so model behavior can be inspected and shared.

## Tech Stack
- Language: Python 3
- App framework: Streamlit
- Data handling: pandas, numpy
- ML: scikit-learn (`RandomForestClassifier`, `RandomForestRegressor`, `LogisticRegression`, `LinearRegression`, `train_test_split`)
- Explainability: SHAP
- Plotting: matplotlib
- Stats utility: scipy (for drift detection helper)

## Architecture Overview
The codebase is organized into app-level orchestration + reusable core modules.

- `app.py`: Main Streamlit UI and flow orchestration
  - Upload data
  - Profile dataset
  - Detect problem type
  - Train model
  - Compute metrics
  - Run SHAP global/local explanations
  - Save experiment outputs
- `core/shap_engine.py`
  - Builds SHAP explainer and computes SHAP values
  - Normalizes SHAP tensor shape for plotting
- `core/insight_engine.py`
  - Generates short model/SHAP textual insights
- `core/storage_engine.py`
  - Initializes storage directories
  - Saves datasets and experiment artifacts (`metrics.json`, `config.json`, `shap_summary.json`, `report.txt`)
- `core/model_trainer.py`, `core/data_processor.py`, `core/evaluation_engine.py`, `core/bias_detector.py`, `core/drift_detector.py`
  - Auxiliary utilities for training, typing, stability, bias, and drift checks
- `report/report_generator.py`
  - Text report generator utility

Data/artifact flow:
1. CSV upload -> dataframe
2. Target/feature split -> train/test split
3. Model fit/predict -> metrics
4. SHAP explainability -> plots + summary insight
5. Persist experiment outputs under `storage/experiments/<experiment_id>/`

## Installation & Setup
1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Run Streamlit.

Example:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

If `requirements.txt` is missing, install at least:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib shap scipy
```

## Usage Guide
1. Launch app: `streamlit run app.py`
2. Upload a CSV file.
3. Select a target column.
4. (Optional) If target is numeric, enable binning to convert to classes.
5. Choose model strategy:
   - `Auto Select`
   - `Random Forest`
   - `Linear/Logistic`
6. Review metrics:
   - Classification: accuracy, weighted F1
   - Regression: R2
7. Inspect SHAP:
   - Global beeswarm
   - Feature interaction scatter
   - Local waterfall explanation
8. Save experiment to generate report and download text output.

Example scenarios:
- Classification: churn prediction with categorical target labels
- Regression: house-price prediction with numeric target

## API Reference
This project currently provides a Streamlit UI, not HTTP endpoints.

Internal callable interfaces:
- `core/shap_engine.py`
  - `create_explainer(model, X_train)` -> SHAP explainer
  - `compute_shap_values(explainer, X_test, sample_size)` -> `(shap_values, X_explain)`
  - `prepare_shap_for_plot(shap_values, class_index=None)` -> shape-normalized SHAP values
- `core/storage_engine.py`
  - `init_storage()` -> ensures storage dirs exist
  - `save_dataset(df, experiment_id)` -> dataset CSV path
  - `save_experiment(experiment_id, metrics, config, shap_summary)` -> experiment directory path

## Environment Variables
No `.env` variables are required for the current implementation.

Optional conventions you may add later:
- `STORAGE_BASE_DIR` for custom artifact location
- `APP_TITLE` for custom Streamlit title

## Contributing Guide
1. Fork and create a feature branch.
2. Keep changes modular (`core/` for reusable logic, `app.py` for UI orchestration).
3. Add tests for bug fixes and non-trivial logic.
4. Run formatting/linting/tests locally before opening PR.
5. Open a PR with:
   - problem statement
   - approach
   - before/after behavior
   - validation evidence (screenshots or test output)

Suggested local checks:

```bash
python -m py_compile app.py core/*.py report/*.py explain/*.py
```

## License
No license file is currently present. Add a `LICENSE` file (for example MIT, Apache-2.0, or proprietary) before redistribution.

## Bugs Found and Fixes Applied
The following issues were identified and fixed in `app.py`:

1. Problem-type misclassification risk
- Issue: Problem type was inferred using `len(np.unique(y)) < 20`, which can misclassify numeric targets.
- Fix: Replaced with robust `sklearn.utils.multiclass.type_of_target`-based detection plus numeric fallback logic.
- Why: Prevents wrong model/metric path selection.

2. Training failure with missing values
- Issue: Models were trained directly on data that could contain `NaN` in features/target.
- Fix: Added explicit target null removal and row filtering for feature nulls before split/train.
- Why: Avoids runtime training errors and ensures consistent preprocessing.

3. SHAP interaction step could use undefined/failed explanation
- Issue: Interaction plotting could execute after SHAP failure and rely on invalid state.
- Fix: Added safe initialization and guard (`shap_values is not None`) before interaction plotting.
- Why: Prevents cascading errors and improves UX messaging.

4. Reproducibility gaps in model setup
- Issue: Random forest models lacked deterministic seeds in the app path.
- Fix: Added `random_state=42` and `n_jobs=-1` for both RF classifier/regressor.
- Why: Improves reproducibility and training performance.

5. Code redundancy and readability issues
- Issue: Large blocks of commented-out legacy code made maintenance hard.
- Fix: Rewrote `app.py` to keep only active logic and grouped helper sections.
- Why: Reduces noise, improves maintainability, and lowers risk of editing wrong sections.

6. Complex SHAP shape handling clarity
- Issue: 3D SHAP tensor handling was present but not clearly explained.
- Fix: Added focused inline comments where SHAP tensors are normalized for plotting.
- Why: Makes critical explanation logic easier to maintain and review.

## Scaling Guide

### 1. Current Bottlenecks
What is most likely to break first under load:
- Single-process Streamlit runtime: limited concurrent request handling and shared in-memory state pressure.
- CPU-heavy inference/explainability: SHAP computation is the most expensive path and can saturate CPU quickly.
- Local filesystem storage (`storage/`): artifact writes are not distributed and become fragile across replicas.
- Synchronous request flow: long-running model training/explanation blocks user interactions.
- No queueing/rate limits: burst traffic can overwhelm app workers.

### 2. Database Scaling
Current app uses filesystem; move to managed DB/object store for scale.

Recommended data architecture:
- Metadata DB: PostgreSQL (experiments, configs, user/session metadata)
- Artifact store: S3/GCS/Azure Blob (datasets, reports, large explanation outputs)
- Cache: Redis (hot experiment reads, precomputed summaries, session/state cache)

Scaling patterns:
- Indexing:
  - `experiments(experiment_id)` unique index
  - `experiments(created_at)` for recency queries
  - `(user_id, created_at)` composite index for user dashboards
- Caching:
  - Cache report payloads and SHAP summaries by experiment id
  - TTL cache for repeated identical inference/explanation requests
- Read replicas:
  - Add one read replica when dashboard/report reads dominate writes
- Sharding/partitioning:
  - Partition experiment table by date or tenant at high scale
  - Shard only when single-node limits are reached and query patterns are stable

### 3. Backend Scaling
Even with Streamlit, treat compute-heavy tasks as backend jobs.

Recommended approach:
- Horizontal scaling first:
  - Containerize app and run multiple replicas behind a load balancer.
  - Use stateless app pods; store state in Redis/DB/object storage.
- Vertical scaling second:
  - Increase CPU/RAM for SHAP/training workers when model complexity grows.
- Offload heavy tasks:
  - Move model training + SHAP generation to async workers (Celery/RQ + Redis/SQS).
  - UI submits job and polls status instead of blocking request thread.
- Load balancing:
  - Use ALB/Cloud Load Balancer/Application Gateway with health checks.
- Guardrails:
  - Add request rate limits, per-user concurrency caps, and job queue backpressure.

### 4. Frontend Scaling
For Streamlit-based UI:
- Put CDN in front of static assets and downloadable reports.
- Lazy load expensive visual sections:
  - Compute SHAP only when user opens explanation section.
  - Defer local explanation/waterfall until row selected.
- Precompute common views and cache summaries.
- SSR/SSG:
  - Not directly applicable to Streamlit pages.
  - If public marketing/docs portal is added, use Next.js with SSR/SSG and CDN caching.

### 5. Infrastructure (Cloud Recommendation)
AWS reference stack (similar mapping possible on GCP/Azure):
- Compute/UI: ECS Fargate or EKS for Streamlit containers
- Load balancing: Application Load Balancer
- Async jobs: SQS + worker service (ECS/EKS)
- Database: RDS PostgreSQL (+ read replica later)
- Cache: ElastiCache Redis
- Artifacts: S3
- Secrets: AWS Secrets Manager
- Monitoring: CloudWatch + OpenTelemetry/Grafana
- CI/CD: GitHub Actions -> ECR -> ECS/EKS deploy

Equivalent services:
- GCP: Cloud Run/GKE, Cloud SQL, Memorystore, GCS, Secret Manager, Cloud Monitoring
- Azure: Container Apps/AKS, Azure Database for PostgreSQL, Azure Cache for Redis, Blob Storage, Key Vault, Azure Monitor

### 6. Cost Estimate (Rough Monthly, USD)
Assumptions:
- Moderate SHAP usage, not all users run heavy explanations continuously.
- Includes app, DB, cache, object storage, networking, monitoring.

Estimated ranges:
- 1k users/month (MVP/early production): $120-$450
- 10k users/month (growing production): $700-$2,500
- 100k users/month (large production): $6,000-$25,000+

Major cost drivers:
- CPU time for model training + SHAP
- Data transfer and object storage volume
- Database size + read/write throughput
- Worker fleet size for background jobs

### 7. Roadmap (MVP -> Production-Grade)
1. MVP hardening
- Containerize app
- Move local storage to managed object storage + PostgreSQL metadata
- Add Redis cache

2. Reliability baseline
- Deploy 2+ app replicas behind load balancer
- Add health checks, centralized logs, basic alerts
- Add backup and retention policies

3. Performance phase
- Introduce async job queue for training/SHAP
- Add request throttling and per-user quotas
- Cache frequent reports and SHAP summaries

4. Scale-out phase
- Add DB read replica
- Split worker pools by workload type (training vs explainability)
- Add autoscaling policies for app and workers

5. Enterprise-grade phase
- Multi-AZ, disaster recovery runbook
- Tenant isolation and stricter authn/authz
- Cost observability dashboards + chargeback tags
- SLOs/SLIs with incident response process

## Competitive Landscape and Niche Strategy

### 1) Ten Similar Apps/Companies

| Company / App | What They Do | Tech Stack (Publicly Known / Likely) | Business Model | Scale | Why They Succeed | Your Niche Opportunity |
|---|---|---|---|---|---|---|
| DataRobot | Enterprise AI platform with model development, governance, and explainability | Python/Java ecosystem, cloud-native platform, enterprise integrations | Enterprise SaaS licenses and services | Large enterprise footprint | End-to-end workflow + enterprise trust/compliance | Position as lightweight, transparent, self-hostable alternative for small teams/academia |
| Dataiku | Collaborative AI/analytics platform with governance and explainability | Python + Spark/SQL integrations, multi-cloud enterprise deployment | Enterprise subscriptions | Global enterprise adoption | Strong collaboration UX across business + data teams | Focus on simpler onboarding and lower operational overhead |
| H2O.ai (Driverless AI + H2O ecosystem) | AutoML + explainable modeling for enterprise and open-source users | JVM/Python, distributed ML, open-source + enterprise products | Open-core + enterprise platform/commercial support | Large OSS + enterprise reach | Strong open-source credibility + high-performance ML tooling | Differentiate on interactive explainability UX and opinionated “best-practice defaults” |
| Fiddler AI | AI observability and explainability in production (ML + LLM) | Cloud platform, monitoring pipelines, explainability methods including SHAP-like approaches | Enterprise SaaS | Mid-to-large enterprise deployments | Strong production monitoring and governance story | Niche into pre-production experimentation + educational explainability workflows |
| Arize AI | ML/LLM observability platform for monitoring, debugging, and explainability | Cloud observability stack, telemetry pipelines, analytics tooling | SaaS (enterprise + usage-based patterns) | Broad startup-to-enterprise usage | Excellent observability depth and practitioner tooling | Focus on local/offline-first explainability for regulated or air-gapped environments |
| Evidently AI | Open-source ML/LLM evaluation and monitoring framework + commercial offering | Python OSS library, notebooks, dashboards, monitoring integrations | Open-source + commercial platform/services | Large OSS adoption; growing commercial | Developer trust through OSS + easy integration | Differentiate with full app experience (not just library) + experiment/report lifecycle |
| IBM watsonx.governance | Enterprise AI governance, risk, compliance, and explainability | IBM cloud stack, governance tooling, enterprise connectors | Enterprise software licensing/services | Very large enterprise customer base | Governance depth and procurement friendliness | Offer faster, lower-cost, domain-specific explainability for teams not needing big-suite governance |
| AWS SageMaker Clarify | Bias detection and explainability integrated into SageMaker workflows | AWS managed services stack (SageMaker, S3, IAM, etc.) | Cloud consumption pricing | Hyperscale cloud customer base | Native integration with AWS ML lifecycle | Win by being cloud-agnostic and simpler for non-AWS-heavy teams |
| Google Vertex AI Explainable AI | Feature attribution/explainability integrated into Vertex AI | GCP managed AI services, BigQuery/Vertex integrations | Cloud consumption pricing | Hyperscale cloud customer base | Tight GCP integration + managed ops | Win with better cross-cloud portability and local prototyping speed |
| Seldon (Seldon Core / MLOps) | Model deployment/monitoring/ops with explainability-related integrations | Kubernetes-native MLOps stack, open-source + enterprise | Open-core + enterprise platform/support | Significant OSS + enterprise presence | Strong Kubernetes-native production story | Focus on UX-first explainability for analysts, not only platform engineers |

### 2) What This Means for Your Project

Where your project can carve a strong niche:
- Explainability-first product, not observability add-on:
  - Keep the user journey centered on “why did this prediction happen?” rather than infra-heavy monitoring first.
- Education + governance bridge:
  - Most tools are either enterprise-heavy or library-heavy. You can occupy the middle with guided workflows and audit-friendly outputs.
- Lightweight and deploy-anywhere:
  - Streamlit + modular Python can target labs, SMEs, and internal teams needing fast setup.
- Opinionated reporting:
  - Your `save_experiment` and generated reports can become a compliance-ready artifact pipeline (versioned, reproducible, shareable).
- Domain specialization:
  - Add vertical packs (finance risk, healthcare triage, manufacturing QA) with prebuilt metrics/explanations/templates.

Suggested next differentiation steps:
1. Add model cards + decision logs per experiment.
2. Add fairness/bias and drift views directly in the UI flow.
3. Add asynchronous SHAP jobs and cached explanation store for scale.
4. Add role-based review workflow (analyst -> validator -> approver).
5. Provide one-click “regulatory report bundle” export.
