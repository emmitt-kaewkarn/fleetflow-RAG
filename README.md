# FleetFlow Agent (POC)

## 1. Overview
FleetFlow Agent is a conversational system that can query fleet data (read-only SQL) and perform operational actions (e.g., create a job) through a simple chat interface. This POC focuses on a Streamlit UI with a LangChain-based agent and a small set of tools.

## 2. Current Tech Stack

| Component | Technology/Approach |
| :--- | :--- |
| UI | Streamlit (app/ui/streamlit_app.py) |
| Agent Runtime | LangChain + LangGraph-style streaming |
| LLM Provider | OpenAI via OPENAI_API_KEY; OpenRouter supported via OPENROUTER_API_KEY (auto-mapped) |
| Streaming | In-app streaming via agent.stream (updates/messages modes) |
| Database | MariaDB, accessed via SQLAlchemy + PyMySQL |
| SQL Access | Safe, parameterized SELECTs (read-only) |
| Write Actions | HTTP API (Account API) via httpx |
| Config | .env file for DB, LLM, and API settings |
| Dev Helpers | Makefile (macOS/Linux) and run.bat (Windows) |

Notes
- There is no FastAPI or SSE endpoint in this POC. All interaction is through Streamlit.
- The agent supports streaming updates and token/messages within the Streamlit app.

## 3. Agent Capabilities

### 3.1 Read Operations (SQL Querying)
- Safe, read-only queries against core tables using the SQL tool in `app/agents/tools/sql_query.py`.
- The agent includes a compact schema hint parsed from `FF_SCHEMA.txt` to keep queries grounded.

### 3.2 Write Operations (Action Tools)
- Job Creation Tool: `create_job` in `app/agents/tools/job_create.py`.
  - Purpose: Create a job via FleetFlow Account API with a minimal, chat-friendly payload.
  - Endpoint and auth:
    - Base URL: `ACCOUNT_API_BASE` (default `https://dev-account.fleetflow.co`)
    - Jobs path: `ACCOUNT_API_JOBS_PATH` (default `/jobs`; retries `/api/jobs` on HTTP 405)
    - Auth: `ACCOUNT_API_TOKEN` optional in dev; enable for staging/prod.
    - TLS: Set `ACCOUNT_API_VERIFY_SSL=false` in dev if you encounter certificate issues.
  - Environment defaults (auto-filled unless provided):
    - `created_by` ← `ACCOUNT_USER_ID`
    - `company_id` ← `COMPANY_ID`
    - `job_type_id` ← `JOB_TYPE_ID`
  - Minimum payload:
    - `service_fee` (number)
    - `service_date` (YYYY-MM-DD)
    - Flags: `is_discounted`, `is_taxable`, `is_reserved` (default false)
    - `destinations[]` with `address`, `latitude`, `longitude`, `location_order` (A, B, C)
  - Destination rules (current behavior):
    - Lat/long required for every destination (geocoding disabled in this POC).
    - `location_type` auto-filled when missing: first = `pick_up`, others = `drop_off`.
    - `use_job_customer` is enforced `true` for every destination.
    - The agent applies the same `contact_name` and `contact_phone` to every destination (collected from the user).
    - `use_location_customer` booleans are converted to strings `'true'`/`'false'`.
    - `customer_id` is no longer sent in the POST payload (kept in input schema for user-facing prompts when needed).
  - Other payload defaults:
    - `discount_amount` defaults to `0.0` when not provided.
    - `pay_with_driver` defaults to `false` when not provided.
  - Returns: A concise string indicating success (`ID`, `Code`, `Status`) or a readable error with HTTP status and validation details.

## 4. Environment Configuration (.env)

Common variables:
- LLM
  - `OPENAI_API_KEY` (primary)
  - `OPENAI_BASE_URL` optional
  - `OPENROUTER_API_KEY` (auto-maps to OpenAI client if `OPENAI_API_KEY` is not set)
  - `LLM_MODEL` (default `gpt-4o-mini`)
  - `SUMMARY_MODEL` (default `gpt-4o-mini`)
- Database
  - `DATABASE_URL` (e.g., `mariadb://user:pass@localhost:3306/fleetfast_dev`)
- Account API (job creation)
  - `ACCOUNT_API_BASE` (default `https://dev-account.fleetflow.co`)
  - `ACCOUNT_API_TOKEN` (optional in dev)
  - `ACCOUNT_API_VERIFY_SSL` (`true`/`false`; set `false` in dev if needed)
  - `ACCOUNT_API_JOBS_PATH` (default `/jobs`)
  - `COMPANY_ID`, `ACCOUNT_USER_ID`, `JOB_TYPE_ID`
- Prompt
  - `SYSTEM_PROMPT` if you want to override; otherwise `SYSTEM_PROMPT.txt` is loaded.

## 5. Running Locally

### macOS/Linux (Makefile)
- Create venv and install deps: `make setup`
- Install/update deps: `make install`
- Run Streamlit app: `make run`
- Clean venv and cache: `make clean`

Manual commands
- `python3 -m venv venv && source venv/bin/activate`
- `pip install -r requirements.txt`
- `streamlit run app/ui/streamlit_app.py --server.headless true --server.port 8501`

## 6. Streaming Modes
- In the Streamlit UI, choose between:
  - `updates`: shows agent steps (model/tool) without revealing hidden chain-of-thought.
  - `messages`: token-level/message streaming when supported by the model/provider.

## 7. Roadmap / Future Work
- Optional FastAPI service with `/chat/stream` endpoint (SSE) for integration.
- Additional write tools (e.g., update_job_status, inspections) once backend contracts are finalized.
- Enable geocoding in future (Google Maps API) to convert addresses to lat/long.