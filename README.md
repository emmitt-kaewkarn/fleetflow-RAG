# FleetFlow Agent Design

## 1. Objective
To build a conversational agent that can query a fleet management database and create/update records through a simple, interactive chat interface.

## 2. Architecture & Tech Stack

| Component | Technology/Approach |
| :--- | :--- |
| **Web Framework** | FastAPI | 
| **Agent Runtime** | LangChain | 
| **LLM Provider** | OpenRouter | 
| **Streaming** | Server-Sent Events (SSE) | 
| **Database** | MariaDB (fleetfast_dev) | 
| **Read Access** | Parameterized SQL SELECTs | 
| **Write Access** | Whitelisted Stored Procedures |
| **Testing UI** | Streamlit |
| **Environment Config** | `.env` file (hard-coded `COMPANY_ID`, `ACCOUNT_USER_ID`, and `JOB_TYPE_ID` for MVP). For write tools, configure `ACCOUNT_API_BASE`. `ACCOUNT_API_TOKEN` is optional in dev (no auth required) and used in staging/prod if enabled. Set `ACCOUNT_API_VERIFY_SSL=false` in dev if you encounter certificate issues. Optionally set `ACCOUNT_API_JOBS_PATH` (default `/jobs`; the tool will retry `/api/jobs` automatically if `/jobs` returns 405).

## 3. Agent Capabilities

### 3.1. Read Operations (SQL Querying)
The agent will answer questions by executing safe, read-only SQL queries against the database. It will focus on core tables like `fleet_jobs`, `drivers`, and `vehicles`.

### 3.2. Write Operations (Action Tools)
The agent will use a set of predefined tools to create or update records. These tools will call specific, whitelisted stored procedures, ensuring no direct database manipulation from the agent.

**Core Tools:**
- `create_job(payload)`
- `update_job_status(job_id, new_status)`
- `create_inspection(payload)`
- `log_mileage(payload)`

#### Job Creation Tool (create_job) — Summary

Purpose
- Create a FleetFlow job by POSTing to the Account API with a minimal, chat-friendly payload.

Endpoint and auth
- Base URL: `ACCOUNT_API_BASE` (default `https://dev-account.fleetflow.co`)
- Jobs path: `ACCOUNT_API_JOBS_PATH` (default `/jobs`). If the server returns HTTP 405 at `/jobs`, the tool automatically retries `/api/jobs`.
- Auth (dev): `ACCOUNT_API_TOKEN` optional; unauthenticated requests may be allowed in dev.
- TLS: set `ACCOUNT_API_VERIFY_SSL=false` in dev if you hit certificate errors.

Environment defaults (auto-filled unless provided)
- `created_by` ← `ACCOUNT_USER_ID`
- `company_id` ← `COMPANY_ID`
- `job_type_id` ← `JOB_TYPE_ID`

Minimum payload
- `service_fee` (number)
- `service_date` (YYYY-MM-DD)
- Flags (default false): `is_discounted`, `is_taxable`, `is_reserved`
- `destinations[]` (array of stops) with:
  - `address` (string)
  - `latitude` (number) and `longitude` (number) — geocoding is disabled; lat/long are required
  - `location_order` (string, e.g., `A`, `B`, `C`)

Destination rules
- Lat/long must be provided for every destination (no geocoding in this POC).
- `location_type` is required by backend; when not provided, the tool auto-fills:
  - First destination → `pick_up`
  - All subsequent destinations → `drop_off`
- `use_location_customer` must be a string per API; the tool converts booleans to `'true'`/`'false'`.

Other payload fields (optional)
- `contact_phone`, `customer_id`, `vehicle_id`, `load_type`, `load_details`, `equipment_ids`, `job_note`, etc.
- The tool ensures backend-expectation keys exist:
  - `discount_amount` defaults to `0.0` when not provided
  - `pay_with_driver` defaults to `false` when not provided

Returns
- A concise string indicating success with `ID`, `Code`, and `Status`, or a readable error containing HTTP status and validation details.

Testing
- Standalone test script: `app/agents/tools/test_job_create.py`
  - Activate venv and run: `source venv/bin/activate && python -m app.agents.tools.test_job_create`
  - Reads `.env`, sets dev-safe defaults, streams progress, and writes to `app/agents/tools/test_job_create.out.txt`.
  - Adjust the sample payload in the script as needed.

Notes
- For staging/production, ensure `ACCOUNT_API_TOKEN` is set and TLS verification is enabled (`ACCOUNT_API_VERIFY_SSL=true`).
- If your deploy exposes jobs under a different route, set `ACCOUNT_API_JOBS_PATH` accordingly (e.g., `/api/jobs`).

### 3.3. Smart Features

- **Location Services (Google Maps API)**: For this POC, geocoding is disabled. Users must provide exact latitude/longitude for each destination. In future iterations, Google Maps geocoding may be enabled to convert addresses to lat/long.

- **Interactive Input Gathering**: If a user's request is missing information, the agent will not fail. Instead, it will politely ask follow-up questions to gather all the necessary details before proceeding. It will then confirm the collected information with the user before executing the action.

## 4. API Endpoint

- **Endpoint**: `POST /chat/stream`
- **Request**:
  ```json
  { "user_message": "string", "session_id": "string" }
  ```
- **Response**: A `text/event-stream` response where the agent's messages are streamed to the client.

## 5. MVP Roadmap

1.  **Setup**: Configure the database, FastAPI, and LangChain.
2.  **Implement Read Tool**: Build the SQL query tool for answering questions.
3.  **Implement Write Tools**: Build the action tools for creating and updating records.
4.  **Build Agent**: Assemble the agent with the tools and a simple conversational flow.
5.  **Develop API**: Create the `/chat/stream` endpoint to expose the agent.