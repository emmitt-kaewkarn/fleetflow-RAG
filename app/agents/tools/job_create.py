"""
FleetFlow Job Creation Tool (POC)

Provides a LangChain tool to create a fleet job via the FleetFlow Account API.
This tool is designed for chat UX: it auto-fills required IDs from environment,
requires exact latitude/longitude for destinations (geocoding disabled), and returns concise success/error summaries.

Environment variables:
- ACCOUNT_API_BASE: Base URL for the Account API (default: https://dev-account.fleetflow.co)
- ACCOUNT_API_TOKEN: Bearer token for authenticated requests (required for production)
- COMPANY_ID: Default company_id (MVP)
- ACCOUNT_USER_ID: Default created_by (MVP)
- JOB_TYPE_ID: Default job_type_id for new jobs (MVP)
- GOOGLE_MAPS_API_KEY: Optional key for geocoding destination addresses (disabled in POC)
- ACCOUNT_API_VERIFY_SSL: Set to 'false' in dev to disable TLS certificate verification
- ACCOUNT_API_JOBS_PATH: Jobs creation path (default '/jobs')

Dependencies: pydantic, httpx (add to requirements.txt), langchain
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, List, Optional, Literal, Dict

try:
    import httpx  # type: ignore
except Exception:
    # If httpx is not installed, we'll return a clear error at runtime
    httpx = None

from pydantic import BaseModel, Field  # type: ignore

try:
    from langchain.tools import tool  # type: ignore
except Exception:
    # Fallback: allow import without langchain present; actual agent use will require it
    def tool(*args, **kwargs):  # type: ignore
        def _decorator(fn):
            return fn
        return _decorator


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class DestinationInput(BaseModel):
    """A single stop in the job route.

    Important:
    - Lat/long are required for each destination (geocoding is disabled in this POC).
    - If `location_type` is not provided, the tool auto-fills: first stop=`pick_up`, others=`drop_off`.
    - `use_location_customer` must be a string per API; the tool converts booleans to 'true'/'false'.
    """
    address: str  # Street address or place name
    latitude: Optional[float] = Field(default=None, description="Latitude of destination")
    longitude: Optional[float] = Field(default=None, description="Longitude of destination")
    location_type: Optional[str] = Field(default=None, description="ENUM('pick_up','drop_off'). Auto-filled when missing.")
    customer_id: Optional[str] = Field(default=None, description="Customer ID associated with this destination")
    use_job_customer: bool = Field(default=True, description="Use the job's customer for this destination (default: true)")
    is_anonymous: bool = Field(default=False, description="Mark destination as anonymous (default: false)")
    use_location_customer: bool = Field(default=False, description="Use location-specific customer (default: false)")
    contact_name: Optional[str] = Field(default=None, description="Contact name at destination (ask user)")
    contact_phone: Optional[str] = Field(default=None, description="Contact phone at destination (ask user)")
    location_order: str  # Sequence order of destination in the route (A, B, C)
    note: Optional[str] = Field(default=None, description="Notes for this destination (default: none)")
    no_expected_arrival: bool = Field(default=True, description="No expected arrival is provided (default: true)")
    expected_arrival_at: Optional[str] = Field(default=None, description="Expected arrival (Y-m-d H:i)")


class JobCreateInput(BaseModel):
    """Input schema for creating a fleet job via Account API.

    Server expects at minimum:
    - service_fee, is_discounted, is_taxable, is_reserved, service_date
    Tool behavior:
    - Auto-fills created_by, company_id, and job_type_id from environment when not provided
    - Ensures `discount_amount` and `pay_with_driver` keys exist to avoid backend 'undefined key' errors
    - Requires lat/long for each destination (no geocoding)
    """

    # Auto-filled unless provided
    created_by: Optional[str] = Field(default=None, description="Account user ID who initiates the job (defaults from env)")
    company_id: Optional[str] = Field(default=None, description="Company ID (defaults from env)")

    # Core required fields
    # job_type_id defaults from environment (JOB_TYPE_ID) if not provided
    job_type_id: Optional[str] = Field(default=None, description="Job type ID (defaults from env JOB_TYPE_ID)")
    service_fee: float  # Base service fee
    is_discounted: bool = False  # Default: no discount
    discount_type: Optional[Literal['fixed', 'percentage']] = Field(default=None, description="Discount type")
    discount_amount: Optional[float] = Field(default=None, description="Discount amount")
    is_taxable: bool = False  # Default: no tax
    tax_rate: Optional[float] = Field(default=None, description="Tax rate if taxable")
    job_note: Optional[str] = Field(default=None, description="Notes for the job")
    vehicle_shipment_details: Optional[Any] = Field(default=None, description="Shipment details (free-form JSON)")
    vehicle_id: Optional[str] = Field(default=None, description="Vehicle ID")
    is_reserved: bool = False  # Default: not reserved
    service_date: str  # Service date (YYYY-MM-DD)
    load_type: Optional[Literal['vehicle', 'misc']] = Field(default=None, description="Type of load")
    load_details: Optional[Any] = Field(default=None, description="Load details (free-form JSON)")
    pay_with_driver: Optional[bool] = Field(default=None, description="Whether payment is handled by driver")
    pay_with_driver_point: Optional[int] = Field(default=None, description="Points used if pay with driver")
    equipment_ids: Optional[List[str]] = Field(default=None, description="List of equipment IDs")
    destinations: Optional[List[DestinationInput]] = Field(default=None, description="Ordered list of destinations")

    # Optional additional fields
    # Note: contact_name is used to populate each destination's contact_name for consistency; not sent at job-level
    contact_name: Optional[str] = Field(default=None, description="Primary contact name to apply to all destinations")
    contact_phone: Optional[str] = Field(default=None, description="Primary contact phone for the job")
    customer_id: Optional[str] = Field(default=None, description="Customer ID for the job")


def _env(key: str, default: str = "") -> str:
    """Read an environment variable with a default."""
    val = os.getenv(key)
    return val if val else default

def _env_bool(key: str, default_true: bool = True) -> bool:
    """Read a boolean-like environment variable. 'false' (case-insensitive) => False; else True."""
    raw = os.getenv(key)
    if raw is None:
        return default_true
    return raw.lower() != "false"


def _writer(writer: Any, message: str) -> None:
    """Stream a message if a writer is available."""
    if writer:
        try:
            writer(message)
        except Exception:
            pass


def _prepare_payload(job: JobCreateInput) -> dict:
    """Build the POST payload, auto-filling env defaults.

    - created_by, company_id, job_type_id are filled from env when missing
    - discount_amount defaults to 0.0
    - pay_with_driver defaults to false
    - Each destination must have lat/long; location_type is auto-filled: first=pick_up, others=drop_off
    - use_location_customer is converted from boolean to string per API contract
    """
    created_by = job.created_by or _env("ACCOUNT_USER_ID")
    company_id = job.company_id or _env("COMPANY_ID")
    env_job_type_id = _env("JOB_TYPE_ID")
    job_type_id = job.job_type_id or env_job_type_id
    if not job_type_id:
        raise ValueError("job_type_id is required. Set JOB_TYPE_ID in .env or provide job_type_id explicitly.")

    payload: Dict[str, Any] = {
        "created_by": created_by,
        "company_id": company_id,
        "job_type_id": job_type_id,
        "contact_phone": job.contact_phone,
        "service_fee": job.service_fee,
        "is_discounted": job.is_discounted,
        "discount_type": job.discount_type,
        "discount_amount": job.discount_amount if job.discount_amount is not None else 0.0,
        "is_taxable": job.is_taxable,
        "tax_rate": job.tax_rate,
        "job_note": job.job_note,
        "vehicle_shipment_details": job.vehicle_shipment_details,
        "vehicle_id": job.vehicle_id,
        "is_reserved": job.is_reserved,
        "service_date": job.service_date,
        "load_type": job.load_type,
        "load_details": job.load_details,
        "pay_with_driver": job.pay_with_driver if job.pay_with_driver is not None else False,
        "pay_with_driver_point": job.pay_with_driver_point,
        "equipment_ids": job.equipment_ids,
    }

    # Note: 'customer_id' intentionally omitted from payload per latest backend contract.

    # Remove None values to keep payload clean
    payload = {k: v for k, v in payload.items() if v is not None}

    # Destinations: require lat/lng and normalize additional fields
    if job.destinations:
        normalized: List[Dict[str, Any]] = []
        # Determine uniform contact details to apply across all destinations
        uniform_name = job.contact_name or (job.destinations[0].contact_name if job.destinations and job.destinations[0].contact_name else None)
        uniform_phone = job.contact_phone or (job.destinations[0].contact_phone if job.destinations and job.destinations[0].contact_phone else None)
        for idx, dest in enumerate(job.destinations, start=1):
            dp = dest.dict()
            lat, lng = dp.get("latitude"), dp.get("longitude")
            if lat is None or lng is None:
                raise ValueError(
                    f"Destination {idx} ('{dp.get('address','')}') is missing latitude/longitude. "
                    "Please provide exact lat/long for all locations."
                )

            # Auto-fill location_type when missing
            if not dp.get("location_type"):
                dp["location_type"] = "pick_up" if idx == 1 else "drop_off"

            # Enforce using job customer for each destination per latest requirement
            dp["use_job_customer"] = True

            # Apply uniform contact name and phone across all destinations (per requirement)
            if uniform_name is not None:
                dp["contact_name"] = uniform_name
            if uniform_phone is not None:
                dp["contact_phone"] = uniform_phone

            # Convert boolean to string for use_location_customer
            if isinstance(dp.get("use_location_customer"), bool):
                dp["use_location_customer"] = "true" if dp["use_location_customer"] else "false"

            normalized.append(dp)

        payload["destinations"] = normalized

    return payload

def _normalize_jobs_path(jobs_path: str) -> str:
    """Ensure jobs_path begins with '/'"""
    return jobs_path if jobs_path.startswith("/") else f"/{jobs_path}"

def _build_headers(token: str) -> Dict[str, str]:
    """Create HTTP headers with optional Bearer token."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def _extract_json(resp: Any) -> Dict[str, Any]:
    """Safely extract JSON or return a minimal dict with status/text."""
    try:
        ct = resp.headers.get("content-type", "")
        if "application/json" in ct:
            return resp.json()
        return {"status_code": resp.status_code, "text": resp.text}
    except Exception:
        return {"status_code": getattr(resp, "status_code", None), "text": getattr(resp, "text", "")}

def _post_job(base_url: str, jobs_path: str, headers: Dict[str, str], payload: Dict[str, Any], verify_ssl: bool, writer: Any) -> Any:
    """POST job to <base>/<jobs_path>, falling back to /api/jobs on 405."""
    # Guard for static analyzers and runtime: ensure httpx is available
    if httpx is None:
        raise RuntimeError("HTTP client 'httpx' is not available. Please add httpx to requirements and install it.")
    url = f"{base_url.rstrip('/')}{jobs_path}"
    _writer(writer, "Submitting job to FleetFlow Account API…")
    resp = httpx.post(url, headers=headers, json=payload, timeout=20.0, verify=verify_ssl)
    if resp.status_code == 405 and jobs_path == "/jobs":
        alt_url = f"{base_url.rstrip('/')}/api/jobs"
        _writer(writer, "POST /jobs returned 405; retrying at /api/jobs…")
        resp = httpx.post(alt_url, headers=headers, json=payload, timeout=20.0, verify=verify_ssl)
    return resp

def _get_latest_job(base_url: str, jobs_path: str, headers: Dict[str, str], company_id: Optional[str], verify_ssl: bool, writer: Any) -> Optional[Dict[str, Any]]:
    """GET latest job list filtered by company_id, with /api/jobs fallback; return first item if present."""
    # Guard for static analyzers and runtime: ensure httpx is available
    if httpx is None:
        raise RuntimeError("HTTP client 'httpx' is not available. Please add httpx to requirements and install it.")
    list_url = f"{base_url.rstrip('/')}{jobs_path}"
    params = {"company_id": company_id} if company_id else {}
    _writer(writer, "Fetching latest job for company via GET /jobs…")
    resp = httpx.get(list_url, headers=headers, params=params, timeout=20.0, verify=verify_ssl)
    if resp.status_code in (404, 405) and jobs_path == "/jobs":
        _writer(writer, "GET /jobs not available; retrying at /api/jobs…")
        alt_url = f"{base_url.rstrip('/')}/api/jobs"
        resp = httpx.get(alt_url, headers=headers, params=params, timeout=20.0, verify=verify_ssl)
    data = _extract_json(resp)

    # Try common shapes: {data: [...]}, list, or single dict
    items = None
    if isinstance(data, dict):
        items = data.get("data")
        if isinstance(items, list) and items:
            return items[0]
        if isinstance(items, dict):
            return items
        if isinstance(data.get("id"), (str, int)) and data.get("code"):
            return data
    elif isinstance(data, list) and data:
        return data[0]
    return None


@tool(
    "create_job",
    description=(
        "Create a FleetFlow job via the Account API. Provide required fields. "
        "Missing IDs (created_by/company_id) default from env. job_type_id is sourced from JOB_TYPE_ID in .env unless provided. "
        "Destinations must include exact latitude/longitude; geocoding is disabled in this POC."
    ),
    args_schema=JobCreateInput,
)
def create_job(
    runtime: Any = None,  # optional runtime for streaming; omitted when called by standard Tool executors
    **kwargs,
) -> str:
    """POST /jobs to the FleetFlow Account API and return a concise result.

    Returns a readable summary string. On error, includes server validation messages.
    """
    if httpx is None:
        return "HTTP client 'httpx' is not available. Please add httpx to requirements and install it."

    # Build input model (validated by pydantic) and prepare payload
    try:
        job_input = JobCreateInput(**kwargs)
        payload = _prepare_payload(job_input)
    except Exception as e:
        return f"Failed to prepare payload: {e}"

    # Resolve API configuration from environment
    base_url = _env("ACCOUNT_API_BASE", "https://dev-account.fleetflow.co")
    token = _env("ACCOUNT_API_TOKEN")
    verify_ssl = _env_bool("ACCOUNT_API_VERIFY_SSL", True)
    jobs_path = _normalize_jobs_path(_env("ACCOUNT_API_JOBS_PATH", "/jobs"))
    headers = _build_headers(token)

    writer = getattr(runtime, "stream_writer", None)
    if httpx is None:
        return "HTTP client 'httpx' is not available. Please add httpx to requirements and install it."

    # 1) Create the job (POST), with fallback to /api/jobs on 405
    try:
        resp = _post_job(base_url, jobs_path, headers, payload, verify_ssl, writer)
        data = _extract_json(resp)
    except Exception as e:
        logger.error("Job creation error: %s", e)
        return f"❌ Request failed: {e}"

    # 2) Handle success/failure
    if 200 <= getattr(resp, "status_code", 500) < 300:
        # Extract created job fields if present
        job_obj = data.get("data") if isinstance(data, dict) else None
        job_obj = job_obj or (data if isinstance(data, dict) else {})
        job_id = job_obj.get("id")
        code = job_obj.get("code")
        status = job_obj.get("job_status")

        # 3) Verification: GET latest job for the company
        latest_job = None
        try:
            latest_job = _get_latest_job(base_url, jobs_path, headers, payload.get("company_id"), verify_ssl, writer)
        except Exception as e:
            logger.warning(f"Failed to fetch latest job listing: {e}")

        # Build a compact verification summary
        ver = ""
        if latest_job:
            lj_code = latest_job.get("code")
            lj_status = latest_job.get("job_status")
            lj_service_date = latest_job.get("service_date")
            lj_fee = latest_job.get("service_fee")
            lj_dest = latest_job.get("destinations")
            lj_dest_count = len(lj_dest) if isinstance(lj_dest, list) else None
            ver = (
                " Verified latest job via GET /jobs: "
                + (f"Code={lj_code}. " if lj_code else "")
                + (f"Status={lj_status}. " if lj_status else "")
                + (f"ServiceDate={lj_service_date}. " if lj_service_date else "")
                + (f"ServiceFee={lj_fee}. " if lj_fee is not None else "")
                + (f"Destinations={lj_dest_count}." if lj_dest_count is not None else "")
            ).strip()
        else:
            ver = " (Note: latest job listing could not be verified.)"

        return (
            "✅ Job created successfully. "
            + (f"ID: {job_id}. " if job_id else "")
            + (f"Code: {code}. " if code else "")
            + (f"Status: {status}." if status else "")
            + ver
        ).strip()
    else:
        # Show validation errors in a readable format
        msg = data.get("message") if isinstance(data, dict) else None
        errors = data.get("errors") if isinstance(data, dict) else None
        pretty_err = json.dumps(errors, ensure_ascii=False, indent=2) if errors else None
        return (
            f"❌ Job creation failed (HTTP {getattr(resp, 'status_code', 'unknown')}). "
            + (f"Message: {msg}. " if msg else "")
            + (f"Errors:\n{pretty_err}" if pretty_err else f"Response: {data}")
        )