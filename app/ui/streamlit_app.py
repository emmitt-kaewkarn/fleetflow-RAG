import os
import uuid
import logging
from typing import Any, Dict
import importlib
import sys
from pathlib import Path

# Dynamically import dependencies to avoid unresolved import diagnostics in editors
def _import_dependencies():
    """Import runtime-only dependencies safely.

    - streamlit: required for the UI. If missing, we raise a clear error.
    - dotenv: optional; if present, we load `.env` automatically.
    """
    try:
        st = importlib.import_module("streamlit")
    except Exception as e:
        raise RuntimeError(
            "Streamlit is not available. Please install it in your active environment (pip install streamlit)"
        ) from e

    # dotenv is optional; if present, load .env
    try:
        dotenv = importlib.import_module("dotenv")
        load_dotenv = getattr(dotenv, "load_dotenv", None)
        if callable(load_dotenv):
            load_dotenv()
    except Exception:
        # Proceed without dotenv
        pass

    return st

st = _import_dependencies()

# Ensure project root is on sys.path so `import app...` works when run via Streamlit
# from the `app/ui` subdirectory or other locations.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# Map OpenRouter -> OpenAI client if needed (silent; no UI notice)
if not os.getenv("OPENAI_API_KEY") and os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY") or ""
if not os.getenv("OPENAI_BASE_URL") and os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

from app.agents.agent import create_main_agent, stream_agent_updates

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


st.set_page_config(page_title="FleetFlow Agent Chat", page_icon="🚌", layout="wide")
st.title("FleetFlow Agent Chat")
st.caption("Chat with the FleetFlow agent. Streaming enabled; conversation summary memory active.")

# Sidebar config
def _load_system_prompt_from_file() -> str | None:
    """Read the default system prompt from SYSTEM_PROMPT.txt if it exists."""
    try:
        prompt_file = PROJECT_ROOT / "SYSTEM_PROMPT.txt"
        if prompt_file.exists():
            text = prompt_file.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception:
        pass
    return None

with st.sidebar:
    st.header("Settings")
    # Quick action to reset the conversation
    def _reset_chat():
        st.session_state.messages = []
        # new thread id to isolate memory for a fresh chat
        st.session_state.thread_id = uuid.uuid4().hex
        st.toast("New chat started")

    st.button("🆕 New chat", on_click=_reset_chat, help="Clear history and start a fresh conversation")

    stream_mode = st.selectbox(
        "Stream mode",
        options=["updates", "messages"],
        index=0,
        help="'updates' streams agent steps; 'messages' streams LLM tokens (when supported)"
    )
    show_steps = st.checkbox(
        "Show live progress (no chain-of-thought)",
        value=True,
        help="Display model/tool steps and results while streaming without exposing hidden reasoning."
    )
    model_id = st.text_input("Model", value=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    summary_model_id = st.text_input("Summary Model", value=os.getenv("SUMMARY_MODEL", "gpt-4o-mini"))
    # Default system prompt: file content if present, else env, else fallback
    default_prompt = _load_system_prompt_from_file() or os.getenv("SYSTEM_PROMPT", "You are FleetFlow, a helpful agent for fleet operations.")
    system_prompt = st.text_area(
        "System Prompt",
        value=default_prompt,
        height=160,
    )

    st.divider()
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    st.write(f"API Key set: {'✅' if api_key_set else '❌'}")
    if not api_key_set:
        st.warning("OPENAI_API_KEY not set. If using OpenRouter, set OPENROUTER_API_KEY in your .env.")


def ensure_agent() -> Any:
    """Create the agent once per session and cache it in session_state."""
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = create_main_agent(
                model=model_id,
                summarizer_model=summary_model_id,
                system_prompt=system_prompt,
            )
        except Exception as e:
            st.error(f"Failed to create agent: {e}")
            raise
    return st.session_state.agent


def ensure_thread_id() -> str:
    """Create a per-session thread_id so the agent can persist memory for this chat."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex
    return st.session_state.thread_id


def init_chat_state():
    """Initialize the chat message history container in session_state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


init_chat_state()
agent = ensure_agent()
thread_id = ensure_thread_id()

# Welcome guidance for new chats
if len(st.session_state.messages) == 0:
    with st.container():
        st.markdown("""
        #### 👋 Welcome to FleetFlow Agent
        Ask operational questions, explore fleet metrics, or run diagnostics. Here are some ideas:
        - "Show me vehicles with maintenance overdue this month"
        - "Summarize incidents for the last 7 days"
        - "What are the top 3 routes by on-time percentage?"
        - "List drivers with more than 3 safety alerts in the past week"
        """
        )

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg.get("role", "assistant")):
        st.markdown(msg.get("content", ""))


def render_updates_chunk(chunk: Dict[str, Any], placeholder):
    """Render a single 'updates' chunk into the assistant placeholder.

    The agent emits a dict of step -> data. Common steps:
    - model: LLM output messages
    - tools: tool call messages
    Other steps may appear depending on middleware.
    """
    # chunk is typically a dict mapping step -> data
    for step, data in chunk.items():
        if step == "model":
            try:
                latest = data["messages"][-1]
                content = getattr(latest, "content", "")
                if content:
                    placeholder.markdown(content)
            except Exception:
                placeholder.markdown("[model step]")
        elif step == "tools":
            # Show last tool message content (if any) with clearer formatting
            try:
                latest = data["messages"][-1]
                content = getattr(latest, "content", "")
                if content:
                    placeholder.markdown("""
```tool
{content}
```
""".format(content=content))
            except Exception:
                placeholder.caption("[tool step]")
        else:
            # Other middleware steps, if present
            placeholder.caption(f"[{step}] {data}")


prompt = st.chat_input("Ask the FleetFlow agent...")
if prompt:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant streaming container
    with st.chat_message("assistant"):
        placeholder = st.empty()
        # Optional progress panel for live step updates
        progress_placeholder = None
        if show_steps:
            with st.expander("Live progress"):
                progress_placeholder = st.empty()
        log_lines = []
        try:
            # Stream updates or messages
            # For simplicity, we handle 'updates' mode by rendering step chunks.
            # For 'messages' mode, we attempt to render content blocks or text.
            text_accum = ""  # accumulate tokens for 'messages' mode for smoother display
            for chunk in stream_agent_updates(
                agent,
                prompt,
                stream_mode=stream_mode,
                thread_id=thread_id,
            ):
                if stream_mode == "updates":
                    render_updates_chunk(chunk, placeholder)
                    # Also accumulate a compact log of steps for the progress panel
                    if progress_placeholder is not None:
                        try:
                            for step, data in chunk.items():
                                if step == "model":
                                    latest = data.get("messages", [])[-1] if data.get("messages") else None
                                    txt = getattr(latest, "content", "") if latest else "(model step)"
                                    if txt:
                                        log_lines.append(f"Model: {txt}")
                                elif step == "tools":
                                    latest = data.get("messages", [])[-1] if data.get("messages") else None
                                    txt = getattr(latest, "content", "") if latest else "(tool step)"
                                    if txt:
                                        log_lines.append(f"Tool: {txt}")
                                else:
                                    # generic middleware step
                                    log_lines.append(f"{step}: {data}")
                        except Exception:
                            log_lines.append("[progress update error]")
                        # Update progress panel
                        progress_placeholder.markdown("\n".join(f"- {l}" for l in log_lines))
                else:
                    # messages mode - chunk likely is a tuple (token, metadata) or a message-like object
                    try:
                        # Try token-style streaming
                        token, _meta = chunk
                        text = getattr(token, "content", "") or (token if isinstance(token, str) else "")
                        if text:
                            text_accum += text
                            placeholder.markdown(text_accum)
                    except Exception:
                        # Fallback to updates-style rendering
                        if isinstance(chunk, dict):
                            render_updates_chunk(chunk, placeholder)
                            if progress_placeholder is not None:
                                # Record generic dict chunk
                                log_lines.append(f"update: {chunk}")
                                progress_placeholder.markdown("\n".join(f"- {l}" for l in log_lines))
                        else:
                            placeholder.write(str(chunk))

        except Exception as e:
            st.error(f"Streaming failed: {e}")
            logger.error("Streaming error: %s", e)
        else:
            # Add the final assistant message to history (best-effort)
            # The final message should be in the agent state; we re-invoke with empty text
            # to fetch the current state and append the latest assistant message.
            try:
                from app.agents.agent import invoke_agent
                result = invoke_agent(agent, "", thread_id=thread_id)
                last_msg = result.get("messages", [])[-1] if result.get("messages") else None
                final_text = getattr(last_msg, "content", "") if last_msg else ""
                if final_text:
                    st.session_state.messages.append({"role": "assistant", "content": final_text})
            except Exception:
                # If invoke fails, append a generic acknowledgement
                st.session_state.messages.append({"role": "assistant", "content": "Response sent."})
                
