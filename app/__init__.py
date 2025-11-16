"""Top-level package for FleetFlow application.

This file ensures Python treats the `app` directory as a package regardless of
environment quirks and tooling differences, simplifying imports like
`from app.agents.agent import create_main_agent`.
"""

__all__ = [
    "agents",
    "database",
    "ui",
]