"""
FleetFlow SQL Tool (POC)

This module provides a simple, easy-to-read set of utilities so the agent can:
- Validate SQL queries against FF_SCHEMA.txt (to avoid unknown tables/columns)
- Execute safe, read-only SELECT queries
- Format results for LLM consumption
- Describe the database schema (FF_SCHEMA.txt first, then live introspection if available)

Kept intentionally straightforward for a proof of concept.
"""

import json
import logging
import sys
import os
import importlib
from typing import Any, Dict, List, Optional, Union

# Logger
logger = logging.getLogger(__name__)

# Ensure imports resolve in simple environments
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# SQLAlchemy components (optional)
text = None
inspect = None
SQLAlchemyError = Exception

try:
    sa = importlib.import_module("sqlalchemy")
    sa_exc = importlib.import_module("sqlalchemy.exc")
    text = getattr(sa, "text", None)
    inspect = getattr(sa, "inspect", None)
    SQLAlchemyError = getattr(sa_exc, "SQLAlchemyError", Exception)
except Exception as e:
    logger.warning(f"SQLAlchemy not available (POC will still run, but DB functions are limited): {e}")

# Database engine
try:
    from app.database.connection import engine
except ImportError as e:
    logger.error(f"Failed to import database engine: {e}")
    engine = None


# Guardrails for validation
DANGEROUS_OPS = [
    'DROP', 'TRUNCATE', 'DELETE', 'UPDATE', 'INSERT',
    'ALTER', 'CREATE', 'GRANT', 'REVOKE', 'EXECUTE',
    'EXEC', 'SP_', 'XP_', 'CALL'
]


class SQLQueryTool:
    """Core SQL tool: validate, execute, and format SELECT queries."""
    
    def __init__(self):
        """Initialize the SQL query tool with database connection."""
        self.engine = engine
        self._table_info_cache = None
        
        # Check if SQLAlchemy is available
        if text is None or inspect is None:
            logger.warning("SQLAlchemy not available. Some functionality may be limited.")
        if self.engine is None:
            logger.warning("Database engine is not available (None). Query execution will be disabled.")
    
    def get_table_info(self) -> Dict[str, Any]:
        """Return table/column info via SQLAlchemy inspector (if available)."""
        try:
            if inspect is None:
                logger.error("SQLAlchemy inspect function not available")
                return {}
            if self.engine is None:
                logger.error("Database engine not available")
                return {}
            inspector = inspect(self.engine)
            tables = {}
            
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                tables[table_name] = [
                    {
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col['nullable'],
                        'default': col['default']
                    }
                    for col in columns
                ]
            
            return tables
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return {}
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a validated, read-only SELECT query and return structured results."""
        try:
            # Always validate before executing to enforce FF_SCHEMA and safety
            validation = self.validate_query(query)
            if not validation.get('valid'):
                return {
                    'success': False,
                    'error': f"Query validation failed: {validation.get('error')}",
                    'data': [],
                    'row_count': 0
                }
            if text is None:
                return {
                    'success': False,
                    'error': "SQLAlchemy text function not available",
                    'data': [],
                    'row_count': 0
                }
            if self.engine is None:
                return {
                    'success': False,
                    'error': "Database engine not available",
                    'data': [],
                    'row_count': 0
                }
            with self.engine.connect() as connection:
                # Execute query with parameters if provided
                if params:
                    result = connection.execute(text(query), params)
                else:
                    result = connection.execute(text(query))
                
                # Get column names
                columns = list(result.keys())
                
                # Fetch all rows
                rows = result.fetchall()
                
                # Convert rows to list of dictionaries
                results = []
                for row in rows:
                    row_dict = {}
                    for i, column in enumerate(columns):
                        value = row[i]
                        # Handle datetime and other complex types
                        if hasattr(value, 'isoformat'):
                            value = value.isoformat()
                        row_dict[column] = value
                    results.append(row_dict)
                
                return {
                    'success': True,
                    'data': results,
                    'columns': columns,
                    'row_count': len(results)
                }
                
        except SQLAlchemyError as e:
            logger.error(f"SQL Error: {e}")
            return {
                'success': False,
                'error': f"SQL Error: {str(e)}",
                'data': [],
                'row_count': 0
            }
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}",
                'data': [],
                'row_count': 0
            }
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """Validate query safety and schema usage (FF_SCHEMA-first).

        Allows read-only SQL like SELECT, WITH, EXPLAIN, SHOW, DESCRIBE.
        Blocks dangerous operations (DML/DDL) defined in DANGEROUS_OPS.
        Schema enforcement is primarily applied to SELECT-style queries.
        """
        # Convert to uppercase for checking
        upper_query = query.upper().strip()

        # Block dangerous operations
        for operation in DANGEROUS_OPS:
            if operation in upper_query:
                return {
                    'valid': False,
                    'error': f"Query contains dangerous operation: {operation}",
                    'warning': "Only SELECT queries are allowed"
                }

        # Enforce FF_SCHEMA.txt tables (and optionally columns)
        schema = self._load_schema_from_file()
        if schema:
            parsed = self._parse_tables_and_columns(query)
            unknown_tables = [t for t in parsed['tables'] if t not in schema]
            if unknown_tables:
                # Build hint of allowed tables
                allowed_tables = ", ".join(sorted(schema.keys()))
                return {
                    'valid': False,
                    'error': f"Query references tables not in FF_SCHEMA: {', '.join(unknown_tables)}",
                    'warning': "Only tables defined in FF_SCHEMA.txt are allowed",
                    'hint': f"Allowed tables: {allowed_tables}"
                }
            # Optional column checks: only enforce when table.column is explicitly used
            unknown_columns = []
            alias_map = parsed.get('aliases', {})
            for tbl_or_alias, col in parsed['qualified_columns']:
                # Resolve alias to real table if available
                tbl = alias_map.get(tbl_or_alias, tbl_or_alias)
                cols = schema.get(tbl, [])
                # If schema lists ellipsis '…', skip strict column enforcement
                if '…' in cols:
                    continue
                if col not in cols:
                    unknown_columns.append(f"{tbl}.{col}")
            if unknown_columns:
                # Build per-table column hints
                table_hints = []
                for t in sorted(set([c.split('.')[0] for c in unknown_columns])):
                    cols = ", ".join(schema.get(t, [])) or "(none)"
                    table_hints.append(f"{t}: {cols}")
                return {
                    'valid': False,
                    'error': f"Query references columns not in FF_SCHEMA: {', '.join(unknown_columns)}",
                    'warning': "Use only columns defined in FF_SCHEMA.txt or update the schema file",
                    'hint': "Columns by table:\n- " + "\n- ".join(table_hints)
                }

            # Heuristic: if only ONE table is used and no aliases, enforce unqualified column names
            unique_tables = list({t for t in parsed['tables']})
            if len(unique_tables) == 1:
                target_table = unique_tables[0]
                allowed_cols = set(schema.get(target_table, []))
                if '…' not in allowed_cols:
                    bad_unqualified = [c for c in parsed.get('unqualified_columns', []) if c not in allowed_cols]
                    if bad_unqualified:
                        return {
                            'valid': False,
                            'error': f"Unqualified columns not in FF_SCHEMA for {target_table}: {', '.join(bad_unqualified)}",
                            'warning': "Qualify columns with table alias or use only schema-defined columns",
                            'hint': f"Allowed columns for {target_table}: {', '.join(sorted(allowed_cols))}"
                        }

        return {
            'valid': True,
            'error': None,
            'warning': None
        }
    
    def format_results_for_llm(self, results: Dict[str, Any], max_rows: int = 10) -> str:
        """Format results into a compact, LLM-friendly string."""
        if not results['success']:
            return f"Query failed: {results['error']}"
        
        if results['row_count'] == 0:
            return "Query returned no results."
        
        data = results['data']
        columns = results['columns']
        
        # Limit rows for LLM context
        limited_data = data[:max_rows]
        
        # Create formatted output
        output = f"Query returned {results['row_count']} rows. Showing first {len(limited_data)} rows:\n\n"
        
        # Add column headers
        output += " | ".join(columns) + "\n"
        output += "-" * (len(" | ".join(columns))) + "\n"
        
        # Add data rows
        for row in limited_data:
            row_values = [str(row.get(col, '')) for col in columns]
            output += " | ".join(row_values) + "\n"
        
        if results['row_count'] > max_rows:
            output += f"\n... and {results['row_count'] - max_rows} more rows"
        
        return output

    def _load_schema_from_file(self, max_columns_per_table: int = 100) -> Dict[str, List[str]]:
        """Proxy to the module-level loader to avoid duplicate logic."""
        return load_ff_schema(max_columns_per_table=max_columns_per_table)

    def _parse_tables_and_columns(self, query: str) -> Dict[str, Any]:
        """
        Best-effort SQL parsing to extract referenced table names and qualified columns.
        - tables: set of table names from FROM/JOIN clauses
        - qualified_columns: list of (table, column) from SELECT list where 'table.column' form is used
        """
        import re
        q = query
        # Normalize whitespace
        qn = re.sub(r"\s+", " ", q)
        # Capture FROM and JOIN tokens with optional aliases
        tables: List[str] = []
        aliases: Dict[str, str] = {}
        # FROM <table> [AS] <alias>
        for m in re.finditer(r"\bFROM\s+`?(\w+)`?(?:\s+(?:AS\s+)?`?(\w+)`?)?", qn, flags=re.IGNORECASE):
            tbl = m.group(1)
            alias = m.group(2)
            tables.append(tbl)
            if alias:
                aliases[alias] = tbl
        # JOIN <table> [AS] <alias>
        for m in re.finditer(r"\bJOIN\s+`?(\w+)`?(?:\s+(?:AS\s+)?`?(\w+)`?)?", qn, flags=re.IGNORECASE):
            tbl = m.group(1)
            alias = m.group(2)
            tables.append(tbl)
            if alias:
                aliases[alias] = tbl
        # SELECT list qualified columns
        qualified_columns: List[tuple] = []
        sel_match = re.search(r"SELECT\s+(.*?)\s+FROM\s", qn, flags=re.IGNORECASE)
        if sel_match:
            select_list = sel_match.group(1)
            # split by commas, find tokens with table.column
            for part in select_list.split(','):
                part = part.strip()
                m = re.match(r"`?(\w+)`?\.`?(\w+)`?", part)
                if m:
                    qualified_columns.append((m.group(1), m.group(2)))
        # Unqualified columns (heuristic): tokens without dot and not functions
        unqualified_columns: List[str] = []
        if sel_match:
            for part in sel_match.group(1).split(','):
                tok = part.strip()
                # skip function calls, wildcards, literals and expressions
                if '(' in tok or ')' in tok or '.' in tok or tok == '*' or tok.upper().startswith('COUNT'):
                    continue
                # Strip simple aliases (AS name)
                tok = re.sub(r"\s+AS\s+`?(\w+)`?", "", tok, flags=re.IGNORECASE)
                # Capture a simple identifier at start
                m2 = re.match(r"`?(\w+)`?", tok)
                if m2:
                    unqualified_columns.append(m2.group(1))
        return {"tables": list({t for t in tables}), "qualified_columns": qualified_columns, "aliases": aliases, "unqualified_columns": unqualified_columns}


def sql_query_tool(query: str, max_results: int = 50) -> str:
    """Validate + run a read-only SQL query, returning a readable text table."""
    tool = SQLQueryTool()
    
    # Validate query
    validation = tool.validate_query(query)
    if not validation['valid']:
        return f"Query validation failed: {validation['error']}"
    
    # Execute query
    results = tool.execute_query(query)
    
    # Format results for LLM with specified max_results
    return tool.format_results_for_llm(results, max_results)


def get_database_schema() -> str:
    """Describe available tables/columns (FF_SCHEMA-first, then live DB if available)."""
    tool = SQLQueryTool()

    # 1) Canonical schema (from FF_SCHEMA.txt)
    ff_schema = tool._load_schema_from_file()
    output_parts: List[str] = []
    if ff_schema:
        output_parts.append("Schema overview (from FF_SCHEMA.txt):\n")
        for table_name, columns in ff_schema.items():
            cols_txt = ", ".join(columns) if columns else "(columns unavailable)"
            output_parts.append(f"- {table_name}: {cols_txt}")
        output_parts.append("")
    else:
        output_parts.append("FF_SCHEMA.txt not found or could not be parsed.")

    # 2) Live database introspection (optional)
    table_info = tool.get_table_info()
    if table_info:
        output_parts.append("Live database introspection (SQLAlchemy Inspector):\n")
        for table_name, columns in table_info.items():
            output_parts.append(f"Table: {table_name}")
            for column in columns:
                nullable = "NULL" if column['nullable'] else "NOT NULL"
                default = f" DEFAULT {column['default']}" if column['default'] else ""
                output_parts.append(f"  - {column['name']}: {column['type']} {nullable}{default}")
            output_parts.append("")
    else:
        output_parts.append("(Live DB schema unavailable or engine not connected.)")

    return "\n".join(output_parts)


def load_ff_schema(max_columns_per_table: int = 100) -> Dict[str, List[str]]:
    """Parse FF_SCHEMA.txt and return {table: [columns...]}."""
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        schema_path = os.path.join(root, "FF_SCHEMA.txt")
        if not os.path.exists(schema_path):
            return {}
        txt = open(schema_path, "r", encoding="utf-8").read()
        import re
        blocks = re.findall(r"CREATE\s+TABLE\s+`?(\w+)`?\s*\((.*?)\)\s*ENGINE", txt, flags=re.IGNORECASE | re.DOTALL)
        out: Dict[str, List[str]] = {}
        for tbl, block in blocks:
            cols: List[str] = []
            for line in block.splitlines():
                s = line.strip()
                if not s or s.upper().startswith(("PRIMARY KEY", "CONSTRAINT", "UNIQUE", "KEY")):
                    continue
                m = re.match(r"`([^`]+)`\s+", s)
                if m:
                    cols.append(m.group(1))
            if max_columns_per_table and len(cols) > max_columns_per_table:
                cols = cols[:max_columns_per_table] + ["…"]
            out[tbl] = cols
        return out
    except Exception as e:
        logger.warning(f"Failed to load FF_SCHEMA.txt: {e}")
        return {}

def render_schema_hint(schema: Dict[str, List[str]], max_tables: int = 40) -> str:
    """Produce a compact schema overview for inclusion in system prompts."""
    if not schema:
        return ""
    lines: List[str] = [
        "",
        "Database schema overview (do not invent tables; use only these):",
    ]
    items = list(schema.items())
    if max_tables and len(items) > max_tables:
        items = items[:max_tables] + [("…", [])]
    for table, cols in items:
        if table == "…":
            lines.append("- …")
            continue
        cols_txt = ", ".join(cols) if cols else "(columns unavailable)"
        lines.append(f"- {table}: {cols_txt}")
    return "\n".join(lines)


def run_describe_table(table_name: str) -> str:
    """Run a live DESCRIBE/SHOW COLUMNS query for a specific table (read-only).

    Safety:
    - Validates the table identifier (alphanumeric/underscore) to prevent injection.
    - Uses SQLAlchemy's text() if available.
    - Returns a readable string description or an error message.
    """
    import re
    safe = re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name or "")
    if not safe:
        return (
            "Invalid table name. Only alphanumeric and underscore characters are allowed, "
            "starting with a letter or underscore."
        )

    if text is None or engine is None:
        return "Live DESCRIBE unavailable (SQLAlchemy or engine not initialized)."

    try:
        sql = f"SHOW COLUMNS FROM `{table_name}`"
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()
            if not rows:
                return f"No columns returned for table '{table_name}'."
            # Expected keys: Field, Type, Null, Key, Default, Extra
            headers = list(result.keys())
            out_lines = [f"Live column details for '{table_name}':", " | ".join(headers), "-" * len(" | ".join(headers))]
            for row in rows:
                vals = [str(row[i]) for i in range(len(headers))]
                out_lines.append(" | ".join(vals))
            return "\n".join(out_lines)
    except SQLAlchemyError as e:
        logger.error(f"DESCRIBE error: {e}")
        return f"DESCRIBE failed for '{table_name}': {e}"
    except Exception as e:
        logger.error(f"Unexpected DESCRIBE error: {e}")
        return f"Unexpected error describing '{table_name}': {e}"


def describe_table(table_name: str) -> str:
    """Provide a combined schema view for a single table.

    Order of information:
    1) FF_SCHEMA.txt columns (canonical, stable)
    2) Live DESCRIBE/SHOW COLUMNS (if available)
    3) SQLAlchemy Inspector fallback (if available)
    """
    import re
    safe = re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name or "")
    if not safe:
        return (
            "Invalid table name. Only alphanumeric and underscore characters are allowed, "
            "starting with a letter or underscore."
        )

    parts: List[str] = []

    # 1) FF_SCHEMA.txt
    schema = load_ff_schema()
    if schema.get(table_name):
        cols_txt = ", ".join(schema[table_name]) if schema[table_name] else "(columns unavailable)"
        parts.append(f"Schema (FF_SCHEMA.txt) for '{table_name}':\n- {table_name}: {cols_txt}\n")
    else:
        parts.append(f"'{table_name}' not found in FF_SCHEMA.txt.")

    # 2) Live DESCRIBE
    live = run_describe_table(table_name)
    if live:
        parts.append(live)

    # 3) Inspector fallback
    if inspect is not None and engine is not None:
        try:
            insp = inspect(engine)
            cols = insp.get_columns(table_name)
            if cols:
                parts.append(f"\nInspector columns for '{table_name}':")
                for c in cols:
                    nullable = "NULL" if c.get("nullable") else "NOT NULL"
                    default = f" DEFAULT {c.get('default')}" if c.get('default') else ""
                    parts.append(f"- {c.get('name')}: {c.get('type')} {nullable}{default}")
        except Exception as e:
            logger.warning(f"Inspector failed for '{table_name}': {e}")

    return "\n".join(parts)


# Example usage (optional)
if __name__ == "__main__":
    print("Testing: get_database_schema()\n")
    print(get_database_schema())