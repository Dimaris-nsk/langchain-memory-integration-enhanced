"""
PostgreSQL importer for LangChain memory data.
"""

import re
from datetime import datetime

import psycopg

from .base import BaseImporter, Conversation, ConversationMessage


class PostgreSQLImporter(BaseImporter):
    """Importer for PostgreSQL-based LangChain chat message history."""

    def __init__(self, table_name: str = "message_store") -> None:
        """
        Initialize PostgreSQL importer.

        Args:
            table_name: Name of the table containing chat messages
        """
        # Validate table name to prevent SQL injection
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
            msg = f"Invalid table name: {table_name}"
            raise ValueError(msg)
        self.table_name = table_name

    def validate_connection(self, connection_string: str) -> bool:
        """Validate PostgreSQL connection."""
        try:
            with psycopg.connect(connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception:
            return False

    def import_conversations(self, connection_string: str, **kwargs) -> list[Conversation]:
        """
        Import conversations from PostgreSQL.

        Args:
            connection_string: PostgreSQL connection string
            **kwargs: Additional parameters:
                - session_ids: List of specific session IDs to import
                - limit: Maximum number of conversations to import

        Returns:
            List of Conversation objects
        """
        conversations = []
        session_ids = kwargs.get("session_ids")
        limit = kwargs.get("limit")

        try:
            with psycopg.connect(connection_string) as conn:
                with conn.cursor() as cur:
                    # Build query and parameters
                    query, params = self._build_query(session_ids, limit)
                    cur.execute(query, params)

                    # Group messages by session_id
                    sessions_data = {}
                    for row in cur.fetchall():
                        session_id, message_type, content, created_at = row

                        if session_id not in sessions_data:
                            sessions_data[session_id] = []

                        # Convert LangChain message type to unified format
                        role = self._convert_message_type(message_type)

                        message = ConversationMessage(
                            role=role,
                            content=content,
                            timestamp=created_at,
                        )
                        sessions_data[session_id].append(message)

                    # Convert to Conversation objects
                    for session_id, messages in sessions_data.items():
                        # Sort messages by timestamp
                        messages.sort(key=lambda m: m.timestamp or datetime.min)

                        conversation = Conversation(
                            session_id=session_id,
                            messages=messages,
                            created_at=messages[0].timestamp if messages else None,
                            updated_at=messages[-1].timestamp if messages else None,
                        )
                        conversations.append(conversation)

        except Exception as e:
            msg = f"Failed to import from PostgreSQL: {e!s}"
            raise Exception(msg)

        return conversations

    def _build_query(self, session_ids: list[str] | None, limit: int | None) -> tuple[str, list]:
        """Build SQL query with parameters to prevent SQL injection."""
        # Base query with identifier quoting for table name
        # Table name already validated in __init__
        query = f"""
        SELECT session_id, type, data->>'content' as content, created_at
        FROM {self.table_name}
        WHERE 1=1
        """
        params = []

        if session_ids:
            # Use parameterized query for session_ids
            placeholders = ', '.join(['%s'] * len(session_ids))
            query += f" AND session_id IN ({placeholders})"
            params.extend(session_ids)

        query += " ORDER BY session_id, created_at"

        if limit:
            # Validate limit is a positive integer
            if not isinstance(limit, int) or limit <= 0:
                msg = f"Invalid limit value: {limit}"
                raise ValueError(msg)
            query += " LIMIT %s"
            params.append(limit)

        return query, params

    def _convert_message_type(self, message_type: str) -> str:
        """Convert LangChain message type to unified format."""
        type_mapping = {
            "human": "human",
            "ai": "ai",
            "system": "system",
            "HumanMessage": "human",
            "AIMessage": "ai",
            "SystemMessage": "system",
        }
        return type_mapping.get(message_type, "human")

    def get_supported_formats(self) -> list[str]:
        """Return supported PostgreSQL table formats."""
        return ["langchain_postgres", "custom_schema"]
