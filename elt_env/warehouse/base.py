"""Destination-agnostic warehouse interface.

The rest of the environment ONLY talks to these two classes. Supporting a new
warehouse (Snowflake, Databricks, ...) = implement them once; tools, loader,
grader and the RL env stay unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Warehouse(ABC):
    """One isolated warehouse namespace, owned by exactly one rollout."""

    name: str  # adapter name, e.g. "duckdb"

    @property
    @abstractmethod
    def model_schema(self) -> str:
        """Schema where dbt materializes the agent's models."""

    @abstractmethod
    def write_table(self, schema: str, table: str, df: pd.DataFrame) -> int:
        """Create/replace schema.table with df (used by the loader). Returns row count."""

    @abstractmethod
    def query(self, sql: str, max_rows: int) -> tuple[pd.DataFrame, bool]:
        """Run agent SQL (read-only)."""

    @abstractmethod
    def fetch_table(self, schema: str, table: str) -> pd.DataFrame | None:
        """Full table for grading (trusted code path). None if it does not exist."""

    @abstractmethod
    def list_tables(self) -> list[tuple[str, str]]:
        """All (schema, table) pairs in this namespace."""

    @abstractmethod
    def dbt_profile(self) -> dict:
        """The dbt `outputs.<target>` block pointing at this namespace."""

    @abstractmethod
    def drop(self) -> None:
        """Delete the namespace and everything in it."""


class Destination(ABC):
    """Factory for isolated warehouses of one type."""

    @abstractmethod
    def create_warehouse(self, namespace: str) -> Warehouse:
        ...
