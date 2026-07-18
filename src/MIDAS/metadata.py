"""Mixed-frequency panel with per-column metadata.

``MetadataPanel`` is the data structure the whole framework revolves around: a
monthly frame, a quarterly frame, and, for each column, the three facts the
machinery needs, its **frequency**, its economic **group** (for factor blocks and
grouped penalties), and its **publication delay** (for pseudo-real-time masking).

It is schema-agnostic: build one with :meth:`from_frames` from any date-indexed
panels and a list of :class:`VariableMeta`. Reading a particular provider's
spreadsheet into that form is the application's job, not the package's (see the
DSAPM project's ``utils`` loader for an example).

Date convention expected by the rest of the package: monthly rows are dated at the
first day of the reference month, quarterly rows at the first day of the quarter's
end month.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class VariableMeta:
    """Metadata for a single panel column."""

    column: str                     # column name as it appears in the panel
    frequency: str                  # "M" or "Q"
    group: str                      # economic block
    publication_delay_days: int     # release lag after the reference period ends
    label: str | None = None
    unit: str | None = None


class MetadataPanel:
    """Monthly and quarterly frames plus per-column :class:`VariableMeta`.

    Attributes
    ----------
    monthly, quarterly:
        Date-indexed panels restricted to the catalogued columns.
    meta:
        ``dict`` mapping column name -> :class:`VariableMeta`.
    """

    def __init__(
        self,
        monthly: pd.DataFrame,
        quarterly: pd.DataFrame,
        meta: dict[str, VariableMeta],
    ) -> None:
        self.monthly = monthly
        self.quarterly = quarterly
        self.meta = meta

    @classmethod
    def from_frames(
        cls,
        monthly: pd.DataFrame,
        quarterly: pd.DataFrame,
        metas: Sequence[VariableMeta],
    ) -> "MetadataPanel":
        """Build from date-indexed frames and a list of :class:`VariableMeta`.

        Only catalogued columns present in the frames are kept, in the order given.
        """
        meta = {m.column: m for m in metas
                if m.column in (monthly.columns if m.frequency.upper() == "M" else quarterly.columns)}
        m_cols = [c for c, v in meta.items() if v.frequency.upper() == "M"]
        q_cols = [c for c, v in meta.items() if v.frequency.upper() == "Q"]
        return cls(
            monthly=monthly.loc[:, m_cols].astype(float),
            quarterly=quarterly.loc[:, q_cols].astype(float),
            meta=meta,
        )

    # ------------------------------------------------------------------ accessors
    def monthly_columns(self) -> list[str]:
        return list(self.monthly.columns)

    def quarterly_columns(self) -> list[str]:
        return list(self.quarterly.columns)

    def all_columns(self) -> list[str]:
        return self.monthly_columns() + self.quarterly_columns()

    def delay_of(self, column: str) -> int:
        return self.meta[column].publication_delay_days

    def group_of(self, column: str) -> str:
        return self.meta[column].group

    def frequency_of(self, column: str) -> str:
        return self.meta[column].frequency

    def delays(self) -> dict[str, int]:
        """Publication delay (days) for every catalogued column."""
        return {c: v.publication_delay_days for c, v in self.meta.items()}

    def columns_by_group(self, *, frequency: str | None = None) -> dict[str, list[str]]:
        """Map economic ``group`` -> columns, optionally filtered by frequency.

        Drives the DFM factor blocks and the sparse-group MIDAS grouping.
        """
        out: dict[str, list[str]] = {}
        for col, v in self.meta.items():
            if frequency is not None and v.frequency != frequency:
                continue
            out.setdefault(v.group, []).append(col)
        return out
