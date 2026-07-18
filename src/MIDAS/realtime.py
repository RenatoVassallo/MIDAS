"""Pseudo-real-time information-set reconstruction.

Given one data vintage and a per-variable publication delay, ``RealtimeEngine``
rebuilds exactly what was knowable at any past ``origin`` by masking every cell
whose release date falls after the origin. The masked panels are wrapped in an
:class:`~MIDAS.base.InformationSet`, the single object every model consumes, so
look-ahead bias is ruled out structurally rather than by convention.

Release-date rule (shared by monthly and quarterly series, because both are
dated at the first day of their reference-period-end month)::

    release_date = reference_period_start + MonthEnd(1) + publication_delay_days

This is the rule already used illustratively in :func:`MIDAS.align.ragged_edge_lattice`,
generalised here to the whole panel at day-level, per-variable granularity so that
begin / middle / end-of-month origins select genuinely different information sets.

Caveats (stated in the proposal, restated here):

* *Pseudo-real-time, not real-time.* We reconstruct the timing of availability,
  not historical data revisions: masked values are the final vintage's numbers.
* ``publication_delay_days`` is a single assumed lag per variable, not a dated
  release calendar. The engine reads it per variable and is structured so a full
  release-calendar table can replace the scalar later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import InformationSet
from .metadata import MetadataPanel


class RealtimeEngine:
    """Reconstructs no-look-ahead information sets from publication delays."""

    def __init__(self, panel: MetadataPanel) -> None:
        self.panel = panel
        self._delays = panel.delays()

    # ---------------------------------------------------------------- core rule
    @staticmethod
    def release_dates(index: pd.DatetimeIndex, delay_days: int) -> pd.DatetimeIndex:
        """Release date for each reference period in ``index`` given ``delay_days``.

        ``index`` holds reference-period-start dates (first of the month for
        monthly series; first day of the quarter's end month for quarterly ones).
        """
        return index + pd.offsets.MonthEnd(1) + pd.Timedelta(days=int(delay_days))

    def _mask_frame(self, frame: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        """Copy ``frame`` with cells released after ``origin`` set to ``NaN``."""
        out = frame.copy()
        period_end = out.index + pd.offsets.MonthEnd(1)
        for col in out.columns:
            release = period_end + pd.Timedelta(days=int(self._delays[col]))
            unreleased = pd.Series(release > origin, index=out.index)
            if unreleased.any():
                out.loc[unreleased, col] = np.nan
        return out

    # ------------------------------------------------------------- public API
    def information_set(
        self,
        origin: str | pd.Timestamp,
        target: str,
        target_period: str | pd.Timestamp | None = None,
    ) -> InformationSet:
        """Information knowable at ``origin`` for nowcasting ``target``.

        Parameters
        ----------
        origin:
            Forecast origin. Any cell with ``release_date > origin`` is masked.
        target:
            Quarterly target column (for example ``"g_invq"``).
        target_period:
            Quarter being nowcast (same dating convention as the quarterly
            index). Optional; recorded on the returned information set.
        """
        origin = pd.Timestamp(origin)
        monthly = self._mask_frame(self.panel.monthly, origin)
        quarterly = self._mask_frame(self.panel.quarterly, origin)
        return InformationSet(
            monthly=monthly,
            quarterly=quarterly,
            target=target,
            origin=origin,
            target_period=None if target_period is None else pd.Timestamp(target_period),
            metadata=self.panel,
        )

    def availability_matrix(
        self,
        origin: str | pd.Timestamp,
        *,
        frequency: str = "M",
        last: int | None = None,
    ) -> pd.DataFrame:
        """Boolean observed/not-observed matrix (rows = columns, cols = periods).

        Convenience for visualising the ragged edge at an ``origin``. ``last``
        keeps only the most recent ``last`` reference periods.
        """
        origin = pd.Timestamp(origin)
        frame = self.panel.monthly if frequency.upper() == "M" else self.panel.quarterly
        period_end = frame.index + pd.offsets.MonthEnd(1)
        data = {}
        for col in frame.columns:
            release = period_end + pd.Timedelta(days=int(self._delays[col]))
            observed = frame[col].notna().to_numpy() & (release <= origin)
            data[col] = pd.Series(observed, index=frame.index)
        matrix = pd.DataFrame(data).T
        if last is not None:
            matrix = matrix.iloc[:, -last:]
        return matrix
