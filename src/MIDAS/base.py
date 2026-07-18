"""Core interfaces for pseudo-real-time mixed-frequency nowcasting.

Three small abstractions tie the framework together:

``InformationSet``
    The data knowable at a single forecast origin, with every not-yet-released
    cell masked to ``NaN``. It is the *only* object a model consumes, so a model
    physically cannot see the future: no look-ahead by construction.

``NowcastResult``
    A point nowcast plus optional density information (standard deviation,
    quantiles) and a free-form ``extra`` bag for model diagnostics such as
    estimated factors, MIDAS weights or the news decomposition.

``BaseNowcaster``
    The uniform ``fit`` / ``nowcast`` contract every model implements, so the
    backtest harness can drive benchmarks, the DFM, sparse MIDAS and future
    models through one call surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:  # avoid a runtime import cycle with metadata.py
    from .metadata import MetadataPanel


@dataclass
class InformationSet:
    """Everything knowable at one forecast ``origin`` (no look-ahead).

    Parameters
    ----------
    monthly, quarterly:
        Masked copies of the monthly and quarterly panels. Both carry a
        ``DatetimeIndex`` following the database convention (monthly = first day
        of the reference month; quarterly = first day of the quarter's end
        month). Cells not yet released at ``origin`` are ``NaN``.
    target:
        Name of the quarterly target column being nowcast (for example
        ``"g_invq"``).
    origin:
        The forecast origin timestamp. Nothing released after this instant is
        visible in ``monthly`` / ``quarterly``.
    target_period:
        The quarterly reference period being nowcast (same dating convention as
        the quarterly index). The harness sets this; it may be ``None`` when the
        information set is inspected on its own.
    metadata:
        Optional back-reference to the :class:`~MIDAS.metadata.MetadataPanel`
        so models can read variable groups, publication delays and frequencies.
    """

    monthly: pd.DataFrame
    quarterly: pd.DataFrame
    target: str
    origin: pd.Timestamp
    target_period: pd.Timestamp | None = None
    metadata: "MetadataPanel | None" = None

    @property
    def y(self) -> pd.Series:
        """The (masked) quarterly target series."""
        return self.quarterly[self.target]

    def observed_quarters(self) -> pd.DatetimeIndex:
        """Quarters whose target value is already released at ``origin``."""
        y = self.quarterly[self.target]
        return self.quarterly.index[y.notna()]

    def monthly_vars(self) -> list[str]:
        """Monthly predictor columns available in this information set."""
        return [c for c in self.monthly.columns]

    def last_observed(self, column: str) -> pd.Timestamp | None:
        """Most recent date at which ``column`` is observed, or ``None``.

        Works for both the monthly and quarterly panels.
        """
        for frame in (self.monthly, self.quarterly):
            if column in frame.columns:
                s = frame[column].dropna()
                return None if s.empty else s.index[-1]
        raise KeyError(f"{column!r} is in neither the monthly nor quarterly panel")


@dataclass
class NowcastResult:
    """A single nowcast: point estimate plus optional density and diagnostics.

    ``std`` and ``quantiles`` are populated only by models that produce a
    predictive distribution; point-only models leave them ``None``. ``extra``
    carries model-specific artefacts (factors, weights, selected variables, the
    news decomposition) for interpretability without widening this contract.
    """

    mean: float
    std: float | None = None
    quantiles: dict[float, float] | None = None
    model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseNowcaster(ABC):
    """Uniform contract implemented by every model in the framework.

    Concrete nowcasters implement :meth:`fit` (estimate on the data available in
    an :class:`InformationSet`) and :meth:`nowcast` (return a
    :class:`NowcastResult` for ``info.target_period``). Keeping this surface
    tiny is deliberate: the backtest harness, forecast combinations and future
    model classes all rely on nothing more than ``fit`` / ``nowcast``.
    """

    #: Human-readable model name; defaults to the class name (see :attr:`name`).
    _name: str | None = None

    @property
    def name(self) -> str:
        return self._name or type(self).__name__

    @abstractmethod
    def fit(self, info: InformationSet) -> "BaseNowcaster":
        """Estimate the model on the data available in ``info``. Returns self."""

    @abstractmethod
    def nowcast(self, info: InformationSet) -> NowcastResult:
        """Produce a nowcast for ``info.target_period``."""

    def fit_nowcast(self, info: InformationSet) -> NowcastResult:
        """Convenience: :meth:`fit` then :meth:`nowcast` on the same info set."""
        return self.fit(info).nowcast(info)
