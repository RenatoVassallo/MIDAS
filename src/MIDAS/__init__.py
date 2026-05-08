"""Teaching-friendly mixed-frequency nowcasting and forecasting tools."""

from .align import align_monthly_to_quarter, ragged_edge_lattice, stack_high_freq_lags
from .data_conflict import TOPIC_COLUMNS, aggregate_conflict_topics, load_conflict_topics
from .data_epu import download_epu, epu_wide, load_epu, load_epu_panel
from .data_macro import COUNTRIES, SERIES, fetch_fred_series, load_macro_panel, load_panel
from .evaluation import dm_test, evaluation_table, mae, mape, pinball_loss, rmse
from .forecasting import make_lags, rolling_forecast
from .midas import (
    BetaMIDASRegressor,
    BetaMidasResult,
    MultiBetaMidasResult,
    almon_weights,
    beta_weights,
    extract_midas_arrays,
    fit_beta_midas,
    fit_beta_midas_multi,
    rolling_beta_midas_forecast,
    stack_midas_features,
    weight_profile_table,
)
from .ml import (
    feature_importance_frame,
    make_boosted_classifier,
    make_boosted_regressor,
    make_xgb_classifier,
    make_xgb_regressor,
    shap_importance_frame,
)
from .panel import PanelArtifacts, PanelBuilder
from .plotting import PALETTE, save_for_slides, shade_periods, use_aer_style
from .recession import build_recession_target, load_recession_panel

__all__ = [
    "COUNTRIES",
    "SERIES",
    "TOPIC_COLUMNS",
    "PALETTE",
    "PanelArtifacts",
    "PanelBuilder",
    "BetaMIDASRegressor",
    "BetaMidasResult",
    "MultiBetaMidasResult",
    "use_aer_style",
    "shade_periods",
    "save_for_slides",
    "align_monthly_to_quarter",
    "ragged_edge_lattice",
    "stack_high_freq_lags",
    "beta_weights",
    "almon_weights",
    "fit_beta_midas",
    "fit_beta_midas_multi",
    "stack_midas_features",
    "extract_midas_arrays",
    "rolling_beta_midas_forecast",
    "weight_profile_table",
    "make_boosted_regressor",
    "make_boosted_classifier",
    "make_xgb_regressor",
    "make_xgb_classifier",
    "feature_importance_frame",
    "shap_importance_frame",
    "rolling_forecast",
    "make_lags",
    "rmse",
    "mae",
    "mape",
    "pinball_loss",
    "dm_test",
    "evaluation_table",
    "fetch_fred_series",
    "load_macro_panel",
    "load_panel",
    "load_conflict_topics",
    "aggregate_conflict_topics",
    "download_epu",
    "load_epu",
    "load_epu_panel",
    "epu_wide",
    "load_recession_panel",
    "build_recession_target",
]
