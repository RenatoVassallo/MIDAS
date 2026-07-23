"""Teaching-friendly mixed-frequency nowcasting and forecasting tools."""

from .align import align_monthly_to_quarter, ragged_edge_lattice, stack_high_freq_lags
from .backtest import (
    make_daily_origin_grid,
    make_origin_grid,
    publication_date,
    quarter_timestamp,
    release_cycle_rmse,
    run_backtest,
    run_horizon_backtest,
    run_release_cycle_backtest,
)
from .bands import bootstrap_quantiles, horizon_bands, release_cycle_bands
from .base import BaseNowcaster, InformationSet, NowcastResult
from .benchmarks import (
    ADLMIDASNowcaster,
    ARNowcaster,
    BridgeNowcaster,
    HistoricalMeanNowcaster,
    MonthlyLevelARNowcaster,
    MonthlyARNowcaster,
    PooledLevelBridgeNowcaster,
    PooledBridgeNowcaster,
    PooledMIDASNowcaster,
    RandomWalkNowcaster,
)
from .adaptive import add_information_index, combine_release_cycle, plot_weight_evolution
from .combine import CombinationNowcaster, combine_backtest
from .data_conflict import TOPIC_COLUMNS, aggregate_conflict_topics, load_conflict_topics
from .data_epu import download_epu, epu_wide, load_epu, load_epu_panel
from .data_macro import COUNTRIES, SERIES, fetch_fred_series, load_macro_panel, load_panel
from .data_nbs_china import NBS_INDICATORS, NBSChinaError, NBSIndicatorSpec, available_nbs_indicators, get_nbs_data
from .dfm import DFMNowcaster
from .gb_trees import GBTreesNowcaster
from .evaluation import cw_test, dm_test, evaluation_table, mae, mape, pinball_loss, rmse
from .forecasting import make_lags, rolling_forecast
from .metadata import MetadataPanel, VariableMeta
from .midas import (
    BetaMIDASRegressor,
    BetaMidasResult,
    MultiBetaMidasResult,
    almon_weights,
    beta_weights,
    extract_midas_arrays,
    fit_beta_midas,
    fit_beta_midas_multi,
    legendre_dictionary,
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
from .nowcast_plots import (
    plot_forecast_fan,
    plot_quarters_grid,
    plot_release_cycle_rmse,
    plot_vintage_panels,
    plot_vintage_tracking,
    plot_rmse_by_origin,
    plot_tracking,
    plot_vintage_evolution,
)
from .panel import PanelArtifacts, PanelBuilder
from .interp_plots import (plot_rank_evolution, plot_relative_rmse,
                           plot_selection_evolution, plot_winner_heatmap)
from .plotting import PALETTE, save_for_slides, shade_periods, use_aer_style
from .realtime import RealtimeEngine
from .recession import build_recession_target, load_recession_panel
from .boosted_midas import BoostedMIDASNowcaster
from .scmidas import SCMIDASNowcaster
from .sparse_midas import SparseGroupLasso, SparseMIDASNowcaster
from .tables import horizon_rmse_table, horizon_table_latex, stars

__all__ = [
    "COUNTRIES",
    "SERIES",
    "TOPIC_COLUMNS",
    "PALETTE",
    "NBS_INDICATORS",
    "PanelArtifacts",
    "PanelBuilder",
    "BaseNowcaster",
    "InformationSet",
    "NowcastResult",
    "MetadataPanel",
    "VariableMeta",
    "RealtimeEngine",
    "RandomWalkNowcaster",
    "HistoricalMeanNowcaster",
    "ARNowcaster",
    "BridgeNowcaster",
    "ADLMIDASNowcaster",
    "MonthlyLevelARNowcaster",
    "MonthlyARNowcaster",
    "PooledMIDASNowcaster",
    "PooledLevelBridgeNowcaster",
    "PooledBridgeNowcaster",
    "DFMNowcaster",
    "GBTreesNowcaster",
    "SCMIDASNowcaster",
    "BoostedMIDASNowcaster",
    "SparseMIDASNowcaster",
    "SparseGroupLasso",
    "CombinationNowcaster",
    "combine_backtest",
    "add_information_index",
    "combine_release_cycle",
    "plot_weight_evolution",
    "legendre_dictionary",
    "run_backtest",
    "run_horizon_backtest",
    "run_release_cycle_backtest",
    "release_cycle_rmse",
    "quarter_timestamp",
    "make_origin_grid",
    "make_daily_origin_grid",
    "publication_date",
    "plot_release_cycle_rmse",
    "plot_vintage_evolution",
    "plot_forecast_fan",
    "plot_vintage_tracking",
    "plot_vintage_panels",
    "horizon_bands",
    "horizon_rmse_table",
    "horizon_table_latex",
    "bootstrap_quantiles",
    "release_cycle_bands",
    "plot_quarters_grid",
    "plot_tracking",
    "plot_rmse_by_origin",
    "BetaMIDASRegressor",
    "BetaMidasResult",
    "MultiBetaMidasResult",
    "use_aer_style",
    "plot_selection_evolution",
    "plot_rank_evolution",
    "plot_winner_heatmap",
    "plot_relative_rmse",
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
    "cw_test",
    "dm_test",
    "evaluation_table",
    "fetch_fred_series",
    "load_macro_panel",
    "load_panel",
    "NBSChinaError",
    "NBSIndicatorSpec",
    "available_nbs_indicators",
    "get_nbs_data",
    "load_conflict_topics",
    "aggregate_conflict_topics",
    "download_epu",
    "load_epu",
    "load_epu_panel",
    "epu_wide",
    "load_recession_panel",
    "build_recession_target",
]
