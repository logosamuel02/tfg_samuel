import plotly.express as px
import plotly.graph_objects as go
import pingouin as pg
import numpy as np
import plotly.figure_factory as ff
import warnings
from typing import List, Dict
from pandas._libs.tslibs.timestamps import Timestamp
from statsmodels.multivariate.multivariate_ols import MultivariateTestResults
from pandas.core.frame import DataFrame
from plotly.graph_objects import Figure

import preprocess as pre
import loaders as load
from export import Config, actualize_figure
from preprocess import clean

warnings.filterwarnings("ignore")
config: Config = Config()


#########################
######### ANOVA #########
#########################


def create_summary_table(
    factor1: str = "distribution",
    factor2: str = "ann",
    level: str = "maximum_accuracy_achieved",
    operation_level: str = "max",
    **karg,
) -> Figure:
    atable = pre.anova_table(factor1, factor2, level, operation_level)
    atable.columns = [clean(var) for var in atable.columns]
    fig: Figure = ff.create_table(atable)
    fig.update_layout(
        title_text=f"Table of {clean(level)} achieved grouped by {clean(factor1)} and {clean(factor2)}"
    )
    if "name" not in karg:
        karg["name"] = "summary_atable"
    actualize_figure(fig, karg["name"])
    return fig


def create_interaction_plot(
    factor1: str = "distribution",
    factor2: str = "ann",
    level: str = "maximum_accuracy_achieved",
    operation_level: str = "max",
    **karg,
) -> Figure:
    atable = pre.anova_table(factor1, factor2, level, operation_level)
    fig: Figure = px.scatter(
        atable,
        x=factor1,
        y="level",
        color=factor2,
    ).update_traces(mode="lines+markers")
    fig.update_layout(
        title_text=f"Interaction between {clean(factor1)} and {clean(factor2)}"
    )
    fig.update_layout(config.plots["interaction_plot"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables[factor1]["legend"],
        yaxis_title_text=config.variables[level]["legend"],
        legend_title_text=config.variables[factor2]["legend"],
    )
    if "name" not in karg:
        karg["name"] = "interaction_plot"
    actualize_figure(fig, karg["name"])
    return fig


def create_box_plot(
    factor1: str = "distribution",
    level: str = "maximum_accuracy_achieved",
    **karg,
) -> Figure:
    df = pre.df_anova()
    fig: Figure = px.box(
        df,
        x=factor1,
        y=level,
        color=factor1,
        points="all",
    )
    fig.update_layout(title_text=f"Distribution of {clean(level)} by {clean(factor1)}")
    fig.update_layout(config.plots["box_plot"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables[factor1]["legend"],
        yaxis_title_text=config.variables[level]["legend"],
        legend_title_text=config.variables[factor1]["legend"],
    )
    if "name" not in karg:
        karg["name"] = "box_plot"
    actualize_figure(fig, karg["name"])
    if "name" not in karg:
        karg["name"] = "box_plot"
    actualize_figure(fig, karg["name"])
    return fig


def create_anova(
    factor1: str = "distribution",
    factor2: str = "ann",
    level: str = "maximum_accuracy_achieved",
    **karg,
) -> Figure:
    df = pre.df_anova()
    atable: DataFrame = pg.anova(
        data=df,
        dv=level,
        between=[factor1, factor2],
        detailed=True,
    ).round(4)
    atable_c: DataFrame = atable.copy()
    atable_c.columns = [clean(var) for var in atable_c.columns]
    fig: Figure = ff.create_table(atable_c)
    fig.update_layout(config.plots["anova_table"]["layout"])
    if "name" not in karg:
        karg["name"] = "anova_table"
    actualize_figure(fig, karg["name"])
    return fig


def create_tuckey_test(
    factor1: str = "distribution", level: str = "maximum_accuracy_achieved", **karg
) -> Figure:
    df = pre.df_anova()
    pg_test: DataFrame = pg.pairwise_tukey(data=df, dv=level, between=factor1).round(3)
    pg_test_c: DataFrame = pg_test.copy()
    pg_test_c.columns = [clean(var) for var in pg_test_c.columns]
    fig: Figure = ff.create_table(pg_test_c)
    fig.update_layout(config.plots["tuckey_test"]["layout"])
    if "name" not in karg:
        karg["name"] = "tuckey_test"
    actualize_figure(fig, karg["name"])
    return fig


def create_nemenyi_test(
    factor1: str = "distribution",
    level: str = "maximum_accuracy_achieved",
    color_scale: str = "Viridis",
    **karg,
) -> Figure:
    nemtable, options = pre.nemenyi_test(factor1, level)
    fig: Figure = px.imshow(
        nemtable,
        x=options,
        y=options,
        color_continuous_scale=color_scale,
        aspect="auto",
    )
    fig.update_traces(text=nemtable, texttemplate="%{text}")
    fig.update_xaxes(side="top")
    fig.update_layout(config.plots["nemenyi_test"]["layout"])
    if "name" not in karg:
        karg["name"] = "nemenyi_test"
    actualize_figure(fig, karg["name"])
    return fig


def create_manova_figure(factor1: str = "distribution", **karg) -> List[Figure]:
    result: MultivariateTestResults = pre.manova_table(factor1)
    var_results: DataFrame = result.results[factor1]["stat"]
    var_results.reset_index(inplace=True)
    var_results = var_results.rename(columns={"index": "tests"})
    fig: Figure = ff.create_table(var_results.round(3))
    fig.update_layout(config.plots["manova_table"]["layout"])
    if "name" not in karg:
        karg["name"] = "manova_table"
    actualize_figure(fig, karg["name"])
    return fig


def create_manova_figure_intercept(
    factor1: str = "distribution", **karg
) -> List[Figure]:
    result: MultivariateTestResults = pre.manova_table(factor1)
    var_results: DataFrame = result.results["Intercept"]["stat"]
    var_results.reset_index(inplace=True)
    var_results = var_results.rename(columns={"index": "tests"})
    fig: Figure = ff.create_table(var_results.round(3))
    fig.update_layout(config.plots["manova_table_intercept"]["layout"])
    if "name" not in karg:
        karg["name"] = "manova_table_intercept"
    actualize_figure(fig, karg["name"])
    return fig


#########################
####### ALGORITHM #######
#########################


def violin_plot(
    box_visible: str = "true",
    meanline_visible: str = "true",
    **karg,
) -> Figure:
    df_bolos, agents = pre.violin_plot()
    fig: Figure = go.Figure()
    for agent in agents:
        fig.add_trace(
            go.Violin(
                x=df_bolos["variable"][df_bolos["variable"] == agent],
                y=df_bolos["value"][df_bolos["variable"] == agent],
                name=agent,
                box_visible=box_visible == "true",
                meanline_visible=meanline_visible == "true",
            )
        )
    fig.update_layout(config.plots["violin_plot"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables["agent"]["legend"],
        yaxis_title_text=config.variables["seconds_to_complete"]["legend"],
        legend_title_text=config.variables["agent"]["legend"],
    )
    if "name" not in karg:
        karg["name"] = "violin_plot"
    actualize_figure(fig, karg["name"])
    return fig


def execution_time_plot(orientation: str = "h", **karg) -> Figure:
    data = pre.execution_time_plot()
    if orientation == "h":
        fig: Figure = px.bar(
            data,
            x="seconds_elapsed",
            y="agent",
            hover_data=["seconds", "seconds_elapsed"],
            color="agent",
            labels={"pop": "seconds"},
            text="seconds",
            orientation=orientation,
        )
    else:
        fig: Figure = px.bar(
            data,
            x="agent",
            y="seconds_elapsed",
            hover_data=["seconds", "seconds_elapsed"],
            color="agent",
            labels={"pop": "seconds"},
            text="seconds",
            orientation=orientation,
        )

    fig.update_layout(config.plots["execution_time_plot"]["layout"])
    fig.update_traces(config.plots["execution_time_plot"]["traces"])
    fig.update_layout(
        xaxis_title_text=config.variables["seconds_elapsed"]["legend"],
        yaxis_title_text=config.variables["agent"]["legend"],
        legend_title_text=config.variables["agent"]["legend"],
    )
    if "name" not in karg:
        karg["name"] = "execution_time_plot"
    actualize_figure(fig, karg["name"])
    return fig


#########################
###### DATA_SPLIT #######
#########################


def create_bubble_plot(phase: str = "train", **karg) -> List[Figure]:
    colors = pre.bubble_colors()
    fig: Figure = go.Figure()
    agents, labels, x, y, colores, sizes, texts = pre.bubble_interprocess(phase)
    fig.add_trace(
        go.Scatter(
            mode="markers+text",
            x=x,
            y=y,
            marker=dict(color=colores, size=sizes, opacity=1),
            text=texts,
            line=dict(color="black", width=1),
            hovertemplate="<br><b>Agent</b>: %{x}<br>"
            + "<br><b>Digit</b>: %{y}<br>"
            + "<br><b>Samples</b>: %{text}<br>",
        )
    )

    fig.update_layout(
        title_text=f"Distribution of samples for {phase}",
        xaxis_title="Agents",
        yaxis_title="Target labels",
    )

    width_factor: int = config.plots["bubble_plot"]["size_factors"]["width"]
    height_factor: int = config.plots["bubble_plot"]["size_factors"]["height"]
    fig.update_layout(xaxis_range=[0, len(agents) + 1])
    fig.update_layout(yaxis_range=[0, len(colors) - 1])
    fig.update_layout(width=width_factor * (len(agents) + 2))
    fig.update_layout(height=height_factor * len(colors))
    fig.update_xaxes(
        ticktext=agents,
        tickvals=list(range(1, len(agents) + 1)),
    )
    fig.update_yaxes(
        ticktext=sorted(labels),
        tickvals=list(range(1, len(labels) + 1)),
    )
    fig.update_layout(config.plots["bubble_plot"]["layout"])
    fig.update_xaxes(config.plots["bubble_plot"]["axes"])
    fig.update_yaxes(config.plots["bubble_plot"]["axes"])
    if "name" not in karg:
        karg["name"] = "bubble_plot"
    actualize_figure(fig, karg["name"])
    return fig


#########################
####### MESSAGES ########
#########################


def heatmap_messages(color_scale: str = "Viridis", **karg) -> Figure:
    agents, data_cross = pre.heatmap_messages()
    fig: Figure = px.imshow(
        data_cross.to_numpy(),
        x=agents,
        y=agents,
        color_continuous_scale=color_scale,
        aspect="auto",
    )
    fig.update_traces(text=data_cross, texttemplate="<b>%{text}</b>")
    fig.update_xaxes(side="top")
    fig.update_layout(config.plots["heatmap_messages"]["layout"])
    if "name" not in karg:
        karg["name"] = "heatmap_messages"
    actualize_figure(fig, karg["name"])
    return fig


def heatmap_sizes(
    unit: str = "MB",
    color_scale: str = "Viridis",
    **karg,
) -> Figure:
    units = {}
    agents, data_cross = pre.heatmap_sizes(unit)
    fig: Figure = px.imshow(
        data_cross.to_numpy(),
        x=agents,
        y=agents,
        color_continuous_scale=color_scale,
        aspect="auto",
    )
    fig.update_traces(text=data_cross, texttemplate="<b>%{text}</b>")
    fig.update_xaxes(side="top")
    fig.update_layout(config.plots["heatmap_sizes"]["layout"])
    if "name" not in karg:
        karg["name"] = "heatmap_sizes"
    actualize_figure(fig, karg["name"])
    return fig


def statistics_messages(**karg) -> Figure:
    stats = pre.statistics_messages()
    fig: Figure = ff.create_table(stats)
    fig.update_layout(config.plots["statistics_messages"]["layout"])
    if "name" not in karg:
        karg["name"] = "statistics_messages"
    actualize_figure(fig, karg["name"])
    return fig


def distribution_messages(color: str = "green", **karg) -> Figure:
    data = pre.distribution_data_df()
    fig: Figure = px.histogram(data, x="timestamp_minutes")
    fig.update_layout(config.plots["distribution_messages"]["layout"])
    fig.update_traces(marker_color=color)
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text="Number of messages",
    )
    if "name" not in karg:
        karg["name"] = "distribution_messages"
    actualize_figure(fig, karg["name"])
    return fig


def distribution_messages_types(**karg) -> Figure:
    data = pre.distribution_data_df()
    fig: Figure = px.histogram(data, x="timestamp_minutes", color="type")
    fig.update_layout(config.plots["distribution_messages_types"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text="Number of messages",
        legend_title_text="Type of messages",
    )
    if "name" not in karg:
        karg["name"] = "distribution_messages_types"
    actualize_figure(fig, karg["name"])
    return fig


def distribution_info(color: str = "green", **karg) -> Figure:
    data = pre.distribution_data_df()
    fig: Figure = px.histogram(
        data, x="timestamp_minutes", y="size", color_discrete_sequence=[color]
    )
    fig.update_layout(config.plots["distribution_info"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text="Amount of information",
    )
    if "name" not in karg:
        karg["name"] = "distribution_info"
    actualize_figure(fig, karg["name"])
    return fig


def distribution_info_type(unit: str = "B", **karg) -> Figure:
    data = pre.distribution_data_df()
    dic_sizes = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    data.size = round(data.size / dic_sizes[unit], 3)
    fig: Figure = px.histogram(data, x="timestamp_minutes", y="size", color="type")
    fig.update_layout(config.plots["distribution_info_type"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text=f"Amount of information {unit}",
        legend_title_text="Type of messages",
    )
    if "name" not in karg:
        karg["name"] = "distribution_info_type"
    actualize_figure(fig, karg["name"])
    return fig


#########################
###### CONVERGENCE ######
#########################


def create_xy_scatter_plot(layer: str = "conv1", **karg) -> Figure:
    layers_opts, data = pre.convergence_df(layer)
    min_x = data[layers_opts[layer][0]].min()
    min_y = data[layers_opts[layer][1]].min()
    max_x = data[layers_opts[layer][0]].max()
    max_y = data[layers_opts[layer][1]].max()
    fig: Figure = px.scatter(
        data,
        x=layers_opts[layer][0],
        y=layers_opts[layer][1],
        animation_frame="algorithm_round",
        color="agent",
        hover_name="agent",
        range_x=[min_x, max_x],
        range_y=[min_y, max_y],
    )
    fig.update_layout(title_text=f"{layer.upper()} layer evolution of agents")
    fig.update_layout(config.plots["xy_scatter"]["layout"])
    fig.update_layout(
        xaxis_title_text=clean(layers_opts[layer][0]),
        yaxis_title_text=clean(layers_opts[layer][1]),
        legend_title_text=config.variables["agent"]["legend"],
    )
    if "name" not in karg:
        karg["name"] = "xy_scatter"
    actualize_figure(fig, karg["name"])
    return fig


def create_scatter_plot(
    layer: str = "conv1", sublayer: str = "weight", **karg
) -> Figure:
    _, data = pre.convergence_df(layer)
    values = f"{layer}.{sublayer}"
    min_y = data[values].max()
    max_y = data[values].max()
    scatter_plot: Figure = px.scatter(
        data,
        x="algorithm_round",
        y=values,
        animation_frame="frame",
        color="agent",
        hover_name="agent",
        range_x=[0, len(data.algorithm_round.unique()) + 1],
        range_y=[min_y, max_y],
    )

    for frame in scatter_plot.frames:
        for data in frame.data:
            data.update(mode="markers", showlegend=True, opacity=1)
            data["x"] = np.take(data["x"], [-1])
            data["y"] = np.take(data["y"], [-1])
    if "name" not in karg:
        karg["name"] = "scatter_plot"
    actualize_figure(scatter_plot, karg["name"])
    return scatter_plot


def create_line_plot(layer: str = "conv1", sublayer: str = "weight", **karg) -> Figure:
    _, data = pre.convergence_df()
    values = f"{layer}.{sublayer}"
    min_y = data[values].min()
    max_y = data[values].max()
    line_plot: Figure = px.line(
        data,
        x="algorithm_round",
        y=values,
        color="agent",
        animation_frame="frame",
        range_x=[0, len(data.algorithm_round.unique()) + 1],
        range_y=[min_y, max_y],
        line_shape="spline",  # make a line graph curvy
    )

    line_plot.update_traces(showlegend=False)  # legend will be from line graph
    for frame in line_plot.frames:
        for data in frame.data:
            data.update(mode="lines", opacity=0.8, showlegend=False)

    if "name" not in karg:
        karg["name"] = "line_plot"
    actualize_figure(line_plot, karg["name"])
    return line_plot


def create_combined_plot(
    scatter_plot: Figure = None,
    line_plot: Figure = None,
    layer: str = "conv1",
    sublayer: str = "weight",
    **karg,
) -> Figure:
    if scatter_plot is None:
        scatter_plot = create_scatter_plot(layer, sublayer)
    if line_plot is None:
        line_plot = create_line_plot(layer, sublayer)
    combined_plot: Figure = go.Figure(
        data=line_plot.data + scatter_plot.data,
        frames=[
            go.Frame(data=line_plot.data + scatter_plot.data, name=scatter_plot.name)
            for line_plot, scatter_plot in zip(line_plot.frames, scatter_plot.frames)
        ],
        layout=line_plot.layout,
    )

    combined_plot.update_yaxes(config.plots["combined_plot"]["yaxes"])

    combined_plot.update_xaxes(config.plots["combined_plot"]["xaxes"])

    combined_plot.update_traces(config.plots["combined_plot"]["traces"])

    combined_plot.layout.updatemenus[0].buttons[0]["args"][1]["frame"]["duration"] = (
        config.plots["combined_plot"]["updatemenus"]["frame_duration"]
    )
    combined_plot.layout.updatemenus[0].buttons[0]["args"][1]["transition"][
        "duration"
    ] = config.plots["combined_plot"]["updatemenus"]["transition_duration"]
    combined_plot.layout.updatemenus[0].buttons[0]["args"][1]["transition"][
        "redraw"
    ] = config.plots["combined_plot"]["updatemenus"]["redraw"]
    combined_plot.update_layout(
        title_text=f"{sublayer.upper()} layer evolution of agents"
    )
    combined_plot.update_layout(
        xaxis_title_text=config.variables["algorithm_round"]["legend"],
        yaxis_title_text=clean(sublayer),
        legend_title_text=config.variables["agent"]["legend"],
    )
    combined_plot.update_layout(config.plots["combined_plot"]["layout"])
    if "name" not in karg:
        karg["name"] = "combined_plot"
    actualize_figure(combined_plot, karg["name"])
    return combined_plot


#########################
####### INFERENCE #######
#########################


def train_by_agent(metric: str = "accuracy", **karg) -> List[Figure]:
    train = load.train_dataset()
    figs = []
    fig: Figure = px.line(
        train,
        x="algorithm_round",
        y=metric,
        color="agent",
        title=f"{config.variables[metric]['legend']} evolution across rounds by agent",
    )
    fig.update_layout(config.plots["train_plot"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables["algorithm_round"]["legend"],
        yaxis_title_text=config.variables[metric]["legend"],
        legend_title_text=config.variables["agent"]["legend"],
    )
    if "name" not in karg:
        karg["name"] = "train_plot"
    actualize_figure(fig, karg["name"])
    return fig


def test_by_agent(metric: str = "accuracy", **karg) -> List[Figure]:
    test = load.inference_dataset()
    metric = f"test_{metric}"
    figs = []
    fig: Figure = px.line(
        test,
        x="algorithm_round",
        y=metric,
        color="agent",
        title=f"{config.variables[metric]['legend']} evolution across rounds by agent",
    )
    fig.update_layout(config.plots["test_plot"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables["algorithm_round"]["legend"],
        yaxis_title_text=config.variables[metric]["legend"],
        legend_title_text=config.variables["agent"]["legend"],
    )
    if "name" not in karg:
        karg["name"] = "test_plot"
    actualize_figure(fig, karg["name"])
    return fig


def train_test_plot(metric: str = "accuracy", **karg) -> List[Figure]:
    train = load.train_dataset()
    test = load.inference_dataset()

    g_train: DataFrame = (
        train.groupby(["algorithm_round"])
        .agg(
            maximum_accuracy=(metric, "max"),
            minimum_accuracy=(metric, "min"),
            mean_accuracy=(metric, "mean"),
        )
        .reset_index()
    )
    g_test: DataFrame = (
        test.groupby(["algorithm_round"])
        .agg(
            maximum_accuracy=(f"test_{metric}", "max"),
            minimum_accuracy=(f"test_{metric}", "min"),
            mean_accuracy=(f"test_{metric}", "mean"),
        )
        .reset_index()
    )

    x: List[int] = list(g_test.algorithm_round.unique())
    x_rev: List[int] = x[::-1]

    nmax: List[float] = g_train.maximum_accuracy.to_list()
    nmin: List[float] = g_train.minimum_accuracy.to_list()
    nmin = nmin[::-1]

    tmax: List[float] = g_test.maximum_accuracy.to_list()
    tmin: List[float] = g_test.minimum_accuracy.to_list()
    tmin = tmin[::-1]

    fig: Figure = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x + x_rev,
            y=nmax + nmin,
            fill="toself",
            fillcolor=config.plots["train_test_plot"]["traces"]["train_fillcolor"],
            line_color=config.plots["train_test_plot"]["traces"]["train_linecolor"],
            name="Train",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x + x_rev,
            y=tmax + tmin,
            fill="toself",
            fillcolor=config.plots["train_test_plot"]["traces"]["test_fillcolor"],
            line_color=config.plots["train_test_plot"]["traces"]["test_linecolor"],
            showlegend=False,
            name="Test",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=g_train.mean_accuracy,
            line_color=config.plots["train_test_plot"]["traces"][
                "mean_train_linecolor"
            ],
            name="Train",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=g_test.mean_accuracy,
            line_color=config.plots["train_test_plot"]["traces"]["mean_test_linecolor"],
            name="Test",
        )
    )

    fig.update_traces(mode="lines")
    fig.update_layout(title_text=f"Netowk's performance {clean(metric)} by round")
    fig.update_layout(config.plots["train_test_plot"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables["algorithm_round"]["legend"],
        yaxis_title_text=config.variables[metric]["legend"],
        legend_title_text=config.variables["agent"]["legend"],
    )
    if "name" not in karg:
        karg["name"] = "train_test_plot"
    actualize_figure(fig, karg["name"])
    return fig


#########################
####### NETWORK #########
#########################


def create_nodes_plot(
    phase: str = "train",
    metric: str = "accuracy",
    msg_type: str = "SEND-LAYERS",
    seed: str = "15",
    **karg,
) -> Figure:
    x_coords, y_coords = pre.create_network_coordinates(seed)
    nodes, edges, timestamps = pre.create_network_artifacts(phase, metric, msg_type)

    if phase != "train":
        metric = f"{phase}_{metric}"
    nodes, range_color = pre.nodes_panel_data(
        nodes, timestamps, x_coords, y_coords, metric
    )

    nodes_plot: Figure = px.scatter(
        nodes,
        x="X",
        y="Y",
        animation_frame="timestamp",
        labels={"timestamp": "Second"},
        text="agent",
        color=metric,
        size="size",
        title=f"{config.variables[metric]['legend']} evolution inside agents network",
        hover_name="agent",
        range_color=range_color,
    )
    nodes_plot.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = (
        config.plots["nodes_plot"]["other"]["frame_duration"]
    )
    xmax, xmin = max(x_coords.values()), min(x_coords.values())
    ymax, ymin = max(y_coords.values()), min(y_coords.values())

    nodes_plot.update_traces(config.plots["nodes_plot"]["traces"])

    i: float = config.plots["nodes_plot"]["other"]["border"]
    nodes_plot.update_xaxes(range=[xmin - i, xmax + i])
    nodes_plot.update_yaxes(range=[ymin - i, ymax + i])

    nodes_plot.update_xaxes(config.plots["nodes_plot"]["axes"])
    nodes_plot.update_yaxes(config.plots["nodes_plot"]["axes"])

    marker_size = config.plots["nodes_plot"]["other"]["marker_size"]
    nodes_plot.for_each_trace(lambda trace: trace.update(marker_size=marker_size))
    nodes_plot.update_layout(
        coloraxis_colorbar_title=config.variables[metric]["legend"]
    )

    nodes_plot.update_layout(config.plots["nodes_plot"]["layout"])
    if "name" not in karg:
        karg["name"] = "nodes_plot"
    actualize_figure(nodes_plot, karg["name"])
    return nodes_plot


def create_edges_plot(
    phase: str = "train",
    metric: str = "accuracy",
    msg_type: str = "SEND-LAYERS",
    seed: str = "15",
    **karg,
) -> Figure:
    x_coords, y_coords = pre.create_network_coordinates(seed)
    nodes, edges, timestamps = pre.create_network_artifacts(
        phase=phase, metric=metric, msg_type=msg_type
    )
    edges = pre.edges_panel_data(edges, timestamps, x_coords, y_coords)

    def new_value(value: int) -> int:
        OldMin: int = edges.weight.min()
        OldMax: int = edges.weight.max()
        OldRange: int = OldMax - OldMin
        NewMin: int = 0
        NewMax: int = config.plots["edges_plot"]["other"]["max_size_lines"]
        NewRange: int = NewMax - NewMin
        return round((((value - OldMin) * NewRange) / OldRange) + NewMin)

    edges_plot: Figure = go.Figure()
    links: List[str] = edges.edge.unique()
    times: List[Timestamp] = edges.timestamp.unique()
    # TRACES
    for link in links:
        x: List[float] = edges[(edges.edge == link)].head(1).X.values[0]
        y: List[float] = edges[(edges.edge == link)].head(1).Y.values[0]
        edges_plot.add_scatter(x=x, y=y, name=str(link), mode="lines", line_width=1)

    # SLIDER
    sliders_dict: Dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Second:",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 25, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }

    # FRAMES
    frames = []
    for time in times:
        data = []
        for link in links:
            row: DataFrame = edges[(edges.edge == link) & (edges.timestamp == time)]
            data.append(
                go.Scatter(
                    x=row.X.values[0],
                    y=row.Y.values[0],
                    name=f"{str(link)} - {row.weight.values[0]}",
                    mode="lines",
                    line_width=new_value(row.weight.values[0]),
                )
            )
        frames.append(
            go.Frame(data=data, traces=list(range(len(links))), name=str(time))
        )
        slider_step: Dict[str, List] = {
            "args": [
                [str(time)],
                {
                    "mode": "immediate",
                    "transition": {
                        "duration": config.plots["edges_plot"]["other"][
                            "transition_duration"
                        ]
                    },
                },
            ],
            "label": str(time),
            "method": "animate",
        }
        sliders_dict["steps"].append(slider_step)

    edges_plot.update(frames=frames)
    updatemenus: List[Dict] = [
        dict(
            buttons=[
                dict(
                    args=[
                        None,
                        {
                            "frame": {
                                "duration": config.plots["edges_plot"]["other"][
                                    "frame_duration"
                                ],
                                "redraw": True,
                            },
                            "fromcurrent": True,
                        },
                    ],
                    label="Play",
                    method="animate",
                ),
                dict(
                    args=[
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    label="Pause",
                    method="animate",
                ),
            ],
            direction="left",
            pad={"r": 10, "t": 87},
            showactive=False,
            type="buttons",
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top",
        )
    ]

    edges_plot.update_layout(
        updatemenus=updatemenus,
        sliders=[sliders_dict],
        title_text=f"{clean(msg_type)} messages evolution inside agents network",
    )
    xmax, xmin = max(x_coords.values()), min(x_coords.values())
    ymax, ymin = max(y_coords.values()), min(y_coords.values())

    i: float = config.plots["edges_plot"]["other"]["border"]
    edges_plot.update_xaxes(range=[xmin - i, xmax + i])
    edges_plot.update_yaxes(range=[ymin - i, ymax + i])

    edges_plot.update_xaxes(config.plots["edges_plot"]["axes"])
    edges_plot.update_yaxes(config.plots["edges_plot"]["axes"])

    edges_plot.update_traces(config.plots["edges_plot"]["traces"])
    edges_plot.update_layout(coloraxis_colorbar_title=f"Test {metric}")
    edges_plot.update_layout(config.plots["edges_plot"]["layout"])
    if "name" not in karg:
        karg["name"] = "edges_plot"
    actualize_figure(edges_plot, karg["name"])
    return edges_plot


def create_network_plot(
    nodes_plot: Figure | None = None,
    edges_plot: Figure | None = None,
    phase: str = "train",
    metric: str = "accuracy",
    msg_type: str = "SEND-LAYERS",
    seed: str = "15",
    **karg,
) -> Figure:
    if nodes_plot is None:
        nodes_plot = create_nodes_plot(phase, metric, msg_type, seed)
    if edges_plot is None:
        edges_plot = create_edges_plot(phase, metric, msg_type, seed)

    x_coords, y_coords = pre.create_network_coordinates(seed)
    nodes, edges, timestamps = pre.create_network_artifacts(phase, metric, msg_type)

    if phase != "train":
        metric = f"{phase}_{metric}"

    nodes, range_color = pre.nodes_panel_data(
        nodes, timestamps, x_coords, y_coords, metric
    )
    combined_plot: Figure = go.Figure(
        data=edges_plot.data + nodes_plot.data,
        frames=[
            go.Frame(data=edges_plot.data + nodes_plot.data, name=nodes_plot.name)
            for edges_plot, nodes_plot in zip(edges_plot.frames, nodes_plot.frames)
        ],
        layout=edges_plot.layout,
    )
    combined_plot.for_each_trace(
        lambda trace: trace.update(
            marker=config.plots["network_plot"]["traces"]["marker"],
        )
    )
    combined_plot.update_layout(
        coloraxis=dict(
            colorbar=dict(x=-0.15, title=config.variables[metric]["legend"]),
            cmin=range_color[0],
            cmax=range_color[1],
        ),
        title_text=f"Network evolution {config.variables[metric]['legend']} and {clean(msg_type)} messages",
    )
    xmax, xmin = max(x_coords.values()), min(x_coords.values())
    ymax, ymin = max(y_coords.values()), min(y_coords.values())
    i: float = 0.12
    combined_plot.update_xaxes(range=[xmin - i, xmax + i])
    combined_plot.update_yaxes(range=[ymin - i, ymax + i])

    combined_plot.update_layout(config.plots["network_plot"]["layout"])
    if "name" not in karg:
        karg["name"] = "network_plot"
    actualize_figure(combined_plot, karg["name"])
    return combined_plot


#########################
######### EXTRA #########
#########################
