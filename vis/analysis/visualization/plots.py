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
        y=level,
        color=factor2,
    ).update_traces(mode="lines+markers")
    fig.update_layout(
        title_text=f"Interaction between {clean(factor1)} and {clean(factor2)}"
    )
    fig.update_layout(config.plots["interaction"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables[factor1]["legend"],
        yaxis_title_text=config.variables[level]["legend"],
        legend_title_text=config.variables[factor2]["legend"],
    )
    return fig


def create_box_plot(factor: str, **karg) -> Figure:
    df = pre.df_anova()
    fig: Figure = px.box(
        df,
        x=factor,
        y="maximum_accuracy_achieved",
        color=factor,
        points="all",
    )
    fig.update_layout(
        title_text=f"Distribution of {clean('maximum_accuracy_achieved')} by {clean(factor)}"
    )
    fig.update_layout(config.plots["anova"]["box"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables[factor]["legend"],
        yaxis_title_text=config.variables["maximum_accuracy_achieved"]["legend"],
        legend_title_text=config.variables[factor]["legend"],
    )
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
    fig.update_layout(config.plots[karg["name"]]["layout"])
    actualize_figure(fig)
    return fig


def create_tuckey_test(
    factor: str = "distribution", level: str = "maximum_accuracy_achieved", **karg
) -> Figure:
    df = pre.df_anova()
    pg_test: DataFrame = pg.pairwise_tukey(data=df, dv=level, between=factor).round(3)
    pg_test_c: DataFrame = pg_test.copy()
    pg_test_c.columns = [clean(var) for var in pg_test_c.columns]
    fig: Figure = ff.create_table(pg_test_c)
    fig.update_layout(config.plots["anova"]["tuckey"]["layout"])
    return fig


def create_nemenyi_test(**karg) -> Figure:
    nemtable, options = pre.nemenyi_test()
    fig: Figure = px.imshow(
        nemtable, x=options, y=options, color_continuous_scale="Viridis", aspect="auto"
    )
    fig.update_traces(text=nemtable, texttemplate="%{text}")
    fig.update_xaxes(side="top")
    fig.update_layout(config.plots["anova"]["nemenyi"]["layout"])
    return fig


def create_manova_figs(var: str, **karg) -> List[Figure]:
    result: MultivariateTestResults = pre.manova_table()
    intercept: DataFrame = result.results["Intercept"]["stat"]
    intercept.reset_index(inplace=True)
    intercept = intercept.rename(columns={"index": "tests"})
    fig1: Figure = ff.create_table(intercept.round(3))
    fig1.update_layout(config.plots["anova"]["manova_inter"]["layout"])

    var_results: DataFrame = result.results[var]["stat"]
    var_results.reset_index(inplace=True)
    var_results = var_results.rename(columns={"index": "tests"})
    fig2: Figure = ff.create_table(var_results.round(3))
    fig2.update_layout(config.plots["anova"]["manova_var"]["layout"])
    return [fig1, fig2]


#########################
####### ALGORITHM #######
#########################


def violin_plot(**karg) -> Figure:
    df_bolos, agents = pre.violin_plot()
    fig: Figure = go.Figure()
    for agent in agents:
        fig.add_trace(
            go.Violin(
                x=df_bolos["variable"][df_bolos["variable"] == agent],
                y=df_bolos["value"][df_bolos["variable"] == agent],
                name=agent,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig.update_layout(config.plots["algorithm"]["violin"]["layout"])
    fig.update_traces(config.plots["algorithm"]["violin"]["traces"])
    fig.update_layout(
        xaxis_title_text=config.variables["agent"]["legend"],
        yaxis_title_text=config.variables["seconds_to_complete"]["legend"],
        legend_title_text=config.variables["agent"]["legend"],
    )
    return fig


def execution_time_plot(**karg) -> Figure:
    data = pre.execution_time_plot()
    fig: Figure = px.bar(
        data,
        x="seconds_elapsed",
        y="agent",
        hover_data=["seconds", "seconds_elapsed"],
        color="agent",
        labels={"pop": "seconds"},
        text="seconds",
        orientation="h",
    )
    fig.update_layout(config.plots["algorithm"]["bar"]["layout"])
    fig.update_traces(config.plots["algorithm"]["bar"]["traces"])
    fig.update_layout(
        xaxis_title_text=config.variables["seconds_elapsed"]["legend"],
        yaxis_title_text=config.variables["agent"]["legend"],
        legend_title_text=config.variables["agent"]["legend"],
    )
    return fig


#########################
###### DATA_SPLIT #######
#########################


def create_bubble_plots(**karg) -> List[Figure]:
    colors = pre.bubble_colors()
    figs = []
    phases: List[str] = ["train", "validation", "test"]
    visible: bool = True
    for phase in phases:
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
                visible=visible,
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

        width_factor: int = config.plots["data_split"]["bubble"]["size_factors"][
            "width"
        ]
        height_factor: int = config.plots["data_split"]["bubble"]["size_factors"][
            "height"
        ]
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
        fig.update_layout(config.plots["data_split"]["bubble"]["layout"])
        fig.update_xaxes(config.plots["data_split"]["bubble"]["axes"])
        fig.update_yaxes(config.plots["data_split"]["bubble"]["axes"])
        figs.append(fig)
    return figs


#########################
####### MESSAGES ########
#########################


def heatmap_messages(**karg) -> Figure:
    agents, data_cross = pre.heatmap_messages()
    fig: Figure = px.imshow(
        data_cross.to_numpy(),
        x=agents,
        y=agents,
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    fig.update_traces(text=data_cross, texttemplate="<b>%{text}</b>")
    fig.update_xaxes(side="top")
    fig.update_layout(config.plots["messages"]["heat_msg"]["layout"])
    return fig


def heatmap_sizes(**karg) -> Figure:
    agents, data_cross = pre.heatmap_sizes()
    fig: Figure = px.imshow(
        data_cross.to_numpy(),
        x=agents,
        y=agents,
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    fig.update_traces(text=data_cross, texttemplate="<b>%{text}</b>")
    fig.update_xaxes(side="top")
    fig.update_layout(config.plots["messages"]["heat_info"]["layout"])
    return fig


def statistics_messages(**karg) -> Figure:
    stats = pre.statistics_messages()
    fig: Figure = ff.create_table(stats)
    fig.update_layout(config.plots["messages"]["table_msg"]["layout"])
    return fig


def distribution_messages(**karg) -> Figure:
    data = pre.distribution_data_df()
    fig: Figure = px.histogram(data, x="timestamp_minutes")
    fig.update_layout(config.plots["messages"]["dist_msg"]["layout"])
    fig.update_traces(config.plots["messages"]["dist_msg"]["traces"])
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text="Number of messages",
    )
    return fig


def distribution_messages_types(**karg) -> Figure:
    data = pre.distribution_data_df()
    fig: Figure = px.histogram(data, x="timestamp_minutes", color="type")
    fig.update_layout(config.plots["messages"]["dist_msg_type"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text="Number of messages",
        legend_title_text="Type of messages",
    )
    return fig


def distribution_info(**karg) -> Figure:
    data = pre.distribution_data_df()
    fig: Figure = px.histogram(
        data, x="timestamp_minutes", y="size", color_discrete_sequence=["indianred"]
    )
    fig.update_layout(config.plots["messages"]["dist_info"]["layout"])
    fig.update_traces(config.plots["messages"]["dist_info"]["traces"])
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text="Amount of information (Bytes)",
    )
    return fig


def distribution_info_type(**karg) -> Figure:
    data = pre.distribution_data_df()
    fig: Figure = px.histogram(data, x="timestamp_minutes", y="size", color="type")
    fig.update_layout(config.plots["messages"]["dist_info_type"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text="Amount of information (Bytes)",
        legend_title_text="Type of messages",
    )
    return fig


#########################
###### CONVERGENCE ######
#########################


def create_xy_scatter_plot(layer: str, **karg) -> Figure:
    layers_opts, data = pre.convergence_df()
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
    fig.update_layout(config.plots["convergence"]["xy_scatter"]["layout"])
    fig.update_layout(
        xaxis_title_text=clean(layers_opts[layer][0]),
        yaxis_title_text=clean(layers_opts[layer][1]),
        legend_title_text=config.variables["agent"]["legend"],
    )
    return fig


def create_scatter_plot(sublayer: str, **karg) -> Figure:
    _, data = pre.convergence_df()
    min_y = data[sublayer].max()
    max_y = data[sublayer].max()
    scatter_plot: Figure = px.scatter(
        data,
        x="algorithm_round",
        y=sublayer,
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
    return scatter_plot


def create_line_plot(sublayer: str, **karg) -> Figure:
    _, data = pre.convergence_df()
    min_y = data[sublayer].min()
    max_y = data[sublayer].max()
    line_plot: Figure = px.line(
        data,
        x="algorithm_round",
        y=sublayer,
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

    return line_plot


def create_combined_plot(
    scatter_plot: Figure, line_plot: Figure, sublayer: str, **karg
) -> Figure:
    combined_plot: Figure = go.Figure(
        data=line_plot.data + scatter_plot.data,
        frames=[
            go.Frame(data=line_plot.data + scatter_plot.data, name=scatter_plot.name)
            for line_plot, scatter_plot in zip(line_plot.frames, scatter_plot.frames)
        ],
        layout=line_plot.layout,
    )

    combined_plot.update_yaxes(config.plots["convergence"]["combined"]["yaxes"])

    combined_plot.update_xaxes(config.plots["convergence"]["combined"]["xaxes"])

    combined_plot.update_traces(config.plots["convergence"]["combined"]["traces"])

    combined_plot.layout.updatemenus[0].buttons[0]["args"][1]["frame"]["duration"] = (
        config.plots["convergence"]["combined"]["updatemenus"]["frame_duration"]
    )
    combined_plot.layout.updatemenus[0].buttons[0]["args"][1]["transition"][
        "duration"
    ] = config.plots["convergence"]["combined"]["updatemenus"]["transition_duration"]
    combined_plot.layout.updatemenus[0].buttons[0]["args"][1]["transition"][
        "redraw"
    ] = config.plots["convergence"]["combined"]["updatemenus"]["redraw"]
    combined_plot.update_layout(
        title_text=f"{sublayer.upper()} layer evolution of agents"
    )
    combined_plot.update_layout(
        xaxis_title_text=config.variables["algorithm_round"]["legend"],
        yaxis_title_text=clean(sublayer),
        legend_title_text=config.variables["agent"]["legend"],
    )
    combined_plot.update_layout(config.plots["convergence"]["combined"]["layout"])
    return combined_plot


#########################
####### INFERENCE #######
#########################


def train_by_agent(**karg) -> List[Figure]:
    train = load.train_dataset()
    metrics: List[str] = ["accuracy", "loss", "precision", "recall", "f1_score"]
    figs = []
    for metric in metrics:
        fig: Figure = px.line(
            train,
            x="algorithm_round",
            y=metric,
            color="agent",
            title=f"{config.variables[metric]['legend']} evolution across rounds by agent",
        )
        fig.update_layout(config.plots["inference"]["train"]["layout"])
        fig.update_layout(
            xaxis_title_text=config.variables["algorithm_round"]["legend"],
            yaxis_title_text=config.variables[metric]["legend"],
            legend_title_text=config.variables["agent"]["legend"],
        )
        figs.append(fig)
    return figs


def test_by_agent(**karg) -> List[Figure]:
    test = load.inference_dataset()
    metrics: List[str] = [
        "test_accuracy",
        "test_loss",
        "test_precision",
        "test_recall",
        "test_f1_score",
    ]
    figs = []
    for metric in metrics:
        fig: Figure = px.line(
            test,
            x="algorithm_round",
            y=metric,
            color="agent",
            title=f"{config.variables[metric]['legend']} evolution across rounds by agent",
        )
        fig.update_layout(config.plots["inference"]["test"]["layout"])
        fig.update_layout(
            xaxis_title_text=config.variables["algorithm_round"]["legend"],
            yaxis_title_text=config.variables[metric]["legend"],
            legend_title_text=config.variables["agent"]["legend"],
        )
        figs.append(fig)
    return figs


def train_test_network(**karg) -> List[Figure]:
    train = load.train_dataset()
    test = load.inference_dataset()
    metrics: List[str] = ["accuracy", "loss", "precision", "recall", "f1_score"]
    figs = []
    for metric in metrics:

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
                fillcolor=config.plots["inference"]["train_test"]["traces"][
                    "train_fillcolor"
                ],
                line_color=config.plots["inference"]["train_test"]["traces"][
                    "train_linecolor"
                ],
                name="Train",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x + x_rev,
                y=tmax + tmin,
                fill="toself",
                fillcolor=config.plots["inference"]["train_test"]["traces"][
                    "test_fillcolor"
                ],
                line_color=config.plots["inference"]["train_test"]["traces"][
                    "test_linecolor"
                ],
                showlegend=False,
                name="Test",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=g_train.mean_accuracy,
                line_color=config.plots["inference"]["train_test"]["traces"][
                    "mean_train_linecolor"
                ],
                name="Train",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=g_test.mean_accuracy,
                line_color=config.plots["inference"]["train_test"]["traces"][
                    "mean_test_linecolor"
                ],
                name="Test",
            )
        )

        fig.update_traces(mode="lines")
        fig.update_layout(title_text=f"Netowk's performance {clean(metric)} by round")
        fig.update_layout(config.plots["inference"]["train_test"]["layout"])
        fig.update_layout(
            xaxis_title_text=config.variables["algorithm_round"]["legend"],
            yaxis_title_text=config.variables[metric]["legend"],
            legend_title_text=config.variables["agent"]["legend"],
        )
        figs.append(fig)
    return figs


#########################
####### NETWORK #########
#########################


def create_nodes_plot(x_coords, y_coords, **karg) -> Figure:
    nodes, edges, timestamps, metric, msg_type = pre.create_network_artifacts()

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
        config.plots["network"]["nodes"]["other"]["frame_duration"]
    )
    xmax, xmin = max(x_coords.values()), min(x_coords.values())
    ymax, ymin = max(y_coords.values()), min(y_coords.values())

    nodes_plot.update_traces(config.plots["network"]["nodes"]["traces"])

    i: float = config.plots["network"]["nodes"]["other"]["border"]
    nodes_plot.update_xaxes(range=[xmin - i, xmax + i])
    nodes_plot.update_yaxes(range=[ymin - i, ymax + i])

    nodes_plot.update_xaxes(config.plots["network"]["nodes"]["axes"])
    nodes_plot.update_yaxes(config.plots["network"]["nodes"]["axes"])

    marker_size = config.plots["network"]["nodes"]["other"]["marker_size"]
    nodes_plot.for_each_trace(lambda trace: trace.update(marker_size=marker_size))
    nodes_plot.update_layout(
        coloraxis_colorbar_title=config.variables[metric]["legend"]
    )

    nodes_plot.update_layout(config.plots["network"]["nodes"]["layout"])
    return range_color, nodes_plot


def create_edges_plot(x_coords, y_coords, **karg) -> Figure:
    nodes, edges, timestamps, metric, msg_type = pre.create_network_artifacts()
    edges = pre.edges_panel_data(edges, timestamps, x_coords, y_coords)

    def new_value(value: int) -> int:
        OldMin: int = edges.weight.min()
        OldMax: int = edges.weight.max()
        OldRange: int = OldMax - OldMin
        NewMin: int = 0
        NewMax: int = config.plots["network"]["edges"]["other"]["max_size_lines"]
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
                        "duration": config.plots["network"]["edges"]["other"][
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
                                "duration": config.plots["network"]["edges"]["other"][
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

    i: float = config.plots["network"]["edges"]["other"]["border"]
    edges_plot.update_xaxes(range=[xmin - i, xmax + i])
    edges_plot.update_yaxes(range=[ymin - i, ymax + i])

    edges_plot.update_xaxes(config.plots["network"]["edges"]["axes"])
    edges_plot.update_yaxes(config.plots["network"]["edges"]["axes"])

    edges_plot.update_traces(config.plots["network"]["edges"]["traces"])
    edges_plot.update_layout(config.plots["network"]["edges"]["layout"])
    return edges_plot


def create_combined_plot(
    nodes_plot: Figure,
    edges_plot: Figure,
    x_coords,
    y_coords,
    range_color,
    metric: str = "accuracy",
    msg_type: str = "SEND-LAYERS",
    **karg,
) -> Figure:
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
            marker=config.plots["network"]["network"]["traces"]["marker"],
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

    combined_plot.update_layout(config.plots["network"]["network"]["layout"])
    return combined_plot


#########################
######### EXTRA #########
#########################
