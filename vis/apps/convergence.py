import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from itertools import groupby
import numpy as np
from pandas.core.frame import DataFrame
from plotly.graph_objects import Figure
from typing import List, Dict
from config import Config, clean

config = Config()


def create_df(config: Config) -> DataFrame:
    df: DataFrame = pd.read_csv(config.experiment_path / r"nn_convergence.csv")
    lst: List[str] = df.layer.unique()
    split_layer_str = lambda x: x.split(".")[0]
    global layers_opts
    layers_opts = {
        split_layer_str(k): list(g)
        for k, g in groupby(sorted(lst, key=split_layer_str), key=split_layer_str)
    }
    df: DataFrame = df[(df.description == "PRE-TRAIN") & (df.epoch_or_iteration == 1)]
    pivot: DataFrame = df.pivot(columns="layer", values="weight")
    df = df[df.layer == layers_opts["conv1"][0]].reset_index()

    for layer in layers_opts.keys():
        df[layers_opts[layer][0]] = (
            pivot[layers_opts[layer][0]]
            .dropna()
            .reset_index()[f"{layers_opts[layer][0]}"]
        )
        df[layers_opts[layer][1]] = (
            pivot[layers_opts[layer][1]]
            .dropna()
            .reset_index()[f"{layers_opts[layer][1]}"]
        )

    df = df.sort_values(
        [
            "algorithm_round",
            "agent",
        ],
        ignore_index=True,
    )
    N_UNIQUE_AGENTS: int = df["agent"].nunique()
    df_indexed: DataFrame = pd.DataFrame()
    for index in np.arange(start=0, stop=len(df) + 1, step=N_UNIQUE_AGENTS):
        df_slicing: DataFrame = df.iloc[:index].copy()
        df_slicing["frame"] = index // N_UNIQUE_AGENTS
        df_indexed: DataFrame = pd.concat([df_indexed, df_slicing])

    return df_indexed


def create_xy_scatter_plot(data: DataFrame, layer: str) -> Figure:
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


def create_scatter_plot(data: DataFrame, sublayer: str) -> Figure:
    scatter_plot: Figure = px.scatter(
        data,
        x="algorithm_round",
        y=sublayer,
        animation_frame="frame",
        color="agent",
        hover_name="agent",
        range_x=[0, len(data.algorithm_round.unique()) + 1],
        range_y=[min_x, max_x],
    )

    for frame in scatter_plot.frames:
        for data in frame.data:
            data.update(mode="markers", showlegend=True, opacity=1)
            data["x"] = np.take(data["x"], [-1])
            data["y"] = np.take(data["y"], [-1])
    return scatter_plot


def create_line_plot(data: DataFrame, sublayer: str) -> Figure:
    line_plot: Figure = px.line(
        data,
        x="algorithm_round",
        y=sublayer,
        color="agent",
        animation_frame="frame",
        range_x=[0, len(data.algorithm_round.unique()) + 1],
        range_y=[min_x, max_x],
        line_shape="spline",  # make a line graph curvy
    )

    line_plot.update_traces(showlegend=False)  # legend will be from line graph
    for frame in line_plot.frames:
        for data in frame.data:
            data.update(mode="lines", opacity=0.8, showlegend=False)

    return line_plot


def create_combined_plot(
    scatter_plot: Figure, line_plot: Figure, sublayer: str
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


def set_globals_layer(data: DataFrame, layer: str) -> None:
    global min_x
    min_x = data[layers_opts[layer][0]].min()
    global min_y
    min_y = data[layers_opts[layer][1]].min()
    global max_x
    max_x = data[layers_opts[layer][0]].max()
    global max_y
    max_y = data[layers_opts[layer][1]].max()


def set_globals_sublayer(data: DataFrame, layer: str, index: int) -> None:
    global min_x
    min_x = data[layers_opts[layer][index]].min()
    global max_x
    max_x = data[layers_opts[layer][index]].max()


def generate(config: Config, action: str = "generate") -> list[str] | None:
    data: DataFrame = create_df(config)
    figs = []
    # for layer in layers_opts.keys():
    layer = "conv1"
    set_globals_layer(data, layer)
    scater_xy: Figure = create_xy_scatter_plot(data, layer)
    figs.append(scater_xy)
    for i, layer_type in enumerate(layers_opts[layer]):
        set_globals_sublayer(data, layer, i)
        scatter_plot: Figure = create_scatter_plot(data, layers_opts[layer][i])
        line_plot: Figure = create_line_plot(data, layers_opts[layer][i])
        combined_plot: Figure = create_combined_plot(
            scatter_plot, line_plot, layers_opts[layer][i]
        )
        figs.append(combined_plot)
    print(f"Created plots for {layer.upper()} layer")

    if action not in ["generate", "download"]:
        return figs

    if action == "generate":
        folder = f"figures/{__name__.split('.')[0]}"
        isExist: bool = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)
        for fig in figs:
            print(1)
            fig.write_json(rf"{folder}/{fig.layout.title.text.replace(' ', '_')}.json")
    else:
        folder = f"{config.output_path}/{__name__.split('.')[0]}"
        isExist: bool = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)
        for fig in figs:
            if len(fig.frames) > 0:
                frame = fig.frames[-1]
                fig.update(data=frame.data)
                fig.layout.sliders[0].update(active=len(fig.frames) - 1)
            fig.write_image(rf"{folder}/{fig.layout.title.text.replace(' ', '_')}.svg")
