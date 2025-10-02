import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from itertools import groupby
import numpy as np
from pandas.core.frame import DataFrame
from plotly.graph_objects import Figure
from typing import List, Dict
from tqdm import tqdm
import time
import preprocess as pre
from export import Config
from preprocess import clean

config = Config()


def create_xy_scatter_plot(layer: str) -> Figure:
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


def create_scatter_plot(sublayer: str) -> Figure:
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


def create_line_plot(sublayer: str) -> Figure:
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


def generate(config: Config, action: str = "generate") -> list[str] | None:
    figs = []
    layer = "conv1"
    scater_xy: Figure = create_xy_scatter_plot(layer)
    figs.append(scater_xy)
    layers_opts, _ = pre.convergence_df()
    for i, layer_type in enumerate(layers_opts[layer]):
        scatter_plot: Figure = create_scatter_plot(layer_type)
        line_plot: Figure = create_line_plot(layer_type)
        combined_plot: Figure = create_combined_plot(
            scatter_plot, line_plot, layer_type
        )
        figs.append(combined_plot)

    if action not in ["generate", "download"]:
        return figs

    if action == "generate":
        folder = f"figures/{__name__.split('.')[-1]}"
        isExist: bool = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)
        for fig in tqdm(figs, desc="Downloading figures"):
            fig.write_json(rf"{folder}/{fig.layout.title.text.replace(' ', '_')}.json")
    else:
        folder = f"{config.output_path}/{__name__.split('.')[-1]}"
        isExist: bool = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)
        for fig in tqdm(figs, desc="Downloading figures"):
            if len(fig.frames) > 0:
                frame = fig.frames[-1]
                fig.update(data=frame.data)
                fig.layout.sliders[0].update(active=len(fig.frames) - 1)
            w, h = fig.layout.width, fig.layout.height
            fig.write_image(
                rf"{folder}/{fig.layout.title.text.replace(' ', '_')}.svg",
                width=w,
                height=h,
            )
