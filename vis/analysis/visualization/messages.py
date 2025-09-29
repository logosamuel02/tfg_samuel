import os
import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff
from pandas.core.frame import DataFrame
from pandas._libs.tslibs.timestamps import Timestamp
from plotly.graph_objects import Figure
from typing import List
import preprocess as pre
from export import Config, clean

config = Config()


def heatmap_messages() -> Figure:
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


def heatmap_sizes() -> Figure:
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


def statistics_messages() -> Figure:
    stats = pre.statistics_messages()
    fig: Figure = ff.create_table(stats)
    fig.update_layout(config.plots["messages"]["table_msg"]["layout"])
    return fig


def distribution_messages() -> Figure:
    data = pre.distribution_data_df()
    fig: Figure = px.histogram(data, x="timestamp_minutes")
    fig.update_layout(config.plots["messages"]["dist_msg"]["layout"])
    fig.update_traces(config.plots["messages"]["dist_msg"]["traces"])
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text="Number of messages",
    )
    return fig


def distribution_messages_types() -> Figure:
    data = pre.distribution_data_df()
    fig: Figure = px.histogram(data, x="timestamp_minutes", color="type")
    fig.update_layout(config.plots["messages"]["dist_msg_type"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text="Number of messages",
        legend_title_text="Type of messages",
    )
    return fig


def distribution_info() -> Figure:
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


def distribution_info_type() -> Figure:
    data = pre.distribution_data_df()
    fig: Figure = px.histogram(data, x="timestamp_minutes", y="size", color="type")
    fig.update_layout(config.plots["messages"]["dist_info_type"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text="Amount of information (Bytes)",
        legend_title_text="Type of messages",
    )
    return fig


def generate(config: Config, action: str = "generate") -> list[str] | None:
    f0: Figure = heatmap_messages()
    f1: Figure = heatmap_sizes()
    f2: Figure = statistics_messages()
    f3: Figure = distribution_messages()
    f4: Figure = distribution_messages_types()
    f5: Figure = distribution_info()
    f6: Figure = distribution_info_type()
    figs: List[Figure] = [f0, f1, f2, f3, f4, f5, f6]

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
