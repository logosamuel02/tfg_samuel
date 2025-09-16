import os
import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff
from pandas.core.frame import DataFrame
from pandas._libs.tslibs.timestamps import Timestamp
from plotly.graph_objects import Figure
from typing import List
from config import Config, save_or_print_figures

config = Config()


def heatmap_messages(data: DataFrame) -> Figure:
    data: DataFrame = data[["sender", "to"]]
    data.sender = list(map(lambda x: x.split("@")[0], data.sender))
    data.to = list(map(lambda x: x.split("@")[0], data.to))
    data_cross = pd.crosstab(index=data.sender, columns=data.to)
    agents: List[str] = data.sender.unique()
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


def heatmap_sizes(data: DataFrame) -> Figure:
    data: DataFrame = data[["sender", "to", "size"]]
    data.sender = list(map(lambda x: x.split("@")[0], data.sender))
    data.to = list(map(lambda x: x.split("@")[0], data.to))
    data_cross: DataFrame = (
        data.groupby(["sender", "to"])["size"]
        .sum()
        .div(1024 * 1024)
        .round(0)
        .unstack()
        .fillna(0)
    )
    agents: List[str] = data.sender.unique()
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


def statistics_messages(data: DataFrame) -> Figure:
    stats: DataFrame = (
        data.groupby("type")
        .agg(
            number_of_messages=("size", "count"),
            total_size=("size", "sum"),
            average_size=("size", "mean"),
            standard_deviation=("size", "std"),
            minimum_size=("size", "min"),
            maximum_size=("size", "max"),
        )
        .reset_index()
        .round(2)
    )
    fig: Figure = ff.create_table(stats)
    fig.update_layout(config.plots["messages"]["table_msg"]["layout"])
    return fig


def create_distribution_data(data: DataFrame) -> Figure:
    data.timestamp = pd.to_datetime(data.timestamp)
    min_date: Timestamp = data.timestamp.min()
    data["timestamp_minutes"] = data.timestamp.apply(
        lambda x: int((x - min_date).total_seconds() / 60.0)
    )
    return data


def distribution_messages(data: DataFrame) -> Figure:
    fig: Figure = px.histogram(data, x="timestamp_minutes")
    fig.update_layout(config.plots["messages"]["dist_msg"]["layout"])
    fig.update_traces(config.plots["messages"]["dist_msg"]["traces"])
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text="Number of messages",
    )
    return fig


def distribution_messages_types(data: DataFrame) -> Figure:
    fig: Figure = px.histogram(data, x="timestamp_minutes", color="type")
    fig.update_layout(config.plots["messages"]["dist_msg_type"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text="Number of messages",
        legend_title_text="Type of messages",
    )
    return fig


def distribution_info(data: DataFrame) -> Figure:
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


def distribution_info_type(data: DataFrame) -> Figure:
    fig: Figure = px.histogram(data, x="timestamp_minutes", y="size", color="type")
    fig.update_layout(config.plots["messages"]["dist_info_type"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text="Amount of information (Bytes)",
        legend_title_text="Type of messages",
    )
    return fig


def generate(config: Config, download: bool = False) -> list[str] | None:
    data: DataFrame = pd.read_csv(config.experiment_path / r"message.csv")
    f0: Figure = heatmap_messages(data)
    f1: Figure = heatmap_sizes(data)
    f2: Figure = statistics_messages(data)
    dist_data: DataFrame = create_distribution_data(data)
    f3: Figure = distribution_messages(dist_data)
    f4: Figure = distribution_messages_types(dist_data)
    f5: Figure = distribution_info(dist_data)
    f6: Figure = distribution_info_type(dist_data)
    figs: List[Figure] = [f0, f1, f2, f3, f4, f5, f6]
    return save_or_print_figures(download, figs)
