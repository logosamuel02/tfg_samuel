import os
import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff

from config import Config, save_or_print_figures

config = Config()


def heatmap_messages(data):
    data = data[["sender", "to"]]
    data.sender = list(map(lambda x: x.split("@")[0], data.sender))
    data.to = list(map(lambda x: x.split("@")[0], data.to))
    data_cross = pd.crosstab(index=data.sender, columns=data.to)
    agents = data.sender.unique()
    fig = px.imshow(
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


def heatmap_sizes(data):
    data = data[["sender", "to", "size"]]
    data.sender = list(map(lambda x: x.split("@")[0], data.sender))
    data.to = list(map(lambda x: x.split("@")[0], data.to))
    data_cross = (
        data.groupby(["sender", "to"])["size"]
        .sum()
        .div(1024 * 1024)
        .round(0)
        .unstack()
        .fillna(0)
    )
    agents = data.sender.unique()
    fig = px.imshow(
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


def statistics_messages(data):
    stats = (
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
    fig = ff.create_table(stats)
    fig.update_layout(config.plots["messages"]["table_msg"]["layout"])
    return fig


def create_distribution_data(data):
    data.timestamp = pd.to_datetime(data.timestamp)
    min_date = data.timestamp.min()
    data["timestamp_minutes"] = data.timestamp.apply(
        lambda x: int((x - min_date).total_seconds() / 60.0)
    )
    return data


def distribution_messages(data):
    fig = px.histogram(data, x="timestamp_minutes")
    fig.update_layout(config.plots["messages"]["dist_msg"]["layout"])
    fig.update_traces(config.plots["messages"]["dist_msg"]["traces"])
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text="Number of messages",
    )
    return fig


def distribution_messages_types(data):
    fig = px.histogram(data, x="timestamp_minutes", color="type")
    fig.update_layout(config.plots["messages"]["dist_msg_type"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text="Number of messages",
        legend_title_text="Type of messages",
    )
    return fig


def distribution_info(data):
    fig = px.histogram(
        data, x="timestamp_minutes", y="size", color_discrete_sequence=["indianred"]
    )
    fig.update_layout(config.plots["messages"]["dist_info"]["layout"])
    fig.update_traces(config.plots["messages"]["dist_info"]["traces"])
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text="Amount of information (Bytes)",
    )
    return fig


def distribution_info_type(data):
    fig = px.histogram(data, x="timestamp_minutes", y="size", color="type")
    fig.update_layout(config.plots["messages"]["dist_info_type"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables["timestamp_minutes"]["legend"],
        yaxis_title_text="Amount of information (Bytes)",
        legend_title_text="Type of messages",
    )
    return fig


def generate(config, download=False):
    data = pd.read_csv(config.experiment_path / r"message.csv")
    f0 = heatmap_messages(data)
    f1 = heatmap_sizes(data)
    f2 = statistics_messages(data)
    dist_data = create_distribution_data(data)
    f3 = distribution_messages(dist_data)
    f4 = distribution_messages_types(dist_data)
    f5 = distribution_info(dist_data)
    f6 = distribution_info_type(dist_data)
    figs = [f0, f1, f2, f3, f4, f5, f6]
    return save_or_print_figures(download, figs)
