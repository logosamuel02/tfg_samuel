import os
import plotly.express as px
import plotly.graph_objects as go, Figure
from typing import List

from export import Config
import preprocess as pre

config = Config()


def violin_plot() -> Figure:
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


def execution_time_plot() -> Figure:
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


def generate(config: Config, action: str = "generate") -> list[str] | None:
    f1: Figure = violin_plot()
    f2: Figure = execution_time_plot()
    figs: List[Figure] = [f1, f2]

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
