import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from more_itertools import sort_together
from pandas.core.frame import DataFrame
from plotly.graph_objects import Figure
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._libs.tslibs.timedeltas import Timedelta
from typing import List
from config import Config, save_or_print_figures

config = Config()


def violin_plot(data: DataFrame) -> Figure:
    agents: List[str] = list(
        map(lambda x: x.split("@")[0], sorted(data.agent.unique()))
    )
    times = []
    for agent in agents:
        agent_data: DataFrame = data[(data.agent == agent + "@localhost")]
        time: List[int] = list(agent_data.seconds_to_complete)
        times.append(time)

    df_bolos: DataFrame = pd.DataFrame(columns=agents)
    for t, a in zip(times, agents):
        df_bolos[a] = t
    df_bolos = df_bolos.melt()
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


def execution_time_plot(data: DataFrame) -> Figure:
    data: DataFrame = data[data.algorithm_round <= 100]
    data.timestamp = pd.to_datetime(data.timestamp)
    agents: List[str] = data.agent.unique()
    times = []
    elapsed = []
    for i, agent in enumerate(agents):
        dates: List[Timestamp] = list(data.timestamp[data.agent == agent])
        rang: Timedelta = dates[-1] - dates[0]
        times.append(round(rang.total_seconds(), 2))
        elapsed.append(dates[-1])

    m: Timestamp = min(elapsed)
    for i, agent in enumerate(agents):
        elapsed[i] = round((elapsed[i] - m).total_seconds(), 2)

    elapsed, agents, times = sort_together((elapsed, agents, times))

    df: DataFrame = pd.DataFrame(
        {
            "seconds": times,
            "agent": list(map(lambda x: x.split("@")[0], agents)),
            "seconds_elapsed": elapsed,
        }
    )
    fig: Figure = px.bar(
        df,
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


def generate(config: Config, download: bool = False) -> list[str] | None:
    data: DataFrame = pd.read_csv(config.experiment_path / r"algorithm.csv")
    f1: Figure = violin_plot(data)
    f2: Figure = execution_time_plot(data)
    figs: List[Figure] = [f1, f2]
    return save_or_print_figures(download, figs, __name__)
