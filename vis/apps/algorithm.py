import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from more_itertools import sort_together

from config import Config, save_or_print_figures

config = Config()


def violin_plot(data):
    agents = list(map(lambda x: x.split("@")[0], sorted(data.agent.unique())))
    times = []
    for agent in agents:
        agent_data = data[(data.agent == agent + "@localhost")]
        time = list(agent_data.seconds_to_complete)
        times.append(time)

    df_bolos = pd.DataFrame(columns=agents)
    for t, a in zip(times, agents):
        df_bolos[a] = t
    df_bolos = df_bolos.melt()
    fig = go.Figure()
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


def execution_time_plot(data):
    data = data[data.algorithm_round <= 100]
    data.timestamp = pd.to_datetime(data.timestamp)
    agents = data.agent.unique()
    times = []
    elapsed = []
    for i, agent in enumerate(agents):
        dates = list(data.timestamp[data.agent == agent])
        rang = dates[-1] - dates[0]
        times.append(round(rang.total_seconds(), 2))
        elapsed.append(dates[-1])

    m = min(elapsed)
    for i, agent in enumerate(agents):
        elapsed[i] = round((elapsed[i] - m).total_seconds(), 2)

    elapsed, agents, times = sort_together((elapsed, agents, times))

    df = pd.DataFrame(
        {
            "seconds": times,
            "agent": list(map(lambda x: x.split("@")[0], agents)),
            "seconds_elapsed": elapsed,
        }
    )
    fig = px.bar(
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


def generate(config, download=False):
    data = pd.read_csv(config.experiment_path / r"algorithm.csv")
    f1 = violin_plot(data)
    f2 = execution_time_plot(data)
    figs = [f1, f2]
    return save_or_print_figures(download, figs)
