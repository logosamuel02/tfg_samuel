import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from more_itertools import sort_together

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
    fig.update_layout(title_text="Seconds to complete training round by agent")
    return fig

def execution_time_plot(data):
    data = data[data.algorithm_round <= 100]
    data.timestamp = pd.to_datetime(data.timestamp)
    agents = data.agent.unique()
    times = []
    elapsed = []
    for i,agent in enumerate(agents):
        dates = list(data.timestamp[data.agent==agent])
        rang = dates[-1] - dates[0]
        times.append(rang.total_seconds())
        elapsed.append(dates[-1])

    m = min(elapsed)
    for i, agent in enumerate(agents):
        elapsed[i] = (elapsed[i] - m).total_seconds()
        
    elapsed, agents, times  = sort_together((elapsed, agents, times))

    df = pd.DataFrame({"seconds":times, "agents": list(map(lambda x: x.split("@")[0], agents)), "seconds elapsed": elapsed})
    fig = px.bar(df, x='seconds elapsed', y='agents',
                hover_data=['seconds', 'seconds elapsed'], 
                color='agents',
                labels={'pop':'seconds'},
                text="seconds",
                title="Tiempos de ejecución ordenados por agente",
                orientation='h')
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    return fig


def generate(config, download=False):
    data = pd.read_csv(config.experiment_path / r"algorithm.csv")
    f1 = violin_plot(data)
    f2 = execution_time_plot(data)
    figs = [f1, f2]
    if download:
        root = "images"
        folder = f"{root}/{__name__.split('.')[0]}"
        isExist = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)
        for i,F in enumerate(figs):
            F.write_image(f"{folder}/{F.layout.title.text.replace(' ', '_')}.svg")
    else:
        updated_figs = []
        for F in figs:
            html_fig = F.to_html(full_html=False)
            html_fig.replace("PNG", "SVG", 1)
            html_fig.replace("png", "svg", 3)
            updated_figs.append(html_fig)
        return updated_figs

