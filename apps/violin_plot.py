from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path


def violin_plot():

    FOLDER = Path("/home/slozgom/tfg/tfg_samuel/xperiments/experimentos_con_cnn")
    experiments_list = list(FOLDER.iterdir())
    PATH = experiments_list[8]

    data = pd.read_csv(PATH / "raw/algorithm.csv")
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
    return fig.to_html(full_html=False)


# fig.write_html("index.html")
# print("Figure CREATED!")
