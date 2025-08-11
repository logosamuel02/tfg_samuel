from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from itertools import groupby


def layer_evolution_plot():
    FOLDER = Path("/home/slozgom/tfg/tfg_samuel/xperiments/experimentos_con_cnn")
    experiments_list = list(FOLDER.iterdir())
    PATH = experiments_list[8]

    df = pd.read_csv(
        "/home/slozgom/tfg/tfg_samuel/xperiments/experimentos_con_cnn/05_non_complete/raw/nn_convergence.csv"
    )
    lst = df.layer.unique()
    f = lambda x: x.split(".")[0]
    layers_opts = {f(k): list(g) for k, g in groupby(sorted(lst, key=f), key=f)}
    df = df[
        (df.description == "PRE-TRAIN")
        & df.layer.isin(layers_opts["conv1"])
        & (df.epoch_or_iteration == 1)
    ]
    pivot = df.pivot(columns="layer", values="weight")
    df = df[df.layer == layers_opts["conv1"][0]].reset_index()
    df[layers_opts["conv1"][0]] = (
        pivot[layers_opts["conv1"][0]]
        .dropna()
        .reset_index()[f"{layers_opts['conv1'][0]}"]
    )
    df[layers_opts["conv1"][1]] = (
        pivot[layers_opts["conv1"][1]]
        .dropna()
        .reset_index()[f"{layers_opts['conv1'][1]}"]
    )
    min_x = df[layers_opts["conv1"][0]].min()
    min_y = df[layers_opts["conv1"][1]].min()
    max_x = df[layers_opts["conv1"][0]].max()
    max_y = df[layers_opts["conv1"][1]].max()
    fig = px.scatter(
        df,
        x=layers_opts["conv1"][0],
        y=layers_opts["conv1"][1],
        animation_frame="algorithm_round",
        color="agent",
        hover_name="agent",
        range_x=[min_x, max_x],
        range_y=[min_y, max_y],
    )

    """
    Second Figure
    """

    df = pd.read_csv(
        "/home/slozgom/tfg/tfg_samuel/xperiments/experimentos_con_cnn/05_non_complete/raw/nn_convergence.csv"
    )
    lst = df.layer.unique()
    f = lambda x: x.split(".")[0]
    layers_opts = {f(k): list(g) for k, g in groupby(sorted(lst, key=f), key=f)}
    df = df[
        (df.description == "PRE-TRAIN")
        & df.layer.isin(layers_opts["conv1"])
        & (df.epoch_or_iteration == 1)
    ]
    pivot = df.pivot(columns="layer", values="weight")
    df = df[df.layer == layers_opts["conv1"][0]].reset_index()
    df[layers_opts["conv1"][0]] = (
        pivot[layers_opts["conv1"][0]]
        .dropna()
        .reset_index()[f"{layers_opts['conv1'][0]}"]
    )
    df[layers_opts["conv1"][1]] = (
        pivot[layers_opts["conv1"][1]]
        .dropna()
        .reset_index()[f"{layers_opts['conv1'][1]}"]
    )
    min_x = df[layers_opts["conv1"][0]].min()
    min_y = df[layers_opts["conv1"][1]].min()
    max_x = df[layers_opts["conv1"][0]].max()
    max_y = df[layers_opts["conv1"][1]].max()
    line_plot = px.line(
        df,
        x="algorithm_round",
        y=layers_opts["conv1"][0],
        color="agent",
        animation_frame="algorithm_round",
        hover_name="agent",
        range_x=[0, 110],
        range_y=[min_x, max_x],
    )
    line_plot.update_traces(showlegend=False)  # legend will be from line graph
    for frame in line_plot.frames:
        for data in frame.data:
            data.update(mode="lines", opacity=0.8, showlegend=False)

    df = df.sort_values(
        [
            "algorithm_round",
            "agent",
        ],
        ignore_index=True,
    )
    N_UNIQUE_AGENTS = df["agent"].nunique()
    df_indexed = pd.DataFrame()
    for index in np.arange(start=0, stop=len(df) + 1, step=N_UNIQUE_AGENTS):
        df_slicing = df.iloc[:index].copy()
        df_slicing["frame"] = index // N_UNIQUE_AGENTS
        df_indexed = pd.concat([df_indexed, df_slicing])

    # Scatter Plot
    scatter_plot = px.scatter(
        df_indexed,
        x="algorithm_round",
        y=layers_opts["conv1"][0],
        animation_frame="frame",
        color="agent",
        hover_name="agent",
        range_x=[0, 110],
        range_y=[min_x, max_x],
    )

    for frame in scatter_plot.frames:
        for data in frame.data:
            data.update(mode="markers", showlegend=True, opacity=1)
            data["x"] = np.take(data["x"], [-1])
            data["y"] = np.take(data["y"], [-1])

    # Line Plot
    line_plot = px.line(
        df_indexed,
        x="algorithm_round",
        y=layers_opts["conv1"][0],
        color="agent",
        animation_frame="frame",
        range_x=[0, 110],
        range_y=[min_x, max_x],
        line_shape="spline",  # make a line graph curvy
    )
    line_plot.update_traces(showlegend=False)  # legend will be from line graph
    for frame in line_plot.frames:
        for data in frame.data:
            data.update(mode="lines", opacity=0.8, showlegend=False)

    line_plot.update_traces(showlegend=False)  # legend will be from line graph
    for frame in line_plot.frames:
        for data in frame.data:
            data.update(mode="lines", opacity=0.8, showlegend=False)

    # Stationary combined plot
    combined_plot = go.Figure(
        data=line_plot.data + scatter_plot.data,
        frames=[
            go.Frame(data=line_plot.data + scatter_plot.data, name=scatter_plot.name)
            for line_plot, scatter_plot in zip(line_plot.frames, scatter_plot.frames)
        ],
        layout=line_plot.layout,
    )

    combined_plot.update_yaxes(
        gridcolor="#7a98cf", griddash="dot", gridwidth=0.5, linewidth=2, tickwidth=2
    )

    combined_plot.update_xaxes(title_font=dict(size=16), linewidth=2, tickwidth=2)

    combined_plot.update_traces(line=dict(width=5), marker=dict(size=25))

    # adjust speed of animation
    combined_plot.layout.updatemenus[0].buttons[0]["args"][1]["frame"]["duration"] = 120
    combined_plot.layout.updatemenus[0].buttons[0]["args"][1]["transition"][
        "duration"
    ] = 50
    combined_plot.layout.updatemenus[0].buttons[0]["args"][1]["transition"][
        "redraw"
    ] = False

    return [F.to_html(full_html=False) for F in [fig, combined_plot]]


# fig.write_html("index.html")
# print("Figure CREATED!")
