import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from itertools import groupby
import numpy as np


def create_df(config):
    df = pd.read_csv(config.experiment_path / r"nn_convergence.csv")
    lst = df.layer.unique()
    f = lambda x: x.split(".")[0]
    global layers_opts
    layers_opts = {f(k): list(g) for k, g in groupby(sorted(lst, key=f), key=f)}
    df = df[(df.description == "PRE-TRAIN") & (df.epoch_or_iteration == 1)]
    pivot = df.pivot(columns="layer", values="weight")
    df = df[df.layer == layers_opts["conv1"][0]].reset_index()

    for layer in layers_opts.keys():
        df[layers_opts[layer][0]] = (
            pivot[layers_opts[layer][0]]
            .dropna()
            .reset_index()[f"{layers_opts[layer][0]}"]
        )
        df[layers_opts[layer][1]] = (
            pivot[layers_opts[layer][1]]
            .dropna()
            .reset_index()[f"{layers_opts[layer][1]}"]
        )

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

    return df_indexed


def create_xy_scatter_plot(data, layer):
    fig = px.scatter(
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
    return fig


def create_scatter_plot(data, sublayer):
    scatter_plot = px.scatter(
        data,
        x="algorithm_round",
        y=sublayer,
        animation_frame="frame",
        color="agent",
        hover_name="agent",
        range_x=[0, len(data.algorithm_round.unique()) + 1],
        range_y=[min_x, max_x],
    )

    for frame in scatter_plot.frames:
        for data in frame.data:
            data.update(mode="markers", showlegend=True, opacity=1)
            data["x"] = np.take(data["x"], [-1])
            data["y"] = np.take(data["y"], [-1])
    return scatter_plot


def create_line_plot(data, sublayer):
    line_plot = px.line(
        data,
        x="algorithm_round",
        y=sublayer,
        color="agent",
        animation_frame="frame",
        range_x=[0, len(data.algorithm_round.unique()) + 1],
        range_y=[min_x, max_x],
        line_shape="spline",  # make a line graph curvy
    )
    line_plot.update_traces(showlegend=False)  # legend will be from line graph
    for frame in line_plot.frames:
        for data in frame.data:
            data.update(mode="lines", opacity=0.8, showlegend=False)

    return line_plot


def create_combined_plot(scatter_plot, line_plot, sublayer):
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
    combined_plot.update_layout(
        title_text=f"{sublayer.upper()} layer evolution of agents"
    )
    return combined_plot


def set_globals_layer(data, layer):
    global min_x
    min_x = data[layers_opts[layer][0]].min()
    global min_y
    min_y = data[layers_opts[layer][1]].min()
    global max_x
    max_x = data[layers_opts[layer][0]].max()
    global max_y
    max_y = data[layers_opts[layer][1]].max()


def set_globals_sublayer(data, layer, index):
    global min_x
    min_x = data[layers_opts[layer][index]].min()
    global max_x
    max_x = data[layers_opts[layer][index]].max()


def generate(config, download=False):
    data = create_df(config)
    figs = []
    for layer in layers_opts.keys():
        set_globals_layer(data, layer)
        scater_xy = create_xy_scatter_plot(data, layer)
        figs.append(scater_xy)
        for i, layer_type in enumerate(layers_opts[layer]):
            set_globals_sublayer(data, layer, i)
            scatter_plot = create_scatter_plot(data, layers_opts[layer][i])
            line_plot = create_line_plot(data, layers_opts[layer][i])
            combined_plot = create_combined_plot(
                scatter_plot, line_plot, layers_opts[layer][i]
            )
            figs.append(combined_plot)
        print(f"Created plots for {layer.upper()} layer")
    if download:
        root = "images"
        folder = f"{root}/{__name__.split('.')[0]}"
        isExist = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)
        for i, F in enumerate(figs):
            F.write_image(f"{folder}/{F.layout.title.text.replace(' ', '_')}.svg")
    else:
        updated_figs = []
        for F in figs:
            html_fig = F.to_html(full_html=False)
            html_fig = html_fig.replace("PNG", "SVG", 1)
            html_fig = html_fig.replace("png", "svg", 3)
            updated_figs.append(html_fig)
        return updated_figs
