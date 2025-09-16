import os
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import branca.colormap as cm
from matplotlib.colors import to_hex

from config import Config, save_or_print_figures

config = Config()


def bubble_plots(data):
    agents = data.agent.unique()

    cmaps = [
        cm.linear.Pastel1_03.scale(0, 2),
        cm.linear.Pastel1_04.scale(0, 3),
        cm.linear.Pastel1_05.scale(0, 4),
        cm.linear.Pastel1_06.scale(0, 5),
        cm.linear.Pastel1_07.scale(0, 6),
        cm.linear.Pastel1_08.scale(0, 7),
        cm.linear.Pastel1_09.scale(0, 8),
    ]
    dic_cmaps = {str(i + 3): cmap for i, cmap in enumerate(cmaps)}
    labels = data.label.unique()

    try:
        cmap = dic_cmaps[str(len(labels))]
    except KeyError:
        if len(labels) < 3:
            cmap = cm.linear.Pastel1_03.scale(0, len(labels) - 1)
        else:
            cmap = cm.linear.Pastel1_09.scale(0, len(labels) - 1)

    colors = ["white"] + [cmap(i) for i in range(len(labels))] + ["white"]
    colors = list(map(to_hex, colors))

    figs = []
    phases = ["train", "validation", "test"]
    visible = True
    for phase in phases:
        fig = go.Figure()
        labels = data.label.unique()
        X = np.zeros((len(labels) + 2, len(agents) + 2))
        for i, agent in enumerate(agents):
            split_data = data[(data.agent == agent) & (data.description == phase)]
            for x, (_, row) in enumerate(split_data.iterrows()):
                X[row.label + 1, i + 1] = row["count"]

        scale = 10
        M, N = X.shape
        X_sizes = X.copy()
        for i in range(M):
            xmin, xmax = X_sizes[i, :].min(), X_sizes[i, :].max()
            tmin, tmax = (
                config.plots["data_split"]["bubble"]["size_factors"]["min_bubble_size"],
                config.plots["data_split"]["bubble"]["size_factors"]["max_bubble_size"],
            )
            X_sizes[i, :] = (X_sizes[i, :] - xmin) / (xmax - xmin) * (
                tmax - tmin
            ) + tmin
        x = []
        y = []
        colores = []
        sizes = []
        texts = []
        for j in range(N):
            for i in range(M):
                color = colors[i]
                if X[i, j] != 0:
                    x.append(j)
                    y.append(i)
                    colores.append(color)
                    sizes.append(X_sizes[i, j])
                    texts.append(f"{int(X[i,j])}")
        fig.add_trace(
            go.Scatter(
                mode="markers+text",
                x=x,
                y=y,
                marker=dict(color=colores, size=sizes, opacity=1),
                text=texts,
                line=dict(color="black", width=1),
                visible=visible,
                hovertemplate="<br><b>Agent</b>: %{x}<br>"
                + "<br><b>Digit</b>: %{y}<br>"
                + "<br><b>Samples</b>: %{text}<br>",
            )
        )

        fig.update_layout(
            title_text=f"Distribution of samples for {phase}",
            xaxis_title="Agents",
            yaxis_title="Target labels",
        )

        width_factor = config.plots["data_split"]["bubble"]["size_factors"]["width"]
        height_factor = config.plots["data_split"]["bubble"]["size_factors"]["height"]
        fig.update_layout(xaxis_range=[0, len(agents) + 1])
        fig.update_layout(yaxis_range=[0, len(colors) - 1])
        fig.update_layout(width=width_factor * (len(agents) + 2))
        fig.update_layout(height=height_factor * len(colors))
        fig.update_xaxes(
            ticktext=agents,
            tickvals=list(range(1, len(agents) + 1)),
        )
        fig.update_yaxes(
            ticktext=sorted(labels),
            tickvals=list(range(1, len(labels) + 1)),
        )
        fig.update_layout(config.plots["data_split"]["bubble"]["layout"])
        fig.update_xaxes(config.plots["data_split"]["bubble"]["axes"])
        fig.update_yaxes(config.plots["data_split"]["bubble"]["axes"])
        figs.append(fig)
    return figs


def generate(config, download=False):
    data = pd.read_csv(config.experiment_path / r"data_split.csv")
    figs = bubble_plots(data)
    return save_or_print_figures(download, figs)
