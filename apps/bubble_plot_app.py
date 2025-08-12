import os
from pathlib import Path
from pandas import read_csv
import numpy as np
import plotly.graph_objects as go


def bubble_plot(download=False):
    FOLDER = Path("/home/slozgom/tfg/tfg_samuel/xperiments/experimentos_con_cnn")
    experiments_list = list(FOLDER.iterdir())
    PATH = experiments_list[8]

    colors = [
        "white",
        "#aec4ff",
        "#ff9090",
        "#95ff90",
        "#ffb74a",
        "#e2b1ff",
        "#fef575",
        "#e2e2e2",
        "#ffdcf7",
        "#f5deb3",
        "#99fff0",
        "white",
    ]
    data = read_csv(PATH.joinpath("raw/data_split.csv"))
    agents = data.agent.unique()

    fig = go.Figure()

    phases = ["train", "validation", "test"]
    visible = True
    for phase in phases:
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
            tmin, tmax = 20, 55
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
        visible = False

    # Create the dropdown menu
    buttons = []

    # Add a button for each agent to the dropdown
    for i, phase in enumerate(phases):
        visibility = [False] * len(phases)
        visibility[i] = True
        buttons.append(
            dict(
                label=phase,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": {"text": f"Categorical bubble plot for {phase}"}},
                ],
            )
        )

    # Update the figure layout with the dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                active=0, buttons=buttons, x=0.01, xanchor="left", y=1.06, yanchor="top"
            )
        ],
        title_text="Categorical bubble plot for train",
        xaxis_title="Agents",
        yaxis_title="Labels",
    )

    fig.update_layout(xaxis_range=[0, len(agents) + 1])
    fig.update_layout(yaxis_range=[0, len(colors) - 1])
    fig.update_layout(width=143 * (len(agents) + 2))
    fig.update_layout(height=67 * len(colors))
    fig.update_layout(plot_bgcolor="rgb(256,256,256)")
    fig.update_xaxes(
        ticktext=agents,
        tickvals=list(range(1, len(agents) + 1)),
        showgrid=True,
        gridwidth=1,
        gridcolor="grey",
    )
    fig.update_yaxes(
        ticktext=sorted(labels),
        tickvals=list(range(1, len(labels) + 1)),
        showgrid=True,
        gridwidth=1,
        gridcolor="grey",
    )

    if download:
        root = "images"
        folder = f"{root}/{__name__.split(".")[0]}"
        isExist = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)
        for i,F in enumerate([fig]):
            F.write_image(f"{folder}/{F.layout.title.text.replace(" ", "_")}.svg")
    else:
        return fig.to_html(full_html=False)


# fig.write_html("index.html")
# print("Figure CREATED!")
