import os
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import numpy.typing as npt
import branca.colormap as cm
from matplotlib.colors import to_hex
from pandas.core.frame import DataFrame
from plotly.graph_objects import Figure
from typing import List, Dict
from tqdm import tqdm
import time
import preprocess as pre
from export import Config

config = Config()


def create_bubble_plots() -> List[Figure]:
    colors = pre.bubble_colors()
    figs = []
    phases: List[str] = ["train", "validation", "test"]
    visible: bool = True
    for phase in phases:
        fig: Figure = go.Figure()
        agents, labels, x, y, colores, sizes, texts = pre.bubble_interprocess(phase)
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

        width_factor: int = config.plots["data_split"]["bubble"]["size_factors"][
            "width"
        ]
        height_factor: int = config.plots["data_split"]["bubble"]["size_factors"][
            "height"
        ]
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


def generate(config: Config, action: str = "generate") -> list[str] | None:
    figs: List[Figure] = create_bubble_plots()

    if action not in ["generate", "download"]:
        return [[fig.layout.title.text, fig.to_html(full_html=False)] for fig in figs]

    if action == "generate":
        folder = f"figures/{__name__.split('.')[-1]}"
        isExist: bool = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)
        for fig in tqdm(figs, desc="Downloading figures"):
            fig.write_json(rf"{folder}/{fig.layout.title.text.replace(' ', '_')}.json")
    else:
        folder = f"{config.output_path}/{__name__.split('.')[-1]}"
        isExist: bool = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)
        for fig in tqdm(figs, desc="Downloading figures"):
            if len(fig.frames) > 0:
                frame = fig.frames[-1]
                fig.update(data=frame.data)
                fig.layout.sliders[0].update(active=len(fig.frames) - 1)
            w, h = fig.layout.width, fig.layout.height
            fig.write_image(
                rf"{folder}/{fig.layout.title.text.replace(' ', '_')}.svg",
                width=w,
                height=h,
            )
