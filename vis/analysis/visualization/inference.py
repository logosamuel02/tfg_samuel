import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pandas.core.frame import DataFrame
from plotly.graph_objects import Figure
from typing import List
import preprocess as pre
import loaders as load
from tqdm import tqdm
import time
from export import Config
from preprocess import clean

config = Config()


def train_by_agent() -> List[Figure]:
    train = load.train_dataset()
    metrics: List[str] = ["accuracy", "loss", "precision", "recall", "f1_score"]
    figs = []
    for metric in metrics:
        fig: Figure = px.line(
            train,
            x="algorithm_round",
            y=metric,
            color="agent",
            title=f"{config.variables[metric]['legend']} evolution across rounds by agent",
        )
        fig.update_layout(config.plots["inference"]["train"]["layout"])
        fig.update_layout(
            xaxis_title_text=config.variables["algorithm_round"]["legend"],
            yaxis_title_text=config.variables[metric]["legend"],
            legend_title_text=config.variables["agent"]["legend"],
        )
        figs.append(fig)
    return figs


def test_by_agent() -> List[Figure]:
    test = load.inference_dataset()
    metrics: List[str] = [
        "test_accuracy",
        "test_loss",
        "test_precision",
        "test_recall",
        "test_f1_score",
    ]
    figs = []
    for metric in metrics:
        fig: Figure = px.line(
            test,
            x="algorithm_round",
            y=metric,
            color="agent",
            title=f"{config.variables[metric]['legend']} evolution across rounds by agent",
        )
        fig.update_layout(config.plots["inference"]["test"]["layout"])
        fig.update_layout(
            xaxis_title_text=config.variables["algorithm_round"]["legend"],
            yaxis_title_text=config.variables[metric]["legend"],
            legend_title_text=config.variables["agent"]["legend"],
        )
        figs.append(fig)
    return figs


def train_test_network() -> List[Figure]:
    train = load.train_dataset()
    test = load.inference_dataset()
    metrics: List[str] = ["accuracy", "loss", "precision", "recall", "f1_score"]
    figs = []
    for metric in metrics:

        g_train: DataFrame = (
            train.groupby(["algorithm_round"])
            .agg(
                maximum_accuracy=(metric, "max"),
                minimum_accuracy=(metric, "min"),
                mean_accuracy=(metric, "mean"),
            )
            .reset_index()
        )
        g_test: DataFrame = (
            test.groupby(["algorithm_round"])
            .agg(
                maximum_accuracy=(f"test_{metric}", "max"),
                minimum_accuracy=(f"test_{metric}", "min"),
                mean_accuracy=(f"test_{metric}", "mean"),
            )
            .reset_index()
        )

        x: List[int] = list(g_test.algorithm_round.unique())
        x_rev: List[int] = x[::-1]

        nmax: List[float] = g_train.maximum_accuracy.to_list()
        nmin: List[float] = g_train.minimum_accuracy.to_list()
        nmin = nmin[::-1]

        tmax: List[float] = g_test.maximum_accuracy.to_list()
        tmin: List[float] = g_test.minimum_accuracy.to_list()
        tmin = tmin[::-1]

        fig: Figure = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x + x_rev,
                y=nmax + nmin,
                fill="toself",
                fillcolor=config.plots["inference"]["train_test"]["traces"][
                    "train_fillcolor"
                ],
                line_color=config.plots["inference"]["train_test"]["traces"][
                    "train_linecolor"
                ],
                name="Train",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x + x_rev,
                y=tmax + tmin,
                fill="toself",
                fillcolor=config.plots["inference"]["train_test"]["traces"][
                    "test_fillcolor"
                ],
                line_color=config.plots["inference"]["train_test"]["traces"][
                    "test_linecolor"
                ],
                showlegend=False,
                name="Test",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=g_train.mean_accuracy,
                line_color=config.plots["inference"]["train_test"]["traces"][
                    "mean_train_linecolor"
                ],
                name="Train",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=g_test.mean_accuracy,
                line_color=config.plots["inference"]["train_test"]["traces"][
                    "mean_test_linecolor"
                ],
                name="Test",
            )
        )

        fig.update_traces(mode="lines")
        fig.update_layout(title_text=f"Netowk's performance {clean(metric)} by round")
        fig.update_layout(config.plots["inference"]["train_test"]["layout"])
        fig.update_layout(
            xaxis_title_text=config.variables["algorithm_round"]["legend"],
            yaxis_title_text=config.variables[metric]["legend"],
            legend_title_text=config.variables["agent"]["legend"],
        )
        figs.append(fig)
    return figs


def generate(config: Config, action: str = "generate") -> list[str] | None:
    f1: List[Figure] = train_by_agent()
    f2: List[Figure] = test_by_agent()
    f3: List[Figure] = train_test_network()
    figs: List[Figure] = f1 + f2 + f3

    if action not in ["generate", "download"]:
        return figs

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
