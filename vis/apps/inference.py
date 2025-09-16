import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pandas.core.frame import DataFrame
from plotly.graph_objects import Figure
from typing import List
from config import Config, clean, save_or_print_figures

config = Config()


def train_by_agent(train: DataFrame) -> List[Figure]:
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


def test_by_agent(test: DataFrame) -> List[Figure]:
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


def train_test_network(train: DataFrame, test: DataFrame) -> List[Figure]:
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


def generate(config: Config, download: bool = False) -> list[str] | None:
    train: DataFrame = pd.read_csv(config.experiment_path / r"nn_train.csv")
    test: DataFrame = pd.read_csv(config.experiment_path / r"nn_inference.csv")
    f1: List[Figure] = train_by_agent(train)
    f2: List[Figure] = test_by_agent(test)
    f3: List[Figure] = train_test_network(train, test)
    figs: List[Figure] = f1 + f2 + f3
    return save_or_print_figures(download, figs)
