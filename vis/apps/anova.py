import os
import re
import pandas as pd
import numpy as np
import scikit_posthocs as sp
from pathlib import Path
import plotly.express as px
import pingouin as pg
import plotly.figure_factory as ff
from statsmodels.multivariate.manova import MANOVA
from statsmodels.multivariate.multivariate_ols import MultivariateTestResults
from pandas.core.frame import DataFrame
from plotly.graph_objects import Figure
from typing import List, Dict
import numpy.typing as npt

from config import Config, clean, save_or_print_figures

config: Config = Config()

experiment_variables: List[str] = [
    "uuid4",
    "algorithm",
    "algorithm_rounds",
    "consensus_iterations",
    "training_epochs",
    "xmpp_domain",
    "graph_path",
    "dataset",
    "distribution",
    "ann",
    "seed",
]

variables: List[str] = ["agent", "experiment"]

agent_variables: List[str] = [
    "minimum_loss_achieved",
    "maximum_accuracy_achieved",
    "maximum_recall_achieved",
    "maximum_precision_achieved",
    "maximum_f1_achieved",
    "mean_seconds_by_round",
]


def create_df(config: Config) -> DataFrame:
    df: DataFrame = pd.DataFrame(
        columns=experiment_variables + variables + agent_variables
    )
    for root, dirs, files in os.walk(config.source_path):
        if root.endswith("raw"):
            root: Path = Path(root)
            dataset: DataFrame = pd.read_csv(root.joinpath(r"nn_inference.csv"))
            descriptive: str = root.parent.name

            with open(root.joinpath(r"general.log"), encoding="utf-8") as file:
                my_data: str = file.read()
            line = re.findall(r"Experiment details: <Experiment (.+)>\n", my_data)[0]
            splits: List[str] = line.split(",")
            splits: Dict[str, str] = dict(map(lambda i: i.strip().split("="), splits))
            experiment_vals: List[str] = list(splits.values())

            times: DataFrame = pd.read_csv(root.joinpath(r"algorithm.csv"))
            agents: List[str] = dataset.agent.unique()
            for ag in agents:
                agent_times: DataFrame = times[(times.agent == ag + "@localhost")]
                time: float = agent_times.seconds_to_complete.mean()

                data: DataFrame = dataset[(dataset.agent == ag)]
                numeric: List[float] = list(
                    map(
                        max,
                        [
                            data.test_accuracy,
                            data.test_recall,
                            data.test_precision,
                            data.test_f1_score,
                        ],
                    )
                )
                numerics: List[float] = [data.test_loss.min()] + numeric + [time]
                row: List[str | float] = (
                    experiment_vals + [ag] + [descriptive] + numerics
                )
                df.loc[len(df)] = row
    return df


def create_atable(df: DataFrame) -> DataFrame:
    atable: DataFrame = (
        df.groupby(["distribution", "ann"])
        .agg(maximum_accuracy=("maximum_accuracy_achieved", "max"))
        .reset_index()
    )
    return atable


def create_atable_fig(atable: DataFrame) -> Figure:
    atable_c: DataFrame = atable.copy()
    atable_c.columns = [clean(var) for var in atable_c.columns]
    fig: Figure = ff.create_table(atable_c)
    fig.update_layout(
        title_text="Table of Minimum Loss achieved grouped by Type and Network"
    )
    return fig


def create_interaction_plot(atable: DataFrame, var1: str, var2: str) -> Figure:
    fig: Figure = px.scatter(
        atable,
        x=var1,
        y="maximum_accuracy",
        color=var2,
    ).update_traces(mode="lines+markers")
    fig.update_layout(title_text=f"Interaction between {clean(var1)} and {clean(var2)}")
    fig.update_layout(config.plots["anova"]["interaction"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables[var1]["legend"],
        yaxis_title_text=config.variables["maximum_accuracy"]["legend"],
        legend_title_text=config.variables[var2]["legend"],
    )
    return fig


def create_box_plot(df: DataFrame, var: str) -> Figure:
    fig: Figure = px.box(
        df,
        x=var,
        y="maximum_accuracy_achieved",
        color=var,
        points="all",
    )
    fig.update_layout(
        title_text=f"Distribution of {clean('maximum_accuracy_achieved')} by {clean(var)}"
    )
    fig.update_layout(config.plots["anova"]["box"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables[var]["legend"],
        yaxis_title_text=config.variables["maximum_accuracy_achieved"]["legend"],
        legend_title_text=config.variables[var]["legend"],
    )
    return fig


def create_anova(df: DataFrame) -> Figure:
    atable: DataFrame = pg.anova(
        data=df,
        dv="maximum_accuracy_achieved",
        between=["distribution", "ann"],
        detailed=True,
    ).round(4)
    atable_c: DataFrame = atable.copy()
    atable_c.columns = [clean(var) for var in atable_c.columns]
    fig: Figure = ff.create_table(atable_c)
    fig.update_layout(config.plots["anova"]["anova"]["layout"])
    return fig


def create_tuckey_test(df: DataFrame) -> Figure:
    pg_test: DataFrame = pg.pairwise_tukey(
        data=df, dv="maximum_accuracy_achieved", between="distribution"
    ).round(3)
    pg_test_c: DataFrame = pg_test.copy()
    pg_test_c.columns = [clean(var) for var in pg_test_c.columns]
    fig: Figure = ff.create_table(pg_test_c)
    fig.update_layout(config.plots["anova"]["tuckey"]["layout"])
    return fig


def create_nemenyi_test(df: DataFrame) -> Figure:
    options: List[str] = df.distribution.unique()
    array = []
    for opt in options:
        opt_values: float = df.maximum_accuracy_achieved[(df.distribution == opt)]
        array.append(opt_values)
    data_nem: npt.NDArray[np.float64] = np.array(array)
    nemtable: npt.NDArray[np.float64] = sp.posthoc_nemenyi_friedman(
        data_nem.T
    ).to_numpy()
    fig: Figure = px.imshow(
        nemtable, x=options, y=options, color_continuous_scale="Viridis", aspect="auto"
    )
    fig.update_traces(text=nemtable, texttemplate="%{text}")
    fig.update_xaxes(side="top")
    fig.update_layout(config.plots["anova"]["nemenyi"]["layout"])
    return fig


def create_manova(df: DataFrame) -> MultivariateTestResults:
    manova: MANOVA = MANOVA.from_formula(
        "minimum_loss_achieved + maximum_accuracy_achieved + maximum_recall_achieved + maximum_precision_achieved + maximum_f1_achieved ~ distribution",
        data=df,
    )
    result: MultivariateTestResults = manova.mv_test()
    return result


def create_manova_figs(result: MultivariateTestResults, var: str) -> List[Figure]:
    intercept: DataFrame = result.results["Intercept"]["stat"]
    intercept.reset_index(inplace=True)
    intercept = intercept.rename(columns={"index": "tests"})
    fig: Figure = ff.create_table(intercept.round(3))
    fig.update_layout(config.plots["anova"]["manova_inter"]["layout"])

    var_results: DataFrame = result.results[var]["stat"]
    var_results.reset_index(inplace=True)
    var_results = var_results.rename(columns={"index": "tests"})
    fig2: Figure = ff.create_table(var_results.round(3))
    fig2.update_layout(config.plots["anova"]["manova_var"]["layout"])

    return [fig, fig2]


def generate(config: Config, download: bool = False) -> list[str] | None:
    data: DataFrame = create_df(config)
    atable: DataFrame = create_atable(data)
    f1: Figure = create_atable_fig(atable)
    f2: List[Figure] = [
        create_interaction_plot(atable, "distribution", "ann"),
        create_interaction_plot(atable, "ann", "distribution"),
    ]
    f3: Figure = create_box_plot(data, "distribution")
    f4: Figure = create_anova(data)
    f5: Figure = create_tuckey_test(data)
    f6: Figure = create_nemenyi_test(data)
    manova: MultivariateTestResults = create_manova(data)
    f7: List[Figure] = create_manova_figs(manova, "distribution")
    figs: List[Figure] = [f1] + f2 + [f3, f4, f5, f6] + f7
    return save_or_print_figures(download, figs, __name__)
