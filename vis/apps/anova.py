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

from config import Config, clean, save_or_print_figures

config = Config()

experiment_variables = [
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

variables = ["agent", "experiment"]

agent_variables = [
    "minimum_loss_achieved",
    "maximum_accuracy_achieved",
    "maximum_recall_achieved",
    "maximum_precision_achieved",
    "maximum_f1_achieved",
    "mean_seconds_by_round",
]


def create_df(config):
    df = pd.DataFrame(columns=experiment_variables + variables + agent_variables)
    for root, dirs, files in os.walk(config.source_path):
        if root.endswith("raw"):
            root = Path(root)
            dataset = pd.read_csv(root.joinpath(r"nn_inference.csv"))
            descriptive = root.parent.name

            with open(root.joinpath(r"general.log"), encoding="utf-8") as file:
                my_data = file.read()
            line = re.findall(r"Experiment details: <Experiment (.+)>\n", my_data)[0]
            splits = line.split(",")
            splits = dict(map(lambda i: i.strip().split("="), splits))
            experiment_vals = list(splits.values())

            times = pd.read_csv(root.joinpath(r"algorithm.csv"))
            agents = dataset.agent.unique()
            for ag in agents:
                agent_times = times[(times.agent == ag + "@localhost")]
                time = agent_times.seconds_to_complete.mean()

                data = dataset[(dataset.agent == ag)]
                numeric = list(
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
                numerics = [data.test_loss.min()] + numeric + [time]
                row = experiment_vals + [ag] + [descriptive] + numerics
                df.loc[len(df)] = row
    return df


def create_atable(df):
    atable = (
        df.groupby(["distribution", "ann"])
        .agg(max_acc=("maximum_accuracy_achieved", "max"))
        .reset_index()
    )
    return atable


def create_atable_fig(atable):
    atable_c = atable.copy()
    atable_c.columns = [clean(var) for var in atable_c.columns]
    fig = ff.create_table(atable_c)
    fig.update_layout(
        title_text="Table of Minimum Loss achieved grouped by Type and Network"
    )
    return fig


def create_interaction_plot(atable, var1, var2):
    fig = px.scatter(
        atable,
        x=var1,
        y="max_acc",
        color=var2,
    ).update_traces(mode="lines+markers")
    fig.update_layout(title_text=f"Interaction between {clean(var1)} and {clean(var2)}")
    fig.update_layout(config.plots["anova"]["interaction"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables[var1]["legend"],
        yaxis_title_text=config.variables["max_acc"]["legend"],
        legend_title_text=config.variables[var2]["legend"],
    )
    return fig


def create_box_plot(df, var):
    fig = px.box(
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


def create_anova(df):
    atable = pg.anova(
        data=df,
        dv="maximum_accuracy_achieved",
        between=["distribution", "ann"],
        detailed=True,
    ).round(4)
    atable_c = atable.copy()
    atable_c.columns = [clean(var) for var in atable_c.columns]
    fig = ff.create_table(atable_c)
    fig.update_layout(config.plots["anova"]["anova"]["layout"])
    return fig


def create_tuckey_test(df):
    pg_test = pg.pairwise_tukey(
        data=df, dv="maximum_accuracy_achieved", between="distribution"
    ).round(3)
    pg_test_c = pg_test.copy()
    pg_test_c.columns = [clean(var) for var in pg_test_c.columns]
    fig = ff.create_table(pg_test_c)
    fig.update_layout(config.plots["anova"]["tuckey"]["layout"])
    return fig


def create_nemenyi_test(df):
    options = df.distribution.unique()
    array = []
    for opt in options:
        opt_values = df.maximum_accuracy_achieved[(df.distribution == opt)]
        array.append(opt_values)
    data_nem = np.array(array)
    nemtable = sp.posthoc_nemenyi_friedman(data_nem.T).to_numpy()
    fig = px.imshow(
        nemtable, x=options, y=options, color_continuous_scale="Viridis", aspect="auto"
    )
    fig.update_traces(text=nemtable, texttemplate="%{text}")
    fig.update_xaxes(side="top")
    fig.update_layout(config.plots["anova"]["nemenyi"]["layout"])
    return fig


def create_manova(df):
    manova = MANOVA.from_formula(
        "minimum_loss_achieved + maximum_accuracy_achieved + maximum_recall_achieved + maximum_precision_achieved + maximum_f1_achieved ~ distribution",
        data=df,
    )
    result = manova.mv_test()
    return result


def create_manova_figs(result, var):
    intercept = result.results["Intercept"]["stat"]
    intercept.reset_index(inplace=True)
    intercept = intercept.rename(columns={"index": "tests"})
    fig = ff.create_table(intercept.round(3))
    fig.update_layout(config.plots["anova"]["manova_inter"]["layout"])

    var_results = result.results[var]["stat"]
    var_results.reset_index(inplace=True)
    var_results = var_results.rename(columns={"index": "tests"})
    fig2 = ff.create_table(var_results.round(3))
    fig2.update_layout(config.plots["anova"]["manova_var"]["layout"])

    return [fig, fig2]


def generate(config, download=False):
    data = create_df(config)
    atable = create_atable(data)
    f1 = create_atable_fig(atable)
    f2 = [
        create_interaction_plot(atable, "distribution", "ann"),
        create_interaction_plot(atable, "ann", "distribution"),
    ]
    f3 = create_box_plot(data, "distribution")
    f4 = create_anova(data)
    f5 = create_tuckey_test(data)
    f6 = create_nemenyi_test(data)
    manova = create_manova(data)
    f7 = create_manova_figs(manova, "distribution")
    figs = [f1] + f2 + [f3, f4, f5, f6] + f7
    return save_or_print_figures(download, figs)
