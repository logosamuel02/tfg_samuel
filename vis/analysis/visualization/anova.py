import os
from typing import List
import plotly.express as px
import pingouin as pg
import plotly.figure_factory as ff
from statsmodels.multivariate.multivariate_ols import MultivariateTestResults
from pandas.core.frame import DataFrame
from plotly.graph_objects import Figure
from tqdm import tqdm
import time

import preprocess as pre
from export import Config
from preprocess import clean

config: Config = Config()


def create_atable_fig() -> Figure:
    atable = pre.anova_table()
    atable.columns = [clean(var) for var in atable.columns]
    fig: Figure = ff.create_table(atable)
    fig.update_layout(
        title_text="Table of Minimum Loss achieved grouped by Type and Network"
    )
    return fig


def create_interaction_plot(
    factor1: str = "distribution",
    factor2: str = "ann",
    level: str = "maximum_accuracy",
) -> Figure:
    atable = pre.anova_table()
    fig: Figure = px.scatter(
        atable,
        x=factor1,
        y=level,
        color=factor2,
    ).update_traces(mode="lines+markers")
    fig.update_layout(
        title_text=f"Interaction between {clean(factor1)} and {clean(factor2)}"
    )
    fig.update_layout(config.plots["anova"]["interaction"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables[factor1]["legend"],
        yaxis_title_text=config.variables[level]["legend"],
        legend_title_text=config.variables[factor2]["legend"],
    )
    return fig


def create_box_plot(factor: str) -> Figure:
    df = pre.df_anova()
    fig: Figure = px.box(
        df,
        x=factor,
        y="maximum_accuracy_achieved",
        color=factor,
        points="all",
    )
    fig.update_layout(
        title_text=f"Distribution of {clean('maximum_accuracy_achieved')} by {clean(factor)}"
    )
    fig.update_layout(config.plots["anova"]["box"]["layout"])
    fig.update_layout(
        xaxis_title_text=config.variables[factor]["legend"],
        yaxis_title_text=config.variables["maximum_accuracy_achieved"]["legend"],
        legend_title_text=config.variables[factor]["legend"],
    )
    return fig


def create_anova(
    factor1: str = "distribution",
    factor2: str = "ann",
    level: str = "maximum_accuracy_achieved",
) -> Figure:
    df = pre.df_anova()
    atable: DataFrame = pg.anova(
        data=df,
        dv=level,
        between=[factor1, factor2],
        detailed=True,
    ).round(4)
    atable_c: DataFrame = atable.copy()
    atable_c.columns = [clean(var) for var in atable_c.columns]
    fig: Figure = ff.create_table(atable_c)
    fig.update_layout(config.plots["anova"]["anova"]["layout"])
    return fig


def create_tuckey_test(
    factor: str = "distribution",
    level: str = "maximum_accuracy_achieved",
) -> Figure:
    df = pre.df_anova()
    pg_test: DataFrame = pg.pairwise_tukey(data=df, dv=level, between=factor).round(3)
    pg_test_c: DataFrame = pg_test.copy()
    pg_test_c.columns = [clean(var) for var in pg_test_c.columns]
    fig: Figure = ff.create_table(pg_test_c)
    fig.update_layout(config.plots["anova"]["tuckey"]["layout"])
    return fig


def create_nemenyi_test() -> Figure:
    nemtable, options = pre.nemenyi_test()
    fig: Figure = px.imshow(
        nemtable, x=options, y=options, color_continuous_scale="Viridis", aspect="auto"
    )
    fig.update_traces(text=nemtable, texttemplate="%{text}")
    fig.update_xaxes(side="top")
    fig.update_layout(config.plots["anova"]["nemenyi"]["layout"])
    return fig


def create_manova_figs(var: str) -> List[Figure]:
    result: MultivariateTestResults = pre.manova_table()
    intercept: DataFrame = result.results["Intercept"]["stat"]
    intercept.reset_index(inplace=True)
    intercept = intercept.rename(columns={"index": "tests"})
    fig1: Figure = ff.create_table(intercept.round(3))
    fig1.update_layout(config.plots["anova"]["manova_inter"]["layout"])

    var_results: DataFrame = result.results[var]["stat"]
    var_results.reset_index(inplace=True)
    var_results = var_results.rename(columns={"index": "tests"})
    fig2: Figure = ff.create_table(var_results.round(3))
    fig2.update_layout(config.plots["anova"]["manova_var"]["layout"])
    return [fig1, fig2]


def generate(config: Config, action: str = "generate") -> list[str] | None:
    f1 = create_atable_fig()
    f2 = create_interaction_plot("distribution", "ann")
    f3 = create_interaction_plot("ann", "distribution")
    f4 = create_box_plot("distribution")
    f5 = create_anova()
    f6 = create_tuckey_test()
    f7 = create_nemenyi_test()
    f8 = create_manova_figs("distribution")
    figs = [f1, f2, f3, f4, f5, f6, f7] + f8

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
