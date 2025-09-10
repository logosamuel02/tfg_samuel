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


def create_atable(df):
    atable = pg.anova(
        data=df,
        dv="maximum_accuracy_achieved",
        between=["distribution", "ann"],
        detailed=True,
    ).round(4)
    return atable


def create_atable_fig(atable):
    fig = ff.create_table(atable)
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
        title="Interaction plot: Network and Type",
        template="seaborn",
        labels={
            "x": "Type",
            "response": "Avg Minimum loss achieved",
            "trace": "Network",
        },
    ).update_traces(mode="lines+markers")
    fig.update_layout(title_text="Interaction plot: Network and Type")
    return fig


def create_box_plot(df, var):
    fig = px.box(
        df,
        x=var,
        y="maximum_accuracy_achieved",
        color=var,
        points="all",
        template="seaborn",
    )
    fig.update_layout(title_text="Distribution distribution")
    return fig


def create_anova(df):
    atable = pg.anova(
        data=df,
        dv="maximum_accuracy_achieved",
        between=["distribution", "ann"],
        detailed=True,
    ).round(4)
    fig = ff.create_table(atable)
    fig.update_layout(title_text="ANOVA table")
    return fig


def create_tuckey_test(df):
    pg_test = pg.pairwise_tukey(
        data=df, dv="maximum_accuracy_achieved", between="distribution"
    ).round(3)
    fig = ff.create_table(pg_test)
    fig.update_layout(title_text="Post-hoc Tuckey Test")
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
    fig.update_layout(title_text="Post-hoc Nemenyi Test")
    return fig


def create_manova(df):
    manova = MANOVA.from_formula(
        "minimum_loss_achieved + maximum_accuracy_achieved + maximum_recall_achieved + maximum_precision_achieved + maximum_f1_achieved ~ distribution",
        data=df,
    )
    result = manova.mv_test()
    return result


def create_manova_figs(result):
    intercept = result.results["Intercept"]["stat"]
    intercept.reset_index(inplace=True)
    intercept = intercept.rename(columns={"index": "tests"})
    fig = ff.create_table(intercept)
    fig.update_layout(title_text="MANOVA table Intercept")

    distribution = result.results["distribution"]["stat"]
    distribution.reset_index(inplace=True)
    distribution = distribution.rename(columns={"index": "tests"})
    fig2 = ff.create_table(distribution)
    fig2.update_layout(title_text="MANOVA table Distribution")

    return [fig, fig2]


def generate(config, download=False):
    data = create_df(config)
    atable = create_atable(data)
    f = create_atable_fig(atable)
    f2 = [
        create_interaction_plot(atable, "distribution", "ann"),
        create_interaction_plot(atable, "ann", "distribution"),
    ]
    f3 = create_box_plot(atable)
    f4 = create_anova(data)
    f5 = create_tuckey_test(data)
    f6 = create_nemenyi_test(data)
    manova = create_manova(data)
    f7 = create_manova_figs(manova)

    if download:
        root = "images"
        folder = f"{root}/{__name__.split('.')[0]}"
        isExist = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)
        for i, F in enumerate(figs):
            F.write_image(f"{folder}/{F.layout.title.text.replace(' ', '_')}.svg")
    else:
        return [F.to_html(full_html=False) for F in figs]


def figures(download=False):
    FOLDER = Path(
        r"C:/Users/samue/OneDrive/Escritorio/Tareas UNI/tfg/tfg_samuel/vis/xperiments/experimentos_con_cnn"
    )

    template = "seaborn"
    experiment_variables = ["agent", "algorithm", "n_agents", "type", "network"]
    agent_variables = [
        "minimum_loss_achieved",
        "maximum_accuracy_achieved",
        "maximum_recall_achieved",
        "maximum_precision_achieved",
        "maximum_f1_achieved",
        "mean_seconds_by_round",
    ]
    network_variables = [
        "minimum_loss_net",
        "maximum_acc_net",
        "maximum_recall_net",
        "maximum_precision_net",
        "maximum_f1_net",
    ]
    df = pd.DataFrame(columns=experiment_variables + agent_variables)

    def max_network(data):
        df_numeric = pd.DataFrame(columns=network_variables)
        rounds = data.algorithm_round.unique()
        for r in rounds:
            round_data = data[(data.algorithm_round == r)]
            df_numeric.loc[len(df_numeric),] = list(
                map(
                    lambda x: sum(x) / len(x),
                    [
                        data.test_loss,
                        data.test_accuracy,
                        data.test_recall,
                        data.test_precision,
                        data.test_f1_score,
                    ],
                )
            )
        return [min(df_numeric.best_loss_net)].extend(
            list(
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
        )

    # Modificar esta lista
    experiments_list = [x for x in list(FOLDER.iterdir()) if "10" in x.name]
    for PATH in experiments_list:
        dataset = pd.read_csv(PATH.joinpath(r"raw/nn_inference.csv"))
        agents = dataset.agent.unique()
        dataset = dataset[(dataset.algorithm_round <= 100)]
        descriptive = PATH.name.split("_")
        times = pd.read_csv(PATH.joinpath(r"raw/algorithm.csv"))
        times = times[(times.algorithm_round <= 100)]
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
            row = [ag] + ["acol"] + descriptive + numerics
            df.loc[len(df)] = row

    data = df[["type", "network", "minimum_loss_achieved"]]

    ### DO data.RESET_INDEX(inplace=TRUE)
    lista = []
    for i, t in enumerate(data.type.unique()):
        for j, n in enumerate(data.network.unique()):
            p = data.minimum_loss_achieved[(data.type == t) & (data.network == n)]
            lista.append([t, n, p.mean()])

    print("######## before create figures")

    atable = pd.DataFrame(lista, columns=["type", "network", "Minimum loss average"])
    F0 = ff.create_table(atable)

    f = pd.DataFrame(lista, columns=["x", "trace", "response"])
    fig1 = px.scatter(
        f,
        x="x",
        y="response",
        color="trace",
        title="Interaction plot: Network and Type",
        template=template,
        labels={
            "x": "Type",
            "response": "Avg Minimum loss achieved",
            "trace": "Network",
        },
    ).update_traces(mode="lines+markers")

    fig2 = px.scatter(
        f,
        x="trace",
        y="response",
        color="x",
        title="Interaction plot: Type and Network",
        template=template,
        labels={
            "x": "Network",
            "response": "Avg Minimum loss achieved",
            "trace": "Type",
        },
    ).update_traces(mode="lines+markers")

    fig3 = px.box(
        data,
        x="type",
        y="minimum_loss_achieved",
        color="type",
        points="all",
        template=template,
    )
    fig4 = px.box(
        data,
        x="network",
        y="minimum_loss_achieved",
        color="network",
        points="all",
        template=template,
    )
    print("######## before create sub plots")

    F1 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Type vs Loss",
            "Network vs Loss",
        ),
    )
    F2 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Type vs Loss",
            "Network vs Loss",
        ),
    )

    print("######## before update axis")

    F1.update_xaxes(title_text="Type", row=1, col=1)
    F1.update_xaxes(title_text="Network", row=1, col=2)
    F2.update_xaxes(title_text="Type", row=1, col=1)
    F2.update_xaxes(title_text="Network", row=1, col=2)

    F1.update_yaxes(title_text="Minimum Loss achieved", row=1, col=1)
    F1.update_yaxes(title_text="Minimum Loss achieved", row=1, col=2)
    F2.update_yaxes(title_text="Minimum Loss achieved", row=1, col=1)
    F2.update_yaxes(title_text="Minimum Loss achieved", row=1, col=2)

    print("######## before join")
    for i, x_fig in enumerate([fig1, fig2]):
        x_traces = []
        for trace in range(len(x_fig["data"])):
            t = x_fig["data"][trace]
            t.showlegend = 0 == i
            x_traces.append(t)
        for traces in x_traces:
            F1.append_trace(traces, row=1, col=i + 1)

    for i, x_fig in enumerate([fig3, fig4]):
        x_traces = []
        for trace in range(len(x_fig["data"])):
            t = x_fig["data"][trace]
            t.showlegend = 0 == i
            x_traces.append(t)
        for traces in x_traces:
            F2.append_trace(traces, row=1, col=i + 1)

    print("######## before table")

    atable = pg.anova(
        data=data,
        dv="minimum_loss_achieved",
        between=["type", "network"],
        detailed=True,
    ).round(4)

    F3 = ff.create_table(atable)

    print("######## after F3")

    pg_test = pg.pairwise_tukey(
        data=data, dv="minimum_loss_achieved", between="network"
    ).round(3)
    F4 = ff.create_table(pg_test)

    print("######## after F4")

    g1 = data.minimum_loss_achieved[(data.network == "complete")]
    g2 = data.minimum_loss_achieved[(data.network == "ring")]
    g3 = data.minimum_loss_achieved[(data.network == "sw")]
    data_nem = np.array([g1, g2, g3])
    x = ["complete", "ring", "sw"]
    nemtable = sp.posthoc_nemenyi_friedman(data_nem.T).to_numpy()
    F5 = px.imshow(nemtable, x=x, y=x, color_continuous_scale="Viridis", aspect="auto")
    F5.update_traces(text=nemtable, texttemplate="%{text}")
    F5.update_xaxes(side="top")

    print("######## after F5")

    manova = MANOVA.from_formula(
        "minimum_loss_achieved + maximum_accuracy_achieved + maximum_recall_achieved + maximum_precision_achieved + maximum_f1_achieved ~ network",
        data=df,
    )
    result = manova.mv_test()
    intercept = result.results["Intercept"]["stat"]
    intercept.reset_index(inplace=True)
    intercept = intercept.rename(columns={"index": "tests"})
    network = result.results["network"]["stat"]
    network.reset_index(inplace=True)
    network = network.rename(columns={"index": "tests"})
    F6 = ff.create_table(intercept)
    F7 = ff.create_table(network)

    for F in [F1, F2]:
        F.update_layout(height=600, width=1800)

    titles = [
        "Table of Minimum Loss achieved grouped by Type and Network",
        "Interaction plots",
        "Box plots",
        "ANOVA table",
        "Post-hoc Tuckey Test",
        "Post-hoc Nemenyi Test",
        "MANOVA table Intercept",
        "MANOVA table Network",
    ]

    for F, title in zip([F0, F1, F2, F3, F4, F5, F6, F7], titles):
        F.update_layout(title_text=title)
        F.update_layout({"margin": {"t": 50}})

    print("######## before return")

    if download:
        root = "images"
        folder = f"{root}/{__name__.split('.')[0]}"
        isExist = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)
        for i, F in enumerate([F0, F1, F2, F3, F4, F5, F6, F7]):
            F.write_image(f"{folder}/{F.layout.title.text.replace(' ','_')}.svg")
    else:
        return [F.to_html(full_html=False) for F in [F0, F1, F2, F3, F4, F5, F6, F7]]


# fig.write_html("index.html")
# print("Figure CREATED!")
