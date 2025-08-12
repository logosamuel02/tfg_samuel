import os
import pandas as pd
import numpy as np
import scikit_posthocs as sp
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pingouin as pg
import plotly.figure_factory as ff
from statsmodels.multivariate.manova import MANOVA


def figures(download=False):
    FOLDER = Path("/home/slozgom/tfg/tfg_samuel/xperiments/experimentos_con_cnn")

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
        dataset = pd.read_csv(PATH.joinpath("raw/nn_inference.csv"))
        agents = dataset.agent.unique()
        dataset = dataset[(dataset.algorithm_round <= 100)]
        descriptive = PATH.name.split("_")
        times = pd.read_csv(PATH.joinpath("raw/algorithm.csv"))
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
        "MANOVA table Network"
    ]

    for F, title in zip([F0, F1, F2, F3, F4, F5, F6, F7], titles):
        F.update_layout(title_text=title)
        F.update_layout({"margin": {"t": 50}})

    print("######## before return")

    if download:
        root = "images"
        folder = f"{root}/{__name__.split(".")[0]}"
        isExist = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)
        for i,F in enumerate([F0, F1, F2, F3, F4, F5, F6, F7]):
            F.write_image(f"{folder}/{F.layout.title.text.replace(" ", "_")}.svg")
    else:
        return [F.to_html(full_html=False) for F in [F0, F1, F2, F3, F4, F5, F6, F7]]


# fig.write_html("index.html")
# print("Figure CREATED!")
