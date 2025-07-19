import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import pingouin as pg
from statsmodels.graphics.factorplots import interaction_plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.tools as tls
import plotly.express as px

FOLDER = Path("/home/slozgom/personal/tfg_project/xperiments/experimentos_con_cnn")

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
data.groupby(["type", "network"])["minimum_loss_achieved"].agg(["mean", "std"])

lista = []
for i, t in enumerate(data.type.unique()):
    for j, n in enumerate(data.network.unique()):
        p = data.minimum_loss_achieved[(data.type == t) & (data.network == n)]
        lista.append([t, n, p.mean()])
f = pd.DataFrame(lista, columns=["x", "trace", "response"])
fig1 = px.scatter(f, x="x", y="response", color="trace")
fig2 = px.line(f, x="x", y="response", color="trace")

fig1_traces = []
fig2_traces = []
for trace in range(len(fig1["data"])):
    fig1_traces.append(fig1["data"][trace])
for trace in range(len(fig2["data"])):
    t = fig2["data"][trace]
    t.showlegend = False
    fig2_traces.append(t)

fig = make_subplots(
    rows=1,
    cols=2,
    start_cell="top-left",
)

for traces in fig1_traces:
    fig.append_trace(traces, row=1, col=1)
for traces in fig2_traces:
    fig.append_trace(traces, row=1, col=2)

fig.update_layout(height=500, width=1100)
fig.write_html("index.html")
print("Figure CREATED!")
