import os
import pandas as pd
import loaders as load
from typing import List
from more_itertools import sort_together
from pandas.core.frame import DataFrame
from plotly.graph_objects import Figure
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._libs.tslibs.timedeltas import Timedelta
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
from itertools import groupby
import branca.colormap as cm
from matplotlib.colors import to_hex
from networkx.classes import Graph
import networkx as nx
import warnings

from export import Config

warnings.filterwarnings("ignore")

config = Config()


def clean(var: str) -> str:
    var: str = " ".join(var.split("_"))
    var = var.replace(".", " ")
    return var.capitalize()


def violin_plot():
    data = load.algorithm_dataset()
    agents: List[str] = list(
        map(lambda x: x.split("@")[0], sorted(data.agent.unique()))
    )
    times = []
    for agent in agents:
        agent_data: DataFrame = data[(data.agent == agent + "@localhost")]
        time: List[int] = list(agent_data.seconds_to_complete)
        times.append(time)

    df_bolos: DataFrame = pd.DataFrame(columns=agents)
    for t, a in zip(times, agents):
        df_bolos[a] = t
    df_bolos = df_bolos.melt()
    return df_bolos, agents


def execution_time_plot():
    data = load.algorithm_dataset()
    data.timestamp = pd.to_datetime(data.timestamp)
    agents: List[str] = data.agent.unique()
    times = []
    elapsed = []
    for i, agent in enumerate(agents):
        dates: List[Timestamp] = list(data.timestamp[data.agent == agent])
        rang: Timedelta = dates[-1] - dates[0]
        times.append(round(rang.total_seconds(), 2))
        elapsed.append(dates[-1])

    m: Timestamp = min(elapsed)
    for i, agent in enumerate(agents):
        elapsed[i] = round((elapsed[i] - m).total_seconds(), 2)

    elapsed, agents, times = sort_together((elapsed, agents, times))

    df: DataFrame = pd.DataFrame(
        {
            "seconds": times,
            "agent": list(map(lambda x: x.split("@")[0], agents)),
            "seconds_elapsed": elapsed,
        }
    )
    return df


def df_anova() -> DataFrame:
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


def anova_table(
    factor1: str = "distribution",
    factor2: str = "ann",
    level: str = "maximum_accuracy_achieved",
    operation_level: str = "max",
) -> DataFrame:
    df = df_anova()
    atable: DataFrame = (
        df.groupby([factor1, factor2])
        .agg(maximum_accuracy=(level, operation_level))
        .reset_index()
    )
    return atable


def manova_table(factor: str = "distribution") -> MultivariateTestResults:
    df = df_anova()
    manova: MANOVA = MANOVA.from_formula(
        f"maximum_accuracy_achieved + maximum_recall_achieved + maximum_precision_achieved + maximum_f1_achieved ~ {factor}",
        data=df,
    )
    result: MultivariateTestResults = manova.mv_test()
    return result


def nemenyi_test():
    df = df_anova()
    options: List[str] = df.distribution.unique()
    array = []
    for opt in options:
        opt_values: float = df.maximum_accuracy_achieved[(df.distribution == opt)]
        array.append(opt_values)
    data_nem: npt.NDArray[np.float64] = np.array(array)
    nemtable: npt.NDArray[np.float64] = sp.posthoc_nemenyi_friedman(
        data_nem.T
    ).to_numpy()
    return nemtable, options


def convergence_df() -> DataFrame:
    df: DataFrame = pd.read_csv(config.experiment_path / r"nn_convergence.csv")
    lst: List[str] = df.layer.unique()
    split_layer_str = lambda x: x.split(".")[0]
    global layers_opts
    layers_opts = {
        split_layer_str(k): list(g)
        for k, g in groupby(sorted(lst, key=split_layer_str), key=split_layer_str)
    }
    df: DataFrame = df[(df.description == "PRE-TRAIN") & (df.epoch_or_iteration == 1)]
    pivot: DataFrame = df.pivot(columns="layer", values="weight")
    df = df[df.layer == layers_opts["conv1"][0]].reset_index()

    for layer in layers_opts.keys():
        df[layers_opts[layer][0]] = (
            pivot[layers_opts[layer][0]]
            .dropna()
            .reset_index()[f"{layers_opts[layer][0]}"]
        )
        df[layers_opts[layer][1]] = (
            pivot[layers_opts[layer][1]]
            .dropna()
            .reset_index()[f"{layers_opts[layer][1]}"]
        )

    df = df.sort_values(
        [
            "algorithm_round",
            "agent",
        ],
        ignore_index=True,
    )
    N_UNIQUE_AGENTS: int = df["agent"].nunique()
    df_indexed: DataFrame = pd.DataFrame()
    for index in np.arange(start=0, stop=len(df) + 1, step=N_UNIQUE_AGENTS):
        df_slicing: DataFrame = df.iloc[:index].copy()
        df_slicing["frame"] = index // N_UNIQUE_AGENTS
        df_indexed: DataFrame = pd.concat([df_indexed, df_slicing])
    config.layers_opts = layers_opts
    return df_indexed


def bubble_colors():
    data = load.data_split_dataset()
    agents: List[str] = data.agent.unique()

    cmaps: List[cm.LinearColormap] = [
        cm.linear.Pastel1_03.scale(0, 2),
        cm.linear.Pastel1_04.scale(0, 3),
        cm.linear.Pastel1_05.scale(0, 4),
        cm.linear.Pastel1_06.scale(0, 5),
        cm.linear.Pastel1_07.scale(0, 6),
        cm.linear.Pastel1_08.scale(0, 7),
        cm.linear.Pastel1_09.scale(0, 8),
    ]
    dic_cmaps: Dict[str, cm.LinearColormap] = {
        str(i + 3): cmap for i, cmap in enumerate(cmaps)
    }
    labels: List[str] = data.label.unique()

    try:
        cmap: cm.LinearColormap = dic_cmaps[str(len(labels))]
    except KeyError:
        if len(labels) < 3:
            cmap: cm.LinearColormap = cm.linear.Pastel1_03.scale(0, len(labels) - 1)
        else:
            cmap: cm.LinearColormap = cm.linear.Pastel1_09.scale(0, len(labels) - 1)

    colors: List[str] = ["white"] + [cmap(i) for i in range(len(labels))] + ["white"]
    colors = list(map(to_hex, colors))
    return colors


def bubble_interprocess(phase: str = "train"):
    data = load.data_split_dataset()
    colors = bubble_colors()
    labels: List[str] = data.label.unique()
    agents = sorted(data.agents.unique())

    X: npt.NDArray[np.int64] = np.zeros((len(labels) + 2, len(agents) + 2))
    for i, agent in enumerate(agents):
        split_data: DataFrame = data[
            (data.agent == agent) & (data.description == phase)
        ]
        for x, (_, row) in enumerate(split_data.iterrows()):
            X[row.label + 1, i + 1] = row["count"]

    scale: int = 10
    M, N = X.shape
    X_sizes: npt.NDArray[np.int64] = X.copy()
    for i in range(M):
        xmin, xmax = X_sizes[i, :].min(), X_sizes[i, :].max()
        tmin, tmax = (
            config.plots["data_split"]["bubble"]["size_factors"]["min_bubble_size"],
            config.plots["data_split"]["bubble"]["size_factors"]["max_bubble_size"],
        )
        X_sizes[i, :] = (X_sizes[i, :] - xmin) / (xmax - xmin) * (tmax - tmin) + tmin
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
    return agents, labels, x, y, colores, sizes, texts


def heatmap_messages():
    data = load.message_dataset()
    data: DataFrame = data[["sender", "to"]]
    data.sender = list(map(lambda x: x.split("@")[0], data.sender))
    data.to = list(map(lambda x: x.split("@")[0], data.to))
    data_cross = pd.crosstab(index=data.sender, columns=data.to)
    agents: List[str] = data.sender.unique()
    return agents, data_cross


def heatmap_sizes():
    data = load.message_dataset()
    data: DataFrame = data[["sender", "to", "size"]]
    data.sender = list(map(lambda x: x.split("@")[0], data.sender))
    data.to = list(map(lambda x: x.split("@")[0], data.to))
    data_cross: DataFrame = (
        data.groupby(["sender", "to"])["size"]
        .sum()
        .div(1024 * 1024)
        .round(0)
        .unstack()
        .fillna(0)
    )
    agents: List[str] = data.sender.unique()
    return agents, data_cross


def statistics_messages():
    data = load.message_dataset()
    stats: DataFrame = (
        data.groupby("type")
        .agg(
            number_of_messages=("size", "count"),
            total_size_kB=("size", "sum"),
            average_size_kb=("size", "mean"),
            standard_deviation_kb=("size", "std"),
            minimum_size_kb=("size", "min"),
            maximum_size_kb=("size", "max"),
        )
        .round(2)
        .reset_index()
    )
    stats.loc[:, ~stats.columns.isin(["number_of_messages", "type"])] = (
        stats.loc[:, ~stats.columns.isin(["number_of_messages", "type"])] / 1024
    ).round(2)
    # stats["number_of_messages"] = stats["number_of_messages"] * 1024
    stats_c: DataFrame = stats.copy()
    stats_c.columns = [clean(var) for var in stats_c.columns]
    return stats_c


def distribution_data_df() -> Figure:
    data = load.message_dataset()
    data.timestamp = pd.to_datetime(data.timestamp)
    min_date: Timestamp = data.timestamp.min()
    data["timestamp_minutes"] = data.timestamp.apply(
        lambda x: int((x - min_date).total_seconds() / 60.0)
    )
    return data


def create_network_coordinates():
    # CONVERT DATA
    messages_mod: DataFrame = load.message_dataset()
    messages_mod = messages_mod[
        ["sender", "to"]
    ]  # [(messages_mod.algorithm_round <= 100)
    messages_mod.sender = messages_mod.sender.apply(lambda x: x.split("@")[0])
    messages_mod.to = messages_mod.to.apply(lambda x: x.split("@")[0])

    agents: List[str] = sorted(messages_mod.sender.unique())

    # GENERATE COORDINATES
    cross: DataFrame = pd.crosstab(index=messages_mod.sender, columns=messages_mod.to)
    G: Graph = nx.random_geometric_graph(len(agents), 0)
    G = nx.relabel_nodes(G, {i: a for i, a in enumerate(agents)})
    tuples: DataFrame = cross.stack().reset_index()
    tuples = tuples[tuples[0] > 0]
    tuples = list(tuples.itertuples(index=False))
    G.add_weighted_edges_from(tuples)

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]["pos"]
        node_x.append(x)
        node_y.append(y)

    np.random.seed(2)

    x_dictionary: Dict[str : npt.NDArray[np.float64]] = {
        x: node_x[i] for i, x in enumerate(agents)
    }
    y_dictionary: Dict[str : npt.NDArray[np.float64]] = {
        x: node_y[i] for i, x in enumerate(agents)
    }
    return x_dictionary, y_dictionary


def create_edges_df(msg_type: str = "SEND-LAYERS") -> DataFrame:
    edges: DataFrame = load.message_dataset()
    edges = edges[["sender", "to", "timestamp", "algorithm_round"]][
        edges.type == msg_type
    ]
    edges[["sender", "domain"]] = edges.sender.str.split("@", expand=True)
    edges[["to", "domain2"]] = edges.to.str.split("@", expand=True)
    edges = edges.drop(["domain", "domain2", "algorithm_round"], axis=1)
    edges["edge"] = list(map(lambda x: tuple(sorted(x)), zip(edges.sender, edges.to)))
    edges["weight"] = edges.groupby(["edge"]).cumcount().add(1)
    edges["timestamp"] = pd.to_datetime(edges.timestamp)
    edges["timestamp"] = edges.timestamp.dt.strftime("%Y/%m/%d %H:%M:%S")
    edges["timestamp"] = pd.to_datetime(edges.timestamp)
    first_date: Timestamp = edges.timestamp.min()
    edges["timestamp"] = edges.timestamp.apply(
        lambda x: round((x - first_date) / pd.Timedelta(seconds=1), 2)
    )
    return msg_type, edges


def create_nodes_df(metric: str = "test_accuracy") -> DataFrame:
    nodes: DataFrame = load.inference_dataset()
    nodes = nodes[["agent", "timestamp", metric]]

    nodes["timestamp"] = pd.to_datetime(nodes.timestamp)
    nodes["timestamp"] = nodes.timestamp.dt.strftime("%Y/%m/%d %H:%M:%S")
    nodes["timestamp"] = pd.to_datetime(nodes.timestamp)
    first_date: Timestamp = nodes.timestamp.min()
    nodes["timestamp"] = nodes.timestamp.apply(
        lambda x: round((x - first_date) / pd.Timedelta(seconds=1), 2)
    )
    return metric, nodes


def create_network_artifacts():
    metric, nodes = create_nodes_df()
    msg_type = edges = create_edges_df()
    timestamps: List[float] = sorted(
        list(set(nodes.timestamp.to_list() + edges.timestamp.to_list()))
    )
    return nodes, edges, timestamps, metric, msg_type


def nodes_panel_data(nodes: DataFrame, timestamps, x_coords, y_coords, metric):
    nodes["agent_id"] = nodes.groupby(nodes.columns.tolist(), sort=False).ngroup() + 1
    mux: DataFrame = pd.MultiIndex.from_product(
        [timestamps, np.array(nodes.agent.unique())],
        names=["timestamp", "agent"],
    )

    nodes = nodes.set_index(["timestamp", "agent"]).reindex(mux).reset_index()
    for agent in nodes.agent.unique():
        nodes[nodes.agent == agent] = nodes[nodes.agent == agent].fillna(method="ffill")
    nodes = nodes.fillna(0)

    def set_value(
        row_number: str, assigned_value: Dict[str, List[float]]
    ) -> List[float]:
        return assigned_value[row_number]

    nodes["X"] = nodes["agent"].apply(set_value, args=(x_coords,))
    nodes["Y"] = nodes["agent"].apply(set_value, args=(y_coords,))
    global range_color
    range_color = [
        float(nodes[metric].min()),
        float(nodes[metric].max()),
    ]
    nodes["size"] = [35 for x in range(len(nodes))]
    return nodes, range_color


def edges_panel_data(edges: DataFrame, timestamps, x_coords, y_coords):
    ## CONVERT TO PANEL DATA
    keep: DataFrame = edges.groupby(["timestamp", "edge"])["timestamp"].idxmax()
    edges: DataFrame = edges.loc[keep]
    mux: DataFrame = pd.MultiIndex.from_product(
        [timestamps, np.array(edges.edge.unique())],
        names=["timestamp", "edge"],
    )
    edges = edges.set_index(["timestamp", "edge"]).reindex(mux).reset_index()
    for agent in edges.edge.unique():
        edges[edges.edge == agent] = edges[edges.edge == agent].fillna(method="ffill")
    edges = edges.fillna(0)

    ## LAST EDGE TRANSFORMATIONS
    def set_value(
        row_number: str, assigned_value: Dict[str, List[float]]
    ) -> List[float]:
        return assigned_value[row_number]

    edges["sender"] = list(map(lambda x: x[0], edges.edge))
    edges["to"] = list(map(lambda x: x[1], edges.edge))
    edges["X_sender"] = edges["sender"].apply(set_value, args=(x_coords,))
    edges["Y_sender"] = edges["sender"].apply(set_value, args=(y_coords,))
    edges["X_to"] = edges["to"].apply(set_value, args=(x_coords,))
    edges["Y_to"] = edges["to"].apply(set_value, args=(y_coords,))
    edges["X"] = list(
        map(lambda x: [x[0], x[1]], list(zip(edges.X_sender, edges.X_to)))
    )
    edges["Y"] = list(
        map(lambda x: [x[0], x[1]], list(zip(edges.Y_sender, edges.Y_to)))
    )
    edges = edges.drop(["sender", "to", "X_sender", "Y_sender", "X_to", "Y_to"], axis=1)
    return edges
