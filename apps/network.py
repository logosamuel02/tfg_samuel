from pathlib import Path
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import networkx as nx


def network_plot():
    FOLDER = Path("/home/slozgom/tfg/tfg_samuel/xperiments/experimentos_con_cnn")
    experiments_list = list(FOLDER.iterdir())
    PATH = experiments_list[8]

    data = pd.read_csv(PATH / "raw/message.csv")
    data.sender = data.sender.apply(lambda x: x.split("@")[0])
    data.to = data.to.apply(lambda x: x.split("@")[0])
    agents = sorted(data.sender.unique())
    data = data[["sender", "to"]][(data.algorithm_round <= 100)]
    cross = pd.crosstab(index=data.sender, columns=data.to)
    G = nx.random_geometric_graph(len(agents), 0)
    G = nx.relabel_nodes(G, {i: a for i, a in enumerate(agents)})
    tuples = cross.stack().reset_index()
    tuples = tuples[tuples[0] > 0]
    tuples = list(tuples.itertuples(index=False))
    G.add_weighted_edges_from(tuples)

    # Edges
    edges = pd.read_csv(
        "/home/slozgom/tfg/tfg_samuel/xperiments/experimentos_con_cnn/05_non_complete/raw/message.csv"
    )
    edges = edges[["sender", "to", "timestamp", "algorithm_round"]][
        edges.type == "SEND-LAYERS"
    ]
    edges

    # Nodes
    nodes = pd.read_csv(
        "/home/slozgom/tfg/tfg_samuel/xperiments/experimentos_con_cnn/05_non_complete/raw/nn_inference.csv"
    )
    nodes = nodes[["agent", "timestamp", "test_accuracy"]]

    nodes.timestamp = pd.to_datetime(nodes.timestamp)
    nodes.timestamp = nodes.timestamp.dt.strftime("%Y/%m/%d %H:%M:%S")
    nodes["agent_id"] = nodes.groupby(nodes.columns.tolist(), sort=False).ngroup() + 1
    mux = pd.MultiIndex.from_product(
        [nodes["timestamp"].unique(), np.array(nodes.agent.unique())],
        names=["timestamp", "agent"],
    )

    nodes = nodes.set_index(["timestamp", "agent"]).reindex(mux).reset_index()
    for agent in nodes.agent.unique():
        nodes[nodes.agent == agent] = nodes[nodes.agent == agent].fillna(method="ffill")
    nodes = nodes.fillna(0)

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]["pos"]
        node_x.append(x)
        node_y.append(y)

    def set_value(row_number, assigned_value):
        return assigned_value[row_number]

    np.random.seed(2)
    # coords = np.random.rand(5, 2)
    x_dictionary = {x: node_x[i] for i, x in enumerate(nodes.agent.unique())}
    y_dictionary = {x: node_y[i] for i, x in enumerate(nodes.agent.unique())}

    # Add a new column named 'Price'
    nodes["X"] = nodes["agent"].apply(set_value, args=(x_dictionary,))
    nodes["Y"] = nodes["agent"].apply(set_value, args=(y_dictionary,))
    range_color = [float(nodes.test_accuracy.min()), float(nodes.test_accuracy.max())]
    nodes["size"] = [10 for x in range(len(nodes))]
    scatter_plot = px.scatter(
        nodes,
        x="X",
        y="Y",
        animation_frame="timestamp",
        text="agent",
        color="test_accuracy",
        size="size",
        hover_name="agent",
        range_color=range_color,
    )
    scatter_plot.update_traces(textposition="top center")
    scatter_plot.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 50

    # EDGE TRACE
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]["pos"]
        x1, y1 = G.nodes[edge[1]]["pos"]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
        showlegend=False,
    )

    scatter_plot.add_trace(edge_trace)
    scatter_plot.update_layout(width=750)
    scatter_plot.update_layout(height=750)
    scatter_plot.update_layout(
        title_text="Test accuracy evolution inside agents network"
    )

    return scatter_plot.to_html(full_html=False)


# fig.write_html("index.html")
# print("Figure CREATED!")
