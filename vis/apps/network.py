import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def create_coordinates(messages):
    # CONVERT DATA
    messages_mod = messages.copy()
    messages_mod = messages_mod[["sender", "to"]][(messages_mod.algorithm_round <= 100)]
    messages_mod.sender = messages_mod.sender.apply(lambda x: x.split("@")[0])
    messages_mod.to = messages_mod.to.apply(lambda x: x.split("@")[0])

    agents = sorted(messages_mod.sender.unique())

    # GENERATE COORDINATES
    cross = pd.crosstab(index=messages_mod.sender, columns=messages_mod.to)
    G = nx.random_geometric_graph(len(agents), 0)
    G = nx.relabel_nodes(G, {i: a for i, a in enumerate(agents)})
    tuples = cross.stack().reset_index()
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

    x_dictionary = {x: node_x[i] for i, x in enumerate(agents)}
    y_dictionary = {x: node_y[i] for i, x in enumerate(agents)}
    return x_dictionary, y_dictionary

def create_nodes_plot(inference, x_coords, y_coords):
    nodes = inference.copy()
    nodes = nodes[["agent", "timestamp", "test_accuracy"]]

    nodes["timestamp"] = pd.to_datetime(nodes.timestamp)
    nodes["timestamp"] = nodes.timestamp.dt.strftime("%Y/%m/%d %H:%M:%S")
    nodes["timestamp"] = pd.to_datetime(nodes.timestamp)
    first_date = nodes.timestamp.min()
    nodes["timestamp"] = nodes.timestamp.apply(
        lambda x: round((x - first_date) / pd.Timedelta(seconds=1), 2)
    )
    ## CONVERT TO PANEL DATA
    nodes["agent_id"] = nodes.groupby(nodes.columns.tolist(), sort=False).ngroup() + 1
    mux = pd.MultiIndex.from_product(
        [nodes["timestamp"].unique(), np.array(nodes.agent.unique())],
        names=["timestamp", "agent"],
    )

    nodes = nodes.set_index(["timestamp", "agent"]).reindex(mux).reset_index()
    for agent in nodes.agent.unique():
        nodes[nodes.agent == agent] = nodes[nodes.agent == agent].fillna(method="ffill")
    nodes = nodes.fillna(0)
    
    def set_value(row_number, assigned_value):
        return assigned_value[row_number]
    
    nodes["X"] = nodes["agent"].apply(set_value, args=(x_coords,))
    nodes["Y"] = nodes["agent"].apply(set_value, args=(y_coords,))
    global range_color
    range_color = [float(nodes.test_accuracy.min()), float(nodes.test_accuracy.max())]
    nodes["size"] = [35 for x in range(len(nodes))]
    
    # CREATE NODES PLOT
    
    nodes_plot = px.scatter(
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
    nodes_plot.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 50
    xmax, xmin = max(x_coords.values()), min(x_coords.values())
    ymax, ymin = max(y_coords.values()), min(y_coords.values())
    i = 0.07
    nodes_plot.update_traces(textposition="top center", textfont_size=17, textfont_family="Arial Black")
    nodes_plot.layout.template = 'plotly_white'
    nodes_plot.update_xaxes(title=None, range=[xmin-i, xmax+i], showgrid=False, zeroline=False, showticklabels=False)
    nodes_plot.update_yaxes(title=None, range=[ymin-i, ymax+i], showgrid=False, zeroline=False, showticklabels=False)
    nodes_plot.for_each_trace(lambda trace: trace.update(marker_size=35))
    nodes_plot.update_layout(
        title_text="Test accuracy evolution inside agents network",
        width=800,
        height=800,
    )
    nodes_plot.update_layout(coloraxis_colorbar_x=-0.15, coloraxis_colorbar_title="Test accuracy")
    return nodes_plot
    
def create_edges_plot(messages, x_coords, y_coords):
    # EDGES DF
    edges = messages.copy()
    edges = edges[["sender", "to", "timestamp", "algorithm_round"]][
        edges.type == "SEND-LAYERS"
    ]
    edges[["sender", "domain"]] = edges.sender.str.split("@", expand=True)
    edges[["to", "domain2"]] = edges.to.str.split("@", expand=True)
    edges = edges.drop(["domain", "domain2", "algorithm_round"], axis=1)
    edges["edge"] = list(map(lambda x: tuple(sorted(x)), zip(edges.sender, edges.to)))
    edges["weight"] = edges.groupby(["edge"]).cumcount().add(1)
    edges["timestamp"] = pd.to_datetime(edges.timestamp)
    edges["timestamp"] = edges.timestamp.dt.strftime("%Y/%m/%d %H:%M:%S")
    edges["timestamp"] = pd.to_datetime(edges.timestamp)
    first_date = edges.timestamp.min()
    edges["timestamp"] = edges.timestamp.apply(
        lambda x: round((x - first_date) / pd.Timedelta(seconds=1), 2)
    )

    ## CONVERT TO PANEL DATA
    keep = edges.groupby(["timestamp", "edge"])["timestamp"].idxmax()
    edges = edges.loc[keep]
    mux = pd.MultiIndex.from_product(
        [edges["timestamp"].unique(), np.array(edges.edge.unique())],
        names=["timestamp", "edge"],
    )
    edges = edges.set_index(["timestamp", "edge"]).reindex(mux).reset_index()
    for agent in edges.edge.unique():
        edges[edges.edge == agent] = edges[edges.edge == agent].fillna(method="ffill")
    edges = edges.fillna(0)
    
    

    ## LAST EDGE TRANSFORMATIONS
    def set_value(row_number, assigned_value):
        return assigned_value[row_number]
    
    edges["sender"] = list(map(lambda x: x[0], edges.edge))
    edges["to"] = list(map(lambda x: x[1], edges.edge))
    edges["X_sender"] = edges["sender"].apply(set_value, args=(x_coords,))
    edges["Y_sender"] = edges["sender"].apply(set_value, args=(y_coords,))
    edges["X_to"] = edges["to"].apply(set_value, args=(x_coords,))
    edges["Y_to"] = edges["to"].apply(set_value, args=(y_coords,))
    edges["X"] = list(map(lambda x: [x[0], x[1]], list(zip(edges.X_sender, edges.X_to))))
    edges["Y"] = list(map(lambda x: [x[0], x[1]], list(zip(edges.Y_sender, edges.Y_to))))
    edges = edges.drop(["sender", "to", "X_sender", "Y_sender", "X_to", "Y_to"], axis=1)
    
    # CREATE EDGES PLOT
    def new_value(value):
        OldMin = edges.weight.min()
        OldMax = edges.weight.max()
        OldRange = OldMax - OldMin
        NewMin = 0
        NewMax = 20
        NewRange = 20 - 0
        return round((((value - OldMin) * NewRange) / OldRange) + NewMin)


    edges_plot = go.Figure()
    links = edges.edge.unique()
    times = edges.timestamp.unique()
    # TRACES
    for link in links:
        x = edges[(edges.edge == link)].head(1).X.values[0]
        y = edges[(edges.edge == link)].head(1).Y.values[0]
        edges_plot.add_scatter(x=x, y=y, name=str(link), mode="lines", line_width=1)

    # SLIDER
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Second:",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 25, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }

    # FRAMES
    frames = []
    for time in times:
        data = []
        for link in links:
            row = edges[(edges.edge == link) & (edges.timestamp == time)]
            data.append(
                go.Scatter(
                    x=row.X.values[0],
                    y=row.Y.values[0],
                    name=f"{str(link)} - {row.weight.values[0]}",
                    mode="lines",
                    line_width=new_value(row.weight.values[0]),
                )
            )
        frames.append(go.Frame(data=data, traces=list(range(len(links))), name=str(time)))
        slider_step = {
            "args": [
                [str(time)],
                {
                    "mode": "immediate",
                    "transition": {"duration": 25},
                },
            ],
            "label": str(time),
            "method": "animate",
        }
        sliders_dict["steps"].append(slider_step)


    edges_plot.update(frames=frames)
    updatemenus = [
        dict(
            buttons=[
                dict(
                    args=[
                        None,
                        {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True},
                    ],
                    label="Play",
                    method="animate",
                ),
                dict(
                    args=[
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    label="Pause",
                    method="animate",
                ),
            ],
            direction="left",
            pad={"r": 10, "t": 87},
            showactive=False,
            type="buttons",
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top",
        )
    ]

    edges_plot.update_layout(
        width=800,
        height=800,
        updatemenus=updatemenus,
        sliders=[sliders_dict],
    )
    xmax, xmin = max(x_coords.values()), min(x_coords.values())
    ymax, ymin = max(y_coords.values()), min(y_coords.values())
    i = 0.07
    edges_plot.update_xaxes(range=[xmin-i, xmax+i])
    edges_plot.update_yaxes(range=[ymin-i, ymax+i])
    edges_plot.update_traces(textfont_size=20)

    edges_plot.layout.template = 'plotly_white'
    edges_plot.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    edges_plot.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    edges_plot.update_layout(legend_title="Number of messages")
    return edges_plot

def create_combined_plot(nodes_plot, edges_plot, x_coords, y_coords):
    # Stationary combined plot
    combined_plot = go.Figure(
        data=edges_plot.data + nodes_plot.data,
        frames=[
            go.Frame(data=edges_plot.data + nodes_plot.data, name=nodes_plot.name)
            for edges_plot, nodes_plot in zip(edges_plot.frames, nodes_plot.frames)
        ],
        layout=edges_plot.layout,
    )
    combined_plot.for_each_trace(lambda trace: trace.update(
        marker=dict(size=35, opacity=1, line=dict(color="DarkSlateGrey")),
        )
    )
    combined_plot.update_layout(title_text="Network evolution")
    combined_plot.update_layout(coloraxis=dict(colorbar=dict(x=-0.15, title="Test accuracy"), cmin=range_color[0], cmax=range_color[1]))
    xmax, xmin = max(x_coords.values()), min(x_coords.values())
    ymax, ymin = max(y_coords.values()), min(y_coords.values())
    i = 0.12
    combined_plot.update_xaxes(range=[xmin-i, xmax+i])
    combined_plot.update_yaxes(range=[ymin-i, ymax+i])
    return combined_plot


def generate(config, download=False):
    messages = pd.read_csv(config.experiment_path / r"message.csv")
    inference = pd.read_csv(config.experiment_path / r"nn_inference.csv")
    
    x_coords, y_coords = create_coordinates(messages)
    nodes_plot = create_nodes_plot(inference, x_coords, y_coords)
    edges_plot = create_edges_plot(messages, x_coords, y_coords)
    combined_plot = create_combined_plot(nodes_plot, edges_plot, x_coords, y_coords)
    
    figs = [combined_plot, nodes_plot, edges_plot]
    if download:
        root = "images"
        folder = f"{root}/{__name__.split('.')[0]}"
        isExist = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)
        for i,F in enumerate(figs):
            F.write_image(f"{folder}/{F.layout.title.text.replace(' ', '_')}.svg")
    else:
        updated_figs = []
        for F in figs:
            html_fig = F.to_html(full_html=False)
            html_fig.replace("PNG", "SVG", 1)
            html_fig.replace("png", "svg", 3)
            updated_figs.append(html_fig)
        return updated_figs

