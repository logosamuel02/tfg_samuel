import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import numpy as np
import numpy.typing as npt
import warnings
from pandas.core.frame import DataFrame
from plotly.graph_objects import Figure

from pandas._libs.tslibs.timestamps import Timestamp
from typing import List, Dict
import preprocess as pre
from export import Config, clean

config = Config()

warnings.filterwarnings("ignore")


def create_nodes_plot(x_coords, y_coords) -> Figure:
    nodes, edges, timestamps, metric, msg_type = pre.create_network_artifacts()

    nodes, range_color = pre.nodes_panel_data(
        timestamps, nodes, timestamps, x_coords, y_coords, metric
    )

    nodes_plot: Figure = px.scatter(
        nodes,
        x="X",
        y="Y",
        animation_frame="timestamp",
        labels={"timestamp": "Second"},
        text="agent",
        color=metric,
        size="size",
        title=f"{config.variables[metric]['legend']} evolution inside agents network",
        hover_name="agent",
        range_color=range_color,
    )
    nodes_plot.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = (
        config.plots["network"]["nodes"]["other"]["frame_duration"]
    )
    xmax, xmin = max(x_coords.values()), min(x_coords.values())
    ymax, ymin = max(y_coords.values()), min(y_coords.values())

    nodes_plot.update_traces(config.plots["network"]["nodes"]["traces"])

    i: float = config.plots["network"]["nodes"]["other"]["border"]
    nodes_plot.update_xaxes(range=[xmin - i, xmax + i])
    nodes_plot.update_yaxes(range=[ymin - i, ymax + i])

    nodes_plot.update_xaxes(config.plots["network"]["nodes"]["axes"])
    nodes_plot.update_yaxes(config.plots["network"]["nodes"]["axes"])

    marker_size = config.plots["network"]["nodes"]["other"]["marker_size"]
    nodes_plot.for_each_trace(lambda trace: trace.update(marker_size=marker_size))
    nodes_plot.update_layout(
        coloraxis_colorbar_title=config.variables[metric]["legend"]
    )

    nodes_plot.update_layout(config.plots["network"]["nodes"]["layout"])
    return range_color, nodes_plot


def create_edges_plot(x_coords, y_coords) -> Figure:
    nodes, edges, timestamps, metric, msg_type = pre.create_network_artifacts()
    edges = pre.edges_panel_data(edges, timestamps, x_coords, y_coords)

    def new_value(value: int) -> int:
        OldMin: int = edges.weight.min()
        OldMax: int = edges.weight.max()
        OldRange: int = OldMax - OldMin
        NewMin: int = 0
        NewMax: int = config.plots["network"]["edges"]["other"]["max_size_lines"]
        NewRange: int = NewMax - NewMin
        return round((((value - OldMin) * NewRange) / OldRange) + NewMin)

    edges_plot: Figure = go.Figure()
    links: List[str] = edges.edge.unique()
    times: List[Timestamp] = edges.timestamp.unique()
    # TRACES
    for link in links:
        x: List[float] = edges[(edges.edge == link)].head(1).X.values[0]
        y: List[float] = edges[(edges.edge == link)].head(1).Y.values[0]
        edges_plot.add_scatter(x=x, y=y, name=str(link), mode="lines", line_width=1)

    # SLIDER
    sliders_dict: Dict = {
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
            row: DataFrame = edges[(edges.edge == link) & (edges.timestamp == time)]
            data.append(
                go.Scatter(
                    x=row.X.values[0],
                    y=row.Y.values[0],
                    name=f"{str(link)} - {row.weight.values[0]}",
                    mode="lines",
                    line_width=new_value(row.weight.values[0]),
                )
            )
        frames.append(
            go.Frame(data=data, traces=list(range(len(links))), name=str(time))
        )
        slider_step: Dict[str, List] = {
            "args": [
                [str(time)],
                {
                    "mode": "immediate",
                    "transition": {
                        "duration": config.plots["network"]["edges"]["other"][
                            "transition_duration"
                        ]
                    },
                },
            ],
            "label": str(time),
            "method": "animate",
        }
        sliders_dict["steps"].append(slider_step)

    edges_plot.update(frames=frames)
    updatemenus: List[Dict] = [
        dict(
            buttons=[
                dict(
                    args=[
                        None,
                        {
                            "frame": {
                                "duration": config.plots["network"]["edges"]["other"][
                                    "frame_duration"
                                ],
                                "redraw": True,
                            },
                            "fromcurrent": True,
                        },
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
        updatemenus=updatemenus,
        sliders=[sliders_dict],
        title_text=f"{clean(msg_type)} messages evolution inside agents network",
    )
    xmax, xmin = max(x_coords.values()), min(x_coords.values())
    ymax, ymin = max(y_coords.values()), min(y_coords.values())

    i: float = config.plots["network"]["edges"]["other"]["border"]
    edges_plot.update_xaxes(range=[xmin - i, xmax + i])
    edges_plot.update_yaxes(range=[ymin - i, ymax + i])

    edges_plot.update_xaxes(config.plots["network"]["edges"]["axes"])
    edges_plot.update_yaxes(config.plots["network"]["edges"]["axes"])

    edges_plot.update_traces(config.plots["network"]["edges"]["traces"])
    edges_plot.update_layout(config.plots["network"]["edges"]["layout"])
    return edges_plot


def create_combined_plot(
    nodes_plot: Figure,
    edges_plot: Figure,
    x_coords,
    y_coords,
    range_color,
    metric: str = "test_accuracy",
    msg_type: str = "SEND-LAYERS",
) -> Figure:
    combined_plot: Figure = go.Figure(
        data=edges_plot.data + nodes_plot.data,
        frames=[
            go.Frame(data=edges_plot.data + nodes_plot.data, name=nodes_plot.name)
            for edges_plot, nodes_plot in zip(edges_plot.frames, nodes_plot.frames)
        ],
        layout=edges_plot.layout,
    )
    combined_plot.for_each_trace(
        lambda trace: trace.update(
            marker=config.plots["network"]["network"]["traces"]["marker"],
        )
    )
    combined_plot.update_layout(
        coloraxis=dict(
            colorbar=dict(x=-0.15, title=config.variables[metric]["legend"]),
            cmin=range_color[0],
            cmax=range_color[1],
        ),
        title_text=f"Network evolution {config.variables[metric]['legend']} and {clean(msg_type)} messages",
    )
    xmax, xmin = max(x_coords.values()), min(x_coords.values())
    ymax, ymin = max(y_coords.values()), min(y_coords.values())
    i: float = 0.12
    combined_plot.update_xaxes(range=[xmin - i, xmax + i])
    combined_plot.update_yaxes(range=[ymin - i, ymax + i])

    combined_plot.update_layout(config.plots["network"]["network"]["layout"])
    return combined_plot


def generate(config: Config, action: str = "generate") -> list[str] | None:
    x_coords, y_coords = pre.create_network_coordinates()
    range_color, nodes_plot = create_nodes_plot(x_coords, y_coords)
    edges_plot: Figure = create_edges_plot(x_coords, y_coords)
    combined_plot: Figure = create_combined_plot(
        nodes_plot, edges_plot, x_coords, y_coords, range_color
    )

    figs: List[Figure] = [combined_plot, nodes_plot, edges_plot]

    if action not in ["generate", "download"]:
        return figs

    if action == "generate":
        folder = f"figures/{__name__.split('.')[0]}"
        isExist: bool = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)
        for fig in figs:
            print(1)
            fig.write_json(rf"{folder}/{fig.layout.title.text.replace(' ', '_')}.json")
    else:
        folder = f"{config.output_path}/{__name__.split('.')[0]}"
        isExist: bool = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)
        for fig in figs:
            if len(fig.frames) > 0:
                frame = fig.frames[-1]
                fig.update(data=frame.data)
                fig.layout.sliders[0].update(active=len(fig.frames) - 1)
            fig.write_image(rf"{folder}/{fig.layout.title.text.replace(' ', '_')}.svg")
