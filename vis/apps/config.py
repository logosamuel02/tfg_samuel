import os
import json
import yaml
from plotly.graph_objects import Figure
import plotly.io as pio
from pydantic import DirectoryPath
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
import PIL
import io

load_dotenv()


def clean(var: str) -> str:
    var: str = " ".join(var.split("_"))
    var = var.replace(".", " ")
    return var.capitalize()


def load_figure(filename: str):
    return pio.read_json(filename)


def load_module_figures(module: str):
    root_fig: str = "figures"
    figures_folder: str = rf"{root_fig}/{module.split('.')[0]}"
    isExistFig: bool = os.path.exists(figures_folder)
    if not isExistFig:
        os.makedirs(figures_folder)
    return [
        load_figure(rf"{x}")
        for x in Path(figures_folder).iterdir()
        if x.suffix == ".json"
    ]


def get_module_figures(module: str):
    figs: list[Figure] = load_module_figures(module)
    html_tuples = []
    for fig in figs:
        title: str = fig.layout.title.text
        html_fig: str = fig.to_html(full_html=False)
        html_fig = html_fig.replace("PNG", "SVG", 1)
        html_fig = html_fig.replace("png", "svg", 3)
        html_tuples.append([title, html_fig])
    return html_tuples


def download_figures(
    download: str, module: str, root_img: str = "images"
) -> List[str] | None:
    figs = load_module_figures(module)
    img_folder: str = rf"{root_img}/{module.split('.')[0]}"
    isExistImg: bool = os.path.exists(img_folder)
    if not isExistImg:
        os.makedirs(img_folder)

    if download == "svg":
        for i, fig in enumerate(figs):
            if len(fig.frames) > 0:
                frame = fig.frames[-1]
                fig.update(data=frame.data)
                fig.layout.sliders[0].update(active=len(fig.frames) - 1)
            fig.write_image(
                rf"{img_folder}/{fig.layout.title.text.replace(' ', '_')}.svg"
            )

    elif download == "gif":
        for F in figs:
            if len(F.frames) > 0:
                filename = rf"{img_folder}/{F.layout.title.text.replace(' ', '_')}.gif"
                frames = []
                frame_duration: int = config.gif["frame_duration"]
                n_frames_by_plot: int = config.gif["n_of_frames"]
                cut: int = len(F.frames) // n_frames_by_plot

                selected = [
                    [i, frame] for i, frame in enumerate(F.frames) if i % cut == 0
                ]
                pbar = tqdm(total=len(selected))
                for slider_pos, frame in selected:
                    F.update(data=frame.data)
                    F.layout.sliders[0].update(active=slider_pos)
                    frames.append(PIL.Image.open(io.BytesIO(F.to_image(format="png"))))
                    pbar.update(1)
                pbar.close()
                # Create the gif file.
                frames[0].save(
                    filename,
                    save_all=True,
                    append_images=frames[1:],
                    optimize=True,
                    duration=frame_duration,
                    loop=0,
                )


@dataclass
class Config:
    source_path: DirectoryPath = (
        r"C:/Users/samue/OneDrive/Escritorio/Tareas UNI/tfg/tfg_samuel/vis/xperiments"
    )
    experiment_path: DirectoryPath = field(init=False)
    output_path: str = "images"
    fig_buttons: Dict = field(init=False)
    plots: str = "config.yaml"
    variables: str = "variables.yaml"
    gif: Dict = field(init=False)

    def __post_init__(self):
        self.source_path: Path = Path(rf"{self.source_path}")
        self.experiment_path: Path = (
            self.source_path / r"experimentos_con_cnn/05_non_complete/raw"
        )
        self.fig_buttons: Dict = dict(
            dragmode="drawopenpath",
            newshape_line_color="cyan",
            modebar_add=[
                "drawline",
                "drawopenpath",
                "drawclosedpath",
                "drawcircle",
                "drawrect",
                "eraseshape",
            ],
        )
        with open(rf"{self.plots}", "r") as file:
            self.plots = yaml.load(file, Loader=yaml.SafeLoader)

        with open(rf"{self.variables}", "r") as f:
            self.variables = yaml.load(f, Loader=yaml.SafeLoader)

        self.gif = {"n_of_frames": 3, "frame_duration": 2}
