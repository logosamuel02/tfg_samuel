import os
import json
from plotly.graph_objects import Figure
from pydantic import BaseModel, DirectoryPath
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


def save_or_print_figures(
    download: bool, figs: List[Figure], module: str
) -> List[str] | None:
    root: str = "images"
    folder: str = f"{root}/{module.split('.')[0]}"
    isExist: bool = os.path.exists(folder)
    if not isExist:
        os.makedirs(folder)
    if download == "svg":
        for i, F in enumerate(figs):
            if len(F.frames) > 0:
                frame = F.frames[-1]
                F.update(data=frame.data)
                F.layout.sliders[0].update(active=len(F.frames) - 1)
            F.write_image(rf"{folder}/{F.layout.title.text.replace(' ', '_')}.svg")

    elif download == "gif":
        for F in figs:
            if len(F.frames) > 0:
                filename = rf"{folder}/{F.layout.title.text.replace(' ', '_')}.gif"
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
    else:
        updated_figs: List = []
        for F in figs:
            title: str = F.layout.title.text
            html_fig: str = F.to_html(full_html=False)
            html_fig = html_fig.replace("PNG", "SVG", 1)
            html_fig = html_fig.replace("png", "svg", 3)
            updated_figs.append([title, html_fig])
        return updated_figs


class Config(BaseModel):
    source_path: DirectoryPath = (
        r"C:/Users/samue/OneDrive/Escritorio/Tareas UNI/tfg/tfg_samuel/vis/xperiments"
    )
    experiment_path: DirectoryPath = None
    fig_config: Dict = None
    fig_buttons: Dict = None
    plots: Dict = None
    variables: Dict = None
    gif: Dict = None

    def model_post_init(self, __context):
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
        with open(r"plots_config.json") as file:
            self.plots: Dict = json.load(file)

        with open(r"variables.json") as file:
            self.variables: Dict = json.load(file)

        self.gif = {"n_of_frames": 3, "frame_duration": 2}


config = Config()
print(config.model_dump())
