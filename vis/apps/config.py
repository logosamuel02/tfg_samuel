import os
import json
from plotly.graph_objects import Figure
from pydantic import BaseModel, DirectoryPath
from dotenv import load_dotenv
from typing import List
from pathlib import Path

load_dotenv()


def clean(var: str) -> str:
    var = " ".join(var.split("_"))
    var = var.replace(".", " ")
    return var.capitalize()


def save_or_print_figures(download: bool, figs: List[Figure]) -> List[str] | None:
    if download:
        root = "images"
        folder = f"{root}/{__name__.split('.')[0]}"
        isExist = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)
        for i, F in enumerate(figs):
            F.write_image(f"{folder}/{F.layout.title.text.replace(' ', '_')}.svg")
    else:
        updated_figs = []
        for F in figs:
            title = F.layout.title.text
            html_fig = F.to_html(full_html=False)
            html_fig = html_fig.replace("PNG", "SVG", 1)
            html_fig = html_fig.replace("png", "svg", 3)
            updated_figs.append([title, html_fig])
        return updated_figs


class Config(BaseModel):
    source_path: DirectoryPath = (
        r"C:/Users/samue/OneDrive/Escritorio/Tareas UNI/tfg/tfg_samuel/vis/xperiments"
    )
    experiment_path: None = None
    fig_config: None = None
    fig_buttons: None = None
    plots: None = None
    variables: None = None

    def model_post_init(self, __context):
        self.source_path = Path(rf"{self.source_path}")
        self.experiment_path = (
            self.source_path / r"experimentos_con_cnn/05_non_complete/raw"
        )
        self.fig_buttons = dict(
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
            self.plots = json.load(file)

        with open(r"variables.json") as file:
            self.variables = json.load(file)


config = Config()
print(config.model_dump())
