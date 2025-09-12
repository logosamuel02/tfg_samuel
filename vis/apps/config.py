import os
import json
from pydantic import BaseModel, DirectoryPath
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def clean(var: str) -> str:
    var = " ".join(var.split("_"))
    var = var.replace(".", " ")
    return var.capitalize()


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
