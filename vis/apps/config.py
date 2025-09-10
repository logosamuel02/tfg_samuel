import os
from pydantic import BaseModel, DirectoryPath
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


class Config(BaseModel):
    source_path: DirectoryPath = (
        r"C:/Users/samue/OneDrive/Escritorio/Tareas UNI/tfg/tfg_samuel/vis/xperiments"
    )
    experiment_path: None = None
    fig_config: None = None
    fig_buttons: None = None

    def model_post_init(self, __context):
        self.source_path = Path(rf"{self.source_path}")
        self.experiment_path = (
            self.source_path / r"experimentos_con_cnn/05_non_complete/raw"
        )
        self.fig_config = {
            "toImageButtonOptions": {
                "format": "svg",  # one of png, svg, jpeg, webp
                "filename": "custom_image",
                # 'height': 500,
                # 'width': 700,
                "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }
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


config = Config()
print(config.model_dump())
