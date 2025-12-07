import pandas as pd
from pathlib import Path
from export import Config

paths = [
    r"C:/Users/samue/OneDrive/Escritorio/Tareas UNI/tfg/tfg_samuel/vis/xperiments/experimentos_con_cnn/01_non_complete/raw",
    r"C:/Users/samue/OneDrive/Escritorio/Tareas UNI/tfg/tfg_samuel/vis/xperiments/experimentos_con_cnn/02_non_complete/raw",
    r"C:/Users/samue/OneDrive/Escritorio/Tareas UNI/tfg/tfg_samuel/vis/xperiments/experimentos_con_cnn/05_non_complete/raw",
    r"C:/Users/samue/OneDrive/Escritorio/Tareas UNI/tfg/tfg_samuel/vis/xperiments/experimentos_con_cnn/10_non_complete/raw",
]

config = Config(
    source_path=r"C:/Users/samue/OneDrive/Escritorio/Tareas UNI/tfg/tfg_samuel/vis/xperiments/experimentos_con_cnn",
    experiment_path=paths[2],
)


def algorithm_dataset():
    return pd.read_csv(Path(config.experiment_path / r"algorithm.csv"))


def general_dataset():
    return pd.read_csv(Path(config.experiment_path / r"general.log"))


def data_split_dataset():
    return pd.read_csv(Path(config.experiment_path / r"data_split.csv"))


def message_dataset():
    return pd.read_csv(Path(config.experiment_path / r"message.csv"))


def convergence_dataset():
    return pd.read_csv(Path(config.experiment_path / r"nn_convergence.csv"))


def inference_dataset():
    return pd.read_csv(Path(config.experiment_path / r"nn_inference.csv"))


def train_dataset():
    return pd.read_csv(Path(config.experiment_path / r"nn_train.csv"))
