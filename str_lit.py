import streamlit as st
import plotly.express as px
import pandas as pd
from pathlib import Path
from pydantic import BaseModel
import numpy as np

FOLDER = Path("/home/slozgom/personal/tfg_project/xperiments/experimentos_con_cnn")

experiment_variables = ["agent", "algorithm", "n_agents", "type", "network"]
agent_variables = [
    "minimum_loss_achieved",
    "maximum_accuracy_achieved",
    "maximum_recall_achieved",
    "maximum_precision_achieved",
    "maximum_f1_achieved",
    "mean_seconds_by_round",
]
network_variables = [
    "minimum_loss_net",
    "maximum_acc_net",
    "maximum_recall_net",
    "maximum_precision_net",
    "maximum_f1_net",
]
df = pd.DataFrame(columns=experiment_variables + agent_variables)


def max_network(data):
    df_numeric = pd.DataFrame(columns=network_variables)
    rounds = data.algorithm_round.unique()
    for r in rounds:
        round_data = data[(data.algorithm_round == r)]
        df_numeric.loc[len(df_numeric),] = list(
            map(
                lambda x: sum(x) / len(x),
                [
                    data.test_loss,
                    data.test_accuracy,
                    data.test_recall,
                    data.test_precision,
                    data.test_f1_score,
                ],
            )
        )
    return [min(df_numeric.best_loss_net)].extend(
        list(
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
    )


# Modificar esta lista
experiments_list = [x for x in list(FOLDER.iterdir()) if "10" in x.name]
for PATH in experiments_list:
    dataset = pd.read_csv(PATH.joinpath("raw/nn_inference.csv"))
    agents = dataset.agent.unique()
    dataset = dataset[(dataset.algorithm_round <= 100)]
    descriptive = PATH.name.split("_")
    times = pd.read_csv(PATH.joinpath("raw/algorithm.csv"))
    times = times[(times.algorithm_round <= 100)]
    for ag in agents:
        agent_times = times[(times.agent == ag + "@localhost")]
        time = agent_times.seconds_to_complete.mean()

        data = dataset[(dataset.agent == ag)]
        numeric = list(
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
        numerics = [data.test_loss.min()] + numeric + [time]
        row = [ag] + ["acol"] + descriptive + numerics
        df.loc[len(df)] = row


class Variable(BaseModel):
    variable: str
    legend: str
    description: str


def get_var(path_to_csv: str, var: str) -> Variable:
    df2 = pd.read_csv(path_to_csv)
    row = df2[(df2.variable == var)].values[0]
    return Variable(variable=row[0], legend=row[1], description=row[2])


st.title("Tabla descriptiva")

st.subheader("Selecciona columnas")
options_exp = st.multiselect(
    "Categoric variables",
    experiment_variables,
)

options_num = st.multiselect(
    "Numeric variables",
    agent_variables,
)

if options_num:
    st.subheader("Ajusta los rangos para las variables numéricas")
    my_round = lambda x: round(x, 3)
    dic_ranges = {}
    for var in options_num:
        col = df[var]
        MIN = col.min()
        MAX = col.max()
        r = MAX - MIN
        # ranges = np.arange(MIN, MAX, (MAX - MIN) / 30).tolist()
        ranges = st.slider(var, MIN, MAX, (MIN, MAX))
        dic_ranges[var] = ranges

# options = [
#    get_var(path_to_csv="variables.csv", var=i).variable for i in agent_variables
# ]

st.subheader("Agrupa los datos")

col1, col2 = st.columns(2)
with col1:
    selection = st.segmented_control(
        "Variables", experiment_variables, selection_mode="multi"
    )

with col2:
    operations = st.radio(
        "Operations",
        ["Sum", "Mean", "Median", "Max", "Min"],
        index=None,
    )


def add_aggregations(df, selection, opearation):
    return df.groupby(selection).agg()


if not options_num and not options_exp:
    st.dataframe(
        df,
        on_select="rerun",
    )
else:
    if options_exp:
        if options_num:
            df2 = df[options_exp + options_num]
        else:
            df2 = df[options_exp + agent_variables]
    elif options_num:
        df2 = df[experiment_variables + options_num]

    for var in options_num:
        min_value, max_value = dic_ranges[var]
        df2 = df2.query(f"{var} >= {min_value} and {var} <= {max_value}")
    st.dataframe(
        df2,
        on_select="rerun",
    )
