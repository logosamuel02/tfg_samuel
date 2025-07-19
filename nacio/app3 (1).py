import plotly.graph_objects as go
import pandas as pd
import numpy as np

df = pd.read_csv("nn_train.csv")
df_agg = df.groupby("algorithm_round")["accuracy"].mean().reset_index()


# Define transformation functions
def log_transform():
    """Calculates the natural logarithm. Adds a small constant to avoid log(0)."""
    return np.log(df_agg["accuracy"] + 1e-6)


def cumulative_sum(data):
    """Calculates the cumulative sum of the data."""
    return data.cumsum()


def moving_average(data, window=5):
    """Calculates the simple moving average to smooth the data."""
    return data.rolling(window=window).mean()


# Create the figure and add the initial trace
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_agg["algorithm_round"],
        y=df_agg["accuracy"],
        mode="lines+markers",
        name="Original Accuracy",
    )
)

# Define the transformation buttons
# Each button's 'args' updates the 'y' data of the trace and the y-axis title.
# 1 button for the original data and 3 for the transformations.
transform_buttons = list(
    [
        dict(
            label="Original",
            method="update",
            args=[
                {"y": [df_agg["accuracy"]]},
                {"yaxis.title.text": "Accuracy"},
            ],
        ),
        dict(
            label="Logarithmic",
            method="update",
            args=[
                {"y": [log_transform()]},
                {"yaxis.title.text": "Log(Accuracy)"},
            ],
        ),
        dict(
            label="Cumulative Sum",
            method="update",
            args=[
                {"y": [cumulative_sum(df_agg["accuracy"])]},
                {"yaxis.title.text": "Cumulative Sum of Accuracy"},
            ],
        ),
        dict(
            label="Moving Average",
            method="update",
            args=[
                {"y": [moving_average(df_agg["accuracy"])]},
                {"yaxis.title.text": "Smoothed Accuracy (5-Round Average)"},
            ],
        ),
    ]
)

# Update the layout with the new buttons
fig.update_layout(
    title_text="Interactive Data Transformations",  # Initial title
    xaxis_title="Algorithm Round",  # Initial x-axis title
    yaxis_title="Accuracy",  # Initial y-axis title
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            active=0,  # The 'Original' button is selected by default
            buttons=transform_buttons,
            x=0.5,
            xanchor="center",
            y=1.15,
            yanchor="top",
        ),
    ],
)

fig.write_html("index.html")
