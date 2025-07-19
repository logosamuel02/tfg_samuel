import pandas as pd
import plotly.graph_objects as go

# Load and prepare the data
df = pd.read_csv("nn_train.csv")
df = df[df.epoch == 1].copy()

agents = sorted(df["agent"].unique())
fig = go.Figure()

# Pre-calculate both original and transformed Y-values
original_y_values = []
transformed_y_values = []
all_x_values = []

for agent in agents:
    df_agent = df[df["agent"] == agent].sort_values("algorithm_round")
    all_x_values.append(df_agent["algorithm_round"])
    original_y_values.append(df_agent["accuracy"])
    transformed_y_values.append(
        df_agent["accuracy"] * 2 + 1 + df_agent["loss"]
    )  # whatever...


# Add initial traces to the figure using original data
for i, agent in enumerate(agents):
    fig.add_trace(
        go.Scatter(
            x=all_x_values[i],
            y=original_y_values[i],
            mode="lines+markers",
            name=agent,
        )
    )

# Define the dropdown buttons for agent selection
dropdown_buttons = []
dropdown_buttons.append(
    dict(
        label="All Agents",
        method="update",
        args=[
            {"visible": [True] * len(agents)},
            {"title": {"text": "Accuracy for All Agents"}},
        ],
    )
)

for i, agent in enumerate(agents):
    visibility = [False] * len(agents)
    visibility[i] = True
    dropdown_buttons.append(
        dict(
            label=agent,
            method="update",
            args=[
                {"visible": visibility},
                {"title": {"text": f"Accuracy for Agent: {agent}"}},
            ],
        )
    )

# Define the transformation buttons
transform_buttons = list(
    [
        dict(
            label="Original Accuracy",
            method="update",
            args=[
                {"y": original_y_values},
                {"yaxis.title.text": "Accuracy"},
            ],
        ),
        dict(
            label="Transformed",
            method="update",
            args=[
                {"y": transformed_y_values},
                {"yaxis.title.text": "Transformed"},
            ],
        ),
    ]
)

# Update layout with BOTH controls inside the 'updatemenus' list
fig.update_layout(
    title_text="Accuracy for All Agents",
    xaxis_title="Algorithm Round",
    yaxis_title="Accuracy",
    legend_title="Agents",
    updatemenus=[
        # Control 1: Dropdown for agent selection
        dict(
            type="dropdown",
            active=0,
            buttons=dropdown_buttons,
            direction="down",
            x=0.01,
            xanchor="left",
            y=1.2,
            yanchor="top",
        ),
        # Control 2: Buttons for data transformation
        dict(
            type="buttons",
            active=0,
            buttons=transform_buttons,
            direction="right",
            x=0.25,
            xanchor="left",
            y=1.2,
            yanchor="top",
        ),
    ],
)

# Save the interactive plot to an HTML file
fig.write_html("index.html")
