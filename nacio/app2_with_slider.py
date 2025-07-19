import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("nn_train.csv")
df = df[df.epoch == 1].copy()

agents = sorted(df["agent"].unique())

fig = go.Figure()

# Add all traces initially
for agent in agents:
    df_agent = df[df["agent"] == agent]
    df_x = df_agent["algorithm_round"]
    df_y = df_agent["accuracy"]
    fig.add_trace(
        go.Scatter(
            x=df_x,
            y=df_y,
            mode="lines+markers",
            name=agent,  # nombre leyenda
            visible=True,  # Start with all traces visible
        )
    )

# Create the dropdown menu
buttons = []

# Button to show all agents
buttons.append(
    dict(
        label="All",
        method="update",
        args=[
            {"visible": [True] * len(agents)},  # all traces visible
            {
                "title": {"text": "Accuracy for All Agents"},
                "showlegend": True,
            },
        ],
    )
)

# Add a button for each agent to the dropdown
for i, agent in enumerate(agents):
    visibility = [False] * len(agents)
    visibility[i] = True
    buttons.append(
        dict(
            label=agent,
            method="update",
            args=[
                {"visible": visibility},
                {"title": {"text": f"Accuracy for Agent: {agent}"}},
            ],
        )
    )

# Create the step slider for rounds
rounds = sorted(df["algorithm_round"].unique())
min_round = rounds[0]

sliders = [
    dict(
        active=len(rounds) - 1,  # Set slider to the last round initially
        currentvalue={"prefix": "Data up to Round: "},
        pad={"t": 50},  # Add padding above the slider
        steps=[],
    )
]

for r in rounds:
    sliders[0]["steps"].append(
        dict(
            method="relayout",
            # Update the x-axis range.
            args=[{"xaxis.range": [min_round - 1, r + 1]}],
            label=str(r),
        )
    )

# Update the figure layout with the dropdown menu
fig.update_layout(
    updatemenus=[
        dict(active=0, buttons=buttons, x=0.01, xanchor="left", y=1.15, yanchor="top")
    ],
    title_text="Accuracy for All Agents",
    xaxis_title="Algorithm Round",
    yaxis_title="Accuracy",
    sliders=sliders,
    xaxis_rangeslider_visible=True,
)

fig.write_html("index.html")
