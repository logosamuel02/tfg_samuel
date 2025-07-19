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

# Update the figure layout with the dropdown menu
fig.update_layout(
    updatemenus=[
        dict(active=0, buttons=buttons, x=0.01, xanchor="left", y=1.15, yanchor="top")
    ],
    title_text="Accuracy for All Agents",
    xaxis_title="Algorithm Round",
    yaxis_title="Accuracy",
    xaxis_rangeslider_visible=True,
)

fig.write_html("index.html")
