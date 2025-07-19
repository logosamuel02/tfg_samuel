import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Scatter(y=[4, 2, 1], mode="lines"), row=1, col=1)
fig.add_trace(go.Bar(y=[2, 1, 3]), row=1, col=2)

fig = go.Figure(
    data=[
        go.Scatter(
            x=df_initial["algorithm_round"],
            y=df_initial["accuracy"],
            mode="lines+markers",
            name=initial_agent,  # legend name: not necessary here
        )
    ]
)

# Create the dropdown menu
buttons = []

# Add a button for each individual agent to the dropdown
for agent in agents:
    df_agent = df[df["agent"] == agent]
    buttons.append(
        dict(
            label=agent,
            method="update",
            args=[
                # Update data for the first (and only) trace
                {
                    "y": [df_agent["accuracy"]],
                    "x": [df_agent["algorithm_round"]],
                    "name": [agent],
                },
                # Update the layout title
                {"title": {"text": f"Accuracy for Agent: {agent}"}},
            ],
        )
    )

# Update the figure layout with the dropdown menu
fig.update_layout(
    updatemenus=[
        dict(active=0, buttons=buttons, x=0.01, xanchor="left", y=1.15, yanchor="top")
    ],
    title_text=f"Accuracy for Agent: {initial_agent}",
    xaxis_title="Algorithm Round",
    yaxis_title="Accuracy",
)

fig.write_html("index.html")
