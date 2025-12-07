import random

values = []
init = [0.2, 0.4, 0.6, 0.8]
back = [0.2, 0.4, 0.6, 0.8]
values.append(init)
e = 0.1

for i in range(1, 16):
    new_init = []
    for i, value in enumerate(init):
        s = 0
        for j, x in enumerate(init):
            if j != i:
                s += x - value
        new_init.append((init[i] + e * s))

    values.append(new_init)
    init = new_init

import plotly.express as px

fig = px.line(values)

fig = px.line(values)
fig.update_layout(font_size=20)
fig.update_layout(
    xaxis_title="# iter",
    xaxis_title_font_size=25,
    yaxis_title="Xi",
    yaxis_title_font_size=25,
    title_text="Syncronous consensus",
    title_font_size=25,
    legend_title_text="Agents",
    legend_font_size=20,
)
fig.write_image("sync_consesnus.svg")
