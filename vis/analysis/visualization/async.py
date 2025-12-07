import random
import numpy as np

e = 0.1
init = [0.2, 0.4, 0.6, 0.8]
opts = np.array([0, 1, 2, 3])
values = []
for i in range(0, 35):
    values.append(init.copy())
    for x in range(2):
        r = random.choice(opts)
        l = [0, 1, 2, 3]
        l.pop(r)
        for j in l:
            init[r] = (1 - e) * init[r] + e * init[j]


import plotly.express as px
import pandas as pd

df = pd.DataFrame(values)
fig = px.line(df)

fig.update_layout(font_size=20)
fig.update_layout(
    xaxis_title="# iter",
    xaxis_title_font_size=25,
    yaxis_title="Xi",
    yaxis_title_font_size=25,
    title_text="Asyncronous consensus",
    title_font_size=25,
    legend_title_text="Agents",
    legend_font_size=20,
)
fig.write_image("async_consesnus.svg")
