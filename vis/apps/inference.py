import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def train_by_agent(train):
    metrics = ["accuracy", "loss", "precision", "recall", "f1_score"]
    figs = []
    for metric in metrics:
        figs.append(px.line(train, x="algorithm_round", y=metric, color="agent", title=f"Train {metric} evolution across rounds by agent"))
    return figs

def test_by_agent(test):
    metrics = ["test_accuracy", "test_loss", "test_precision", "test_recall", "test_f1_score"]
    figs = []
    for metric in metrics:
        figs.append(px.line(test, x="algorithm_round", y=metric, color="agent", title=f"{metric} evolution across rounds by agent"))
    return figs

def train_test_network(train, test):
    metrics = metrics = ["accuracy", "loss", "precision", "recall", "f1_score"]
    figs = []
    for metric in metrics:
        
        g_train = train.groupby(["algorithm_round"]).agg(maximum_accuracy=(metric, 'max'),
                                                        minimum_accuracy=(metric, 'min'),
                                                        mean_accuracy=(metric, 'mean')).reset_index()
        g_test = test.groupby(["algorithm_round"]).agg(maximum_accuracy=(f'test_{metric}', 'max'),
                                                        minimum_accuracy=(f'test_{metric}', 'min'),
                                                        mean_accuracy=(f'test_{metric}', 'mean')).reset_index()

        x = list(g_test.algorithm_round.unique())
        x_rev = x[::-1]

        nmax = g_train.maximum_accuracy.to_list()
        nmin = g_train.minimum_accuracy.to_list()
        nmin = nmin[::-1]

        tmax = g_test.maximum_accuracy.to_list()
        tmin = g_test.minimum_accuracy.to_list()
        tmin = tmin[::-1]


        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x+x_rev,
            y=nmax+nmin,
            fill='toself',
            fillcolor='rgba(0,176,246,0.2)',
            line_color='rgba(255,255,255,0)',
            name='Train',
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=x+x_rev,
            y=tmax+tmin,
            fill='toself',
            fillcolor='rgba(231,107,243,0.2)',
            line_color='rgba(255,255,255,0)',
            showlegend=False,
            name='Test',
        ))
        fig.add_trace(go.Scatter(
            x=x, y=g_train.mean_accuracy,
            line_color='rgb(0,176,246)',
            name='Train',
        ))
        fig.add_trace(go.Scatter(
            x=x, y=g_test.mean_accuracy,
            line_color='rgb(231,107,243)',
            name='Test',
        ))

        fig.update_traces(mode='lines')
        fig.update_layout(title_text=f"Netowk's performance {metric} by round")
        figs.append(fig)
    return figs


def generate(config, download=False):
    train = pd.read_csv(config.experiment_path / r"nn_train.csv")
    test = pd.read_csv(config.experiment_path / r"nn_inference.csv")
    f1 = train_by_agent(train)
    f2 = test_by_agent(test)
    f3 = train_test_network(train, test)
    figs = f1 + f2 + f3
    if download:
        root = "images"
        folder = f"{root}/{__name__.split('.')[0]}"
        isExist = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)
        for i,F in enumerate(figs):
            F.write_image(f"{folder}/{F.layout.title.text.replace(' ', '_')}.svg")
    else:
        updated_figs = []
        for F in figs:
            html_fig = F.to_html(full_html=False)
            html_fig.replace("PNG", "SVG", 1)
            html_fig.replace("png", "svg", 3)
            updated_figs.append(html_fig)
        return updated_figs