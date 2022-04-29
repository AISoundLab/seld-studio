import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output
import pathlib
import os
import plotly.graph_objects as go
from model.models.SELDNet import Seldnet_augmented
import pandas as pd

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

def get_options(names: list):
    options = []
    for n in names:
        gt_filename = f'{pathlib.Path(n).stem}.csv'
        gt_filepath = str(pathlib.Path('assets', 'ground-truth', gt_filename))
        options.append(
            {'label': n, 'value': gt_filepath}
        )
    return options

script_dir = pathlib.Path(__file__).parent.resolve()
audio_filenames = os.listdir(pathlib.Path(script_dir, 'assets', 'audio'))
audio_options = get_options(audio_filenames)

scene = dict(dict(
            xaxis = dict(range=[-2.5,2.5], autorange=False),
            yaxis = dict(range=[-2,2], autorange=False),
            zaxis = dict(range=[-1.5,1.5], autorange=False)))

def get_init_fig():
    # microphone
    new_fig = go.Figure(
        layout=go.Layout(
            scene=scene,
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                            method="animate",
                            args=[None])])]
        ),
        data=[
            go.Scatter3d(
                text='microphone',
                x=[0], 
                y=[0], 
                z=[0],
                mode='markers',
                name='microphone'), 
            go.Scatter3d(
                    visible=False,
                    x=[0], 
                    y=[0], 
                    z=[0],
                    mode='markers',
                    name='ground truth')]
    )

    return new_fig
  

fig = get_init_fig()
gt_df_dict = {} 
graph_init = False
for opt in audio_options:
    gt_df_dict[opt['value']] = pd.read_csv(opt['value'])

switches = html.Div(
    [
        dbc.Label("Toggle a bunch"),
        dbc.Checklist(
            options=[
                {"label": "Ground Truth", "value": 1, 'disabled': True},
                {"label": "Prediction", "value": 2, 'disabled': True},
            ],
            value=[1],
            id="switches-plot",
            switch=True,
        ),
    ]
)

controls = dbc.Card(
    dbc.CardBody(
        [
        html.Div(
            [
                dbc.Label("Audio File"),
                dbc.Select(
                    id="select-audio",
                    options=audio_options
                )
            ]
        ),  
        switches
        ]
    )
)

viewer = dbc.Card(
    [
        dbc.CardHeader("Viewer"),
        dbc.CardBody(
            [
                dcc.Graph(
                    id='graph',
                    figure=fig,
                )
            ]
        ),
        dbc.CardFooter(
            [
            html.Div(
                [
                    dbc.Label(
                        id="label-audio"
                    ),
                ]
            ), 
            ]
        )
    ]
)



app.layout = dbc.Container(
            [
                dbc.CardGroup([
                        viewer, 
                        controls, 
                ]
                      
                ),
            ],
            fluid=True
        )

@app.callback(
    Output(component_id='graph', component_property='figure'),
    Input(component_id='select-audio', component_property='value'),
)
def audio_selected(select_value):
    if select_value is not None:
        df = gt_df_dict[select_value]
        new_fig = get_init_fig()
        frames = []
        
        # ground truth
        for i in range(len(df)):
            frames.append(
                go.Frame(data=[
                        go.Scatter3d(
                            text='microphone',
                            x=[0], 
                            y=[0], 
                            z=[0],
                            mode='markers',
                            name='microphone'),
                        go.Scatter3d(
                            visible=df.iloc[i]['class'] != 'NOTHING',
                            text=df.iloc[i]['class'],
                            x=[df.iloc[i]['x']], 
                            y=[df.iloc[i]['y']], 
                            z=[df.iloc[i]['z']],
                            mode='markers',
                            name='ground-truth')]
                        ))

        new_fig.update(dict(frames=frames))

        return new_fig

    return fig


if __name__ == "__main__":
    app.run_server( debug=True)