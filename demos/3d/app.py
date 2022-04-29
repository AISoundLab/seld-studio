import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output
import pathlib
import os
import librosa
import pandas as pd
import plotly.express as px
import torch
import numpy as np
from model.utils import predictions_list, get_eval_model, get_inputs, predictions_list, spectrum_fast

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

script_dir = pathlib.Path(__file__).parent.resolve()
audio_filenames = os.listdir(pathlib.Path(script_dir, 'assets', 'audio'))
inputs = get_inputs(audio_filenames)

model_path = str(pathlib.Path(script_dir, 'assets', 'model', 'SELDNet_checkpoint.pt'))
model = get_eval_model(model_path)

def make_fig_df(df, legend):
    return pd.DataFrame({
            'legend': legend, 
            'class': df['class'].replace('NOTHING', '[silence]'),
            'x': df['x'],
            'y': df['y'],
            'z': df['z'],
            'time(s)': df['frame']/10,
            })

def make_figure(gt_df=None, pred_df=None, n_frames=1):
    # nothing loaded yet
    df = pd.DataFrame({
        'legend':'microphone',
        'class': ['-' for _ in range(n_frames)],
        'x':[0 for _ in range(n_frames)],
        'y':[0 for _ in range(n_frames)],
        'z':[0 for _ in range(n_frames)],
        'time(s)': [i/10 for i in range(n_frames)],
    })

    if gt_df is not None and pred_df is not None:
        new_gt_df = make_fig_df(gt_df, 'ground-truth')
        new_pred_df = make_fig_df(pred_df, 'prediction')
        df = pd.concat((df, new_gt_df, new_pred_df))
    range_x= [-3.0, 3.0]
    range_y= [-3.0, 3.0]
    range_z= [-2.0, 2.0]
    new_fig = px.scatter_3d(df, x='x', y='y', z='z', color='legend', text='class', 
    animation_frame='time(s)', opacity=0.8,
                        range_x=range_x, range_y=range_y, range_z=range_z, height=600)

    if gt_df is not None:
        new_fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 100

    return new_fig
    
gt_df_dict = {} 
tensor_dict = {}
for input in inputs:
    gt_df_dict[input['name']] = pd.read_csv(input['gt_path'])
    tensor,_ = librosa.load(input['audio_path'], sr=32000, mono=False)
    tensor = spectrum_fast(tensor, nperseg=512, noverlap=112, window="hamming", output_phase=False)
    tensor= torch.tensor(tensor).float().unsqueeze(0)
    tensor_dict[input['name']] = tensor

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

controls = html.Div(
            [
                dbc.Label("Audio File"),
                dbc.Select(
                    id="select-audio",
                    options=list(map(lambda x : x['option'], inputs))
                )
            ])

viewer = dbc.Card(
    [
        dbc.CardHeader("Viewer"),
        dbc.CardBody(
            [
                dcc.Loading(
                    dcc.Graph(
                    id='graph',
                ),type='cube'
                )
               
            ]
        ),
        dbc.CardFooter(
            [
            html.Div(
                [
                    controls
                ]
            ), 
            ]
        )
    ]
)



app.layout = dbc.Container(
            [
                viewer
            ],
        )

@app.callback(
    Output(component_id='graph', component_property='figure'),
    Input(component_id='select-audio', component_property='value'),
)
def audio_selected(select_value):
    if select_value is not None:
        gt_df = gt_df_dict[select_value]
        x = tensor_dict[select_value]
        with torch.no_grad():
            sed, doa = model(x)
        sed = sed.cpu().numpy().squeeze()
        doa = doa.cpu().numpy().squeeze()
        predictions = predictions_list(sed, doa, len(gt_df))
        pred_df = pd.DataFrame(predictions, columns=['frame', 'class', 'x', 'y', 'z'])
        new_fig = make_figure(gt_df, pred_df, len(gt_df))

        return new_fig
    

    return make_figure()


if __name__ == "__main__":
    app.run_server(debug=True)