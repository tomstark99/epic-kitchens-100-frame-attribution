from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import dash
from dash import Dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import base64

from apps.esv_dashboard.result import Result, ShapleyValueResults

class Visualiser:
    def __init__(
        self,
        results: ShapleyValueResults,
        colours: Dict[str, Any],
        verb2str: Dict[int, str],
        noun2str: Dict[int, str],
        verb_noun: Dict[str, Any],
        verb_noun_classes: Dict[str, Any],
        verb_noun_narration: Dict[str, Any],
        dataset_dir: Path,
        title: str = "ESV Dashboard for Epic",
    ):
        self.results = results
        self.colours = colours
        self.verb2str = verb2str
        self.noun2str = noun2str
        self.dataset_dir = dataset_dir
        self.title = title
        self.verb_noun = verb_noun
        self.verb_noun_classes = verb_noun_classes
        self.verb_noun_narration = verb_noun_narration

        for verb in verb_noun_narration:
            unique = {}
            for noun, nar_id in verb_noun_narration[verb]:
                if noun in unique:
                    unique[noun].append(nar_id)
                else:
                    unique[noun] = [nar_id]

            verb_noun_narration[verb] = unique

        self.default_state = {
            "n_frames": self.results.max_n_frames,
            "default_frames": 4,
            "verb": 0,
        }

    def attach_to_app(self, app: Dash):
        def app_layout():
            return self.render_layout()

        app.layout = app_layout
        self.attach_callbacks(app)

    def attach_callbacks(self, app: Dash):

        @app.callback(
            Output('noun','options'),
            Input('verb','value')
        )
        def update_noun_list_from_selected_verb(verb_class):
            noun_list=self.verb_noun[self.verb2str[verb_class]]
            noun_class=self.verb_noun_classes[verb_class]
            
            assert len(noun_list) == len(noun_class)
            
            return [{'label':i, 'value':j} for i,j in zip(noun_list,noun_class)]

        @app.callback(
            Output('noun','value'),
            Input('noun', 'options')
        )
        def return_noun_from_options(noun):
            return noun[0]['value']

        @app.callback(
            Output('video', 'options'),
            Input('noun', 'value'),
            Input('verb', 'value')
        )
        def update_video_list_from_selected_noun(noun, verb):
            return [{'label':i, 'value':i} for i in self.verb_noun_narration[verb][noun]]

        @app.callback(
            Output('video','value'),
            Input('video','options')
        )
        def return_video_from_options(video):
            return video[0]['value']

        @app.callback(
            Output('display-selected-values', 'children'),
            Input('noun', 'value'),
            Input('verb', 'value'),
            Input('video', 'value')
        )
        def output_selections(noun, verb, video):
            return 'Selected Verb: {}, Selected Noun: {}, Video {}'.format(verb, noun, video)

        @app.callback(
            Output('video-container', 'children'),
            Input('frame-slider','value'),
            Input('video', 'value')
        )
        def update_video(frame, narr_id):
            path = self.dataset_dir / f'videos/{narr_id}.webm'
            encode = base64.b64encode(open(path,'rb').read())

            return html.Video(src='data:video/webm;base64,{}'.format(encode.decode()),autoPlay=True, loop=True, style={'borderRadius':'10px'})

        @app.callback(
            Output('frame-container', 'children'),
            Output('selected-frame', 'children'),
            Input('esv-preds-plot','hoverData'),
            Input('video', 'value')
        )
        def update_frame(hover_data, narr_id):

            if hover_data is None or hover_data['points'][0]['x'] > self.results[narr_id].max_frame:
                frame_idx = 0
            else:
                frame_idx = hover_data['points'][0]['x']

            path = self.dataset_dir / f'{narr_id}/'f'frame_{frame_idx:06d}.jpg'
            encode = base64.b64encode(open(path,'rb').read())

            return html.Img(src='data:image/png;base64,{}'.format(encode.decode()), style={'borderRadius':'10px'}), f"Selected frame: {frame_idx}"

        @app.callback(
            Output('model-preds-bar','figure'),
            Input('frame-slider','value'),
            Input('video','value')
        )
        def update_predictions(n_frames, narr_id):
            result = self.results[narr_id]
            v_df, n_df = self.get_preds_df(result, n_frames)

            return self.plot_preds(v_df, n_df)

        @app.callback(
            Output('esv-preds-plot','figure'),
            Input('frame-slider','value'),
            Input("model-preds-bar", "clickData"),
            Input('video','value')
        )
        def update_esvs(n_frames, click_data, narr_id):
            result = self.results[narr_id]
            if click_data is not None:
                cls = click_data['points'][0]['customdata']
                return self.plot_esvs(result, n_frames, cls)
            else:
                return self.plot_esvs(result, n_frames)

    def render_layout(self):
        return html.Div([
            html.Div([
                html.H1(children="ESVs Dashboard for Epic", style={'width':'100%', 'marginTop':'20px'}),
                # html.Hr(style={'marginBottom':'0px'}),
                html.Div([
                    html.Div([
                        html.Div([
                            html.H4(children='Select a verb'),
                            dcc.Dropdown(
                                id='verb',
                                options=[{'label':i, 'value':j} for i,j in zip(self.verb_noun.keys(), self.verb_noun_classes.keys())],
                                value=self.default_state['verb']
                            )
                        ], style={'width':'20%', 'float':'left','display':'inline-block'}),
                        html.Div([
                            html.H4(children='Select a noun'),
                            dcc.Dropdown(
                                id='noun'
                            )
                        ], style={'width':'20%','display':'inline-block','paddingLeft':'40px'}),
                        html.Div([
                            html.H4(children='Select a video'),
                            dcc.Dropdown(
                                id='video'
                            )
                        ], style={'width':'20%','display':'inline-block','paddingLeft':'40px'}),
                        html.Div([
                            html.H4(children='Select number of frames'),
                            dcc.Slider(
                                id='frame-slider',
                                min=self.results.sequence_idxs[0].shape[1],
                                max=self.results.sequence_idxs[-1].shape[1],
                                marks={str(i):str(i) for i in range(self.results.sequence_idxs[0].shape[1], self.results.sequence_idxs[-1].shape[1]+1)},
                                value=self.default_state['default_frames']
                            )
                        ], style={'width':'90%'})
                    ], style={'background':self.colours['rgb']['blue_20'], 'borderRadius':'20px', 'marginRight':'50px', 'padding':'10px 30px 20px 30px'}),
                    # html.Hr(style={'marginBottom':'0px'}),
                    html.Div([
                        html.Div([
                            html.H3("Model Predictions"),
                            dcc.Graph(
                                id="model-preds-bar",
                                config={"displayModeBar": False},
                                responsive=True,
                            )
                        ], id="model-preds-bar-container", style={'width':'40%','display':'inline-block','paddingRight':'5%'}),
                        html.Div([
                            html.H3("ESV Predictions"),
                            dcc.Graph(
                                id="esv-preds-plot",
                                config={"displayModeBar": False},
                                responsive=True,
                            )
                        ], id="esv-preds-plot-container", style={'width':'50%', 'display':'inline-block'})
                    ], style={'width':'100%'})
                ],style={'width':'75%','display':'inline-block'}),
                html.Div([
                    html.Div([
                        html.Div([
                                html.Span("Original Video:"),
                                html.Div(id='video-container', style={'width':'100%'})
                            ]
                        ),
                        html.Div([
                                html.Span(id='selected-frame'),
                                html.Div(id='frame-container')
                            ],style={'padding':'10px 0'}
                        ),
                        html.Div(id='display-selected-values', style={'width':'100%','paddingTop':'10px', 'fontStyle':'italic'})
                    ], style={'width':'100%'})
                ],style={'width':'25%', 'float':'right','display':'inline-block'}),
            ],style={'paddingLeft':'50px','paddingRight':'50px'})
        ])

    def get_preds_df(self, result: Result, n_frames: int):

        scores = result.scores[n_frames - 1]

        verb_pred = list(scores['verb'].squeeze().argsort()[::-1][:10])
        noun_pred = list(scores['noun'].squeeze().argsort()[::-1][:10])

        if result.label['verb'] not in verb_pred:
            verb_pred = verb_pred[:-1] + [result.label['verb']]

        if result.label['noun'] not in noun_pred:
            noun_pred = noun_pred[:-1] + [result.label['noun']]

        entries = []
        for i, (v_cls, n_cls) in enumerate(zip(verb_pred, noun_pred)):
            verb_name = self.verb2str[v_cls]
            noun_name = self.noun2str[n_cls]

            entries.append({
                'verb': {
                    'Idx': i,
                    'Class': verb_name,
                    'ClassId': v_cls,
                    'Score': scores['verb'].squeeze()[v_cls],
                    'Type': 'verb',
                    'Colour': self.colours['hex']['blue']
                },
                'noun': {
                    'Idx': i,
                    'Class': noun_name,
                    'ClassId': n_cls,
                    'Score': scores['noun'].squeeze()[n_cls],
                    'Type': 'noun',
                    'Colour': self.colours['hex']['red']
                }
            })

        return pd.DataFrame([e['verb'] for e in entries]), pd.DataFrame([e['noun'] for e in entries])

    def plot_esvs(self, result: Result, n_frames: int, alt_class: Optional[int] = None):
        
        pred_class = result.label
        
        if alt_class != pred_class and alt_class is not None:
            dicts = [pred_class, alt_class]
        else:
            dicts = [pred_class]

        classes = {}
        for k in pred_class.keys():
            classes[k] = [d[k] for d in dicts]

        # print(result.sequence_idxs[n_frames-1], n_frames)
    
        entries = {'verb':[],'noun':[]}
        for key, cls in classes.items():
            for c in cls:
                for i in range(n_frames):
                    entries[key].append({
                        'segment': i + 1,
                        'Frame': result.sequence_idxs[n_frames-1][i],
                        'ESV': result.esvs[n_frames-1][key].squeeze(0)[i, c],
                        'Class': self.verb2str[c] if key == 'verb' else self.noun2str[c],
                        'Colour': self.colours['hex']['blue'] if key == 'verb' else self.colours['hex']['red']
                    })

        v_df = pd.DataFrame(entries['verb'][:n_frames]) 
        n_df = pd.DataFrame(entries['noun'][:n_frames])

        fig = go.Figure()

        for name, df in zip(['verb','noun'],[v_df, n_df]):
            fig.add_trace(go.Scatter(
                name=f'<b>Predicted {name}</b>',
                x=df.Frame,
                y=df.ESV,
                hovertemplate='<br>Frame: %{x}<br>'+ f'{name}: ' + '%{text}<br>' + 'Score: %{y}',
                text=[f'{c}' for c in df.Class],
                line_shape='spline',
                opacity=1,
                line_color=df.Colour.values[0]
            ))

        if alt_class != pred_class and alt_class is not None:
            av_df = pd.DataFrame(entries['verb'][n_frames:]) 
            an_df = pd.DataFrame(entries['noun'][n_frames:])
            for name, df in zip(['a_verb','a_noun'],[av_df, an_df]):
                fig.add_trace(go.Scatter(
                    name=f'<b>{name}</b>',
                    x=df.Frame,
                    y=df.ESV,
                    hovertemplate='<br>Frame: %{x}<br>'+ f'{name}: ' + '%{text}<br>' + 'Score: %{y}',
                    text=[f'{c}' for c in df.Class],
                    line_shape='spline',
                    opacity=0.3,
                    line_color=df.Colour.values[0]
                ))

        fig.add_hline(y=0)

        fig.update_traces(mode="markers+lines")
        fig.update_layout(
            margin_r=0,
            margin_b=20,
            xaxis_title="Frame",
            yaxis_title="Score",
            hovermode="x unified",
            legend={"yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
            transition={"duration": 400, "easing": "cubic-in-out"},
        )

        return fig

    def plot_preds(self, v_df, n_df):
        
        fig = go.Figure()

        for name, df in zip(['verb','noun'],[v_df, n_df]):
            fig.add_trace(go.Bar(
                name=f'<b>Predicted {name}</b>',
                x=df.Idx,
                y=df.Score,
                hovertemplate="Score: %{y}<br>" + "Class: %{text}",
                text=[f"{c}" for c in df.Class],
                customdata=[{'verb':v, 'noun':n} for v, n in zip(v_df.ClassId, n_df.ClassId)],
                marker_color=df.Colour
            ))

        fig.update_layout(
            autosize=False,
            width=2000,
            height=800,
            xaxis_title="Verb, Noun labels",
            yaxis_title="Score",
            legend={"yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
            transition={"duration": 400, "easing": "cubic-in-out"}
        )

        fig.update_xaxes(
            tickmode="array",
            tickvals=v_df.Idx,
            ticktext=v_df.Class + ', ' + n_df.Class,
            tickangle=45,
            automargin=True,
        )

        return fig

    def get_result(self, cls: int, example_idx: int) -> Result:
        idx = self.results.class_example_idxs_lookup[cls][example_idx]
        return self.results[idx]
