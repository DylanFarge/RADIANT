from dash import register_page, html, dcc, callback, Input, Output, State, ctx
import plotly.express as px
from processing import data
import plotly.graph_objects as go
import numpy as np
import dash_bootstrap_components as dbc
from flask_login import current_user
from assets.html_ids import AnalysisIDs as ids

register_page(__name__, path='/analysis')

layout = html.Div(
    [
        html.Div(
            style={
                'display': 'flex',
                'flexDirection': 'row',
                'justifyContent': 'space-evenly',
                'marginBottom': '2%',
            },
            children=[
                html.Div(
                    style={
                        'border': '1px solid black',
                        'borderRadius': '20px',
                        'backgroundColor': '#111111',
                        'width': '100%',
                        'height': '550px',
                        'marginRight':'1%',
                    },
                    children=[
                        html.Div(
                            style={
                                'margin': '2%',
                            },
                            children=[
                                dcc.Dropdown(
                                    # id = "catalog_dropdown",
                                    id = ids.cats,
                                    style={
                                        'color': 'black',
                                        'backgroundColor': 'black',
                                    },
                                    placeholder="Select Catalog",
                                    value="MiraBest",
                                    clearable=False,
                                ),
                            ]
                        ),
                        dcc.Loading(
                            type="default",
                            children=[
                                html.Div(
                                    style={
                                        'display': 'flex',
                                        'flexDirection': 'row',
                                        'justifyContent': 'space-evenly',
                                        'marginTop':'2%',
                                    },
                                    children=[
                                        html.Button(
                                            # id = 'log_x',
                                            id = ids.log_x,
                                            children=["Toggle Log X"],
                                            style={
                                                'border': '1px solid black',
                                                'borderRadius': '10px',
                                            },
                                        ),
                                        html.Button(
                                            # id = 'log_y',
                                            id = ids.log_y,
                                            children=["Toggle Log Y"],
                                            style={
                                                'border': '1px solid black',
                                                'borderRadius': '10px',
                                            },
                                        ),
                                    ]
                                ),
                                dcc.Graph(
                                    # id = "distance_graph",
                                    id = ids.distance_graph,
                                    style={
                                        'height': '100%',
                                    },
                                ),
                            ]
                        ),
                    ]
                ),
                html.Div(
                    style={
                        'border': '1px solid black',
                        'borderRadius': '20px',
                        'width': '100%',
                        'backgroundColor': '#111111',
                        'marginLeft':'1%',
                    },
                    children=[
                        html.Div(
                            style={
                                'display': 'flex',
                                'flexDirection': 'row',
                                'justifyContent': 'space-evenly',
                                'marginTop':'2%',
                            },
                            children=[
                                html.Div(
                                    style={
                                        'width': '20%',
                                    },
                                    children=[dcc.Dropdown(
                                        id = ids.sim_cat,
                                        multi=False,
                                        clearable=False,
                                        value="All Catalogs",
                                        style={
                                            # 'color': 'blue',
                                            'backgroundColor': 'black',
                                            'width': '100%',
                                        },
                                    ),]
                                ),
                                html.Button(
                                    # id = 'log_x_snap2',
                                    id = ids.log_x_snap2,
                                    children=["Toggle Log X"],
                                    style={
                                        'border': '1px solid black',
                                        'borderRadius': '10px',
                                    },
                                ),
                                html.Button(
                                    # id = 'log_y_snap2',
                                    id = ids.log_y_snap2,
                                    children=["Toggle Log Y"],
                                    style={
                                        'border': '1px solid black',
                                        'borderRadius': '10px',
                                    },
                                ),
                            ]
                        ),
                        dcc.Loading(
                            children = [
                            dcc.Graph(
                                # id ="snapshot_0002",
                                id = ids.snapshot_0002,
                            ),
                            html.Div(
                                style={
                                    'display': 'flex',
                                    'flexDirection': 'row',
                                    'justifyContent': 'space-between',
                                    'padding':"1% 2% 1% 2%",
                                    'backgroundColor': '#222222',
                                    'borderRadius': '20px',
                                },
                                children=[
                                    html.Div(
                                        style={
                                            'display': 'flex',
                                            'flexDirection': 'row',
                                        },
                                        children=[
                                            html.H6("Max: "),
                                            html.Div([dcc.Input(
                                                # id ='ceiling',
                                                id = ids.ceiling,
                                                type="number", 
                                                value=0.002, 
                                                required=True, 
                                                style={'width':'100%', 'height':'60%'},
                                                max=99,
                                            )], style={'width': '30%'}),
                                            html.H6("degrees.")
                                        ]
                                    ),
                                    html.Div(
                                        style={
                                            'display': 'flex',
                                            'flexDirection': 'row',
                                        },
                                        children=[
                                            html.H6("Steps: "),
                                            html.Div([dcc.Input(
                                                # id ="steps",
                                                id = ids.steps,
                                                type="number", 
                                                value=0.0001, 
                                                required=True, 
                                                style={'width': '100%','height':'60%'},
                                                max = 99,
                                            ),], style={'width': '30%'}),
                                            html.H6("degrees.")
                                        ]
                                    ),
                                    html.Button("Process",id = ids.process)
                                ]
                            ),]
                        ),
                        # dcc.Interval(
                        #     # id ="progress-interval", 
                        #     id = ids.process_interval,
                        #     n_intervals=0, 
                        #     interval=250
                        # ),
                        html.Center(
                            hidden=True, 
                            id = ids.progress_bar_div,
                            style = {
                                "display": "flex",
                                "flexDirection": "row",
                            },
                            children = [
                                dbc.Progress(
                                    # id ="progress-bar",
                                    id = ids.progress_bar,
                                    value=0,
                                    striped=True,
                                    animated=True,
                                    style={
                                        "width": "100%",
                                        "margin": "auto",
                                    }
                                ),
                                html.Button(
                                    "Cancel", 
                                    style={
                                        "marginLeft":"1%",
                                        "marginRight":"1%"
                                    },
                                    id = ids.cancel,
                                ),
                            ], 
                        ),
                    ]
                ),
            ]
        ),
        html.Div(
            style={
                'display': 'flex',
                'flexDirection': 'row',
                'justifyContent': 'space-evenly',
            },
            children=[
                html.Div(
                    style={
                        'border': '1px solid black',
                        'borderRadius': '20px',
                        'backgroundColor': '#111111',
                        'width': '100%',
                        'marginRight':'1%',
                    },
                    children=[
                        dcc.Tabs(
                            colors={
                                'background': 'black',
                                # 'border': 'blue',
                                'primary': '#08F',
                            },
                            children=[
                                dcc.Tab(
                                    label="Morphology Composition",
                                    style={
                                        'borderRadius': '0 20px 0 20px',
                                    },
                                    selected_style={
                                        'backgroundColor': '#111111',
                                        'color': '#ffffff',
                                        'borderRadius': '20px',
                                    },
                                    children=[
                                        dcc.Loading(
                                            type="default",
                                            children=[
                                                dcc.Graph(
                                                    # id ="type_graph",
                                                    id = ids.type_graph,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                dcc.Tab(
                                    label="Catalog Composition",
                                    style={
                                        'borderRadius': '20px 0 20px 0',
                                    },
                                    selected_style={
                                        'backgroundColor': '#111111',
                                        'color': '#ffffff',
                                        'borderRadius': '20px',
                                    },
                                    children=[
                                        dcc.Loading([
                                            html.Center(
                                                children=[
                                                    html.H5("Catalog composition by morphology"),
                                                    dcc.Graph(
                                                        # id ="sunburst",
                                                        id = ids.sunburst,
                                                        style={
                                                            'width': '80%',
                                                            'margin': 'auto',
                                                        },
                                                    )
                                                ]
                                            ),
                                        ])
                                    ]
                                ),
                            ]   
                        )
                    ]
                ),
                html.Div(
                    style={
                        'border': '1px solid black',
                        'borderRadius': '20px',
                        'backgroundColor': '#111111',
                        'width': '100%',
                        'marginRight':'1%',
                    },
                    children=[
                        dcc.Tabs(
                            colors={
                                'background': 'black',
                                # 'border': 'blue',
                                'primary': '#08F',
                            },
                            children=[
                                dcc.Tab(
                                    label="Venn Overlap",
                                    style={
                                        'borderRadius': '0 20px 0 20px',
                                    },
                                    selected_style={
                                        'backgroundColor': '#111111',
                                        'color': '#ffffff',
                                        'borderRadius': '20px',
                                    },
                                    children=[
                                        dcc.Loading([
                                            html.Div(
                                                style={
                                                    'display': 'flex',
                                                    'flexDirection': 'row',
                                                    'justifyContent': 'space-evenly',
                                                },
                                                children=[
                                                    html.Div(
                                                        style={
                                                            'width': '100%',
                                                        },
                                                        children=[
                                                            dcc.Dropdown(
                                                                # id = "venn_dropdown1",
                                                                id = ids.venn_1,
                                                                style={
                                                                    'color': 'black',
                                                                    'backgroundColor': 'black',
                                                                },
                                                                placeholder="Select Catalog",
                                                                value="MiraBest",
                                                                clearable=False,
                                                            ),
                                                        ]
                                                    ),
                                                    html.Div(
                                                        style={
                                                            'width': '100%',
                                                        },
                                                        children=[
                                                            dcc.Dropdown(
                                                                # id = "venn_dropdown2",
                                                                id = ids.venn_2,
                                                                style={
                                                                    'color': 'black',
                                                                    'backgroundColor': 'black',
                                                                },
                                                                placeholder="Select Catalog",
                                                                value="FRICAT",
                                                                clearable=False,
                                                            ),
                                                        ]
                                                    ),
                                                    html.Div(
                                                        style={
                                                            'width': '100%',
                                                            'backgroundColor': '#111111',
                                                        },
                                                        children=[dcc.Input(id = ids.venn_input,type="number", value=1e-4, style={'width': '100%', 'height':'100%',})]
                                                    ),
                                                ]
                                            ),
                                            html.Center(
                                                # id ="snapshot_002",
                                                id = ids.snapshot_002,
                                                children=[html.Img(
                                                    src="src/assets/venn.png",
                                                    style={
                                                        'width': '80%',
                                                        'marginTop': '1%',
                                                        'borderRadius': '25px',
                                                        'border': '1px solid black',
                                                    }
                                                ),]
                                            ),
                                            html.Center([html.H5(
                                                # id ='venn_text',
                                                id = ids.venn_text,
                                                style={
                                                    'marginTop': '5%',
                                                    'whiteSpace': 'pre',
                                                }
                                            )])
                                        ]),
                                    ]
                                ),
                                dcc.Tab(
                                    label="Heatmap Overlap",
                                    style={
                                        'borderRadius': '20px 0 20px 0',
                                    },
                                    selected_style={
                                        'backgroundColor': '#111111',
                                        'color': '#ffffff',
                                        'borderRadius': '20px',
                                    },
                                    children=[
                                        dcc.Loading([
                                            html.Div(
                                                style=dict(
                                                    display="flex",
                                                    flexDirection="row",
                                                    justifyContent="space-evenly",
                                                    marginTop="2%",
                                                    marginBottom="2%",
                                                ),
                                                children=[
                                                    html.H5("Heatmap of overlap between catalogs"),
                                                    dcc.Input(
                                                        id=ids.heatmap_input,
                                                        placeholder="Threshold",
                                                        type="number",
                                                        value=0.0001,
                                                    ),
                                                    html.Button(
                                                        "Toggle Log",
                                                        id=ids.log_heatmap,
                                                        n_clicks=0
                                                    )
                                                ]
                                            ),
                                            html.Center(
                                                id=ids.heatmap,
                                            )
                                        ])
                                    ]
                                ),
                            ]   
                        )
                    ]
                ),
                # html.Div(
                #     style={
                #         'border': '1px solid black',
                #         'borderRadius': '20px',
                #         'display': 'flex',
                #         'flexDirection': 'column',
                #         'backgroundColor': '#111111',
                #         'width': '100%',
                #         'height': '100%',
                #         'marginLeft':'1%',
                #         'paddingTop':"1%",
                #     },
                #     children=[
                #         html.Div(
                #             style={
                #                 'display': 'flex',
                #                 'flexDirection': 'row',
                #                 'justifyContent': 'space-evenly',
                #             },
                #             children=[
                #                 html.Div(
                #                     style={
                #                         'width': '100%',
                #                     },
                #                     children=[
                #                         dcc.Dropdown(
                #                             # id = "venn_dropdown1",
                #                             id = ids.venn_1,
                #                             style={
                #                                 'color': 'black',
                #                                 'backgroundColor': 'black',
                #                             },
                #                             placeholder="Select Catalog",
                #                             value="MiraBest",
                #                             clearable=False,
                #                         ),
                #                     ]
                #                 ),
                #                 html.Div(
                #                     style={
                #                         'width': '100%',
                #                     },
                #                     children=[
                #                         dcc.Dropdown(
                #                             # id = "venn_dropdown2",
                #                             id = ids.venn_2,
                #                             style={
                #                                 'color': 'black',
                #                                 'backgroundColor': 'black',
                #                             },
                #                             placeholder="Select Catalog",
                #                             value="FRI",
                #                             clearable=False,
                #                         ),
                #                     ]
                #                 ),
                #                 html.Div(
                #                     style={
                #                         'width': '100%',
                #                         'backgroundColor': '#111111',
                #                     },
                #                     children=[dcc.Input(id = ids.venn_input,type="number", value=1e-4, style={'width': '100%', 'height':'100%',})]
                #                 ),
                #             ]
                #         ),
                #         html.Center(
                #             # id ="snapshot_002",
                #             id = ids.snapshot_002,
                #             children=[html.Img(
                #                 src="src/assets/venn.png",
                #                 style={
                #                     'width': '80%',
                #                     'marginTop': '1%',
                #                     'borderRadius': '25px',
                #                     'border': '1px solid black',
                #                 }
                #             ),]
                #         ),
                #         html.Center([html.H5(
                #             # id ='venn_text',
                #             id = ids.venn_text,
                #             style={
                #                 'marginTop': '5%',
                #                 'whiteSpace': 'pre',
                #             }
                #         )])
                #     ]
                # ),
                html.Div(id = ids.dummy)
            ]
        ),
    ]
)


# @callback(
#     # Output("progress-bar-div", "hidden"),
#     # Input("process", "n_clicks"),
#     # Input("progress-interval", "n_intervals"),
#     # State("progress-bar", "value"),
#     # State("ceiling", "value"),
#     # State("steps", "value"),
#     Output(ids.progress_bar_div, "hidden"),
#     Input(ids.process, "n_clicks"),
#     Input(ids.process_interval, "n_intervals"),
#     State(ids.progress_bar, "value"),
#     State(ids.ceiling, "value"),
#     State(ids.steps, "value"),
# )
# def show_progress_bar(n_clicks, n_intervals, value, ceil, steps):
#     if ceil == None or steps == None:
#         return True
#     if n_clicks is None:
#         return True
#     if value != None and value >= 100:
#         return True
#     if not current_user.is_simulating:
#         return True
#     return False


# @callback(
#     # Output("progress-bar", "value"),
#     # Output("progress-bar", "label"),
#     # Input("progress-interval", "n_intervals"),
#     # State("progress-bar", "value"),
#     # State("ceiling", "value"),
#     # State("progress-bar", "value"),
#     # State("progress-bar", "label"),
#     Output(ids.progress_bar, "value"),
#     Output(ids.progress_bar, "label"),
#     Input(ids.process_interval, "n_intervals"),
#     State(ids.progress_bar, "value"),
#     State(ids.ceiling, "value"),
#     State(ids.progress_bar, "value"),
#     State(ids.progress_bar, "label"),
# )
# def progress_bar(n_intervals, value, ceil, prev_value, prev_label):
#     if value != None and value >= 100:
#         return None, None
#     if ceil == None:
#         return prev_value, prev_label
    
#     perc = current_user.sim_progress /(10**-ceil)*100
#     perc = 2 if perc < 2 else perc
#     return perc, f"{perc:.0f}%"
    


@callback(
    # Output("sunburst", 'figure'),
    # Input("dummy", "children")
    Output(ids.sunburst, 'figure'),
    Input(ids.dummy, "children")
)
def update_sunburst(dummy):
    return data.getSunburst()


@callback(
#     Output("type_graph", 'figure'),
#     Input("dummy", "children")
    Output(ids.type_graph, 'figure'),
    Input(ids.dummy, "children")
)
def update_type_graph(dummy):
    return data.getTypesGraph()


@callback(
    Output(ids.snapshot_002, 'children'),
    Output(ids.venn_text, "children"),
    Input(ids.venn_input, "value"),
    Input(ids.venn_1, "value"),
    Input(ids.venn_2, "value"),
)
def update_snapshot_002(threshold, venn1, venn2):
    ven, nums = data.overlap_venn(threshold, venn1, venn2)
    return ven, f"Percentage Duplicates:\t{venn1}({np.round((nums[0]*100),1)}%)\t{venn2}({np.round((nums[1]*100),1)}%)"

@callback(
    Output(ids.heatmap, 'children'),
    Input(ids.heatmap_input, "value"),
    Input(ids.log_heatmap, "n_clicks"),
)
def update_heatmap(threshold, log):
    threshold = 0 if threshold == None else threshold
    is_logged = False if log % 2 == 0 else True
    return data.overlap_heatmap(threshold, is_logged)

@callback(
    # Output("snapshot_0002", 'figure'),
    # Input("process", "n_clicks"),
    # Input("log_x_snap2", "n_clicks"),
    # Input("log_y_snap2", "n_clicks"),
    # State("ceiling", "value"),
    # State("steps", "value"),
    # State("snapshot_0002", "figure"),
    Output(ids.snapshot_0002, 'figure'),
    Input(ids.process, "n_clicks"),
    Input(ids.log_x_snap2, "n_clicks"),
    Input(ids.log_y_snap2, "n_clicks"),
    State(ids.ceiling, "value"),
    State(ids.steps, "value"),
    State(ids.snapshot_0002, "figure"),
    State(ids.sim_cat, "value"),
    background=True,
    cancel=Input(ids.cancel, "n_clicks"),
    progress=[
        Output(ids.progress_bar, "value"),
        Output(ids.progress_bar, "label"),
        Output(ids.progress_bar, "max"),
    ],
    running=[
        (Output(ids.process, "disabled"), True, False),
        (Output(ids.progress_bar_div, "hidden"), False, True),
    ]
)
def update_threshold_simulation(set_progress, process, lx, ly, ceil, steps, prev_fig, cat):
    froms = ctx.triggered[0]["prop_id"].split(".")[0]
    if froms == ids.process:

        if ceil == None or steps == None:
            return prev_fig
        
        print("Running simulation...")
        fig = data.simulate_thresholds(set_progress,0, ceil, steps, cat)
        print("Simulation complete!")

    else:
        if not froms:
            draft_template = go.layout.Template()
            draft_template.layout.annotations = [
                dict(
                    name="draft watermark",
                    text="Example",
                    textangle=-20,
                    opacity=1,
                    font=dict(color="rgba(100,100,100,0.2)", size=100),
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
            ]
            snapshot_002 = np.fromfile("src/pages/components/analysis/snapshot_002.csv", sep=",")
            fig = px.line(y=snapshot_002, x=np.arange(0.00001, 0.002, 0.00001), template="plotly_dark", title="Number of duplicates detected with change in threshold")
            fig.update_layout(paper_bgcolor="rgb(0,0,0,0)",plot_bgcolor="rgb(0,0,0,0)", title_x=0.5, template=draft_template, font=dict(color="white"))
        else:
            fig = go.Figure(prev_fig)
            fig.update_annotations(dict(name="draft watermark",text=""))
        if lx != None and lx%2==1:
            fig.update_xaxes(title_text="Distance thresholds in degrees (LOG)",type="log",gridcolor = "#445")
        else:
            fig.update_xaxes(title_text="Distance thresholds in degrees",type="linear",gridcolor = "#445")
        if ly != None and ly%2==1:
            fig.update_yaxes(title_text="Number of duplicates detected (LOG)",type="log",gridcolor = "#445")
        else:
            fig.update_yaxes(title_text="Number of duplicates detected",type="linear",gridcolor = "#445")
    
    return fig

@callback(
    Output(ids.cats, "options"),
    Output(ids.venn_1, "options"),
    Output(ids.venn_2, "options"),
    Output(ids.sim_cat, "options"),
    Input(ids.dummy, "children"),
)
def update_options(n_clicks):
    options = [{"label": html.Span(cat,style={'color': '#08F',}),
            "value": cat,
            } for cat in current_user.df["Catalog"].unique().tolist()]
    sim_options = options.copy()
    sim_options.insert(0,{"label": html.Span("All Catalogs",style={'color': '#08F',}), "value": "All Catalogs"})
    print(sim_options)
    print(options)
    return options, options, options, sim_options


@callback(
    # Output("distance_graph", 'figure'),
    # Input('log_x', 'n_clicks'),
    # Input('log_y', 'n_clicks'),
    # Input("catalog_dropdown", "value"),
    # State("distance_graph", "figure")
    Output(ids.distance_graph, 'figure'),
    Input(ids.log_x, 'n_clicks'),
    Input(ids.log_y, 'n_clicks'),
    Input(ids.cats, "value"),
    State(ids.distance_graph, "figure")
)
def update_distance_graph(log_x, log_y, catalog, current_figure):
    if ctx.triggered[0]["prop_id"].split(".")[0] not in ["log_x","log_y"]:
        fig = px.line(data.get_closest_dist(catalog), template="plotly_dark", color_discrete_map=data.analysis_colours(get=True),width=675)
    else:
        fig = go.Figure(current_figure)
    fig.update_layout(paper_bgcolor="rgb(0,0,0,0)", title_text="Distance to closest source", title_x=0.5)

    if log_x == None or log_x%2==0:
        fig.update_xaxes(title_text="Index of "+catalog+" sources (LOG)",type="log")
    else:
        fig.update_xaxes(title_text="Index of "+catalog+" sources",type="linear")
    if log_y == None or log_y%2==0:
        fig.update_yaxes(title_text="Distance in degrees (LOG)",type="log")
    else:
        fig.update_yaxes(title_text="Distance in degrees",type="linear")

    return fig