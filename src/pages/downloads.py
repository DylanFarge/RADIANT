from dash import register_page, html, dcc, callback, Input, Output, State, ctx
from processing import data
import dash_bootstrap_components as dbc
from assets.html_ids import DownloadsIDs as ids
from flask_login import current_user

register_page(__name__, path='/downloads')

layout = html.Div(
    style={
        'height': '100%',
    },
    children=[
        html.Div(
            style = {
                'display': 'flex',
                'flexDirection': 'row',
                'justifyContent': 'space-between',
                'marginTop': '3%',
                'marginBottom': '3%',
            },
            children=[
                html.Button(
                    # id = "select-all-catalogs",
                    id = ids.toggle_all_cats,
                    children=["Select All Catalogs"],
                    style={
                        'width': '25%',
                        "marginLeft": "20%",
                        'height':'40%',
                    }
                ),
                html.Div(
                    style={
                        'width': '25%',
                        "marginRight": "20%",
                    },
                    children=[
                        dcc.Dropdown(
                            # id = "catalogs-dropdown",
                            id = ids.cats,
                            placeholder="Select catalogs to download...",
                            multi=True,
                            style={
                                'backgroundColor': 'black',
                                'color': 'black'
                            }
                        )
                    ]
                )
            ]
        ),
        html.Div(
            style = {
                'display': 'flex',
                'flexDirection': 'row',
                'justifyContent': 'space-between',
                'marginBottom': '3%',
            },
            children=[
                html.Button(
                    # id="select-all-morphologies",
                    id = ids.toggle_all_types,
                    children=["Select All Morphologies"],
                    style={
                        'width': '25%',
                        "marginLeft": "20%",
                        'height':'40%',
                    }
                ),
                html.Div(
                    style={
                        'width': '25%',
                        "marginRight": "20%",
                    },
                    children=[
                        dcc.Dropdown(
                            # id="morphologies-dropdown",
                            id = ids.types,
                            multi=True,
                            placeholder="Select morphologies to download...",
                            style={
                                'backgroundColor': 'black',
                                'color': 'black'
                            }
                        )
                    ]
                )
            ]
        ),
        html.Div(
            style = {
                'display': 'flex',
                'flexDirection': 'row',
                'justifyContent': 'space-between',
                'marginBottom': '3%',
            },
            children=[
                html.Div(
                    style={
                        'width': '25%',
                        "margin": "0 0 0 20%",
                        'height':'40%',
                        'display': 'flex',
                        'flexDirection': 'row',
                        'justifyContent': 'space-between',
                    },
                    children = [
                        dcc.Checklist(
                            # id = "include-images",
                            id = ids.include_images,
                            options=[" Include Images"],
                        ),
                        dcc.Input(
                            # id = "image-limit",
                            id = ids.image_limit,
                            placeholder="No Image Limit....",
                            type="number",
                            spellCheck=False,
                        ),
                    ]
                ),
                html.Div(
                    style={
                        'width': '25%',
                        "marginRight": "20%",
                    },
                    children=[
                        dcc.Dropdown(
                            # id = "survey-list",
                            id = ids.surveys,
                            options=data.get_surveys(),
                            placeholder="Select surveys to download from...",
                            multi=True,
                            style={
                                'backgroundColor': 'black',
                                'color': 'black'
                            }
                        )
                    ]
                ),
            ]
        ),
        html.Div(
            style = {
                'display': 'flex',
                'flexDirection': 'row',
                'justifyContent': 'space-between',
                'marginBottom': '3%',
            },
            children=[
                html.Div(
                    style={
                        'width': '25%',
                        "margin": "0 0% 0 20%",
                        'height':'40%',
                        'display': 'flex',
                        'flexDirection': 'row',
                        'justifyContent': 'space-between',
                    },
                    children=[
                        dcc.Checklist(
                            # id = "remove-duplicates",
                            id = ids.remove_duplicates,
                            options=[" Remove Duplicates"],
                        ),
                        dcc.Input(
                            # id = "threshold",
                            id = ids.threshold,
                            placeholder="Threshold....",
                            type="number",
                            spellCheck=False,
                        ),
                    ]
                ),
                html.Div(
                    style={
                        'width': '25%',
                        "marginRight": "20%",
                    },
                    children=[
                        dcc.Dropdown(
                            # id="priority-dropdown",
                            id = ids.priority,
                            multi=True,
                            placeholder="(Optional) List catalogs in order of priority...",
                            style={
                                'backgroundColor': 'black',
                                'color': 'black'
                            }
                        )
                    ]
                )
            ]
        ),
        html.Center([
            dcc.Loading([html.H5(id=ids.num_of_sources),]),
            html.Button(
                children =["Download"], 
                # id="download-button",
                id = ids.download_btn,
            )
        ]),
        dcc.Download(id=ids.download),
        dcc.ConfirmDialog(id=ids.failed),
        # dcc.Interval(id=ids.progress_interval, n_intervals=0, interval=250),
        html.Center([
            html.Button(
                children=["Cancel"],
                id = ids.cancel,
            ),
            dbc.Progress(
                id=ids.progress,
                style={
                    'marginTop': '1%',
                    'width': '50%',
                }, 
                value=0, 
                striped=True, 
                animated=True
            ),
        ],hidden=True,id=ids.progress_bar_center),
        html.Center("Form Not Complete", style={"color": "red"}, hidden=True, id=ids.status)
    ]
)    

@callback(
    # Output("priority-dropdown", "disabled"),
    # Output("priority-dropdown", "value"),
    # Output("priority-dropdown", "options"),
    # Input("remove-duplicates", "value"),
    # Input("catalogs-dropdown", "value"),
    # State("priority-dropdown", "value")
    Output(ids.priority, "disabled"),
    Output(ids.priority, "value"),
    Output(ids.priority, "options"),
    Input(ids.remove_duplicates, "value"),
    Input(ids.cats, "value"),
    State(ids.priority, "value")
)
def update_priority_dropdown(rm, catalogs, curr:list):
    if catalogs == None:
        catalogs = []
        
    if curr == None:
        curr = []

    if not rm:
        return True, [], []
    removed = []

    for cat in curr:
        if cat not in catalogs:
            removed.append(cat)

    for cat in removed:
        curr.remove(cat)

    return False, curr, catalogs

# @callback(
#     # Output("progress-bar-center", "hidden"),
#     # Input("download-button", "n_clicks"),
#     # Input("progress-interval", "n_intervals"),
#     # State("progress", "value"),
#     Output(ids.progress_bar_center, "hidden"),
#     Input(ids.download_btn, "n_clicks"),
#     Input(ids.progress_interval, "n_intervals"),
#     State(ids.progress, "value"),
# )
# def show_progress_bar(n_clicks, n_intervals, progress):
#     if n_clicks and current_user.is_downloading and progress != None and progress < 100:
#         return False
#     return True

# @callback(
#     # Output("progress", "value"),
#     # Output("progress", "label"),
#     # Output("failed", "displayed"),
#     # Output("failed", "message"),
#     # Input("progress-interval", "n_intervals"),
#     # State("num-of-sources", "children"),
#     # State("image-limit", "value"),
#     # State("survey-list", "value"),
#     # State("progress", "value"),
#     Output(ids.progress, "value"),
#     Output(ids.progress, "label"),
#     Output(ids.failed, "displayed"),
#     Output(ids.failed, "message"),
#     Input(ids.progress_interval, "n_intervals"),
#     State(ids.num_of_sources, "children"),
#     State(ids.image_limit, "value"),
#     State(ids.surveys, "value"),
#     State(ids.progress, "value"),
#     prevent_initial_call=True,
# )
# def progress_bar(n_intervals, num_of_sources, image_limit, surveys, progress):
#     if num_of_sources is None:
#         num_of_sources = "Source: 0"
#     num_of_sources = int(num_of_sources.split(" ")[1])

#     if progress != None and progress >= 100:
#         failed = int(current_user.total - current_user.progress)
#         current_user.progress = 0
#         current_user.total = 0

#         if failed > 0:
#             return None, None, True, f"Failed to find {failed} radio images from specified sureys."
#         return None, None, False, None
    
#     if not num_of_sources or not surveys:
#         return None, None, False, None
    
#     if image_limit: 
#         value = int((current_user.progress / image_limit) * 100)
#         if value == 0:
#             value = 2
#         return value, f"{value}%", False, None
#     value = int((current_user.total / (num_of_sources * len(surveys))) * 100)

#     if value == 0:
#         value = 2
#     return value, f"{value}%", False, None

@callback(
    # Output("download", "data"),
    # Input("download-button", "n_clicks"),
    # State("catalogs-dropdown", "value"),
    # State("morphologies-dropdown", "value"),
    # State("include-images", "value"),
    # State("survey-list", "value"),
    # State("image-limit", "value"),
    # State("remove-duplicates", "value"),
    # State("threshold", "value"),
    # State("priority-dropdown", "value")
    Output(ids.download, "data"),
    Output(ids.status, "hidden"),
    Input(ids.download_btn, "n_clicks"),
    State(ids.cats, "value"),
    State(ids.types, "value"),
    State(ids.include_images, "value"),
    State(ids.surveys, "value"),
    State(ids.image_limit, "value"),
    State(ids.remove_duplicates, "value"),
    State(ids.threshold, "value"),
    State(ids.priority, "value"),
    background=True,
    cancel=Input(ids.cancel, "n_clicks"),
    progress=[
        Output(ids.progress,"value"),
        Output(ids.progress, "label"),
        Output(ids.progress, "max"),
    ],
    prevent_initial_call=True,
    running=[
        (Output(ids.download_btn, "disabled"), True, False),
        (Output(ids.progress_bar_center, "hidden"), False, True),
    ]
)
def download(set_progress, n_clicks, catalogs, types, include_images, surveys, image_limit, remove_duplicates, threshold, priority):
    print("DOWNLOADING...")
    if n_clicks:
        if catalogs and types:
            if (include_images and (not surveys or surveys ==[])) or (remove_duplicates and threshold == None):
                print("FORM NOT COMPLETED")
                return None, False
            data_bytes = data.download(set_progress, catalogs, types, include_images, surveys, image_limit, threshold, priority)
            return dcc.send_bytes(data_bytes, "RADIANT.zip"), True
            
        else:
            print("FORM NOT COMPLETED")
            return None, False
    return None, True

@callback(
    # Output("num-of-sources", "children"),
    # Input("catalogs-dropdown", "value"),
    # Input("morphologies-dropdown", "value"),
    # Input("threshold", "value"),
    # Input("priority-dropdown", "value")
    Output(ids.num_of_sources, "children"),
    Input(ids.cats, "value"),
    Input(ids.types, "value"),
    Input(ids.threshold, "value"),
    Input(ids.priority, "value")
)
def display_num_of_sources_to_download(catalogs, types, threshold, prior):
    if catalogs and types:
        cats = data.getCatalogs(catalogs, types)
        
        if threshold != None:
            if prior == None:
                prior = []
            remove = data.remove_duplicates(catalogs, threshold, prior)
            cats = cats[~cats.index.isin(remove)]

        return f"Sources: {len(cats)}"
    return "Sources: 0"

@callback(
    # Output("catalogs-dropdown", "options"),
    # Input("dummy", "children"),
    Output(ids.cats, "options"),
    Input(ids.cats, "id"), # Arbitrary
)
def get_catalog_options(dummy):
    return data.get_catalog_names()

@callback(
    # Output("catalogs-dropdown", "value"),
    # Input("select-all-catalogs", "n_clicks"),
    Output(ids.cats, "value"),
    Input(ids.toggle_all_cats, "n_clicks"),
)
def select_all_catalogs(n_clicks):
    if n_clicks:
        return data.get_catalog_names()
    
@callback(
#     Output("morphologies-dropdown", "value"),
#     Output("morphologies-dropdown", "options"),
#     Input("select-all-morphologies", "n_clicks"),
#     Input("catalogs-dropdown", "value"),
#     State("morphologies-dropdown", "value"),
    Output(ids.types, "value"),
    Output(ids.types, "options"),
    Input(ids.toggle_all_types, "n_clicks"),
    Input(ids.cats, "value"),
    State(ids.types, "value"),
)
def select_all_morphologies(n_clicks, catalogs, curr):
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered == ids.toggle_all_types:
        return data.get_morphology_names(catalogs), data.get_morphology_names(catalogs)
    
    if curr == None:
        curr = []
    remove = []

    for typ in curr:
        if typ not in data.get_morphology_names(catalogs):
            remove.append(typ)

    for typ in remove:
        curr.remove(typ)   

    return curr, data.get_morphology_names(catalogs)
        
@callback(
    # Output("threshold", "disabled"),
    # Output("threshold", "value"),
    # Input("remove-duplicates", "value")
    Output(ids.threshold, "disabled"),
    Output(ids.threshold, "value"),
    Input(ids.remove_duplicates, "value")
)
def toggle_duplicate_options(value):
    if value:
        return False, None
    else:
        return True, None
    
@callback(
    # Output("image-limit", "disabled"),
    # Output("image-limit", "value"),
    # Output("survey-list", "disabled"),
    # Output("survey-list", "value"),
    # Input("include-images", "value")
    Output(ids.image_limit, "disabled"),
    Output(ids.image_limit, "value"),
    Output(ids.surveys, "disabled"),
    Output(ids.surveys, "value"),
    Input(ids.include_images, "value")
)
def toggle_image_options(value):
    if value:
        return False, None, False, None
    else:
        return True, None, True, None