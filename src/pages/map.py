import pandas as pd
import seaborn as sns
import dash_aladin_lite as dal
from dash import exceptions
import base64, datetime, io, os

from processing import data
from astropy import units as u
from urllib.error import HTTPError
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from dash import register_page, html, callback, Output, Input, dcc, State, dash_table, ctx
from assets.html_ids import MapIDs as ids
from flask_login import current_user
from dash import no_update

# The code block below is a bit of a 'hack' to allow unittests to run sufficiently.
#!-----------------------------------------------------------------------------
# global sideImage
# image = Image.open("src/assets/init.png")
# plt.figure(frameon=False)
# plt.imshow(image,cmap='inferno', origin='lower')      
# plt.axis('off')
# buffer = io.BytesIO()
# plt.savefig(buffer, format='png',bbox_inches='tight', pad_inches=0)
# image_png = base64.b64encode(buffer.getvalue()).decode()
# buffer.close()
# sideImage = f"data:image/png;base64,{image_png}"
#!-----------------------------------------------------------------------------

register_page(__name__, path='/map')
# data.setup()
# shapes = {
#     "X": "cross",
#     "FRI": "triangle",
#     "FRII": "triangle",
#     "BENT": "square",
#     "RING": "rhomb",
#     "OTHER": "plus",
#     "COMPACT": "circle",
    
# }


layout = html.Div([
    html.Div(
        style={
            "display": "flex",
            "justifyContent": "space-around",
            "flexDirection": "row",
            "paddingBottom": "15px",
        },
        children=[
            dcc.Dropdown(
                id=ids.cats,
                style={
                    "width": "100%", 
                    "color": "black",
                    "backgroundColor": "black",
                    'height': '100%',
                },
                multi=True,
                placeholder="Select Catalogs",
            ),
            html.Button(
                id = ids.toggle_cats,
                style = {
                    'width': '370px',
                    'height':'40px',
                    'marginLeft':'2%',
                    'marginRight':'2%',
                },
                children=["Toggle all catalogs"]
            ),
            dcc.Dropdown(
                id=ids.types,
                style={
                    "width": "100%", 
                    "color": "black",
                    "backgroundColor": "black",
                    'height': '100%',
                },
                multi=True,
                placeholder="Select Types",
            ),
            html.Button(
                id = ids.toggle_types,
                style = {
                    'width': '370px',
                    'height':'40px',
                    'marginLeft':'2%',
                    'marginRight':'2%',
                },
                children=["Toggle all types"]
            ),
        ]
    ),
    html.Div(
        style={
            'width': '100%', 
            'height': '80vh',
            'display': 'flex',
            'direction': 'row',
            'justifyContent': 'space-evenly',
        },
        children=[
            dal.DashAladinLite(
                id = ids.map,
                target="83.82 -5.39",
                fov=3,
                style={
                    'height': '100%',
                    'width': '100%'
                },
                options={
                    "showFullscreebControl": True,
                    "allowFullZoomout": True,
                    "showReticle": False,
                },
            ),
            html.Div(
                style={
                    'borderColor': 'black',
                    'borderStyle': 'solid',
                    'height': '100%',
                    'width': '60%',
                    'marginLeft': '2%',
                },
                children=[
                    html.Div(
                        style={
                            'height':"65%",
                        },
                        children=[dcc.Loading(
                        type="cube",
                        children=[
                            html.Center(
                                children=[html.Img(id=ids.radio_image, style={
                                    'width': '80%',
                                    'marginTop': '1%',
                                    'borderRadius': '25px',
                                    'border': '1px solid black',
                                }), dcc.Store(id=ids.radio_store)]
                            ),
                            html.H6(
                                id=ids.img_target,
                                style={'margin': '10px auto auto auto', 'width': '100%', 'textAlign': 'center'},
                            ), 
                        ]
                    )]),
                    html.Div(
                        style={
                            'height':"37%", 
                            'justifyContent': 'space-evenly',
                        },
                        children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "row",
                                "justifyContent": "space-around",
                                'margin': '0 0 5% 0',
                            },
                            children=[
                                dcc.Dropdown(
                                    id=ids.survey,
                                    style={
                                        'backgroundColor': 'black',
                                        'color': 'black',
                                        'width': '10vw',
                                    },
                                    placeholder="Surveys",
                                    options=data.get_surveys(),
                                    value = "NVSS",
                                    clearable=False,
                                ),
                                dcc.Checklist(
                                    id=ids.auto_fov,
                                    options={'true': "Auto Scale"}
                                ),
                            ]
                        ),
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "row",
                                "justifyContent": "space-around",
                                "margin": "0 0 2% 3%"
                            },
                            children=[
                                html.H6("Pixels(nxn)"),
                                html.Div([dcc.Slider(50,2000,1,
                                    id=ids.pixels,
                                    value = 300,
                                    marks={300:"Standard"},
                                    tooltip={"placement": "bottom"}
                                )],
                                    style={'width': '100%'},
                                )
                            ]
                        ),
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "row",
                                "justifyContent": "space-around",
                                "margin": "0 0 2% 3%"
                            },
                            children=[
                                html.H6("Contrast"),
                                html.Div([dcc.Slider(0,5,0.2,
                                    id=ids.contrast,
                                    value = 1,
                                    marks=None,
                                    tooltip={"placement": "bottom"},
                                    updatemode='drag',
                                )],
                                    style={'width': '100%', 'marginLeft': '13px'},
                                )
                            ]
                        ),
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "row",
                                "justifyContent": "space-around",
                                "margin": "0 0 2% 3%"
                            },
                            children=[
                                html.H6(children=["Brightness"]),
                                html.Div([dcc.Slider(0,5,0.2,
                                    id=ids.brightness,
                                    value = 1,
                                    marks=None,
                                    tooltip={"placement": "bottom"},
                                    updatemode='drag',
                                )],
                                    style={'width': '100%'},
                                )
                            ]
                        ),
                        html.Center([html.Button(
                            id=ids.get_image,
                            style={
                                "margin": "0 0 2% 0"
                            },
                            children='Get Image',
                        )])
                    ])
                ]
            )
        ]
    ),
    dcc.Upload(
            id = ids.cat_upload,
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'marginTop': '10px'
            },
            children=html.Div([
                'Drag and Drop Your Own Catalog or ', 
                html.A('Select File')
            ])
        ),
        html.Div(
            hidden = True,
            id = ids.opts_upload,
            style={
                "display": "flex", 
                "flexDirection": "row", 
                "justifyContent": "space-around",
                "paddingBottom": "10px",
                "paddingTop": "10px",
            },
            children = [
                html.Div(children=[
                    html.H6("Which column is the RA(deg)?"),
                    dcc.Dropdown(
                        id=ids.rad_upload,
                        style={'color': 'black'},
                        placeholder="Select column"
                    )]
                ),
                html.Div(children=[
                    html.H6("Which column is the DEC(deg)?"),
                    dcc.Dropdown(
                        id=ids.decd_upload,
                        style={'color': 'black'},
                        placeholder="Select a column"
                    )]
                ),
                html.Div(children=[
                    html.H6("Which column is the Types?"),
                    dcc.Dropdown(
                        id=ids.type_upload,
                        style={'color': 'black'},
                        placeholder="Select a column"
                    )]
                ),
                html.Button(
                    id = ids.btn_upload,
                    style={'width': "8%", 'height':'5%', 'marginTop': 'auto', 'marginBottom': 'auto' },
                    children="Process",
                )
            ]    
        )
,
    dcc.Loading(id = ids.cat_dump),
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "row",
                "margin": "10px 40% 0 40%",
            },
            children=[
                html.Div(
                    style={
                        'width': '100%',
                    },
                    children=[
                        dcc.Dropdown(
                            id=ids.delete_cats,
                            placeholder="Select Catalog To Delete",
                            style={
                                'backgroundColor': 'black',
                                'color': 'black',
                            },
                            multi= True,
                        )
                    ] 
                ),
                html.Button(
                    style={
                        'marginLeft': '5%',
                    },
                    id = ids.delete_btn,
                    children="Delete",
                    hidden=True,
                ),
            ]
        ),
        # dcc.Loading(
        #     fullscreen=True,            
        #     children=[html.Div(id="dummy")]
        # )
    ]
)     
  
  
@callback(
    Output(ids.types, "options"),
    Input(ids.btn_upload, "n_clicks"),
    Input(ids.delete_btn, "n_clicks"),
)
def update_type_option(upload, delete):
    return sorted(current_user.df["Type"].unique().tolist())


@callback(
    # Output("delete_catalogs", "options"),
    # Output("delete_button", "hidden"),
    # Input("delete_button", "n_clicks"),
    # Input("delete_catalogs", "value"),
    # Input('upload-button', 'n_clicks'),
    # State("delete_catalogs", "options"),
    # State('upload_rad', 'value'),
    # State('upload_decd', 'value'),
    # State('upload_type', 'value'),
    # State('upload_new_catalog', 'contents'),
    # State('upload_new_catalog', 'filename'),
    Output(ids.delete_cats, "options"),
    Output(ids.delete_btn, "hidden"),
    Input(ids.delete_btn, "n_clicks"),
    Input(ids.delete_cats, "value"),
    Input(ids.btn_upload, "n_clicks"),
    State(ids.delete_cats, "options"),
    State(ids.rad_upload, "value"),
    State(ids.decd_upload, "value"),
    State(ids.type_upload, "value"),
    State(ids.cat_upload, "contents"),
    State(ids.cat_upload, "filename"),
)
def remove_catalog(delete_btn, rm_cats, upload, delete_options, ra, dec, types, contents, filename):
    '''
    This function is used to update the options for the dropdown when selecting a user-uploaded catalog to delete.
    '''
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered == ids.delete_cats:
        if rm_cats is None or rm_cats == []:
            return delete_options, True
        return delete_options, False
    
    elif triggered == ids.btn_upload:
        # Check to see if all the required fields were selected
        if None in [ra, dec, types, contents, filename]:
            return delete_options, True
        delete_options.append(filename.split(".")[0])
        return delete_options, False
    
    options = []
    
    for file in os.listdir(f"src/database/{current_user.id}"):
        if 'fits' in file:
            name = file.split(".")[0]
            if rm_cats is None:
                rm_cats = []
            if name not in rm_cats:
                options.append(name)

    return options, True


@callback(
    # Output("radio_image", "children"),
    # Output("radio_image_target", "children"),
    # Input("get_image", "n_clicks"),
    # Input('contrast', 'value'),
    # Input('brightness', 'value'),
    # Input("aladin-lite", "objectClicked"),
    # State("radio_image", "children"),
    # State("radio_image_target", "children"),
    # State("survey", "value"),
    # State("pixels", "value"),
    # State("aladin-lite", "position"),
    # State("check_fov", "value"),
    # State("aladin-lite", "fov"),
    Output(ids.radio_image, "src"),
    Output(ids.img_target, "children"),
    Output(ids.radio_store, "data"),
    Input(ids.get_image, "n_clicks"),
    Input(ids.contrast, 'value'),
    Input(ids.brightness, 'value'),
    Input(ids.map, "objectClicked"),
    State(ids.radio_image, "src"),
    State(ids.img_target, "children"),
    State(ids.survey, "value"),
    State(ids.pixels, "value"),
    State(ids.map, "position"),
    State(ids.auto_fov, "value"),
    State(ids.map, "fov"),
)
def update_radio_image( btn, cont, bright, selected_source, current_image, current_target, survey, pixels, target, autoFocus, fov):
    '''
    This function is used to update the radio wave image on the right hand side of the page.
    '''
    coming_from = ctx.triggered[0]["prop_id"].split(".")[0]
    if current_image is None:
        ra = 83.82
        dec = -5.39
    
    elif coming_from == ids.get_image:
        ra = target["ra"]
        dec = target["dec"]

    elif selected_source is None:
        return current_image, current_target, no_update
    
    else:
        ra = selected_source["ra"]
        dec = selected_source["dec"]
        autoFocus = None

    if autoFocus == None or autoFocus == []:
        auto = False
    else:
        auto = True

    sideImage = data.getImage(ra, dec, pixels, fov, survey, auto)
    if sideImage is None:
        return "assets/missing.png", "Try Another Survey or Location", "assets/missing.png"
    
    if cont == None:
        cont = 1
    if bright == None:
        bright = 1
    if cont != 1 or bright != 1:
        image_bytes = base64.b64decode(sideImage.split(',')[1])
        im = Image.open(io.BytesIO(image_bytes))
        im = ImageEnhance.Brightness(im).enhance(bright)
        im = ImageEnhance.Contrast(im).enhance(cont)
    else:
        im = sideImage

    coords = SkyCoord(ra, dec, unit="deg")
    
    return im, coords.to_string('hmsdms'), sideImage

@callback(
    Output(ids.radio_image, "src", allow_duplicate=True),
    Input(ids.contrast, "value"),
    Input(ids.brightness, "value"),
    State(ids.radio_store, "data"),
    prevent_initial_call=True
)
def radio_image_filter(contrast, brightness, image_data):
    '''Apply filter to radio image'''
    if contrast == None:
        contrast = 1
    if brightness == None:
        brightness = 1
    if image_data is None or image_data == "assets/missing.png":
        raise exceptions.PreventUpdate
    image_bytes = base64.b64decode(image_data.split(',')[1])
    im = Image.open(io.BytesIO(image_bytes))
    im = ImageEnhance.Brightness(im).enhance(brightness)
    im = ImageEnhance.Contrast(im).enhance(contrast)
    return im
    
    
@callback(
    Output(ids.cats, "value"),
    Input(ids.toggle_cats, "n_clicks"),
)
def select_all_catalogs(btn):
    if btn is None or btn % 2 == 0:
        return []
    return sorted(current_user.df["Catalog"].unique())


@callback(
    Output(ids.types, "value"),
    Input(ids.toggle_types, "n_clicks"),
)
def select_all_types(btn):
    if btn is None or btn % 2 == 0:
        return []
    return sorted(current_user.df["Type"].unique())


@callback(
    # Output('aladin-lite', 'layers'),
    # Output('catalogs', 'options'),
    # Input("all-catalogs-btn", "n_clicks"),
    # Input('catalogs', 'value'),
    # Input('types', 'value'),
    # Input('upload-button', 'n_clicks'),
    # Input("delete_button", "n_clicks"),
    # State('delete_catalogs', 'value'),
    # State('aladin-lite', 'layers'),
    # State('upload_rad', 'value'),
    # State('upload_decd', 'value'),
    # State('upload_type', 'value'),
    # State('upload_new_catalog', 'contents'),
    # State('upload_new_catalog', 'filename'),
    Output(ids.map, 'layers'),
    Output(ids.cats, 'options'),
    Input(ids.toggle_cats, "n_clicks"),
    Input(ids.cats, 'value'),
    Input(ids.types, 'value'),
    Input(ids.btn_upload, 'n_clicks'),
    Input(ids.delete_btn, 'n_clicks'),
    State(ids.delete_cats, 'value'),
    State(ids.map, 'layers'),
    State(ids.rad_upload, 'value'),
    State(ids.decd_upload, 'value'),
    State(ids.type_upload, 'value'),
    State(ids.cat_upload, 'contents'),
    State(ids.cat_upload, 'filename'),
)
def update_layers(all_cat_btn, cats, typs, upload, delete_btn, cats_rm, layers, rad, decd, typ, content, filename):  
    '''
    This function controls the Aladin Lite layers. It is used to add and remove layers from the map.
    '''
    options = sorted(current_user.df["Catalog"].unique().tolist())
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    print(options, cats)

    if triggered == ids.btn_upload:
        print("RESET LAYERS")
        if None in [upload, rad, decd, typ, content, filename]:
            print("was empty")
            return layers, options
        content_string = content.split(',')[1]

        try:
            decodedThe64 = base64.b64decode(content_string)
            if "csv" in filename:

                filecontent = decodedThe64.decode('utf-8')
                if ";" in filecontent:
                    delimiter = ";"
                else:
                    delimiter = ","
                df = pd.read_csv(io.StringIO(filecontent), delimiter=delimiter)
                
            else:
                print("FILE NOT A CSV")
                return layers, options
            
        except Exception as e:
            print(e)
            return layers, options
        
        data.newCatalog(df, filename.split(".")[0], rad, decd, typ)
        layers = None
        options = sorted(current_user.df["Catalog"].unique().tolist())
        data.analyse_new_catalog(filename.split(".")[0])

    elif triggered == ids.toggle_cats:
        if all_cat_btn % 2 == 1:
            print("ALL CATALOGS")
            cats = options
        else:
            print("NO CATALOGS")
            cats = []

    elif triggered == ids.delete_btn:
        print("DELETE LAYERS")
        layers = None
        data.rmCatalog(cats_rm)
        options = sorted(current_user.df["Catalog"].unique().tolist())
    
    catalogs = current_user.df.rename(columns={"RA/deg":"ra", "DEC/deg":"dec"})
    colours = sns.color_palette("Spectral",len(set(catalogs["Catalog"]))).as_hex()
    shapes = {
        "X": "cross",
        "FRI": "triangle",
        "FRII": "triangle",
        "BENT": "square",
        "RING": "rhomb",
        "OTHER": "plus",
        "COMPACT": "circle",
        
    }

    if layers is None:
        layers = []

        for i, cat in enumerate(catalogs["Catalog"].unique()):
            catalog = catalogs[catalogs["Catalog"] == cat]
            
            for typ in catalog["Type"].unique():
                catalog_specific = catalog[catalog["Type"] == typ]
                layers.append({
                    "type": "catalog",
                    "data": catalog_specific.to_dict('records'),
                    "options": {
                        "color": colours[i],
                        "name": str(cat+"-"+typ),
                        "show": False,
                        "onClick": "showPopup",
                        "shape": shapes[typ],
                    }
                })
    else:
        if typs is None:
            typs = []
        for layer in layers:
            catalog, morph = layer["options"]["name"].split("-")

            if catalog in cats and morph in typs:
                layer["options"]["show"] = True
            else:
                layer["options"]["show"] = False

    return layers, options


@callback(
    # Output('upload_rad', 'options'),
    # Output('upload_decd', 'options'),
    # Output('upload_type', 'options'),
    # Output('upload_output', 'children'),
    # Output('upload_options', 'hidden'),
    # Output('upload_new_catalog', 'contents'),
    # Input('upload_new_catalog', 'contents'),
    # Input('upload_new_catalog', 'filename'),
    # Input('upload-button', 'n_clicks'),
    # State('upload_new_catalog', 'last_modified'),
    # State('upload_options', 'hidden'),
    Output(ids.rad_upload, 'options'),
    Output(ids.decd_upload, 'options'),
    Output(ids.type_upload, 'options'),
    Output(ids.cat_dump, 'children'),
    Output(ids.opts_upload, 'hidden'),
    Output(ids.cat_upload, 'contents'),
    Input(ids.cat_upload, 'contents'),
    Input(ids.cat_upload, 'filename'),
    Input(ids.btn_upload, 'n_clicks'),
    State(ids.cat_upload, 'last_modified'),
    State(ids.opts_upload, 'hidden'),
    prevent_initial_call=True,
)
def new_catalog_columns(contents, filename, upload, date, is_hidden):
    '''
    This function is used to update the options for the dropdown when selecting a user-uploaded catalog to delete - including 
    the RA, DEC and Type columns.
    '''
    if contents is None:
        return [], [], [], [], True, None
    content_string = contents.split(',')[1]

    try:
        decodedThe64 = base64.b64decode(content_string)
        if "csv" in filename:

            filecontent = decodedThe64.decode('utf-8')
            if ";" in filecontent:
                delimiter = ";"
            else:
                delimiter = ","
            df = pd.read_csv(io.StringIO(filecontent), delimiter=delimiter)

        else:
            print("FILE NOT A CSV")
            return [], [], [], [], True, None
    except Exception as e:
        print(e)
        return

    if not is_hidden and ctx.triggered[0]["prop_id"] == str(ids.btn_upload+".n_clicks"):
        return [], [], [], [], True, None

    options = df.columns.tolist()
    output_div = html.Div(
        style = {'color': 'black', 'backgroundColor': 'white'},
        children = [
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(date)),
            dash_table.DataTable(
                df.to_dict('records'),
                [{'name': i, 'id': i} for i in df.columns]
            ),
        ]
    )
    return options , options, options, output_div, False, contents


@callback(
    # Output("pixels", "disabled"),
    # Input("check_fov", "value"),
    Output(ids.pixels, "disabled"),
    Input(ids.auto_fov, "value"),
)
def disable_pixel(auto):
    if auto == None or auto == []:
        print("auto is off")
        return False
    print("auto is on")
    return True