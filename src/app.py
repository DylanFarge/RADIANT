from dash import Dash, page_container, html, dcc, Output, Input, State, page_registry, exceptions, DiskcacheManager
import diskcache
from dash_bootstrap_components.themes import DARKLY
from flask import Flask, redirect
from flask_login import LoginManager, current_user, logout_user
from assets.user import User, Credentials
from sys import argv
from assets.html_ids import UniversalIDs as ids
import dash_bootstrap_components as dbc

# Exposing the Flask Server to enable configuring it for logging in
server = Flask(__name__)

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

@server.route("/")
def index():
    return redirect("/map")

@server.route("/<path:path>")
def redir(path):
    if current_user.is_authenticated:
        if path in ["login", "signup"]:
            return redirect("/map")
        else:
            return app.index()
    else:
        if path in ["login", "signup"]:
            return app.index()
        else:
            return redirect("/login")

app = Dash(__name__, use_pages = True, server=server,
    title="RADIANT",
    update_title="Loading...",
    suppress_callback_exceptions=True,
    external_stylesheets=[DARKLY],
    external_scripts=["https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js"],
    background_callback_manager=background_callback_manager
)

app.layout = html.Div(
    id=ids.entire_page,
    style = {"padding": "2vh 10vw"},
    children=[
        html.Div(id=ids.header),
        page_container,
        dcc.Location(id=ids.url),
        dbc.Offcanvas(
            id = ids.sidebar,
            title="User Settings",
            is_open = False,
            placement='end',
            children=html.Center([
                html.Button(
                    "Logout",
                    id = ids.logout_btn,
                    style={
                        'width': "50%",
                        "border": '1px solid black',
                        "borderRadius": '20px',
                        "backgroundColor" : "#08F",
                        "margin":"1vh 1vw 1vh 1vw",
                    }
                ),
                html.Button(
                    'Delete Account',
                    id = ids.delete_btn,
                    style={
                        'width': "50%",
                        "border": '1px solid black',
                        "borderRadius": '20px',
                        "backgroundColor" : "#F55"
                    }
                )
            ])
        ),
        dcc.ConfirmDialog(
            id = ids.delete_alert,
            message = "You are about to delete your account and all data associated with it. This action cannot be undone. ARE YOU SURE?",
        )
    ]
)

# Updating the Flask Server configuration with Secret Key to encrypt the user session cookie
server.config.update(SECRET_KEY='secret-key')

# Login manager object will be used to login / logout users
login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = '/login'

@ login_manager.user_loader
def load_user(username):
    ''' get id from session,then retrieve user object from database with peewee query'''
    try:
        return User(username)
    except:
        print("******* User Files in Database are Corrupted or Missing *******")
    # returns None if the user does not exist

@app.callback(
    Output(ids.delete_alert, "displayed"),
    Input(ids.delete_btn, "n_clicks"),
    prevent_initial_call = True
)
def popup_alert(nclicks):
    return True

@app.callback(
    Output(ids.url, "pathname", allow_duplicate=True),
    Input(ids.delete_alert, "submit_n_clicks_timestamp"),
    prevent_initial_call=True
)
def delte_account(timestamp):
    Credentials.delete_account(current_user.id)
    logout_user()
    return "/login"

@app.callback(
    Output(ids.sidebar, "is_open"),
    Input(ids.sidebar_btn, "n_clicks"),
    State(ids.sidebar, "is_open"),
)
def toggle_sidebar(n_clicks, is_open):
    print("Open Side bar")
    if n_clicks:
        return not is_open
    return is_open

@app.callback(
    Output(ids.url,"pathname"),
    Input(ids.logout_btn,"n_clicks"),
    State(ids.url,"pathname"),
)
def logging(btn, url):
    if current_user.is_authenticated:
        if btn is not None:
            logout_user()
            print("User logged out")
            return "/login"
        
        print("User is logged in")

    elif url not in ["/login", "/signup"]:
        print("User is not logged in, you are being redirected.")
        return "/login"
    raise exceptions.PreventUpdate()
    
@app.callback(
    Output(ids.header,"children"),
    Output(ids.entire_page,"style"),
    Input(ids.url,"pathname"),
    State(ids.entire_page,"style")
)
def header(name, styl):
    if current_user.is_authenticated:
        if styl is None or styl == {}:
            styl = {"padding": "2vh 10vw"}
        return html.Div([
            html.Div(
                style={
                    "display": "flex", 
                    "flexDirection": "row", 
                    'alignItems': 'center'
                },
                children=[
                    html.Div([
                        dcc.Link(
                            page['name']+" | ", 
                            href=page['path'], 
                            style={'color':'#08F'}
                        ) for page in page_registry.values() if page['name'] not in ["Login", "Signup"]
                    ]),
                    html.Div(style={'marginLeft': 'auto'},children=["Reach For The Stars, "+current_user.id+"!"]),
                    html.Button("Options", id=ids.sidebar_btn, style={'marginLeft': '1vw'}),
                ]
            ),
            html.Hr(),
            html.Center(html.H1( (name[1:].title()))),
            html.Hr(),
        ]), styl
    return None, {}

if __name__ == "__main__":
    if argv[1] == "devTools":
        import processing.devTools as dt
        if len(argv) == 2:
            print('''>>>__Options__<<<
                  override_dataframe_build
                  override_analysis_build
                  create_fits <catalog name or "all">
                  remove_user <username>
                  list-users''')
            print("No arguments provided...stopping")

        else:
            option = argv[2]

            if option == "override_dataframe_build":
                dt.override_dataframe_build()

            elif option == "override_analysis_build":
                dt.override_analysis_build()
                
            elif option == "create_fits":
                dt.create_fits(argv[3:])

            elif option == "remove_user":
                dt.remove_user(argv[3:])

            elif option == "list-users":
                [print(user) for user in Credentials.creds.keys()]

            else:
                print("Don't know option",argv[2])
    else:
        app.run(debug=bool(int(argv[1])), port=8000, host="0.0.0.0")