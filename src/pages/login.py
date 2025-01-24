import dash
from dash import html, Output, Input, State, dcc, exceptions, callback
from flask_login import login_user
from assets.user import User, Credentials
from assets.html_ids import LoginIDs as ids
from assets.status_codes import Database as c

dash.register_page(__name__, path="/login")



layout = html.Center(
    style={
        "justifyContent": "center",
        "alignItems": "center",
        "display": "flex",
        "width": "100%", 
        "height": "98vh",
        "background": "linear-gradient(90deg, rgba(0,0,20,1) 0%, rgba(0,0,75,1) 25%,rgba(255,255,255,1) 120%)",
    },
    children=[
        html.Div(
            style={
                "borderRadius": "2vh",
                'border': '1px solid black',
                "width": "90%",
                "height": "80%",
                "display": "flex",
                "flexDirection": "row",
            },
            children = [
                html.Div(
                    style={
                        "width": "100%",
                        "height": "100%",
                        "borderRadius": "2vh 0 0 2vh",
                        "backgroundImage": 'url("/assets/images/login.jpeg")',
                        "backgroundPosition": "center",
                        "backgroundSize": "107vh 100vh",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                    },
                    children =[
                        html.H1(
                            style={
                                "color": "white",
                                "fontSize": "6vw",
                                "fontWeight": "bold",
                                "fontFamily": "Times New Roman",
                                "textShadow": "2px 2px 4px #000000",
                                "letterSpacing": "0.3em",
                            },
                            children="RADIANT"
                        )
                    ]
                ),
                html.Div(
                    style={
                        "width": "70%",
                        "borderRadius": "0 2vh 2vh 0",
                        "backgroundColor": "white",
                        "display": "flex",
                        "flexDirection": "column",
                        "alignItems": "center",
                        "justifyContent": "center",
                    },
                    children=[html.Div(
                        style={
                            "width": "90%",
                            "height": "70%",
                            "display": "flex",
                            "flexDirection": "column",
                            "alignItems": "center",
                            "justifyContent": "space-evenly",
                        },
                        children=[
# ------------------------------------------------------
                        html.H1("Login", 
                                style={
                                    "fontWeight": "bold", 
                                    "color": "black",
                                    "fontFamily": "Times New Roman"
                                }
                            ),
                        html.Hr(
                            style={
                                "width": "80%",
                                "border": "1px solid black",
                            }
                        ),
                        
                        dcc.Input(
                            id=ids.username_field,
                            style={
                                "borderRadius": "1vh",
                                "border": "1px solid black",
                                "width": "50%",
                                "padding": "1vh",
                                "backgroundColor": "#EEF",
                                "color": "black",
                            },
                            placeholder="Username",
                            type="text",
                            required=True,
                        ),
                        dcc.Store(id=ids.username_cache, storage_type="local"),

                        dcc.Input(
                            id=ids.password_field, 
                            style={
                                "borderRadius": "1vh",
                                "border": "1px solid black",
                                "width": "50%",
                                "padding": "1vh",
                                "backgroundColor": "#EEF",
                                "color": "black",
                            },
                            placeholder="Password",
                            type="password",
                            required=True,
                        ),
                        dcc.Store(id=ids.password_cache, storage_type="local"),

                        dcc.Checklist(
                            id=ids.remember_me,
                            style={
                                "color": "black",
                                "fontFamily": "Arial",
                            },
                            options=[
                                {'label': ' Remember me', 'value': True}
                            ],
                            value=[True],
                        ),

                        html.Button(
                            id=ids.login_btn,
                            style={
                                "borderRadius": "1vh",
                                "border": "1px solid black",
                                "width": "20%",
                                "padding": "0.5vh",
                                "backgroundColor": "white",
                                "color": "black",
                            },
                            children="Login",
                        ),
                        dcc.Location(id=ids.login_redirect),

                        dcc.Link("Don't have an account? Signup!", href="/signup", style={"color":"rgba(0,0,100,1)"}),
                        html.Div(id=ids.status),
                        html.Div(id="dummy"),
# ------------------------------------------------------
                    ])]
                )
            ]
        ),
    ]
)

for cred in ["username", "password"]:
    # Get credentials from local storage
    dash.clientside_callback(
        f"""
        function(dummy, {cred}) {{
            console.log("Getting {cred} " + {cred})
            if ({cred} === undefined) {{
                return "";
            }}
            return {cred};
        }}
        """,
        Output(f"{cred}-field", "value"),
        Input("dummy", "children"), # Arbitrary
        State(f"{cred}-cache", "data"),
    )

    # Store credentials in local storage
    dash.clientside_callback(
        f"""
        function(n_clicks, {cred}, {cred}store, remember) {{
            if (n_clicks === undefined) {{
                
                return {cred}store;
            }}
            if (remember.length == 0) {{
                console.log("Remember me not checked")
                return null;
            }}
            console.log("Storing {cred} " + {cred})
            return {cred};
        }}
        """,
        Output(f"{cred}-cache", "data"),
        Input(ids.login_btn, "n_clicks"),
        State(f"{cred}-field", "value"),
        State(f"{cred}-cache", "data"),
        State(ids.remember_me, "value"),
    )

@callback(
    Output(ids.login_redirect, "pathname"),
    Output(ids.status, "children"),
    Input(ids.login_btn, "n_clicks"),
    State(ids.username_field, 'value'), 
    State(ids.password_field, 'value'),
    prevent_initial_call = True
)
def login(n_clicks, username, password):
    if n_clicks is None:
        raise exceptions.PreventUpdate
    
    attempt = Credentials.user_not_found(username, password)

    if attempt == c.SUCCESS:
        user = User(username)
        login_user(user) # store user id in session
        return "/", html.P("Successful", style={"color": "green"})
    elif attempt == c.USERNAME_NOT_FOUND:
        msg = "Could not find username. Please try again or signup."
        print(msg)
        return None, html.P(msg, style={"color": "red"})
    else:
        msg = "Incorrect username or password. Please try again."
        print(msg)
        return None, html.P(msg, style={"color": "red"})