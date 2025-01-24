import dash
import os, shutil
from dash import html, Output, Input, State, dcc, exceptions
from flask_login import login_user
from assets.user import User, Credentials
from assets.html_ids import SignupIDs as ids
from assets.status_codes import Database as c
import pandas as pd

dash.register_page(__name__, path="/signup")



layout = html.Center(
    style={
        "justifyContent": "center",
        "alignItems": "center",
        "display": "flex",
        "width": "100%", 
        "height": "98vh",
        "background": "linear-gradient(90deg, rgba(0,0,20,1) 0%, rgba(0,100,100,1) 25%,rgba(255,255,255,1) 120%)",
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
                        "backgroundImage": 'url("/assets/images/signup.jpeg")',
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
                        html.H1("Signup", style={"fontWeight": "bold", 
                                    "color": "black",
                                    "fontFamily": "Times New Roman"}),
                        html.Hr(
                            style={
                                "width": "80%",
                                "border": "1px solid black",
                            }
                        ),
                        
                        dcc.Input(
                            id= ids.username_field,
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

                        dcc.Input(
                            id=ids.confirm_password_field,
                            style={
                                "borderRadius": "1vh",
                                "border": "1px solid black",
                                "width": "50%",
                                "padding": "1vh",
                                "backgroundColor": "#EEF",
                                "color": "black",
                            },
                            placeholder="Confirm Password",
                            type="password",
                            required=True,
                        ),

                        html.Button(
                            id=ids.signup_btn,
                            style={
                                "borderRadius": "1vh",
                                "border": "1px solid black",
                                "width": "20%",
                                "padding": "0.5vh",
                                "backgroundColor": "white",
                                "color": "black",
                            },
                            children="Signup",
                        ),
                        dcc.Location(id=ids.signup_redirect),

                        dcc.Link("Already have an account? Login!", href="/login",style={"color":"rgba(0,100,100,1)"}),

                        html.Div(id=ids.status),
# ------------------------------------------------------
                    ])]
                )
            ]
        ),
    ]
)

@dash.callback(
    Output(ids.signup_redirect, "pathname"),
    Output(ids.password_field, "value"),
    Output(ids.confirm_password_field, "value"),
    Output(ids.username_field, "value"),
    Output(ids.status, "children"),
    Input(ids.signup_btn, "n_clicks"),
    State(ids.username_field, "value"),
    State(ids.confirm_password_field, "value"),
    State(ids.password_field, "value"),
)
def signup(n_clicks, username, confirm_password, password):
    if n_clicks is None:
        raise exceptions.PreventUpdate
    
    if password != confirm_password:
        print("Passwords do not match")
        return dash.no_update, "", "", username, html.P("Passwords do not match", style={"color": "red"})
    
    attempt = Credentials.user_not_found(username, password)

    if username.isalnum() and attempt == c.USERNAME_NOT_FOUND:
        Credentials.create_user(username, password)
        user = User(username)
        login_user(user)
        return "/map", "", "", "", html.P("User created", style={"color": "green"})
    else:
        print("User already exists")
        return dash.no_update, "", "", "", html.P("User already exists", style={"color": "red"})