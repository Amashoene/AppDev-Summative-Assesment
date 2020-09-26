import dash
import dash_core_components as dcc 	#This library will give us the dashboard elements (pie charts, scatter plots, bar graphs etc)
import dash_html_components as html 	#This library allows us to arrange the elements from dcc in a page as is done using html/css (how the internet generally does it)
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from MyWindModel import data1
from MySolarModel import df2_new

def get_dash(server):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, 
                    server=server,
                    routes_pathname_prefix='/dashapp/',
                    external_stylesheets=external_stylesheets
                    )

    #df = get_data()

    styles = get_styles()

    fig={'data': [{'x': data1['Date'], 'y': data1['PredictedPower_Wind'], 'type': 'bar', 'name': 'PredictedPower_Wind'},{'x': df2_new['Date'], 'y': df2_new['PredictedPower_Solar'], 'type': 'bar', 'name': 'PredictedPower_Solar'},
                    ],'layout': {'title': 'Solar Power Plant and Wind Farm Power Output'}}

    app.layout = html.Div([
        # html.H6("Change the value in the text box to see callbacks in action!"),
        html.A("Go to Home Page", href="/", style=styles["button_styles"]),
        html.Div("Display of the expected power output for the Solar Power Plant and Wind Farm.", id='my-output',
                 style=styles["text_styles"]),
        html.Div(
            dcc.Graph(
                id='example-graph',
                figure=fig
            ),
            style=styles["fig_style"]
        )
    ])

    return app

def get_styles():
    """
    Very good for making the thing beautiful.
    """
    base_styles = {
        "text-align": "center",
        "border": "1px solid #ddd",
        "padding": "7px",
        "border-radius": "2px",
    }
    text_styles = {
        "background-color": "#eee",
        "margin": "auto",
        "width": "50%"
    }
    text_styles.update(base_styles)

    button_styles = {
        "text-decoration": "none",
    }
    button_styles.update(base_styles)

    fig_style = {
        "padding": "10px",
        "width": "80%",
        "margin": "auto",
        "margin-top": "5px"
    }
    fig_style.update(base_styles)
    return {
        "text_styles" : text_styles,
        "base_styles" : base_styles,
        "button_styles" : button_styles,
        "fig_style": fig_style,
    }

