# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


from dash import Dash, dcc, html, Input, Output,callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
import os

app = Dash(__name__)
server = app.server

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

def falling_pattern_filtering_fallingtype(df, target_fall, target_col):
    target_fall = target_fall
    target_array = np.array(df[target_col])
    target_array = target_array.astype(str)

    # Check if "_11" is present in each element
    # Check if any element contains any string in target_fall list
    result = [any(substring in element for substring in target_fall) for element in target_array]
    #result = np.core.defchararray.find(target_array, target_fall) != -1
    return df[result]

current_path = os.path.dirname(os.path.realpath(__file__))
#all_df = pd.read_csv(current_path + "/Asset/all_df.csv")
all_df = pd.read_csv("https://raw.githubusercontent.com/BerryLeeYY/crash-data/main/all_df.csv")




variable_list = ["change_time","falling_velocity", "bouncing_velocity", "change_velocity", "acceleration", "energy_change"]
axis_list = ["combine", "x", "y", "z"]
segment_list = ['T8', 'Right Upper Arm', 'Left Upper Arm', 'Right Forearm',
       'Left Forearm', 'Pelvis', 'Right Upper Leg', 'Left Upper Leg',
       'Neck', 'Right Shoulder', 'Left Shoulder', 'T12']
falling_scenarios_list = ['front', 'right', 'left', 'back']
height_list = ['11 cm', '18 cm', '25 cm']

app.layout = html.Div(children=[
    html.H1(children='EVOC & UM bike-crash measurement dashboard'),

    html.Div
    ([
        html.Div
        ("  ", style={'width': '5%'}),

        html.Div
        ([
            html.H4(children='Variable'),
            dcc.Dropdown(
                        variable_list,
                        "change_velocity", 
                        id='variable_dropdown'
                        )
        ], style={'width': '12%'}),
        html.Div
        ("  ", style={'width': '1%'}),

        html.Div
        ([
            html.H4(children='Axis'),
            dcc.Dropdown(
                        axis_list,
                        'combine', 
                        id='axis_dropdown'
                        )
        ], style={'width': '12%'}),
        html.Div
        ("  ", style={'width': '1%'}),

        html.Div
        ([
            html.H4(children='Segment'),
            dcc.Dropdown(
                        segment_list,
                        'T8', 
                        id='segment_dropdown'
                        )
        ], style={'width': '12%'}),
        html.Div
        ("  ", style={'width': '5%'}),

        html.Div
        ([
            html.H4(children='Fall scenarios'),
            dcc.Checklist(
                        falling_scenarios_list,
                        value =['front'], 
                        id='fall_scenarios_checklist',
                        inline=True,
                        style={ 'fontSize': 18, 'padding': '5px', 'width': '100%'}
                        )
        ], style={'width': '20%'}),
        html.Div
        ("  ", style={'width': '1%'}),

        html.Div
        ([
            html.H4(children='Fall height'),
            dcc.Checklist(
                        height_list, 
                        value = ['25 cm'],
                        id='height_checklist',
                        inline=True,
                        style={ 'fontSize': 18, 'padding': '5px', 'width': '100%'}
                        )
        ], style={'width': '30%'}),
        html.Div
        ("  ", style={'width': '1%'}),

        
        ],  style = {'display': 'flex'}
    ),

    html.Div
    ([
        dcc.Graph(
            id='bar_graph'
        ),

        dcc.Graph(
            id='scatter_graph'
        )
    ],  style = {'display': 'flex'})
])

@callback(
    Output('bar_graph', 'figure'),
    Output('scatter_graph', 'figure'),
    Input('variable_dropdown', 'value'),
    Input('axis_dropdown', 'value'),
    Input('segment_dropdown', 'value'),
    Input('fall_scenarios_checklist', 'value'),
    Input('height_checklist', 'value')
)
def update_bar(variable_value, axis_value, segment_value, fall_scenarios_value, height_value):
    if variable_value == 'change_time':
        column = 'change_time'
    else:
        column = variable_value + '_' + axis_value
    if len(fall_scenarios_value) == 0 :
        fall_scenarios_value = falling_scenarios_list
    if len(height_value) == 0:
        height_value = height_list
    target_segment_df = all_df[all_df["segment"] == segment_value]
    target_segment_scenarios_df = falling_pattern_filtering_fallingtype(df = target_segment_df, target_fall = fall_scenarios_value, target_col = 'direction')
    target_segment_scenarios_height_df = falling_pattern_filtering_fallingtype(df = target_segment_scenarios_df, target_fall = height_value, target_col = 'height')
    yes_df = target_segment_scenarios_height_df[target_segment_scenarios_height_df["airbag_condition"] == "with"]
    no_df = target_segment_scenarios_height_df[target_segment_scenarios_height_df["airbag_condition"] == "without"]
    t_statistic, p_value = stats.ttest_rel(no_df[column], yes_df[column])
    

    target_segment_scenarios_height_with_df = target_segment_scenarios_height_df[target_segment_scenarios_height_df["airbag_condition"]=="with"]
    average_values_with = target_segment_scenarios_height_with_df.groupby('height')[column].mean().reset_index()
    std_values_with = target_segment_scenarios_height_with_df.groupby('height')[column].std().reset_index()

    target_segment_scenarios_height_without_df = target_segment_scenarios_height_df[target_segment_scenarios_height_df["airbag_condition"]=="without"]
    average_values_without = target_segment_scenarios_height_without_df.groupby('height')[column].mean().reset_index()
    std_values_without = target_segment_scenarios_height_without_df.groupby('height')[column].std().reset_index()

    target_bar_fig = go.Figure()
    target_bar_fig.add_trace(go.Bar(
        name='With',
        x= average_values_with["height"], y= average_values_with[column],
        error_y=dict(type='data', array=std_values_with[column])
    ))
    target_bar_fig.add_trace(go.Bar(
        name='Without',
        x= average_values_without["height"], y= average_values_without[column],
        error_y=dict(type='data', array=std_values_without[column])
    ))
    
    target_bar_fig.update_layout(barmode='group', title = "{} p-value: {}".format(segment_value, p_value), title_x=0.5, title_y=0.85, yaxis_title = "aver_{}".format(column))
    target_scatter_fig = px.scatter(target_segment_scenarios_height_df, x="falling_velocity_combine", y=column, color = "airbag_condition", trendline='ols')
    
    
    return [target_bar_fig, target_scatter_fig]


if __name__ == '__main__':
    app.run(debug=True)
