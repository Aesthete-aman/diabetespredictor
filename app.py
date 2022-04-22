#!/usr/bin/env python
# coding: utf-8

# # CP - System Project (Heroku + Python Deployment File)

# <h3> Importing the Libraries </h3>

# In[14]:


#Importing the libraries
import pandas as pd
import numpy as np
import dash
import plotly.express as px

from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_daq as daq
import time


# <h3> Dash Frontend Code </h3>

# In[15]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
server = app.server
app.title = 'Diabetes Predictor'

margin_height = '50px'
size = 100

app.layout = html.Div(children=[
    
    #Header
    html.Div(style={'backgroundColor': 'blue'}, children=[
    html.H1(children='Diabetes Prediction',style={'margin': 'auto','text-align': 'center','color': 'white',"font-family": "Arial","height":'45px','padding': '10px 0'})]),

    #LHS Input Section
    html.Div(children=[
    html.Div(children=[
        html.Div(children=[
            html.Div(children=[html.H4(children="PARAMETERS")],style={"font-weight": "1500","color":"#ff0000","text-align":"left","height":margin_height}),
            
            #Input Section - Text
            
            html.Div(children=[html.H6(children='Age')],style={"font-weight": "900","color":"white","text-align":"left","height":margin_height}),
            html.Div(children=[html.H6(children='Glucose')],style={"font-weight": "900","color":"white","text-align":"left","height":margin_height}),
            html.Div(children=[html.H6(children='Insulin')],style={"font-weight": "900","color":"white","text-align":"left","height":'57px'}),
            html.Div(children=[html.H6(children='Blood Pressure')],style={"font-weight": "900","color":"white","text-align":"left","height":'57px'}),
            html.Div(children=[html.H6(children='BMI')],style={"font-weight": "900","color":"white","text-align":"left","height":'57px'}),
            html.Div(children=[html.H6(children='Pregnancies')],style={"font-weight": "900","color":"white","text-align":"left","height":'55px'}),
            html.Div(children=[html.H6(children='Skin Thickness')],style={"font-weight": "900","color":"white","text-align":"left","height":margin_height}),
            html.Div(children=[html.H6(children='DPF - Diabetes Pedigree Function')],style={"font-weight": "900","color":"white","text-align":"left","height":margin_height}),
        
                #Search Input for searching the Articles
                html.Br(),

                dcc.Loading(id="loading-1",type="circle",fullscreen=True,children=html.Div(id="loading-output-1"))],style={"margin-left":"1rem","text-align": "center","margin-right":"1rem"})],

                #CSS Styling for the LHS Division
                style={"width":"22.5%","height":'565px'}),
        
    
    #Button Section
    html.Div(children=[html.H4(children=" ")],style={'width':'5%'}),
        
    html.Div(children=[html.Br(),html.Br(),html.Br(),
                      daq.NumericInput(id='input_age',min=0,max=100,size=size),html.Br(),
                      daq.NumericInput(id='input_glucose',min=0,max=200,size=size),html.Br(),
                      daq.NumericInput(id='input_insulin',min=0,max=850,size=size),html.Br(),
                      daq.NumericInput(id='input_bp',min=0,max=130,size=size),html.Br(),
                      daq.NumericInput(id='input_bmi',min=0,max=100,size=size),html.Br(),
                      daq.NumericInput(id='input_pregnancy',min=0,max=100,size=size),html.Br(),
                      daq.NumericInput(id='input_st',min=0,max=100,size=size),html.Br(),
                      daq.NumericInput(id='input_dpf',min=0,max=100,size=size)
                      ],style={"width":"20%","color":"white"}),
    
    html.Div(children=[html.H4(children="  ")],style={'color':'white','width':'5%'}),
    
    #Output Section
    html.Div(children=[
        html.Div(html.H1(id='output_box')),
        dbc.Button('Enter', id='submit-button', n_clicks=0, outline=True, color="white") 
    ],style={"width":"47.5%","color":"white",'margin': 'auto','text-align': 'center'})],style={"display":"flex"})
    

],style={'backgroundColor': 'black'})
        

@app.callback([
dash.dependencies.Output("loading-1","children"),
dash.dependencies.Output("output_box","children")],
dash.dependencies.Input('submit-button', 'n_clicks'),
[dash.dependencies.State('input_age','value'),
dash.dependencies.State('input_glucose', 'value'),
dash.dependencies.State('input_insulin','value'),
dash.dependencies.State('input_bp','value'),
dash.dependencies.State('input_bmi','value'),
dash.dependencies.State('input_pregnancy','value'),
dash.dependencies.State('input_st','value'),
dash.dependencies.State('input_dpf','value')])


def first_output(n_clicks,age,glucose,insulin,bp,bmi,pregnancy,st,dpf):
    
    if n_clicks>0:
        
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn import svm        

        # loading the diabetes dataset to a pandas DataFrame
        diabetes_dataset = pd.read_csv('diabetes.csv')

        X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
        Y = diabetes_dataset['Outcome']

        scaler = StandardScaler()
        scaler.fit(X)

        standardized_data = scaler.transform(X)
        X = standardized_data
        Y = diabetes_dataset['Outcome']

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)

        classifier = svm.SVC(kernel = 'linear')

        # training the support vector machine classifier
        classifier.fit(X_train, Y_train)

        input_data = (age,glucose,insulin,bp,bmi,pregnancy,st,dpf)

        # changing the input_data to a numpy array
        input_data_as_numpy_array = np.asarray(input_data)

        # reshape the array as we are predicting for one instance 
        input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

        # standardize the input_data
        std_data = scaler.transform(input_data_reshape)

        prediction = classifier.predict(std_data)

        if (prediction[0] == 0):
          string1 = 'Non Diabetic'
        else:
          string1 = 'Diabetic'
        
        #Time delay to run animation
        time.sleep(2)
        
        return['',string1]
    

if __name__ == '__main__':
    app.run_server()


# In[ ]:




