#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the data
df = pd.read_csv('https://storage.googleapis.com/tsa_final_project/tsa_final_project_folder/ufo_dataset.csv')
ufo_df=df
ufo_df=ufo_df[(ufo_df['date_time'].isna()==False) & (ufo_df['country'].isna()==False)]
ufo_df['shape'].replace({'None':'unknown', 'other': 'unknown'})
ufo_df['shape'] = ufo_df['shape'].fillna(value='unknown')
ufo_df['date_time2']=ufo_df['date_time']
ufo_df.reset_index(inplace=True)
ufo_df.set_index('date_time', inplace=True)
ufo_df.drop(columns=['index'],inplace=True)
ufo_df['date'] = pd.to_datetime(ufo_df['date_time2']).dt.date
ufo_df['dayness'] = pd.to_datetime(ufo_df['date_time2']).dt.hour.between(7, 18, inclusive=True).replace({True: 'day', False: 'night'})
ufo_df['Month_Year']=pd.to_datetime(ufo_df['date_time2']).dt.to_period('M')
ufo_df['ymd'] =pd.to_datetime(ufo_df['date_time2']).astype("datetime64[M]")
df=ufo_df
# Create the app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1('UFO Sightings'),
    html.H2('Time Series Analysis - Final Project'),
    html.H3('Anas AL Omary'),
    
    dcc.DatePickerRange(
        id='date-range-picker',
        min_date_allowed=df['date'].min(),
        max_date_allowed=df['date'].max(),
        initial_visible_month=df['date'].max(),
        start_date=df['date'].min(),
        end_date=df['date'].max(),
        #display_format='MMM Do, YYYY'
    ),
    
    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': country, 'value': country} for country in df.groupby(['country']).size().reset_index(name='counts').sort_values(by='counts', ascending=False).head(15)['country']],
        value=None,
        placeholder='Select a country'
    ),
    
    dcc.Dropdown(
        id='state-dropdown',
        options=[{'label': state, 'value': state} for state in df.groupby(['state']).size().reset_index(name='counts').sort_values(by='counts', ascending=False).head(15)['state']],
        value=None,
        placeholder='Select a state'
    ),
    
    dcc.Dropdown(
        id='city-dropdown',
        options=[{'label': city, 'value': city} for city in df.groupby(['city']).size().reset_index(name='counts').sort_values(by='counts', ascending=False).head(30)['city']],
        value=None,
        placeholder='Select a city'
    ),
    
    html.Button(id='submit-button', n_clicks=0, children='Submit'),
    
    dcc.Graph(id='counts-graph'),
    
    dcc.Graph(id='timeseries-graph')
])

# Define the callback functions for the two graphs
@app.callback(
    [Output(component_id='counts-graph', component_property='figure'),
     Output(component_id='timeseries-graph', component_property='figure')],
    [Input(component_id='submit-button', component_property='n_clicks')],
    [State(component_id='date-range-picker', component_property='start_date'),
     State(component_id='date-range-picker', component_property='end_date'),
     State(component_id='country-dropdown', component_property='value'),
     State(component_id='state-dropdown', component_property='value'),
     State(component_id='city-dropdown', component_property='value')]
)
    
def update_graphs(n_clicks, start_date, end_date, country, state, city):
    # Filter the data based on user input
    filtered_df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
    if country and state and city:
        filtered_df = filtered_df[(filtered_df['country'] == country)&(filtered_df['state'] == state)&(filtered_df['city'] == city)]
    elif country and state:
        filtered_df = filtered_df[(filtered_df['country'] == country)&(filtered_df['state'] == state)]
    elif country and city:
        filtered_df = filtered_df[(filtered_df['country'] == country)&(filtered_df['city'] == city)]
    elif state and city:
        filtered_df = filtered_df[(filtered_df['state'] == state)&(filtered_df['city'] == city)]
    elif country:
        filtered_df = filtered_df[(filtered_df['country'] == country)]
    elif state:
        filtered_df = filtered_df[filtered_df['state'] == state]
    elif city:
        filtered_df = filtered_df[filtered_df['city'] == city]
        
    
    # Create the counts graph
    
    if country and state and city:
        counts_df = filtered_df.groupby(['city']).size().reset_index(name='counts').sort_values(by='counts', ascending=False).head(20)
        counts_fig = px.bar(counts_df, x='city', y='counts', color="city")
    elif country and state:
        counts_df = filtered_df.groupby(['city']).size().reset_index(name='counts').sort_values(by='counts', ascending=False).head(20)
        counts_fig = px.bar(counts_df, x='city', y='counts', color="city")
    elif country and city:
        counts_df = filtered_df.groupby(['city']).size().reset_index(name='counts').sort_values(by='counts', ascending=False).head(20)
        counts_fig = px.bar(counts_df, x='city', y='counts', color="city")
    elif state and city:
        counts_df = filtered_df.groupby(['city']).size().reset_index(name='counts').sort_values(by='counts', ascending=False).head(20)
        counts_fig = px.bar(counts_df, x='city', y='counts', color="city")
    elif country:
        counts_df = filtered_df.groupby(['state']).size().reset_index(name='counts').sort_values(by='counts', ascending=False).head(20)
        counts_fig = px.bar(counts_df, x='state', y='counts', color="state")
    elif state:
        counts_df = filtered_df.groupby(['city']).size().reset_index(name='counts').sort_values(by='counts', ascending=False).head(20)
        counts_fig = px.bar(counts_df, x='city', y='counts', color="city")
    elif city:
        counts_df = filtered_df.groupby(['city']).size().reset_index(name='counts').sort_values(by='counts', ascending=False).head(20)
        counts_fig = px.bar(counts_df, x='city', y='counts', color="city")
    else:
        counts_df = filtered_df.groupby(['country']).size().reset_index(name='counts').sort_values(by='counts', ascending=False).head(20)
        counts_fig = px.bar(counts_df, x='counts', y='country', color="country")
        
    
    
    # Create the timeseries graph
    filtered_df = df[(df['date'] >= pd.to_datetime('01/01/1990')) & (df['date'] <= pd.to_datetime('01/01/2020'))]
    timeseries_df = filtered_df.groupby(['ymd']).size().reset_index(name='count').sort_values(by='ymd', ascending=True)
    df2=timeseries_df
    loaded_model = load_model(r'gs://tsa_final_project/tsa_final_project_folder/my_lstm_model.h5', compile = False)
    #C:\Users\anaso\Google Drive\Indiana_Master_of_Data_Science\Year3_Spring\TimeSeriesAnalysis\FinalProject\Part3\my_lstm_model.h5
    df2.set_index('ymd',inplace=True)
    # Select the most recent 12 months of data
    input_data = df2[-12:]
    time_steps=12
    # Scale the input data using the same scaler used to train the model
    scaler = MinMaxScaler()
    scaled_input_data = scaler.fit_transform(input_data)
    # Reshape the input data to match the shape expected by the LSTM model
    reshaped_input_data = scaled_input_data.reshape((1, time_steps, 1))
    # Make a prediction using the loaded model
    predicted_value = loaded_model.predict(reshaped_input_data)
    # Rescale the predicted value to the original scale
    unscaled_predicted_value = scaler.inverse_transform(predicted_value)
    data = {'ymd': [df2.tail(1).index.values[0],(df2.tail(1).index+ pd.DateOffset(months=1)).values[0]],
        'count': [df2.tail(1)['count'].values[0],unscaled_predicted_value[0][0]]}
    dfpred = pd.DataFrame(data)
    timeseries_df.reset_index(inplace=True)
    timeseries_fig = px.line(timeseries_df, x='ymd', y='count', title='Time Series')
    timeseries_fig.add_scatter(x=dfpred['ymd'], y=dfpred['count'], mode='lines', name='Predicted', line=dict(color='red'))
    timeseries_fig.update_xaxes(rangeslider_visible=True)
    
    # Return the figures to update the graphs
    return counts_fig, timeseries_fig

if __name__ == '__main__':
    #app.run_server(debug=True)
    #app.run_server(debug=True, port=8049,use_reloader=False)
    app.run_server(debug=False, host='0.0.0.0', port=8080, dev_tools_hot_reload=False)


# In[ ]:




