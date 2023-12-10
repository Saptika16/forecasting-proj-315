# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 19:46:18 2023

@author: SAPTIKA
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

msft=pd.read_csv('Stock_Price.csv')
df1=pd.read_csv("Cleaned_Data.csv")
st.title("Forecasting")
st.header("Stock Market Price Forecasting")
#st.set_page_config(page_title='Forecasting',layout='wide')
filename='trained_model.sav'

loaded_model=pickle.load(open(filename, 'rb'))

st.dataframe(msft)

ypred_future=loaded_model.predict(start=9498,end=10018)
futuredate_arima_loaded=pd.DataFrame({'Close':ypred_future})
futuredate_arima_loaded['Date']=pd.date_range(start='2023-11-17',end='2025-11-15',freq='B')
futuredate_arima_loaded.set_index('Date',inplace=True)
future_data_Arima_loaded=pd.concat([df1,futuredate_arima_loaded])
a=pd.DataFrame(future_data_Arima_loaded)
#=pd.DataFrame(future_data_Arima_loaded)
#future_data_Arima_loaded['Date'] = (future_data_Arima_loaded['Date']).astype(str)
a.set_index('Date',inplace=True)
fig=plt.figure(figsize=(12,4))
plt.plot(df1['Close'],color='pink')
plt.plot(futuredate_arima_loaded['Close'],color='b')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.show()
st.pyplot(fig)

