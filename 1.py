
import streamlit as st
import pickle
import pandas as pd
st.set_page_config(page_title='Forecasting',layout='wide')
st.set_option('deprecation.showPyplotGlobalUse',False)
st.title("Forecasting")

msft=pd.read_csv('Stock_Price.csv')
df1=pd.read_csv("Cleaned_Data.csv")
#
st.header("Stock Market Price Forecasting")
filename='trained_model.sav'

loaded_model=pickle.load(open(filename, 'rb'))

column1, column2 = st.columns([1,2])
with column1:
    company=st.selectbox("Pick Company", ["Microsoft"],
                         index=None,
   placeholder="Select company...",)
#with column2:
   
    if company=='Microsoft':
        
        with column2:
            
            plots=st.radio(
                "Plots 👉",
       
                options=["CLeaned Original data",'2 years forecasted data','1 year forecasted data','6 months forecasted data'])
            if plots=='CLeaned Original data':
                column2.write('Microsoft stock data from 1986-03-13 to 2023-11-16 ')
                column2.line_chart(df1['Close'])
                
                column1.dataframe(msft)
            if plots=='2 years forecasted data':
                column2.write('Microsoft stock data from 1986-03-13 to 2025-11-16 ')
                
                ypred_future=loaded_model.predict(start=9498,end=10018)#521 days or 2 years
                futuredate_arima=pd.DataFrame({'Close':ypred_future})
                futuredate_arima=pd.DataFrame({'Close':ypred_future})
                futuredate_arima['Date']=pd.date_range(start='2023-11-17',end='2025-11-15',freq='B')
                future_data_Arima=pd.concat([df1,futuredate_arima])
                future_data_Arima=future_data_Arima.set_index('Date')
                futuredate_arima.set_index('Date',inplace=True)
                st.line_chart(future_data_Arima)
                st.line_chart(futuredate_arima['Close'])
                column1.dataframe(future_data_Arima)


                
            if plots=='1 year forecasted data':
              column2.write('Microsoft stock data from 1986-03-13 to 2024-11-16 ')
              
              ypred_future=loaded_model.predict(start=9498,end=9758)#261 days or 1 year
              futuredate_arima=pd.DataFrame({'Close':ypred_future})
              futuredate_arima=pd.DataFrame({'Close':ypred_future})
              futuredate_arima['Date']=pd.date_range(start='2023-11-17',end='2024-11-15',freq='B')
              future_data_Arima=pd.concat([df1,futuredate_arima])
              future_data_Arima=future_data_Arima.set_index('Date')
              futuredate_arima.set_index('Date',inplace=True)
              st.line_chart(future_data_Arima)
              st.line_chart(futuredate_arima['Close'])
              column1.dataframe(future_data_Arima)  
            if plots=='6 months forecasted data':
               column2.write('Microsoft stock data from 1986-03-13 to 2024-05-16 ')
               
               ypred_future=loaded_model.predict(start=9498,end=9626)#129 days or 6 months
               futuredate_arima=pd.DataFrame({'Close':ypred_future})
               futuredate_arima=pd.DataFrame({'Close':ypred_future})
               futuredate_arima['Date']=pd.date_range(start='2023-11-17',end='2024-05-15',freq='B')
               future_data_Arima=pd.concat([df1,futuredate_arima])
               future_data_Arima=future_data_Arima.set_index('Date')
               futuredate_arima.set_index('Date',inplace=True)
               st.line_chart(future_data_Arima)
               st.line_chart(futuredate_arima['Close'])
               column1.dataframe(future_data_Arima)   
                
       
    #else:
     #   st.write('Please select a company name for stock market price forecast')


#ypred_future=loaded_model.predict(start=9498,end=10227)#730 days
#futuredate_arima['Date'] = pd.to_datetime(futuredate_arima['Date'],format='%m/%d/%Y%H%M')
#future_data_Arima.index=future_data_Arima.index.date
#future_data_Arima_loaded['Date'] = (future_data_Arima_loaded['Date']).astype(str)
#future_data_Arima.set_index('Date',inplace=True)
#fig=plt.figure(figsize=(12,4))
#plt.plot(df1['Close'],color='pink')
#plt.plot(futuredate_arima['Close'],color='b')
#plt.plot(future_data_Arima,color='b')
#plt.xlabel('Time')
#plt.ylabel('Close Price')
#plt.show()
#st.pyplot(fig)
#st.line_chart(future_data_Arima)
#st.write('future_data_Arima')
#st.write(future_data_Arima)
#st.write('futuredate_arima')
#st.write(futuredate_arima)
#st.write('df1')
#   st.write(df1)