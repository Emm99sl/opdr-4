#!/usr/bin/env python
# coding: utf-8

# In[1]:


#data manipulatie
import pandas as pd
import numpy as np
import streamlit as st

#plots
import plotly.graph_objects as go
import plotly.express as px

import plotly.figure_factory as ff

#modellen
from statsmodels.formula.api import ols


#kaarten
from streamlit_folium import folium_static
import folium


# In[2]:


########################## dataset#############################
df1 = pd.read_csv('SampleSuperstore_metcoördinaten.csv')
df2 = pd.read_csv('SalesSuperstore_met_coords.csv')
df1 = df1[['Segment', 'City', 'State',
       'Postal Code', 'Region', 'Category', 'Sub-Category', 'Sales',
       'Quantity', 'Discount', 'Profit', 'geocode', 'latitude', 'longitude']]
df2 = df2[['Order Date', 'Ship Date', 'Segment',
       'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category',
       'Sub-Category', 'Product Name', 'Sales', 'geocode', 'latitude',
       'longitude']]
df = pd.read_csv('SalesSuperstoreMerged_coords.csv')

#Mergen van de dataframe
df3 = pd.merge(df1, df2, on = ['Segment', 'City', 'Postal Code', 'Sub-Category', 'Sales'])

#Checken of de kolommen gelijk zijn van df1 en df2
df3['State_x'].equals(df3['State_y'])
df3['Region_x'].equals(df3['Region_y'])
df3['Category_x'].equals(df3['Category_y'])

#Selecteren van de kolommen
df3 = df3[['Segment', 'City', 'State_x', 'Postal Code', 'Region_x', 'Category_x', 'Sub-Category', 'Sales', 'Quantity', 'Discount', 'Profit','Order Date', 'Ship Date','Product Name']]

#Veranderen van de namen
df3 = df3.rename(columns = {'State_x':'State', 'Region_x':'Region', 'Category_x':'Category'})
df3.head()

#functie om data te filteren uit een kolom
def filter_df(data, kolom, cat):
    cat = data[data[kolom] == cat]
    return cat

#functie om outliers te droppen
def drop_outlier(df, kolom):
    q1 = df[kolom].quantile(0.25)
    q3 = df[kolom].quantile(0.75)
    iqr = q3 - q1
    hoog = (df[kolom] <= (q3 + 1.5*iqr))
    laag = (df[kolom] >= (q1 - 1.5*iqr))
    df_update = df.loc[hoog]
    df_update2 = df_update.loc[laag]
    return df_update2
df['Kosten'] = df['Sales'] - df['Profit']

#Selecteren van de kolommen
df = df[['Segment', 'City', 'State', 'Postal Code', 'Region', 'Category', 'Sub-Category', 'Discount', 'Profit', 'Order Date', 'Ship Date', 'Sales', 'Kosten', 'Product Name', 'geocode', 'latitude', 'longitude']]

#Data omzetten naar datetime type
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df["jaar_order"] = df["Order Date"].dt.year

#Dataframes maken van de verschillende categrieen 
furniture = filter_df(df, 'Category', 'Furniture')
off_supp = filter_df(df, 'Category', 'Office Supplies')
tech = filter_df(df, 'Category', 'Technology')

#dataframe van de verschillende segmenten:
consumer = filter_df(df, 'Segment', 'Consumer')
corporate = filter_df(df, 'Segment', 'Corporate')
home_office = filter_df(df, 'Segment', 'Home Office')
              
#Outliers droppen
df = drop_outlier(df, 'Profit')
df = drop_outlier(df, 'Sales')
df = drop_outlier(df, 'Discount')

#model maken
model_orig = ols('Profit ~ Sales', data = df).fit()

#verklarende variabelen maken
explanatory_data_orig = pd.DataFrame({'Sales': np.arange(1, 228, 8)})

#voorspellende data verkrijgen
pred_data_orig = explanatory_data_orig.assign(Profit = model_orig.predict(explanatory_data_orig))

#voorspellende data gebruiken
little_orig = pd.DataFrame({'Sales': np.arange(228, 600, 31)})

pred_little_orig = little_orig.assign(Profit = model_orig.predict(little_orig))

#transformeren
df['Sales_log'] = np.log(df['Sales'])
df['Profit_log'] = np.log(df['Profit'])

# Droppen van de na waarde in de log (dit komt waarschijnlijk door een profit van 0)
df = df[df['Profit_log'].notna()]

#droppen van de -inf waarden in de log
df = df[df['Profit_log'] != df['Profit_log'].min()]

#Lineair regressie model
model = ols('Profit_log ~ Sales_log', data = df).fit()

#verklarende variabelen maken
explanatory_data = pd.DataFrame({'Sales_log': np.log(np.arange(1, 228, 8)), 
                                'Sales': np.arange(1, 228, 8)})

#prediction maken van verklarende variabelen
pred_data = explanatory_data.assign(Profit_log = model.predict(explanatory_data))
pred_data['Profit'] = 10**pred_data['Profit_log']

#prediction toekomst sales maken
little = pd.DataFrame({'Sales_log': np.log(np.arange(228, 600, 31))})
pred_little = little.assign(Profit_log = model.predict(little))


# In[3]:


st.title("Visual Analytics Eindpresentatie ")
#Dataframe
with st.expander('Samengevoegde dataframes'):
       st.dataframe(df)

tab1, tab2, tab3, tab4, tab5 = st.tabs(['1D Inspecties', '2D Inspecties', 'Geospatiale Inspecties', 'Model', 'Bronverwijzing'])
                     
with tab1:
       st.subheader("1D Inspecties")
       plot_code0 = '''fig0 = px.histogram(df, x = "State", y = "Profit" ,title = "1D Inspectie: Histogram")
       fig0.update_xaxes(title_text = "Staten US")
       fig0.update_yaxes(title_text = "Winst in $")
       fig0.show() '''
       st.code(plot_code0)
       
       fig0 = px.bar(df, x = "jaar_order", y = "Profit", title= "1D Inspectie: Histogram")
       fig0.update_xaxes(title_text = "Tijd in jaren")
       fig0.update_yaxes(title_text = "Winst in $")
       st.plotly_chart(fig0)

       plot_code2 = '''fig2 = px.histogram(df, x = "Segment",title = "1D Inspectie: Histogram", color = "Segment")
       fig2.update_xaxes(title_text = "Categoriën Segment")
       fig2.update_yaxes(title_text = "Aantallen")
       fig2.show()'''
       st.code(plot_code2)

       fig2 = px.histogram(df, x = "Segment",title = "1D Inspectie: Histogram", color = "Segment")
       fig2.update_xaxes(title_text = "Categoriën Segment")
       fig2.update_yaxes(title_text = "Aantallen")
       st.plotly_chart(fig2)

       plot_code3 = '''fig3 = go.Figure()
       fig3.add_trace(go.Histogram(x = tech['Kosten'], nbinsx = 20, name = 'Techology'))
       fig3.add_trace(go.Histogram(x = furniture['Kosten'], name = 'Furniture'))
       fig3.add_trace(go.Histogram(x = off_supp['Kosten'], name = 'Office supply'))
       fig3.update_layout(title_text = 'Kosten van de Superstore per categorie')
       fig3.update_xaxes(title = 'Aantal kosten')
       fig3.update_yaxes(title = 'Aantal keer in dezelfde kosten categorie')
       fig3.show()'''
       st.code(plot_code3)

       fig3 = go.Figure()
       fig3.add_trace(go.Histogram(x = tech['Kosten'], nbinsx = 20, name = 'Techology'))
       fig3.add_trace(go.Histogram(x = furniture['Kosten'], name = 'Furniture'))
       fig3.add_trace(go.Histogram(x = off_supp['Kosten'], name = 'Office supply'))
       fig3.update_layout(title_text = 'Kosten van de Superstore per categorie')
       fig3.update_xaxes(title = 'Aantal kosten')
       fig3.update_yaxes(title = 'Aantal keer in dezelfde kosten categorie')
       st.plotly_chart(fig3)

       plot_code4 = '''fig4 = px.histogram(df, x = "Profit", title = "1D Inspectie: Histogram over de winst", nbins = 25)
       fig4.update_xaxes(title_text = "Winst in $")
       fig4.update_yaxes(title_text = "Aantal")
       fig4.show()'''
       st.code(plot_code4)

       fig4 = px.histogram(df, x = "Profit", title = "1D Inspectie: Histogram over de winst", nbins = 25)
       fig4.update_xaxes(title_text = "Winst in $")
       fig4.update_yaxes(title_text = "Aantal")
       st.plotly_chart(fig4)

with tab2:
       st.subheader("2D Inspecties")

       fig5 = go.Figure()
       fig5.add_traces(go.Scatter(x = consumer['Discount'], y = consumer['Sales'], mode = 'markers', name = 'Consumer', visible = True))
       fig5.add_traces(go.Scatter(x = corporate['Discount'], y = corporate['Sales'], mode = 'markers', name = "Corporate", visible = False))
       fig5.add_traces(go.Scatter(x = home_office['Discount'], y = home_office['Sales'], mode = 'markers', name = 'Home Office', visible = False))

       #dropdownmenu aanmaken
       dropdown_buttons = [{"label":"Consumer", "method":"update","args":[{"visible":[True, False, False]},{"title":"Consumer"}]}, 
       {"label":"Corporate", "method":"update","args":[{"visible":[False, True, False]},{"title":"Corporate"}]},
       {"label":"Home Office", "method":"update","args":[{"visible":[False,False,True]},{"title":"Home Office"}]}]
       #dropdownmenu toevoegen
       fig5.update_layout({"updatemenus":[{"type":"dropdown","x": 1.2,"y":0.9,"showactive":True,"active":0,"buttons": dropdown_buttons}]})
       #titels/labels aanmaken
       fig5.update_layout(title = "2D Inspectie: Scatterplot")
       fig5.update_xaxes(title_text="Discount")
       fig5.update_yaxes(title_text="Opbrengst")
       st.plotly_chart(fig5)

       fig6 = px.scatter(df, x = "Kosten", y = "Sales", color = "Region")
       #titels/labels aanmaken
       fig5.update_layout(title = "2D Inspectie: Scatterplot tussen kosten en opbrengst")
       fig6.update_xaxes(title_text="Regio US")
       fig6.update_yaxes(title_text="Kosten in $")
       st.plotly_chart(fig6)

#KAART 1
with tab3:
       def color_producer(type):
              if type == 'Consumer':
                     return 'green'
              elif type == 'Corporate':
                     return 'red'
              elif type == 'Home Office':
                     return 'blue'


       st.subheader('Kaart van verkochte artikelen per segment')
       m = folium.Map(location = [37.09024, -95.712891], zoom_start = 4.4)

    
    
       for mp in df.iterrows():
              mp_values = mp[1]
              location = [mp_values['latitude'], mp_values['longitude']]
              popup = (str(mp_values['City']))
              color = color_producer(mp_values['Segment'])
              marker = folium.CircleMarker(location = location, popup = popup, color = color)
              marker.add_to(m)
    
       folium_static(m)


       def color_producer2(type):
              if type < 0:
                     return 'red'
              elif 0 <= type <= 10 :
                     return 'black'
              elif 10 < type <= 20:
                     return 'blue'
              elif 20 < type <= 30:   
                     return 'yellow'
              elif 30 < type <= 40:
                     return 'orange'
              elif type > 40:
                     return 'green'

       st.subheader('Kaart van de winst ($) per superstore')
       m2 = folium.Map(location = [37.09024, -95.712891], zoom_start = 4.4)
       
       for mp in df.iterrows():
              mp_values = mp[1]
              location = [mp_values['latitude'], mp_values['longitude']]
              popup = (str(mp_values['City']))
              color = color_producer2(mp_values['Profit'])
              marker = folium.CircleMarker(location = location, popup = popup, color = color)
              marker.add_to(m2) 
       folium_static(m2)

with tab4:
       #Figuur maken van model
       fig7 = go.Figure()
       #Toevoegen van traces van de verschillende stappen in het model 
       #fig.add_trace(go.Scatter(x=df["Sales"], y=df["Profit"], opacity= 0.8, mode = 'markers', name = 'Data'))
       fig7.add_trace(go.Scatter(x=df["Sales_log"], y=df["Profit_log"], opacity= 0.8, mode = 'markers', name = 'Getransformeerde data'))
       fig7.add_trace(go.Scatter(x=pred_data["Sales_log"], y=pred_data["Profit_log"], mode = 'markers', name = 'Voorspelling nu'))
       fig7.add_trace(go.Scatter(x=pred_little["Sales_log"], y=pred_little["Profit_log"], mode = 'markers', name = 'Voorspelling als er meer verkocht wordt'))

       #Assenlabels toevoegen
       fig7.update_layout(title = 'Visualisatie van de voorspelling van de winst aan de hand van de sales')
       fig7.update_yaxes(title = 'De winst van de Superstore')
       fig7.update_xaxes(title = 'Sales')
       st.plotly_chart(fig7)


       st.text('Het originele model heeft een rsquared van 0.25.')
       st.text('Het getransformeerde model heeft een rsquared van 0.65.')
       st.text('De correlatie tussen sales en winst is: 0.75.')

with tab5:
       st.subheader('Bronnen:')
       st.text('Ibrahim Elsayed. (2022). Sample Superstore (Versie V1) [Dataset]. \nhttps://www.kaggle.com/datasets/ibrahimelsayed182/superstore')
       st.text('Rohit Sahoo. (2021). Superstore Sales dataset (Versie V2) [Dataset]. \nhttps://www.kaggle.com/datasets/rohitsahoo/sales-forecasting')
       
