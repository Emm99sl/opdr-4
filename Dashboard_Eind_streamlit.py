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
df = df[['Segment', 'City', 'State', 'Region', 'Category', 'Sub-Category', 'Discount', 'Profit', 'Order Date', 'Sales', 'Kosten', 'latitude', 'longitude']]

#Data omzetten naar datetime type
df['Order Date'] = pd.to_datetime(df['Order Date'])
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
df4 = df.copy()
df4 = drop_outlier(df4, 'Profit')
df4 = drop_outlier(df4, 'Sales')
df4 = drop_outlier(df4, 'Discount')

#model maken
model_orig = ols('Profit ~ Sales', data = df4).fit()

#verklarende variabelen maken
explanatory_data_orig = pd.DataFrame({'Sales': np.arange(1, 228, 8)})

#voorspellende data verkrijgen
pred_data_orig = explanatory_data_orig.assign(Profit = model_orig.predict(explanatory_data_orig))

#voorspellende data gebruiken
little_orig = pd.DataFrame({'Sales': np.arange(228, 600, 31)})

pred_little_orig = little_orig.assign(Profit = model_orig.predict(little_orig))

#transformeren
df4['Sales_log'] = np.log(df4['Sales'])
df4['Profit_log'] = np.log(df4['Profit'])

# Droppen van de na waarde in de log (dit komt waarschijnlijk door een profit van 0)
df4 = df4[df4['Profit_log'].notna()]

#droppen van de -inf waarden in de log
df4 = df4[df4['Profit_log'] != df4['Profit_log'].min()]

#Lineair regressie model
model = ols('Profit_log ~ Sales_log', data = df4).fit()

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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['1D Inspecties', '2D Inspecties', 'Geospatiale Inspectie 1', 'Geospatiale Inspectie 2', 'Model', 'Bronverwijzing'])
                     
with tab1:
       st.subheader("1D Inspectie: 1")
       plot_code0 = '''fig0 = px.histogram(df, x = "jaar_order", y = "Profit" ,title = "Histogram: Winst per jaar")
       fig0.update_xaxes(title_text = "Tijd in jaren")
       fig0.update_yaxes(title_text = "Winst in $")
       fig0.show() '''
       st.code(plot_code0)
       
       fig0 = px.bar(df, x = "jaar_order", y = "Profit", title= "Histogram: Winst per jaar")
       fig0.update_xaxes(title_text = "Tijd in jaren")
       fig0.update_yaxes(title_text = "Winst in $")
       st.plotly_chart(fig0)
       
       st.subheader("1D Inspectie: 2")
       plot_code2 = '''fig2 = px.histogram(df, x = "Segment",title = "Histogram: Aantallen per categorieën 'Segment'", color = "Segment")
       fig2.update_xaxes(title_text = "Categorieën Segment")
       fig2.update_yaxes(title_text = "Aantallen")
       fig2.show()'''
       st.code(plot_code2)

       fig2 = px.histogram(df, x = "Segment",title = "Histogram: Aantallen per categorieën 'Segment'", color = "Segment")
       fig2.update_xaxes(title_text = "Categoriën Segment")
       fig2.update_yaxes(title_text = "Aantallen")
       st.plotly_chart(fig2)
       
       st.subheader("1D Inspectie: 3")
       plot_code3 = '''fig3 = go.Figure()
       fig3.add_trace(go.Histogram(x = tech['Kosten'], nbinsx = 20, name = 'Techology'))
       fig3.add_trace(go.Histogram(x = furniture['Kosten'], name = 'Furniture'))
       fig3.add_trace(go.Histogram(x = off_supp['Kosten'], name = 'Office supply'))
       fig3.update_layout(title_text = 'Kosten van de Superstore per categorie')
       fig3.update_xaxes(title = 'Aantal keer')
       fig3.update_yaxes(title = 'Aantal keer in dezelfde kosten categorie')
       fig3.show()'''
       st.code(plot_code3)

       fig3 = go.Figure()
       fig3.add_trace(go.Histogram(x = tech['Kosten'], nbinsx = 30, name = 'Technologie'))
       fig3.add_trace(go.Histogram(x = furniture['Kosten'], nbinsx = 30, name = 'Meubilair'))
       fig3.add_trace(go.Histogram(x = off_supp['Kosten'], nbinsx = 30, name = 'Kantoor artikelen'))
       slider = [
              {'steps':[
                     {'method': 'update', 'label':'Alle categorieën ', 'args':[{'visible': [True, True, True]}]},
                     {'method': 'update', 'label':'Technologie', 'args':[{'visible': [True, False, False]}]},
                     {'method': 'update', 'label':'Meubilair', 'args':[{'visible': [False, True, False]}]},
                     {'method': 'update', 'label':'Kantoor artikelen', 'args':[{'visible': [False, False, True]}]}]}]
       
       fig3.update_layout(title_text = 'Kosten van de superstore per categorie')
       fig3.update_xaxes(title = 'Aantal kosten')
       fig3.update_yaxes(title = 'Aantal keer in dezelfde kosten categorie')
       fig3.update_layout({'sliders':slider})
       st.plotly_chart(fig3)
       
       st.subheader("1D Inspectie: 4")
       plot_code4 = '''fig4 = px.histogram(df, x = "Profit", title = "Histogram: Winst", nbins = 25)
       fig4.update_xaxes(title_text = "Winst in $")
       fig4.update_yaxes(title_text = "Aantal Superstore")
       fig4.show()'''
       st.code(plot_code4)

       fig4 = px.histogram(df, x = "Profit", title = "Histogram: Winst", nbins = 25)
       fig4.update_xaxes(title_text = "Winst in $")
       fig4.update_yaxes(title_text = "Aantal Superstore")
       st.plotly_chart(fig4)

with tab2:
              
       st.subheader("2D Inspectie: 1")
       plot_code5 = '''fig5 = go.Figure()
       fig5.add_traces(go.Scatter(x = consumer['Discount'], y = consumer['Profit'], mode = 'markers', name = 'Consumer', visible = True))
       fig5.add_traces(go.Scatter(x = corporate['Discount'], y = corporate['Profit'], mode = 'markers', name = "Corporate", visible = False))
       fig5.add_traces(go.Scatter(x = home_office['Discount'], y = home_office['Profit'], mode = 'markers', name = 'Home Office', visible = False))

       #dropdownmenu aanmaken
       dropdown_buttons = [{"label":"Consumer", "method":"update","args":[{"visible":[True, False, False]},{"title":"Consumer"}]}, 
       {"label":"Corporate", "method":"update","args":[{"visible":[False, True, False]},{"title":"Corporate"}]},
       {"label":"Home Office", "method":"update","args":[{"visible":[False,False,True]},{"title":"Home Office"}]}]
       #dropdownmenu toevoegen
       fig5.update_layout({"updatemenus":[{"type":"dropdown","x": 1.2,"y":0.9,"showactive":True,"active":0,"buttons": dropdown_buttons}]})
       #titels/labels aanmaken
       fig5.update_layout(title = "Scatterplot korting tegenover winst")
       fig5.update_xaxes(title_text="Korting")
       fig5.update_yaxes(title_text="Winst in $")
       st.plotly_chart(fig5)'''
       st.code(plot_code5)

       fig5 = go.Figure()
       fig5.add_traces(go.Scatter(x = consumer['Discount'], y = consumer['Profit'], mode = 'markers', name = 'Consumer', visible = True))
       fig5.add_traces(go.Scatter(x = corporate['Discount'], y = corporate['Profit'], mode = 'markers', name = "Corporate", visible = False))
       fig5.add_traces(go.Scatter(x = home_office['Discount'], y = home_office['Profit'], mode = 'markers', name = 'Home Office', visible = False))

       #dropdownmenu aanmaken
       dropdown_buttons = [{"label":"Consumer", "method":"update","args":[{"visible":[True, False, False]},{"title":"Consumer"}]}, 
       {"label":"Corporate", "method":"update","args":[{"visible":[False, True, False]},{"title":"Corporate"}]},
       {"label":"Home Office", "method":"update","args":[{"visible":[False,False,True]},{"title":"Home Office"}]}]
       #dropdownmenu toevoegen
       fig5.update_layout({"updatemenus":[{"type":"dropdown","x": 1.2,"y":0.9,"showactive":True,"active":0,"buttons": dropdown_buttons}]})
       #titels/labels aanmaken
       fig5.update_layout(title = "Scatterplot korting tegenover winst")
       fig5.update_xaxes(title_text="Korting")
       fig5.update_yaxes(title_text="Winst in $")
       st.plotly_chart(fig5)
       
       st.subheader("2D Inspectie: 2")
       plot_code6 = '''fig6 = px.scatter(df, x = "Kosten", y = "Sales", color = "Region")
       #titels/labels aanmaken
       fig5.update_layout(title = "Scatterplot kosten tegenover sales")
       fig6.update_xaxes(title_text="Kosten in $")
       fig6.update_yaxes(title_text="Kosten in $")
       st.plotly_chart(fig6)'''
       st.code(plot_code6)

       fig6 = px.scatter(df, x = "Kosten", y = "Sales", color = "Region")
       #titels/labels aanmaken
       fig6.update_layout(title = "Scatterplot kosten tegenover sales")
       fig6.update_xaxes(title_text="Kosten in $")
       fig6.update_yaxes(title_text="Kosten in $")
       st.plotly_chart(fig6)

#KAART 1
with tab3:
       st.header("Kaart visualisatie: 1")
       
       st.subheader('Kaart van verkochte artikelen per segment')
       col1, col2, col3 = st.columns([10, 1, 3])
       
       with col1:
       
              def color_producer(type):
                     if type == 'Consumer':
                            return 'green'
                     elif type == 'Corporate':
                            return 'red'
                     elif type == 'Home Office':
                            return 'blue'
        
              m = folium.Map(location = [37.09024, -95.712891], zoom_start = 3.4)

              for mp in df.iterrows():
                     mp_values = mp[1]
                     location = [mp_values['latitude'], mp_values['longitude']]
                     popup = (str(mp_values['City']))
                     color = color_producer(mp_values['Segment'])
                     marker = folium.CircleMarker(location = location, popup = popup, color = color)
                     marker.add_to(m)
      
    
              folium_static(m, width = 500, height = 350)
       
       with col2:
              st.color_picker('Consumer', '#346934', label_visibility = 'collapsed')
              st.color_picker('Corporate', '#ff0000', label_visibility = 'collapsed')
              st.color_picker('Home Office', '#0000ff', label_visibility = 'collapsed')
              

       with col3:
              st.write('Consumer')
              st.write('Corporate')
              st.write('Home Office')
          
with tab4:
       
       st.header("Kaart visualisatie: 2")
       st.subheader('Kaart van de winst ($) per superstore')
       col4, col5, col6 = st.columns([10, 1, 4])
       with col4:
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

       
              m2 = folium.Map(location = [37.09024, -95.712891], zoom_start = 3.4)
       
              for mp in df.iterrows():
                     mp_values = mp[1]
                     location = [mp_values['latitude'], mp_values['longitude']]
                     popup = (str(mp_values['City']))
                     color = color_producer2(mp_values['Profit'])
                     marker = folium.CircleMarker(location = location, popup = popup, color = color)
                     marker.add_to(m2) 
              folium_static(m2, width = 450, height = 350)
       
       with col5:
              st.color_picker('Verlies', '#ff0000', label_visibility = 'collapsed')
              st.color_picker('Verlies', '#020202', label_visibility = 'collapsed')
              st.color_picker('Verlies', '#0000ff', label_visibility = 'collapsed')
              st.color_picker('Verlies', '#EBF905', label_visibility = 'collapsed')
              st.color_picker('Verlies', '#F9A005', label_visibility = 'collapsed')
              st.color_picker('Verlies', '#346934', label_visibility = 'collapsed')
              
       with col6:
              st.write('Verlies')
              st.write('0<= winst <=10')
              st.write('10< winst <=20')
              st.write('20< winst <= 30')
              st.write('30< winst <= 40')
              st.write('winst > 40')
     

with tab5:
       st.header("Visualisatie model")
       st.text('De correlatie tussen opbrengst en winst is: 0.48.')
       #Figuur maken van model
       fig7 = go.Figure()
       #Toevoegen van traces van de verschillende stappen in het model 
       #fig.add_trace(go.Scatter(x=df4["Sales"], y=df4["Profit"], opacity= 0.8, mode = 'markers', name = 'Data'))
       fig7.add_trace(go.Scatter(x=df4["Sales_log"], y=df4["Profit_log"], opacity= 0.8, mode = 'markers', name = 'Getransformeerde data'))
       fig7.add_trace(go.Scatter(x=pred_data["Sales_log"], y=pred_data["Profit_log"], mode = 'markers', name = 'Voorspelling nu'))
       fig7.add_trace(go.Scatter(x=pred_little["Sales_log"], y=pred_little["Profit_log"], mode = 'markers', name = 'Voorspelling als er meer verkocht wordt'))

       #Assenlabels toevoegen
       fig7.update_layout(title = 'Visualisatie van de voorspelling van de winst aan de hand van de sales')
       fig7.update_yaxes(title = 'De winst van de Superstore')
       fig7.update_xaxes(title = 'Sales')
       st.plotly_chart(fig7)

       
       st.text('Het originele model heeft een rsquared van 0.25.')
       st.text('Het getransformeerde model heeft een rsquared van 0.65.')
       

with tab6:
       st.subheader('Bronnen:')
       st.text('Ibrahim Elsayed. (2022). Sample Superstore (Versie V1) [Dataset]. \nhttps://www.kaggle.com/datasets/ibrahimelsayed182/superstore')
       st.text('Rohit Sahoo. (2021). Superstore Sales dataset (Versie V2) [Dataset]. \nhttps://www.kaggle.com/datasets/rohitsahoo/sales-forecasting')
       
