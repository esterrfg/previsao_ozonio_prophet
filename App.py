import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import requests
from prophet.serialize import model_from_json
from prophet import Prophet
from prophet.plot import plot_plotly

# Load model
def load_model():
    with open("modelo_o3_prophet.json", "r") as fin:
        model = model_from_json(json.load(fin))  # Load model
        return model

modelo = load_model()

# Adicionando textos ao layout do Streamlit
st.title('Previsão de Níveis de Ozônio (O3) Utilizando a Biblioteca Prophet')

st.caption('''Este projeto utiliza a biblioteca Prophet para prever os níveis de ozônio em ug/m3. O modelo
           criado foi treinado com dados até o dia 05/05/2023 e possui um erro de previsão (RMSE - Erro Quadrático Médio) igual a 17.43 nos dados de teste.
           O usuário pode inserir o número de dias para os quais deseja a previsão, e o modelo gerará um gráfico
           interativo contendo as estimativas baseadas em dados históricos de concentração de O3.
           Além disso, uma tabela será exibida com os valores estimados para cada dia.''')

st.subheader('Insira o número de dias para previsão:')

dias = st.number_input('Dias:', min_value=1, max_value=365, value=30, step=1)

if 'previsao_feita' not in st.session_state:
    st.session_state['previsao_feita'] = False
    st.session_state['dados_previsao'] = None

if st.button('Gerar Previsão'):
    st.session_state['previsao_feita'] = True
    future = modelo.make_future_dataframe(periods=dias,freq='D')	
    forecast = modelo.predict(future)
    st.session_state['dados_previsao'] = forecast#[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(dias)
    
if st.session_state.previsao_feita:
    fig = plot_plotly(modelo, st.session_state.dados_previsao)
    fig.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 1)',  # Define o fundo da área do gráfico como branco
        'paper_bgcolor': 'rgba(255, 255, 255, 1)', # Define o fundo externo ao gráfico como branco
        'title': {'text': "Previsão de Ozônio", 'font': {'color': 'black'}},
        'xaxis': {'title': 'Data', 'title_font': {'color': 'black'}, 'tickfont': {'color': 'black'}},
        'yaxis': {'title': 'Nível de Ozônio (O3 μg/m3)', 'title_font': {'color': 'black'}, 'tickfont': {'color': 'black'}}
    })
    st.plotly_chart(fig)

previsao = st.session_state.dados_previsao
tabela_previsao = previsao[['ds', 'yhat']].tail(dias)
tabela_previsao.columns = ['Data (dia/mês/ano)', 'Previsão de O3 (μg/m3)']
tabela_previsao['Data (dia/mês/ano)'] = pd.to_datetime(tabela_previsao['Data (dia/mês/ano)']).dt.strftime('%d/%m/%Y')
tabela_previsao['Previsão de O3 (μg/m3)'] = tabela_previsao['Previsão de O3 (μg/m3)'].round(2)
tabela_previsao.reset_index(drop=True, inplace=True)
st.write('Tabela contendo as previsões de O3 para os próximos', dias, 'dias.'.format(dias))
st.dataframe(tabela_previsao,height=300)

csv = tabela_previsao.to_csv(index=False)
st.download_button(label='Baixar Tabela de Previsão de O3 como .csv', data=csv, file_name='previsao_o3.csv', mime='text/csv')














