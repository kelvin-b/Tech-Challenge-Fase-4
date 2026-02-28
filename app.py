import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import joblib
import os


def loading():
    return joblib.load('model_obesity.pkl')

st.set_page_config(page_title="Predição de Obesidade - Hospital", layout="wide")

st.title("Sistema de Diagnóstico e Análise de Obesidade")


# ------- Mapeamento dos valores e colunas ----------------------
map_gender = {"Masculino": "Male", "Feminino": "Female"}

map_yes_no = {"Sim": "yes", "Não": "no"}

map_frequency = {
    "Não consome": "no", 
    "Às vezes": "Sometimes", 
    "Frequentemente": "Frequently", 
    "Sempre": "Always"
}

map_mtrans = {
    "Automóvel": "Automobile",
    "Motocicleta": "Motorbike",
    "Bicicleta": "Bike",
    "Transporte Público": "Public_Transportation",
    "Caminhada": "Walking"
}

map_fcvc = {
    "Raramente": 1,
    "Às vezes": 2,
    "Sempre": 3
}

map_faf = {
    "Nenhuma": 0,
    "1 à 2 vezes": 1,
    "3 à 4 vezes": 2,
    "5 vezes ou mais": 3
}

map_tue = {
    "0 à 2h por dia": 0,
    "3 à 5h por dia": 1,
    "Mais de 5h por dia": 2
}

target_translation = {
    "Insufficient_Weight": "Peso Insuficiente",
    "Normal_Weight": "Peso Normal",
    "Overweight_Level_I": "Sobrepeso Grau I",
    "Overweight_Level_II": "Sobrepeso Grau II",
    "Obesity_Type_I": "Obesidade Grau I",
    "Obesity_Type_II": "Obesidade Grau II",
    "Obesity_Type_III": "Obesidade Grau III"
}

column_translation = {
    "Obesity": "Nível de Obesidade",
    "Gender": "Genero",
    "Age": "Idade",
    "Weight": "Peso (kg)",
    "Height": "Altura (m)",
    "TUE": "Tempo eletronicos"
}

# ----------------------------------------------------------------------------------

st.sidebar.header("Dados do Paciente")
age = st.sidebar.number_input("Idade", 14, 61, 25)
height = st.sidebar.number_input("Altura (m)", 1.40, 2.10, 1.70)
weight = st.sidebar.number_input("Peso (kg)", 30, 200, 70)

gender = st.sidebar.selectbox("Gênero", list(map_gender.keys()))
family = st.sidebar.selectbox("Histórico Familiar excesso peso?", list(map_yes_no.keys()))
favc = st.sidebar.selectbox("Consome alimentos calóricos?", list(map_yes_no.keys()))
fcvc = st.sidebar.selectbox("Frequência Consome Vegetais", list(map_fcvc.keys()))

ncp = st.sidebar.slider("Quantas refeições principais?", 1, 4, 3)

caec = st.sidebar.selectbox("Alimentos entre refeições", list(map_frequency.keys()))
smoke = st.sidebar.selectbox("Fuma?", list(map_yes_no.keys()))

water = st.sidebar.slider("Consumo Água (1 à 3 Litros)", 1, 3, 2)

scc = st.sidebar.selectbox("Monitora calorias diária?", list(map_yes_no.keys()))
faf = st.sidebar.selectbox("Frequência semanal de atividade física", list(map_faf.keys()))
tue = st.sidebar.selectbox("Tempo diário usando dispositivos eletrônicos", list(map_tue.keys()))
calc = st.sidebar.selectbox("Consome bebida alcoólica?", list(map_frequency.keys()))
mtrans = st.sidebar.selectbox("Transporte", list(map_mtrans.keys()))

input_data = {
    "Gender": map_gender[gender], 
    "Age": age,
    "Height": height,
    "Weight": weight,
    "family_history": map_yes_no[family],
    "FAVC": map_yes_no[favc],
    "FCVC": map_fcvc[fcvc],
    "NCP": ncp,
    "CAEC": map_frequency[caec],
    "SMOKE": map_yes_no[smoke],
    "CH2O": water,
    "SCC": map_yes_no[scc],
    "FAF": map_faf[faf],
    "TUE": map_tue[tue],
    "CALC": map_frequency[calc],
    "MTRANS": map_mtrans[mtrans]
}
modelo = loading()

if st.sidebar.button("Realizar Predição"):
    try:
        df_input = pd.DataFrame([input_data])
        predicao = modelo.predict(df_input)[0]
        result_port = target_translation.get(predicao, predicao)
        st.success(f"Resultado do Diagnóstico: {result_port}")
    except Exception as e:
        st.error(f"Erro ao processar predição: {e}")

# Plot dos Gráficos
st.divider()
st.subheader("Visão Analítica de Negócio")


def translate_columns():
    df = pd.read_csv('Obesity.csv')
    df['Obesity'] = df['Obesity'].map(target_translation)
    df['Gender'] = df['Gender'].map({"Male" : "Masculino", "Female" : "Feminino"})
    df = df.rename(columns=column_translation)
    return df


df_view = translate_columns()

ordem_clinica = [
    "Peso Insuficiente", "Peso Normal", "Sobrepeso Grau I", 
    "Sobrepeso Grau II", "Obesidade Grau I", "Obesidade Grau II", "Obesidade Grau III"
]

fig1 = px.histogram(df_view, 
    x="Nível de Obesidade", 
    color="Genero", 
    title="Distribuição de Casos por Gênero"
    )
fig1.update_yaxes(title_text="Quantidade de Diagnósticos")
fig1.update_xaxes(categoryorder='array', categoryarray=ordem_clinica)
st.plotly_chart(fig1)

fig2 = px.box(df_view,
    x="Nível de Obesidade",
    y="Idade",
    title="Idade vs Nível de Obesidade"
    )
fig2.update_xaxes(categoryorder='array', categoryarray=ordem_clinica)
st.plotly_chart(fig2)