import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Sistema de Previsão de Churn", page_icon="📉")

# --- CARREGAR OS MODELOS ---
@st.cache_resource
def load_assets():
    modelo_rf = joblib.load('modelo_rf.pkl') # Random Forest
    modelo_lr = joblib.load('modelo_lr.pkl') # Regressão Logística
    scaler = joblib.load('scaler.pkl')
    return modelo_rf, modelo_lr, scaler

try:
    rf_model, lr_model, scaler = load_assets()
except FileNotFoundError:
    st.error("Erro: Arquivos .pkl não encontrados. Rode o notebook para gerar os modelos.")
    st.stop()

# --- TÍTULO E SELEÇÃO DE MODELO ---
st.title("📉 Previsão de Churn - Telecom")
st.markdown("Insira os dados do cliente para verificar o risco.")

# SELETOR DE MODELO (A novidade!)
st.sidebar.header("Configuração do Modelo")
modelo_escolhido = st.sidebar.selectbox(
    "Qual algoritmo usar?",
    ("Random Forest (Recomendado)", "Regressão Logística (Baseline)")
)

# Define qual modelo será usado na previsão
if modelo_escolhido == "Random Forest (Recomendado)":
    model = rf_model
else:
    model = lr_model

st.sidebar.markdown("---")
st.sidebar.header("Dados do Cliente")

# --- FORMULÁRIO (Igual ao anterior) ---
def user_input_features():
    age = st.sidebar.slider("Idade", 18, 80, 30)
    gender = st.sidebar.selectbox("Gênero", ["Feminino", "Masculino"])
    contract = st.sidebar.selectbox("Tipo de Contrato", ["Annual", "Monthly", "Quarterly"])
    subscription = st.sidebar.selectbox("Tipo de Assinatura", ["Basic", "Standard", "Premium"])
    tenure = st.sidebar.slider("Tempo de Contrato (Meses)", 0, 60, 12)
    usage_freq = st.sidebar.number_input("Frequência de Uso (Vezes)", 0, 30, 15)
    support_calls = st.sidebar.number_input("Chamadas ao Suporte", 0, 10, 1)
    payment_delay = st.sidebar.number_input("Atraso no Pagamento (Dias)", 0, 30, 0)
    total_spend = st.sidebar.number_input("Gasto Total ($)", 0.0, 5000.0, 500.0)
    last_interaction = st.sidebar.number_input("Dias desde última interação", 0, 30, 5)

    gender_bin = 1 if gender == "Masculino" else 0
    contract_monthly = 1 if contract == "Monthly" else 0
    contract_quarterly = 1 if contract == "Quarterly" else 0
    sub_premium = 1 if subscription == "Premium" else 0
    sub_standard = 1 if subscription == "Standard" else 0
    
    data = {
        'Age': [age], 'Gender': [gender_bin], 'Tenure': [tenure],
        'Usage Frequency': [usage_freq], 'Support Calls': [support_calls],
        'Payment Delay': [payment_delay], 'Total Spend': [total_spend],
        'Last Interaction': [last_interaction], 'Subscription Type_Premium': [sub_premium],
        'Subscription Type_Standard': [sub_standard], 'Contract Length_Monthly': [contract_monthly],
        'Contract Length_Quarterly': [contract_quarterly]
    }
    return pd.DataFrame(data)

input_df = user_input_features()
st.subheader("Dados do Cliente")
st.write(input_df)

# --- PREVISÃO ---
if st.button("Calcular Risco de Churn"):
    cols_to_scale = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 
                     'Payment Delay', 'Total Spend', 'Last Interaction']
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
    
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]
    
    st.subheader(f"Resultado usando {modelo_escolhido}")
    
    if prediction[0] == 1:
        st.error(f"⚠️ ALERTA: Cliente com ALTO risco de cancelamento!")
        st.metric(label="Probabilidade", value=f"{probability:.2%}")
    else:
        st.success(f"✅ Cliente Seguro.")
        st.metric(label="Probabilidade", value=f"{probability:.2%}")