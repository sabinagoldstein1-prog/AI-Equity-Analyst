import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from engine import (fetch_prices, fetch_fundamentals, run_nlp, 
                    run_valuation, run_ml, run_scoring, run_asset_manager_analysis)
from ai_analyst import generate_company_analysis, generate_executive_summary

st.set_page_config(page_title="AI Equity Analyst", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🤖 AI Equity Analyst")
    tickers_input = st.text_area("Tickers", value="PETR4.SA\nITUB4.SA\nVALE3.SA\nWEGE3.SA")
    perfil = st.selectbox("Perfil", ["conservador", "moderado", "agressivo"])
    run_btn = st.button("Analisar")

# Criação das 7 ABAS (Importante: 7 variáveis)
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏆 Ranking", "📊 Dashboard", "📝 NLP", "💰 Valuation", "🤖 ML", "📋 Dossie IA", "🗂️ Asset Management"
])

if run_btn:
    tickers = [t.strip() for t in tickers_input.split('\n') if t.strip()]
    with st.spinner("Processando..."):
        # 1. Processamento
        prices_all, price_metrics = fetch_prices(tickers)
        fund_df = fetch_fundamentals(tickers)
        nlp_df = run_nlp(tickers)
        val_df = run_valuation(tickers, fund_df)
        ml_df = run_ml(prices_all)
        
        # 2. Scoring e Asset Management
        result = run_scoring(price_metrics, fund_df, nlp_df, val_df, ml_df, perfil)
        risk_data = run_asset_manager_analysis(prices_all)
        
        # 3. Guardar no Session State
        st.session_state.result = result
        st.session_state.risk_data = risk_data
        st.session_state.prices = prices_all

# Renderização das Abas
if "result" in st.session_state:
    res = st.session_state.result
    
    with tab1:
        st.dataframe(res)
    
    with tab2:
        fig = px.scatter(res, x="vol_21", y="score", text="ticker", color="recomendacao")
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.header("📋 Dossiê IA")
        st.write(generate_executive_summary(res))

    # --- NOVA SEÇÃO 3.2 ---
    with tab7:
        st.header("🗂️ Asset Management & Risk Control")
        risk = st.session_state.risk_data
        
        c1, c2, c3 = st.columns(3)
        c1.metric("VaR Diário (95%)", f"{risk['var']:.2%}")
        c2.metric("CVaR (Expected Shortfall)", f"{risk['cvar']:.2%}")
        c3.metric("Ativos Analisados", len(risk['betas']))
        
        st.subheader("Matriz de Correlação")
        fig_corr = px.imshow(risk['corr'], text_auto=".2f", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.subheader("Exposição (Beta)")
        beta_df = pd.DataFrame.from_dict(risk['betas'], orient='index', columns=['Beta'])
        st.bar_chart(beta_df)

else:
    st.info("Digite os tickers e clique em Analisar para começar.")
