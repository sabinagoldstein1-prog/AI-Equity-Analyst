"""
AI Equity Analyst - Projeto Final FGV
Agente de IA Generativa + Chatbot Financeiro
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from engine import (fetch_prices, fetch_fundamentals, run_ml,
                    run_predictive_model, run_trading_system,
                    run_monte_carlo, run_scoring, PERFIS, safe_div)
from ai_analyst import chat_with_gemini, build_context

st.set_page_config(page_title="AI Equity Analyst", page_icon="\U0001f916", layout="wide")
st.markdown("""<style>
.stMetric {background:#1E293B;padding:12px;border-radius:8px;border-left:4px solid #3B82F6;}
.block-container {padding-top:1rem;}
h1 {color:#3B82F6 !important;}
</style>""", unsafe_allow_html=True)

# === SIDEBAR ===
with st.sidebar:
    st.title("\U0001f916 AI Equity Analyst")
    st.caption("Projeto Final - IA no Mercado Financeiro - FGV")
    st.divider()
    tickers_input = st.text_area("Tickers (um por linha)", value="PETR4\nITUB4\nBBDC4\nBBAS3\nVALE3\nWEGE3\nABEV3\nPRIO3\nEGIE3\nEQTL3", height=180)
    perfil = st.selectbox("Perfil", ["conservador","moderado","agressivo"], index=1)
    data_inicio = st.date_input("Data Inicio", value=pd.Timestamp("2021-01-01"))
    st.divider()
    p = PERFIS[perfil]
    st.markdown(f"**Pesos ({perfil}):** Mercado {p['mercado']:.0%} | Qual {p['qualidade']:.0%}")
    run_btn = st.button("\U0001f680 Analisar", type="primary", use_container_width=True)

def parse_tickers(raw):
    tickers = [t.strip().upper() for t in raw.replace(",", " ").replace("\n", " ").split() if t.strip()]
    return [t if t.endswith(".SA") else t + ".SA" for t in tickers]

keys = ["result","prices","fund","ml_metrics","ml_fi","ml_preds","ml_clusters",
        "pred_metrics","pred_fi","trading_curves","trading_summary",
        "mc_carteiras","mc_best","mc_tickers","chat_history","chat_context"]
for k in keys:
    if k not in st.session_state:
        st.session_state[k] = [] if k == "chat_history" else None

if run_btn:
    tickers = parse_tickers(tickers_input)
    if len(tickers) < 2:
        st.error("Insira ao menos 2 tickers.")
        st.stop()
    progress = st.progress(0, "Iniciando...")
    try:
        progress.progress(10, "Precos...")
        st.session_state.prices = fetch_prices(tickers, str(data_inicio))
        tickers_ok = st.session_state.prices["ticker"].unique().tolist()
        progress.progress(25, "Fundamentos...")
        st.session_state.fund = fetch_fundamentals(tickers_ok)
        progress.progress(40, "Random Forest walk-forward...")
        ml = run_ml(st.session_state.prices)
        st.session_state.ml_metrics, st.session_state.ml_fi, st.session_state.ml_preds, st.session_state.ml_clusters = ml
        progress.progress(55, "Modelo preditivo...")
        st.session_state.pred_metrics, st.session_state.pred_fi = run_predictive_model(st.session_state.prices)
        progress.progress(70, "Trading system...")
        tc, ts = run_trading_system(st.session_state.prices)
        st.session_state.trading_curves = tc
        st.session_state.trading_summary = ts
        progress.progress(85, "Monte Carlo...")
        mc, best, mct = run_monte_carlo(st.session_state.prices)
        st.session_state.mc_carteiras = mc
        st.session_state.mc_best = best
        st.session_state.mc_tickers = mct
        progress.progress(95, "Score composto...")
        st.session_state.result = run_scoring(st.session_state.prices, st.session_state.fund, perfil)
        st.session_state.chat_context = build_context(st.session_state.result, st.session_state.fund)
        st.session_state.chat_history = []
        progress.progress(100, "Completo!")
    except Exception as e:
        st.error(f"Erro: {e}")
        import traceback; st.code(traceback.format_exc())
        st.stop()

if st.session_state.result is None:
    st.markdown("# \U0001f916 AI Equity Analyst\n"
                "### Agente de IA Generativa para Analise de Empresas\n\n"
                "1. Digite os tickers na barra lateral\n"
                "2. Escolha o perfil\n"
                "3. Clique **Analisar**\n\n"
                "*Projeto Final - FGV 2026*")
    st.stop()

result = st.session_state.result
prices = st.session_state.prices
fund = st.session_state.fund

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "\U0001f3c6 Ranking","\U0001f4ca Dashboard","\U0001f916 ML",
    "\U0001f3af Preditivo","\U0001f4c8 Trading & Portfolio","\U0001f4ac Chatbot IA"])

# === ABA 1: RANKING ===
with tab1:
    st.header("Ranking Final")
    n_buy = sum("Buy" in str(r) for r in result["recomendacao"])
    n_hold = sum("Hold" in str(r) for r in result["recomendacao"])
    n_sell = sum("Sell" in str(r) for r in result["recomendacao"])
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Empresas", len(result)); c2.metric("Buy", n_buy); c3.metric("Hold", n_hold); c4.metric("Sell", n_sell)
    r = result.sort_values("score", ascending=True)
    colors = ["#22C55E" if "Buy" in rec else "#EF4444" if "Sell" in rec else "#F59E0B" for rec in r["recomendacao"]]
    fig = go.Figure(go.Bar(y=r["ticker"].str.replace(".SA",""), x=r["score"], orientation="h", marker_color=colors,
                           text=[f"{s:.0f} {rec}" for s, rec in zip(r["score"], r["recomendacao"])], textposition="outside"))
    fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(title="Score Composto (0-100)", xaxis_range=[0,115], height=max(300,len(r)*45),
                      template="plotly_dark", paper_bgcolor="#0F172A", plot_bgcolor="#1E293B")
    st.plotly_chart(fig, use_container_width=True)
    cols_ok = [c for c in ["rank","ticker","nome","setor","preco","score","recomendacao","P_L","P_VP"] if c in result.columns]
    st.dataframe(result[cols_ok].sort_values("rank"), use_container_width=True, hide_index=True,
                 column_config={"preco": st.column_config.NumberColumn("Preco", format="R$%.2f"),
                                "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100)})

# === ABA 2: DASHBOARD ===
with tab2:
    st.header("Dashboard de Mercado")
    col1, col2 = st.columns(2)
    with col1:
        d = result.dropna(subset=["vol_21","mom_6m","score"])
        fig = px.scatter(d, x=d["vol_21"]*100, y=d["mom_6m"]*100, size="score", color="recomendacao",
                         text=d["ticker"].str.replace(".SA",""),
                         color_discrete_map={"Buy":"#22C55E","Hold":"#F59E0B","Sell":"#EF4444"},
                         labels={"x":"Volatilidade 21d (%)","y":"Momentum 6m (%)"},title="Risco x Retorno")
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0F172A", plot_bgcolor="#1E293B")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        top5 = result.head(5)["ticker"].tolist()
        fig = go.Figure()
        for t in top5:
            sub = prices[prices["ticker"]==t].dropna(subset=["preco"])
            if sub.empty: continue
            norm = sub["preco"]/sub["preco"].iloc[0]*100
            fig.add_trace(go.Scatter(x=sub["data"], y=norm, name=t.replace(".SA",""), mode="lines"))
        fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.4)
        fig.update_layout(title="Evolucao Top 5 (base 100)", yaxis_title="Performance (%)",
                          template="plotly_dark", paper_bgcolor="#0F172A", plot_bgcolor="#1E293B")
        st.plotly_chart(fig, use_container_width=True)

# === ABA 3: ML ===
with tab3:
    st.header("Machine Learning - Random Forest Walk-Forward")
    col1, col2 = st.columns(2)
    with col1:
        ml_m = st.session_state.ml_metrics
        if ml_m is not None and not ml_m.empty:
            st.subheader("Walk-Forward")
            st.dataframe(ml_m.round(4), use_container_width=True, hide_index=True)
            fig = px.bar(ml_m, x="year", y="spearman_ic", title="Spearman IC por Ano",
                         color="spearman_ic", color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0F172A", plot_bgcolor="#1E293B")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dados insuficientes para walk-forward.")
    with col2:
        fi = st.session_state.ml_fi
        if fi is not None and not fi.empty:
            st.subheader("Feature Importance")
            fi_mean = fi.mean().sort_values(ascending=True)
            fig = go.Figure(go.Bar(y=fi_mean.index, x=fi_mean.values, orientation="h", marker_color="#3B82F6"))
            fig.update_layout(title="Importancia Media", template="plotly_dark", paper_bgcolor="#0F172A", plot_bgcolor="#1E293B")
            st.plotly_chart(fig, use_container_width=True)
    clust = st.session_state.ml_clusters
    if clust is not None and not clust.empty and "perfil" in clust.columns:
        st.subheader("Clusters")
        cl = clust[["ticker","perfil","vol_21","mom_6m","drawdown"]].copy()
        cl["ticker"] = cl["ticker"].str.replace(".SA","")
        for c in ["vol_21","mom_6m","drawdown"]:
            if c in cl.columns: cl[c] = (cl[c]*100).round(1)
        st.dataframe(cl.sort_values("perfil"), use_container_width=True, hide_index=True)
    preds = st.session_state.ml_preds
    if preds is not None and not preds.empty and "pred_ret_12m" in preds.columns:
        st.subheader("Previsao Retorno 12m")
        pp = preds[["ticker","pred_ret_12m","rank_pred"]].dropna().copy()
        pp["ticker"] = pp["ticker"].str.replace(".SA","")
        pp["pred_ret_12m"] = (pp["pred_ret_12m"]*100).round(1)
        pp = pp.sort_values("rank_pred")
        fig = go.Figure(go.Bar(x=pp["ticker"], y=pp["pred_ret_12m"],
                                marker_color=["#22C55E" if v>0 else "#EF4444" for v in pp["pred_ret_12m"]],
                                text=[f"{v:+.1f}%" for v in pp["pred_ret_12m"]], textposition="outside"))
        fig.update_layout(title="Retorno 12m Previsto (%)", template="plotly_dark", paper_bgcolor="#0F172A", plot_bgcolor="#1E293B")
        st.plotly_chart(fig, use_container_width=True)

# === ABA 4: PREDITIVO ===
with tab4:
    st.header("Modelo Preditivo - Previsao Diaria")
    pm = st.session_state.pred_metrics
    pfi = st.session_state.pred_fi
    if pm is not None and not pm.empty:
        col1, col2 = st.columns(2)
        with col1:
            pm_d = pm.copy(); pm_d["ticker"] = pm_d["ticker"].str.replace(".SA","")
            st.dataframe(pm_d, use_container_width=True, hide_index=True)
            fig = px.bar(pm_d, x="ticker", y="auc", title="AUC por Ticker",
                         color="auc", color_continuous_scale="RdYlGn", color_continuous_midpoint=0.5)
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.5)
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0F172A", plot_bgcolor="#1E293B")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if pfi is not None and not pfi.empty:
                fi_mean = pfi.mean().sort_values(ascending=True)
                fig = go.Figure(go.Bar(y=fi_mean.index, x=fi_mean.values, orientation="h", marker_color="#3B82F6"))
                fig.update_layout(title="Feature Importance", template="plotly_dark", paper_bgcolor="#0F172A", plot_bgcolor="#1E293B")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Dados insuficientes.")

# === ABA 5: TRADING & PORTFOLIO ===
with tab5:
    st.header("Trading System & Portfolio Monte Carlo")
    col_t, col_p = st.columns(2)
    with col_t:
        st.subheader("Trading - Medias Moveis")
        ts = st.session_state.trading_summary
        if ts is not None and not ts.empty:
            ts_d = ts.copy(); ts_d["ticker"] = ts_d["ticker"].str.replace(".SA","")
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Buy & Hold", x=ts_d["ticker"], y=ts_d["ret_buyhold"], marker_color="#3B82F6"))
            fig.add_trace(go.Bar(name="Estrategia", x=ts_d["ticker"], y=ts_d["ret_estrategia"], marker_color="#22C55E"))
            fig.update_layout(barmode="group", title="Retorno (%)", template="plotly_dark",
                              paper_bgcolor="#0F172A", plot_bgcolor="#1E293B")
            st.plotly_chart(fig, use_container_width=True)
            tc = st.session_state.trading_curves
            if tc is not None and not tc.empty:
                sel = st.selectbox("Ticker:", ts_d["ticker"].tolist())
                sel_sa = sel+".SA" if not sel.endswith(".SA") else sel
                sub = tc[tc["ticker"]==sel_sa]
                if not sub.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=sub["data"],y=sub["acum_acao"],name="Buy&Hold",line=dict(color="#3B82F6")))
                    fig.add_trace(go.Scatter(x=sub["data"],y=sub["acum_est"],name="Estrategia",line=dict(color="#22C55E")))
                    fig.update_layout(title=f"Evolucao: {sel}", template="plotly_dark",
                                      paper_bgcolor="#0F172A", plot_bgcolor="#1E293B")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dados insuficientes.")
    with col_p:
        st.subheader("Monte Carlo - 10k Carteiras")
        mc = st.session_state.mc_carteiras
        best = st.session_state.mc_best
        mct = st.session_state.mc_tickers
        if mc is not None and not mc.empty and best:
            fig = px.scatter(mc, x="risco", y="retorno", color="sharpe", color_continuous_scale="viridis",
                             labels={"risco":"Risco","retorno":"Retorno","sharpe":"Sharpe"},
                             title="Fronteira Eficiente")
            fig.add_trace(go.Scatter(x=[best["risco"]],y=[best["retorno"]],mode="markers",
                                     marker=dict(size=15,color="red",symbol="star"),name="Melhor"))
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0F172A", plot_bgcolor="#1E293B")
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Sharpe", f"{best['sharpe']:.2f}")
            st.metric("Retorno", f"{best['retorno']*100:.1f}%")
            st.metric("Risco", f"{best['risco']*100:.1f}%")
            st.divider()
            st.markdown("**Alocacao Otima:**")
            for t in mct:
                w = best.get(t, 0)
                st.markdown(f"- {t.replace('.SA','')}: **{w*100:.1f}%**")
        else:
            st.info("Dados insuficientes.")

# === ABA 6: CHATBOT IA ===
with tab6:
    st.header("Chatbot Financeiro - IA Generativa")
    has_key = bool(st.secrets.get("GEMINI_API_KEY", ""))
    if not has_key:
        st.warning("Adicione GEMINI_API_KEY nos Secrets do Streamlit Cloud para ativar o chatbot com IA.")
        st.info("Sem a key, o chatbot nao funciona. Acesse https://aistudio.google.com/apikey para criar uma gratis.")
    else:
        st.success("Gemini conectado! Pergunte sobre qualquer empresa do universo analisado.")
    # Chat history display
    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(msg["user"])
        with st.chat_message("assistant"):
            st.write(msg["assistant"])
    # Chat input
    user_msg = st.chat_input("Pergunte sobre as empresas analisadas...")
    if user_msg and has_key:
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                response = chat_with_gemini(
                    user_msg,
                    st.session_state.chat_context or "",
                    st.session_state.chat_history
                )
            st.write(response)
        st.session_state.chat_history.append({"user": user_msg, "assistant": response})

st.sidebar.divider()
st.sidebar.caption("AI Equity Analyst v2.0 - FGV 2026")
