"""
AI Equity Analyst - Projeto Final FGV
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from engine import (fetch_prices, fetch_fundamentals, run_nlp,
                    run_valuation, run_ml, run_scoring, PERFIS, safe_div)
from ai_analyst import generate_company_analysis, generate_executive_summary

st.set_page_config(page_title="AI Equity Analyst", page_icon="\U0001f916", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.stMetric {background:#1E293B;padding:12px;border-radius:8px;border-left:4px solid #3B82F6;}
.block-container {padding-top:1rem;}
h1 {color:#3B82F6 !important;}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.title("\U0001f916 AI Equity Analyst")
    st.caption("Projeto Final - IA no Mercado Financeiro - FGV")
    st.divider()
    tickers_input = st.text_area("Tickers (um por linha)", value="PETR4\nITUB4\nBBDC4\nBBAS3\nVALE3\nWEGE3\nABEV3\nPRIO3\nEGIE3\nEQTL3", height=180)
    perfil = st.selectbox("Perfil", ["conservador","moderado","agressivo"], index=1)
    data_inicio = st.date_input("Data Inicio", value=pd.Timestamp("2021-01-01"))
    st.divider()
    p = PERFIS[perfil]
    st.markdown(f"**Pesos ({perfil}):** Mercado {p['mercado']:.0%} | Val {p['valuation']:.0%} | NLP {p['nlp']:.0%} | Qual {p['qualidade']:.0%}")
    run_btn = st.button("\U0001f680 Analisar", type="primary", use_container_width=True)

def parse_tickers(raw):
    tickers = [t.strip().upper() for t in raw.replace(",", " ").replace("\n", " ").split() if t.strip()]
    return [t if t.endswith(".SA") else t + ".SA" for t in tickers]

for key in ["result","prices","fund","nlp_data","val_df","sens_df","ml_metrics","ml_fi","ml_preds","ml_clusters"]:
    if key not in st.session_state:
        st.session_state[key] = None

if run_btn:
    tickers = parse_tickers(tickers_input)
    if len(tickers) < 2:
        st.error("Insira ao menos 2 tickers.")
        st.stop()
    progress = st.progress(0, "Iniciando...")
    try:
        progress.progress(10, "Precos de mercado...")
        st.session_state.prices = fetch_prices(tickers, str(data_inicio))
        tickers_ok = st.session_state.prices["ticker"].unique().tolist()
        progress.progress(30, "Fundamentos e multiplos...")
        st.session_state.fund = fetch_fundamentals(tickers_ok)
        progress.progress(45, "Sentimento NLP...")
        st.session_state.nlp_data = run_nlp(st.session_state.fund)
        progress.progress(60, "Valuation e cenarios...")
        st.session_state.val_df, st.session_state.sens_df = run_valuation(st.session_state.fund)
        progress.progress(75, "Random Forest walk-forward...")
        ml = run_ml(st.session_state.prices)
        st.session_state.ml_metrics, st.session_state.ml_fi, st.session_state.ml_preds, st.session_state.ml_clusters = ml
        progress.progress(90, "Score composto...")
        st.session_state.result = run_scoring(st.session_state.prices, st.session_state.fund, st.session_state.nlp_data, st.session_state.val_df, st.session_state.ml_preds, perfil)
        progress.progress(100, "Completo!")
    except Exception as e:
        st.error(f"Erro: {e}")
        st.stop()

if st.session_state.result is None:
    st.markdown("# \U0001f916 AI Equity Analyst\n### Agente de IA Generativa para Analise de Empresas\n\n"
                "1. Digite os tickers na barra lateral\n2. Escolha o perfil\n3. Clique **Analisar**\n\n"
                "*Projeto Final - FGV 2026*")
    st.stop()

result = st.session_state.result
prices = st.session_state.prices
fund = st.session_state.fund
nlp_df = st.session_state.nlp_data
val_df = st.session_state.val_df
sens = st.session_state.sens_df

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["\U0001f3c6 Ranking","\U0001f4ca Dashboard","\U0001f4dd NLP","\U0001f4b0 Valuation","\U0001f916 ML","\U0001f4cb Dossie IA"])

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
    cols_ok = [c for c in ["rank","ticker","nome","setor","preco","score","recomendacao","gap_pct","sent_label"] if c in result.columns]
    st.dataframe(result[cols_ok].sort_values("rank"), use_container_width=True, hide_index=True,
                 column_config={"preco": st.column_config.NumberColumn("Preco", format="R$%.2f"),
                                "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
                                "gap_pct": st.column_config.NumberColumn("Gap", format="%+.1f%%")})

with tab2:
    st.header("Dashboard de Mercado")
    col1, col2 = st.columns(2)
    with col1:
        d = result.dropna(subset=["vol_21","mom_6m","score"])
        fig = px.scatter(d, x=d["vol_21"]*100, y=d["mom_6m"]*100, size="score", color="recomendacao",
                         text=d["ticker"].str.replace(".SA",""),
                         color_discrete_map={"\U0001f7e2 Buy":"#22C55E","\U0001f7e1 Hold":"#F59E0B","\U0001f534 Sell":"#EF4444"},
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

with tab3:
    st.header("NLP - Sentimento e Risco")
    d = nlp_df.sort_values("indice_textual", ascending=True)
    colors = ["#22C55E" if l=="positivo" else "#EF4444" if l=="negativo" else "#F59E0B" for l in d["sent_label"]]
    fig = go.Figure(go.Bar(y=d["ticker"].str.replace(".SA",""), x=d["indice_textual"], orientation="h", marker_color=colors,
                           text=[f"{v:.0f} ({l})" for v,l in zip(d["indice_textual"],d["sent_label"])], textposition="outside"))
    fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(title="Indice Textual (0-100)", xaxis_range=[0,115], height=max(300,len(d)*45),
                      template="plotly_dark", paper_bgcolor="#0F172A", plot_bgcolor="#1E293B")
    st.plotly_chart(fig, use_container_width=True)
    topic_cols = [c for c in nlp_df.columns if c.startswith("topic_")]
    if topic_cols:
        st.subheader("Topicos de Risco")
        dd = nlp_df[["ticker","sent_score","sent_label","indice_textual"]+topic_cols].copy()
        dd["ticker"] = dd["ticker"].str.replace(".SA","")
        st.dataframe(dd.sort_values("indice_textual", ascending=False), use_container_width=True, hide_index=True)

with tab4:
    st.header("Valuation por Multiplos Setoriais")
    col1, col2 = st.columns(2)
    with col1:
        d = val_df.dropna(subset=["impl_price","preco"]).sort_values("gap_pct")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Mercado", y=d["ticker"].str.replace(".SA",""), x=d["preco"], orientation="h", marker_color="#3B82F6"))
        fig.add_trace(go.Bar(name="Implicito", y=d["ticker"].str.replace(".SA",""), x=d["impl_price"], orientation="h", marker_color="#22C55E"))
        fig.update_layout(barmode="group", title="Preco Mercado vs Implicito", template="plotly_dark",
                          paper_bgcolor="#0F172A", plot_bgcolor="#1E293B", height=max(300,len(d)*50))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if not sens.empty:
            piv = sens.pivot_table(index="ticker", columns="scenario", values="gap_pct", aggfunc="first")
            piv.index = piv.index.str.replace(".SA","")
            order = [c for c in ["dove -200bp","base","hawk +200bp","fx up +10%","fx dn -10%"] if c in piv.columns]
            piv = piv[order] if order else piv
            fig = px.imshow(piv, text_auto=".1f", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                            title="Sensibilidade Macro (Gap %)", labels={"color":"Gap %"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0F172A", height=max(300,len(piv)*45))
            st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Machine Learning - Random Forest")
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

with tab6:
    st.header("Dossie - IA Generativa")
    has_key = bool(st.secrets.get("GEMINI_API_KEY", ""))
    if not has_key:
        st.warning("API Key do Gemini nao configurada. Usando analise heuristica. Adicione GEMINI_API_KEY nos Secrets.")
    with st.spinner("Gerando resumo executivo..."):
        summary = generate_executive_summary(result)
    st.markdown(f"### Resumo Executivo\n\n{summary}")
    st.divider()
    for _, row in result.iterrows():
        tk = row["ticker"].replace(".SA","")
        with st.expander(f"**{row['rank']:.0f}. {tk}** - {row.get('nome','')} | {row['recomendacao']} | Score: {row['score']:.0f}"):
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Preco", f"R${row.get('preco',0):.2f}")
            gap = row.get("gap_pct", 0)
            c2.metric("Gap Val.", f"{gap:+.1f}%" if pd.notna(gap) else "N/A", delta=f"{gap:+.1f}%" if pd.notna(gap) else None)
            c3.metric("P/L", f"{row.get('P_L',0):.1f}x" if pd.notna(row.get("P_L")) else "N/A")
            c4.metric("Vol 21d", f"{row.get('vol_21',0)*100:.1f}%")
            with st.spinner(f"Analisando {tk}..."):
                analysis = generate_company_analysis(row.to_dict())
            st.markdown(analysis)
with tab7:
    st.header("🗂️ Asset Management & Risk Control")
    
    if st.session_state.get("prices") is not None:
        risk_data = run_asset_manager_analysis(st.session_state.prices)
        
        # --- LINHA 1: Métricas de Risco de Cauda ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("VaR Histórico Diário (95%)", f"{risk_data['var']:.2%}", 
                      help="Potencial de perda máxima em 95% dos dias.")
        with col2:
            st.metric("CVaR (Expected Shortfall)", f"{risk_data['cvar']:.2%}",
                      help="Média das perdas caso o VaR seja ultrapassado (Risco de Cauda).")
        with col3:
            avg_beta = np.mean(list(risk_data['betas'].values()))
            st.metric("Beta Médio da Carteira", f"{avg_beta:.2f}",
                      help="Sensibilidade da carteira em relação ao mercado analisado.")

        st.divider()

        # --- LINHA 2: Análise de Correlação e Alocação ---
        c_left, c_right = st.columns([2, 1])
        
        with c_left:
            st.subheader("Matriz de Correlação Asset-Level")
            fig_corr = px.imshow(risk_data['corr'], 
                                 text_auto=".2f", 
                                 color_continuous_scale='RdBu_r',
                                 aspect="auto")
            fig_corr.update_layout(template="plotly_dark")
            st.plotly_chart(fig_corr, use_container_width=True)
            st.info("💡 Gestores buscam correlações baixas (azul) para maximizar o benefício da diversificação.")

        with c_right:
            st.subheader("Exposição por Ticker (Beta)")
            beta_df = pd.DataFrame.from_dict(risk_data['betas'], orient='index', columns=['Beta']).sort_values('Beta')
            fig_beta = px.bar(beta_df, x='Beta', orientation='h', 
                               color='Beta', color_continuous_scale='Viridis')
            fig_beta.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_beta, use_container_width=True)

        # --- LINHA 3: Stress Test Simples ---
        st.subheader("⚠️ Stress Test (Cenários Históricos)")
        st.write("Impacto estimado na carteira em eventos de estresse:")
        
        stress_scenarios = {
            "Crise 2008 (Proxy -5%)": -0.05,
            "Circuit Breaker (Proxy -10%)": -0.10,
            "Cenário Otimista (+3%)": 0.03
        }
        
        s_cols = st.columns(len(stress_scenarios))
        for i, (name, impact) in enumerate(stress_scenarios.items()):
            # Cálculo simplificado: impacto * beta médio
            est_impact = impact * avg_beta
            s_cols[i].metric(name, f"{est_impact:+.2%}")

st.sidebar.divider()
st.sidebar.caption("AI Equity Analyst v1.0 - FGV 2026")
