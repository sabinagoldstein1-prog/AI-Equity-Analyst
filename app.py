"""
AI Equity Analyst - Projeto Final FGV
Quant-style interface for institutional asset managers
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from engine import (fetch_prices, fetch_fundamentals, run_ml,
                    run_predictive_model, run_trading_system,
                    run_monte_carlo, run_scoring, PERFIS)
from ai_analyst import chat_with_gemini, build_context

st.set_page_config(page_title="AI Equity Analyst", page_icon="\U0001f4b9", layout="wide")

# === BLOOMBERG-STYLE CSS ===
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');
:root {--bg: #0a0e17; --card: #111827; --border: #1e2d3d; --green: #00d26a; --red: #ff3b3b; --blue: #0ea5e9; --gold: #facc15; --text: #e2e8f0; --muted: #64748b;}
.stApp {background-color: var(--bg) !important;}
.block-container {padding-top: 1rem; max-width: 1400px;}
h1, h2, h3 {font-family: 'Inter', sans-serif !important; color: var(--blue) !important; letter-spacing: -0.5px;}
.stMetric {background: var(--card); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px;}
.stMetric label {font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; color: var(--muted) !important; text-transform: uppercase; letter-spacing: 1px;}
.stMetric [data-testid="stMetricValue"] {font-family: 'JetBrains Mono', monospace !important; font-size: 22px !important;}
.stDataFrame {font-family: 'JetBrains Mono', monospace !important; font-size: 12px;}
div[data-testid="stSidebar"] {background: #070b12 !important; border-right: 1px solid var(--border);}
.stTabs [data-baseweb="tab"] {font-family: 'Inter', sans-serif; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;}
.stButton>button[kind="primary"] {background: linear-gradient(135deg, #0ea5e9, #3b82f6); border: none; font-weight: 700; letter-spacing: 0.5px;}
[data-testid="stChatMessage"] {background: var(--card) !important; border: 1px solid var(--border); border-radius: 8px;}
</style>""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#111827",
                     font=dict(family="JetBrains Mono, monospace", size=11, color="#e2e8f0"),
                     margin=dict(l=40,r=20,t=50,b=40))

# === SIDEBAR ===
with st.sidebar:
    st.markdown("## \U0001f4b9 AI EQUITY ANALYST")
    st.caption("QUANT RESEARCH TERMINAL | FGV 2026")
    st.markdown("---")
    tickers_input = st.text_area("\U0001f4cc UNIVERSE", value="PETR4\nITUB4\nBBDC4\nBBAS3\nVALE3\nWEGE3\nABEV3\nPRIO3\nEGIE3\nEQTL3", height=180)
    perfil = st.selectbox("STRATEGY PROFILE", ["conservador","moderado","agressivo"], index=1)
    data_inicio = st.date_input("LOOKBACK START", value=pd.Timestamp("2021-01-01"))
    st.markdown("---")
    p = PERFIS[perfil]
    st.code(f"WEIGHTS [{perfil.upper()}]\nMKT  = {p['mercado']:.0%}\nQUAL = {p['qualidade']:.0%}", language="yaml")
    run_btn = st.button("\U0001f680 RUN ANALYSIS", type="primary", use_container_width=True)

def parse_tickers(raw):
    tickers = [t.strip().upper() for t in raw.replace(",", " ").replace("\n", " ").split() if t.strip()]
    return [t if t.endswith(".SA") else t + ".SA" for t in tickers]

def fmt_ticker(t):
    return str(t).replace(".SA","")

keys = ["result","prices","fund","ml_metrics","ml_fi","ml_preds","ml_clusters",
        "pred_metrics","pred_fi","trading_curves","trading_summary",
        "mc_carteiras","mc_best","mc_tickers","chat_history","chat_context"]
for k in keys:
    if k not in st.session_state:
        st.session_state[k] = [] if k == "chat_history" else None

if run_btn:
    tickers = parse_tickers(tickers_input)
    if len(tickers) < 2:
        st.error("MIN 2 TICKERS REQUIRED"); st.stop()
    progress = st.progress(0, "INITIALIZING...")
    try:
        progress.progress(10, "FETCHING MARKET DATA...")
        st.session_state.prices = fetch_prices(tickers, str(data_inicio))
        tickers_ok = st.session_state.prices["ticker"].unique().tolist()
        progress.progress(25, "FETCHING FUNDAMENTALS...")
        st.session_state.fund = fetch_fundamentals(tickers_ok)
        progress.progress(40, "TRAINING RF WALK-FORWARD...")
        ml = run_ml(st.session_state.prices)
        st.session_state.ml_metrics, st.session_state.ml_fi, st.session_state.ml_preds, st.session_state.ml_clusters = ml
        progress.progress(55, "PREDICTIVE MODEL...")
        st.session_state.pred_metrics, st.session_state.pred_fi = run_predictive_model(st.session_state.prices)
        progress.progress(70, "BACKTESTING TRADING SYSTEM...")
        tc, ts = run_trading_system(st.session_state.prices)
        st.session_state.trading_curves = tc; st.session_state.trading_summary = ts
        progress.progress(85, "MONTE CARLO SIMULATION...")
        mc, best, mct = run_monte_carlo(st.session_state.prices)
        st.session_state.mc_carteiras = mc; st.session_state.mc_best = best; st.session_state.mc_tickers = mct
        progress.progress(95, "COMPUTING COMPOSITE SCORE...")
        st.session_state.result = run_scoring(st.session_state.prices, st.session_state.fund, perfil)
        st.session_state.chat_context = build_context(st.session_state.result, st.session_state.fund)
        st.session_state.chat_history = []
        progress.progress(100, "ANALYSIS COMPLETE")
    except Exception as e:
        st.error(f"ERROR: {e}")
        import traceback; st.code(traceback.format_exc())
        st.stop()

if st.session_state.result is None:
    st.markdown("""
# \U0001f4b9 AI EQUITY ANALYST
### Quantitative Research Terminal

```
SYSTEM STATUS: READY
ENGINE: 7 tools | RF Walk-Forward | Monte Carlo 10K | Gemini LLM
COVERAGE: B3 Brazilian Equities
```

**WORKFLOW:**
1. Input universe in sidebar
2. Select strategy profile
3. Click **RUN ANALYSIS**

*FGV | IA Aplicada ao Mercado Financeiro | 2026*
    """)
    st.stop()

result = st.session_state.result
prices = st.session_state.prices
fund = st.session_state.fund

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["RANKING","MARKET","ML ENGINE","SIGNALS","PORTFOLIO","AI CHAT"])

# === TAB 1: RANKING ===
with tab1:
    st.markdown("### \U0001f3c6 COMPOSITE RANKING")
    n_buy = sum("BUY" in str(r) for r in result["recomendacao"])
    n_hold = sum("HOLD" in str(r) for r in result["recomendacao"])
    n_sell = sum("SELL" in str(r) for r in result["recomendacao"])
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("UNIVERSE", len(result))
    c2.metric("BUY", n_buy)
    c3.metric("HOLD", n_hold)
    c4.metric("SELL", n_sell)
    avg_score = result["score"].mean()
    c5.metric("AVG SCORE", f"{avg_score:.1f}")
    r = result.sort_values("score", ascending=True)
    colors = ["#00d26a" if "BUY" in rec else "#ff3b3b" if "SELL" in rec else "#facc15" for rec in r["recomendacao"]]
    fig = go.Figure(go.Bar(y=r["ticker"].apply(fmt_ticker), x=r["score"], orientation="h", marker_color=colors,
                           text=[f"{s:.0f} | {rec}" for s, rec in zip(r["score"], r["recomendacao"])], textposition="outside",
                           textfont=dict(family="JetBrains Mono", size=11)))
    fig.add_vline(x=50, line_dash="dash", line_color="#64748b", opacity=0.5)
    fig.update_layout(title="COMPOSITE SCORE (0-100)", xaxis_range=[0,115], height=max(350,len(r)*42), **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    # Table
    display_cols = ["rank","ticker","nome","setor","preco","P_L","P_VP","EV_EBITDA","score","recomendacao"]
    display_ok = [c for c in display_cols if c in result.columns]
    tbl = result[display_ok].sort_values("rank").copy()
    tbl["ticker"] = tbl["ticker"].apply(fmt_ticker)
    # Format numeric columns, replace NaN with "-"
    for c in ["P_L","P_VP","EV_EBITDA"]:
        if c in tbl.columns:
            tbl[c] = tbl[c].apply(lambda v: f"{v:.1f}x" if pd.notna(v) else "-")
    if "preco" in tbl.columns:
        tbl["preco"] = tbl["preco"].apply(lambda v: f"R${v:.2f}" if pd.notna(v) else "-")
    if "nome" in tbl.columns:
        tbl["nome"] = tbl["nome"].fillna("-")
    if "setor" in tbl.columns:
        tbl["setor"] = tbl["setor"].fillna("-")
    st.dataframe(tbl, use_container_width=True, hide_index=True,
                 column_config={"score": st.column_config.ProgressColumn("SCORE", min_value=0, max_value=100, format="%.0f")})

# === TAB 2: MARKET ===
with tab2:
    st.markdown("### \U0001f4ca MARKET DASHBOARD")
    col1, col2 = st.columns(2)
    with col1:
        d = result.dropna(subset=["vol_21","mom_6m","score"])
        fig = px.scatter(d, x=d["vol_21"]*100, y=d["mom_6m"]*100, size="score", color="recomendacao",
                         text=d["ticker"].apply(fmt_ticker),
                         color_discrete_map={"BUY":"#00d26a","HOLD":"#facc15","SELL":"#ff3b3b"},
                         labels={"x":"VOLATILITY 21D (%)","y":"MOMENTUM 6M (%)"},title="RISK vs RETURN")
        fig.update_traces(textposition="top center", textfont=dict(size=10, family="JetBrains Mono"))
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        top5 = result.head(5)["ticker"].tolist()
        fig = go.Figure()
        for t in top5:
            sub = prices[prices["ticker"]==t].dropna(subset=["preco"])
            if sub.empty: continue
            norm = sub["preco"]/sub["preco"].iloc[0]*100
            fig.add_trace(go.Scatter(x=sub["data"], y=norm, name=fmt_ticker(t), mode="lines"))
        fig.add_hline(y=100, line_dash="dash", line_color="#64748b", opacity=0.4)
        fig.update_layout(title="TOP 5 PERFORMANCE (BASE 100)", yaxis_title="CUMULATIVE (%)", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    # Fundamentals table
    st.markdown("#### FUNDAMENTALS SNAPSHOT")
    fund_display = fund[["ticker","nome","setor","preco","P_L","P_VP","EV_EBITDA","returnOnEquity","profitMargins","revenueGrowth"]].copy()
    fund_display["ticker"] = fund_display["ticker"].apply(fmt_ticker)
    for c in ["P_L","P_VP","EV_EBITDA"]:
        fund_display[c] = fund_display[c].apply(lambda v: f"{v:.1f}x" if pd.notna(v) else "-")
    for c in ["returnOnEquity","profitMargins","revenueGrowth"]:
        fund_display[c] = fund_display[c].apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "-")
    fund_display["preco"] = fund_display["preco"].apply(lambda v: f"R${v:.2f}" if pd.notna(v) else "-")
    fund_display.columns = ["TICKER","NAME","SECTOR","PRICE","P/E","P/B","EV/EBITDA","ROE","MARGIN","GROWTH"]
    st.dataframe(fund_display, use_container_width=True, hide_index=True)

# === TAB 3: ML ENGINE ===
with tab3:
    st.markdown("### \U0001f916 ML ENGINE | RANDOM FOREST")
    col1, col2 = st.columns(2)
    with col1:
        ml_m = st.session_state.ml_metrics
        if ml_m is not None and not ml_m.empty:
            st.markdown("#### WALK-FORWARD VALIDATION")
            st.dataframe(ml_m.round(4), use_container_width=True, hide_index=True)
            fig = px.bar(ml_m, x="year", y="spearman_ic", title="SPEARMAN IC BY YEAR",
                         color="spearman_ic", color_continuous_scale=["#ff3b3b","#facc15","#00d26a"], color_continuous_midpoint=0)
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for walk-forward (needs 3+ years).")
    with col2:
        fi = st.session_state.ml_fi
        if fi is not None and not fi.empty:
            st.markdown("#### FEATURE IMPORTANCE")
            fi_mean = fi.mean().sort_values(ascending=True)
            fig = go.Figure(go.Bar(y=fi_mean.index, x=fi_mean.values, orientation="h", marker_color="#0ea5e9"))
            fig.update_layout(title="AVG IMPORTANCE", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
    clust = st.session_state.ml_clusters
    if clust is not None and not clust.empty and "perfil" in clust.columns:
        st.markdown("#### CLUSTER PROFILES")
        cl = clust[["ticker","perfil","vol_21","mom_6m","drawdown"]].copy()
        cl["ticker"] = cl["ticker"].apply(fmt_ticker)
        for c in ["vol_21","mom_6m","drawdown"]:
            cl[c] = cl[c].apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "-")
        st.dataframe(cl.sort_values("perfil"), use_container_width=True, hide_index=True)
    preds = st.session_state.ml_preds
    if preds is not None and not preds.empty and "pred_ret_12m" in preds.columns:
        st.markdown("#### 12M RETURN FORECAST")
        pp = preds[["ticker","pred_ret_12m","rank_pred"]].dropna().copy()
        pp["ticker"] = pp["ticker"].apply(fmt_ticker)
        pp["pred_ret_12m"] = (pp["pred_ret_12m"]*100).round(1)
        pp = pp.sort_values("rank_pred")
        fig = go.Figure(go.Bar(x=pp["ticker"], y=pp["pred_ret_12m"],
                                marker_color=["#00d26a" if v>0 else "#ff3b3b" for v in pp["pred_ret_12m"]],
                                text=[f"{v:+.1f}%" for v in pp["pred_ret_12m"]], textposition="outside"))
        fig.update_layout(title="PREDICTED 12M RETURN (%)", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# === TAB 4: SIGNALS ===
with tab4:
    st.markdown("### \U0001f3af SIGNALS | PREDICTIVE MODEL & TRADING SYSTEM")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### DAILY DIRECTION FORECAST")
        pm = st.session_state.pred_metrics
        if pm is not None and not pm.empty:
            pm_d = pm.copy(); pm_d["ticker"] = pm_d["ticker"].apply(fmt_ticker)
            fig = px.bar(pm_d, x="ticker", y="auc", title="AUC BY TICKER (0.5 = random)",
                         color="auc", color_continuous_scale=["#ff3b3b","#facc15","#00d26a"], color_continuous_midpoint=0.5)
            fig.add_hline(y=0.5, line_dash="dash", line_color="#ff3b3b", opacity=0.5)
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data.")
    with col2:
        st.markdown("#### MOVING AVERAGE CROSSOVER")
        ts = st.session_state.trading_summary
        if ts is not None and not ts.empty:
            ts_d = ts.copy(); ts_d["ticker"] = ts_d["ticker"].apply(fmt_ticker)
            fig = go.Figure()
            fig.add_trace(go.Bar(name="BUY & HOLD", x=ts_d["ticker"], y=ts_d["ret_buyhold"], marker_color="#0ea5e9"))
            fig.add_trace(go.Bar(name="MA STRATEGY", x=ts_d["ticker"], y=ts_d["ret_estrategia"], marker_color="#00d26a"))
            fig.update_layout(barmode="group", title="CUMULATIVE RETURN (%)", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
    # Trading curves
    tc = st.session_state.trading_curves
    ts = st.session_state.trading_summary
    if tc is not None and not tc.empty and ts is not None and not ts.empty:
        sel = st.selectbox("SELECT TICKER:", [fmt_ticker(t) for t in ts["ticker"].tolist()])
        sel_sa = sel + ".SA" if not sel.endswith(".SA") else sel
        sub = tc[tc["ticker"]==sel_sa]
        if not sub.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sub["data"],y=sub["acum_acao"],name="BUY & HOLD",line=dict(color="#0ea5e9",width=2)))
            fig.add_trace(go.Scatter(x=sub["data"],y=sub["acum_est"],name="MA STRATEGY",line=dict(color="#00d26a",width=2)))
            fig.update_layout(title=f"EQUITY CURVE: {sel}", yaxis_title="CUMULATIVE", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

# === TAB 5: PORTFOLIO ===
with tab5:
    st.markdown("### \U0001f3b2 PORTFOLIO OPTIMIZATION | MONTE CARLO")
    mc = st.session_state.mc_carteiras
    best = st.session_state.mc_best
    mct = st.session_state.mc_tickers
    if mc is not None and not mc.empty and best:
        col1, col2 = st.columns([2,1])
        with col1:
            fig = px.scatter(mc, x="risco", y="retorno", color="sharpe", color_continuous_scale="viridis",
                             labels={"risco":"ANNUAL RISK","retorno":"ANNUAL RETURN","sharpe":"SHARPE"},
                             title="EFFICIENT FRONTIER | 10,000 SIMULATIONS")
            fig.add_trace(go.Scatter(x=[best["risco"]],y=[best["retorno"]],mode="markers",
                                     marker=dict(size=18,color="#ff3b3b",symbol="star",line=dict(width=2,color="white")),name="MAX SHARPE"))
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### OPTIMAL PORTFOLIO")
            st.metric("SHARPE RATIO", f"{best['sharpe']:.3f}")
            st.metric("ANNUAL RETURN", f"{best['retorno']*100:.1f}%")
            st.metric("ANNUAL RISK", f"{best['risco']*100:.1f}%")
            st.markdown("---")
            st.markdown("#### ALLOCATION")
            alloc_data = [(fmt_ticker(t), best.get(t,0)*100) for t in mct]
            alloc_data.sort(key=lambda x: -x[1])
            for tk, w in alloc_data:
                bar_width = int(w * 2)
                color = "#00d26a" if w > 15 else "#0ea5e9"
                st.markdown(f"`{tk:>6s}` {'█' * max(1, bar_width)} **{w:.1f}%**")
    else:
        st.info("Insufficient data for Monte Carlo (needs 2+ tickers).")

# === TAB 6: AI CHAT ===
with tab6:
    st.markdown("### \U0001f4ac AI RESEARCH ASSISTANT")
    has_key = bool(st.secrets.get("GEMINI_API_KEY", ""))
    if not has_key:
        st.warning("Add GEMINI_API_KEY in Streamlit Cloud Secrets to enable AI chat. Get free key at https://aistudio.google.com/apikey")
    else:
        st.success("GEMINI CONNECTED | Ask about any ticker in the universe")
    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(msg["user"])
        with st.chat_message("assistant"):
            st.write(msg["assistant"])
    user_msg = st.chat_input("Ask about any company in the universe...")
    if user_msg and has_key:
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("ANALYZING..."):
                response = chat_with_gemini(user_msg, st.session_state.chat_context or "", st.session_state.chat_history)
            st.write(response)
        st.session_state.chat_history.append({"user": user_msg, "assistant": response})

st.sidebar.markdown("---")
st.sidebar.code("AI EQUITY ANALYST v3.0\nFGV | 2026", language="yaml")
