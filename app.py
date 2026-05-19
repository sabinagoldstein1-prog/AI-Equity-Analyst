"""
AI Equity Analyst - Bloomberg-style Quant Research Terminal
Projeto Final FGV | IA Aplicada ao Mercado Financeiro
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from engine import (
    fetch_prices, fetch_fundamentals, run_ml,
    run_predictive_model, run_trading_system,
    run_monte_carlo, run_scoring, PROFILES,
)
from b3_universe import (
    B3_UNIVERSE, PORTFOLIOS, get_all_sectors,
    get_tickers_by_sector, get_ticker_label, get_universe_df,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Equity Analyst | Quant Terminal",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# BLOOMBERG-STYLE CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0a0e17;
    --bg-2: #0f1419;
    --card: #111827;
    --card-2: #1a2332;
    --border: #1e2d3d;
    --green: #00d26a;
    --green-dim: #00d26a33;
    --red: #ff3b3b;
    --red-dim: #ff3b3b33;
    --blue: #0ea5e9;
    --blue-dim: #0ea5e933;
    --gold: #facc15;
    --orange: #f97316;
    --text: #e2e8f0;
    --text-dim: #94a3b8;
    --muted: #64748b;
}

/* Global */
.stApp {
    background-color: var(--bg) !important;
    color: var(--text);
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1500px;
    padding-bottom: 4rem;
}

/* Typography */
h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -0.3px;
    font-weight: 700;
}

h1 {
    font-size: 2rem !important;
    border-bottom: 2px solid var(--blue);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem !important;
}

h2 { font-size: 1.4rem !important; color: var(--blue) !important; }
h3 { font-size: 1.1rem !important; color: var(--blue) !important; }

p, .stMarkdown {
    font-family: 'Inter', sans-serif;
    color: var(--text-dim);
    line-height: 1.6;
}

/* Monospace data */
code, pre, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--card-2) !important;
    border: 1px solid var(--border);
    border-radius: 4px;
}

/* Metrics - Bloomberg cards */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
    border-left: 3px solid var(--blue);
}

[data-testid="stMetric"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    font-weight: 500 !important;
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 24px !important;
    color: var(--text) !important;
    font-weight: 700 !important;
}

[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-2) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

/* Tabs - Bloomberg style */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 0;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    padding: 12px 20px !important;
    color: var(--text-dim) !important;
    border-right: 1px solid var(--border);
    background: transparent !important;
}

.stTabs [aria-selected="true"] {
    color: var(--blue) !important;
    background: var(--card-2) !important;
    border-top: 2px solid var(--blue) !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    border: none !important;
    transition: all 0.2s !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--blue), #2563eb) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0284c7, #1d4ed8) !important;
    transform: translateY(-1px);
}

/* Dataframes - Terminal style */
.stDataFrame {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11.5px !important;
    border: 1px solid var(--border) !important;
}

[data-testid="stDataFrame"] {
    background: var(--card) !important;
}

/* Selectbox & inputs */
.stSelectbox > div > div,
.stTextArea textarea,
.stTextInput input,
.stDateInput input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--blue), var(--green)) !important;
}

/* Alert/Info boxes */
.stAlert {
    background: var(--card) !important;
    border-left: 3px solid var(--blue) !important;
    border-radius: 4px;
    font-family: 'Inter', sans-serif;
}

/* Custom Bloomberg header */
.bloomberg-header {
    background: linear-gradient(135deg, var(--card), var(--card-2));
    border: 1px solid var(--border);
    border-left: 4px solid var(--blue);
    padding: 16px 20px;
    border-radius: 6px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.bloomberg-header .title {
    font-family: 'Inter', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--text);
    text-transform: uppercase;
    letter-spacing: 1px;
}

.bloomberg-header .subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.bloomberg-ticker {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--green);
    letter-spacing: 1px;
}

/* Edu boxes */
.edu-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    padding: 14px 18px;
    border-radius: 4px;
    margin: 12px 0;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: var(--text-dim);
    line-height: 1.6;
}

.edu-box strong {
    color: var(--gold);
    font-weight: 600;
}

.edu-box code {
    background: var(--bg) !important;
    color: var(--blue) !important;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 11px;
}
</style>
""", unsafe_allow_html=True)

# Plotly theme
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0a0e17",
    plot_bgcolor="#111827",
    font=dict(family="JetBrains Mono, monospace", size=11, color="#e2e8f0"),
    margin=dict(l=50, r=20, t=60, b=50),
    title_font=dict(family="Inter, sans-serif", size=14, color="#e2e8f0"),
    xaxis=dict(gridcolor="#1e2d3d", zerolinecolor="#1e2d3d"),
    yaxis=dict(gridcolor="#1e2d3d", zerolinecolor="#1e2d3d"),
)

# ============================================================
# HELPERS
# ============================================================

def fmt_ticker(t):
    return str(t).replace(".SA", "")

def fmt_brl(v):
    return f"R$ {v:,.2f}" if pd.notna(v) else "-"

def fmt_x(v):
    return f"{v:.1f}x" if pd.notna(v) else "-"

def fmt_pct(v):
    return f"{v*100:.1f}%" if pd.notna(v) else "-"

def fmt_pct_signed(v):
    return f"{v*100:+.1f}%" if pd.notna(v) else "-"

def fmt_mcap(v):
    if pd.isna(v): return "-"
    if v >= 1e12: return f"R$ {v/1e12:.1f}T"
    if v >= 1e9: return f"R$ {v/1e9:.1f}B"
    if v >= 1e6: return f"R$ {v/1e6:.1f}M"
    return f"R$ {v:,.0f}"

def bloomberg_header(title, subtitle):
    st.markdown(f"""
    <div class="bloomberg-header">
        <div>
            <div class="title">{title}</div>
            <div class="subtitle">{subtitle}</div>
        </div>
        <div class="bloomberg-ticker">● LIVE | {datetime.now().strftime("%H:%M:%S")}</div>
    </div>
    """, unsafe_allow_html=True)

def edu_box(html_content):
    st.markdown(f'<div class="edu-box">{html_content}</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("# 💹 AI EQUITY ANALYST")
    st.caption("QUANT RESEARCH TERMINAL")
    st.caption(f"v5.1 | FGV 2026")
    st.markdown("---")

    # ===== SELECTION MODE =====
    st.markdown("##### 📊 STOCK SELECTION")
    selection_mode = st.radio(
        "Mode",
        ["📋 Preset Portfolio", "🏭 By Sector", "🎯 Custom Selection"],
        label_visibility="collapsed",
    )

    selected_tickers = []

    if selection_mode == "📋 Preset Portfolio":
        portfolio_name = st.selectbox(
            "Portfolio",
            list(PORTFOLIOS.keys()),
            index=0,
        )
        selected_tickers = PORTFOLIOS[portfolio_name]
        st.caption(f"**{len(selected_tickers)} stocks** in {portfolio_name}")

    elif selection_mode == "🏭 By Sector":
        sectors = get_all_sectors()
        selected_sectors = st.multiselect(
            "Select sectors",
            sectors,
            default=["Financial Services", "Energy"],
        )
        for sec in selected_sectors:
            selected_tickers.extend(get_tickers_by_sector(sec))
        st.caption(f"**{len(selected_tickers)} stocks** across {len(selected_sectors)} sectors")

    else:  # Custom Selection
        all_tickers = sorted(B3_UNIVERSE.keys())
        selected_tickers = st.multiselect(
            "Select tickers",
            all_tickers,
            default=[
                "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA",
                "ABEV3.SA", "WEGE3.SA", "BBAS3.SA", "B3SA3.SA",
            ],
            format_func=lambda t: f"{t.replace('.SA','')} • {B3_UNIVERSE.get(t,('','?'))[1][:12]}",
        )
        st.caption(f"**{len(selected_tickers)} stocks** selected manually")

    # ===== STRATEGY =====
    st.markdown("---")
    st.markdown("##### 🎯 STRATEGY")
    profile = st.selectbox(
        "Profile",
        ["Conservative", "Moderate", "Aggressive"],
        index=1,
        label_visibility="collapsed",
    )

    # ===== LOOKBACK =====
    st.markdown("##### 📅 LOOKBACK")
    data_inicio = st.date_input(
        "Start",
        value=pd.Timestamp("2021-01-01"),
        label_visibility="collapsed",
    )

    # ===== WEIGHTS DISPLAY =====
    st.markdown("---")
    p = PROFILES[profile]
    st.code(
        f"WEIGHTS [{profile.upper()}]\n"
        f"────────────────\n"
        f"MARKET   = {p['market']:.0%}\n"
        f"QUALITY  = {p['quality']:.0%}",
        language="yaml"
    )

    st.markdown("")
    run_btn = st.button("▶ RUN ANALYSIS", type="primary", use_container_width=True)

    # ===== UNIVERSE INFO =====
    st.markdown("---")
    with st.expander("📚 B3 UNIVERSE DATABASE"):
        st.caption(f"**{len(B3_UNIVERSE)} stocks** across **{len(get_all_sectors())} sectors**")
        st.caption("Data: Yahoo Finance")
        if st.checkbox("Show full universe"):
            st.dataframe(get_universe_df(), use_container_width=True, hide_index=True, height=250)

    st.markdown("---")
    st.caption("**Engine:** RF Walk-Forward + Multi-Factor")
    st.caption("**Monte Carlo:** 10,000 simulations")

# ============================================================
# STATE
# ============================================================

state_keys = [
    "result", "prices", "fund", "ml_metrics", "ml_fi",
    "ml_preds", "ml_clusters", "pred_metrics", "pred_fi",
    "trading_curves", "trading_summary", "mc_carteiras",
    "mc_best", "mc_tickers",
]
for k in state_keys:
    if k not in st.session_state:
        st.session_state[k] = None

# ============================================================
# RUN ANALYSIS
# ============================================================

if run_btn:
    tickers = selected_tickers
    if len(tickers) < 2:
        st.error("⚠ Minimum 2 tickers required — select more stocks in the sidebar")
        st.stop()

    progress = st.progress(0, "▸ INITIALIZING QUANT ENGINE...")
    try:
        progress.progress(10, "▸ FETCHING MARKET DATA (Yahoo Finance)...")
        st.session_state.prices = fetch_prices(tickers, str(data_inicio))
        tickers_ok = st.session_state.prices["ticker"].unique().tolist()

        progress.progress(25, "▸ FETCHING FUNDAMENTALS (3-layer fallback)...")
        st.session_state.fund = fetch_fundamentals(tickers_ok, st.session_state.prices)

        progress.progress(45, "▸ TRAINING RANDOM FOREST (Market + Fundamental Features)...")
        ml = run_ml(st.session_state.prices, st.session_state.fund)
        (st.session_state.ml_metrics, st.session_state.ml_fi,
         st.session_state.ml_preds, st.session_state.ml_clusters) = ml

        progress.progress(60, "▸ TRAINING DIRECTIONAL CLASSIFIER...")
        st.session_state.pred_metrics, st.session_state.pred_fi = run_predictive_model(st.session_state.prices)

        progress.progress(75, "▸ BACKTESTING MA CROSSOVER STRATEGY...")
        tc, ts = run_trading_system(st.session_state.prices)
        st.session_state.trading_curves = tc
        st.session_state.trading_summary = ts

        progress.progress(88, "▸ MONTE CARLO PORTFOLIO OPTIMIZATION (10K)...")
        mc, best, mct = run_monte_carlo(st.session_state.prices)
        st.session_state.mc_carteiras = mc
        st.session_state.mc_best = best
        st.session_state.mc_tickers = mct

        progress.progress(95, "▸ COMPUTING COMPOSITE SCORES...")
        st.session_state.result = run_scoring(
            st.session_state.prices, st.session_state.fund, profile)

        progress.progress(100, "✓ ANALYSIS COMPLETE")
    except Exception as e:
        st.error(f"⚠ ERROR: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# ============================================================
# WELCOME SCREEN
# ============================================================

if st.session_state.result is None:
    st.markdown("# 💹 AI Equity Analyst")
    st.markdown("### Quantitative Research Terminal for Asset Managers")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **🎯 PURPOSE**

        Automated equity research for Brazilian stocks.
        Replaces hours of analyst work with seconds of AI.
        """)
    with col_b:
        st.markdown("""
        **⚙️ ENGINE**

        7 quant tools orchestrated:
        prices, fundamentals, ML, predictive,
        trading system, Monte Carlo, scoring.
        """)
    with col_c:
        st.markdown("""
        **📊 OUTPUT**

        BUY/HOLD/SELL recommendations,
        portfolio optimization,
        ML-driven return forecasts.
        """)

    st.markdown("---")
    st.markdown("##### 🚀 GETTING STARTED")

    cgs1, cgs2, cgs3 = st.columns(3)
    with cgs1:
        st.markdown("""
        **1️⃣ SELECT STOCKS**

        Choose from 3 modes:
        - **Preset Portfolio** (Ibovespa Top 10, Bancos, etc.)
        - **By Sector** (multi-sector filter)
        - **Custom Selection** (pick any of 79 stocks)
        """)
    with cgs2:
        st.markdown("""
        **2️⃣ SET PROFILE**

        - **Conservative** — favor quality, low vol
        - **Moderate** — balanced approach
        - **Aggressive** — favor momentum, growth
        """)
    with cgs3:
        st.markdown("""
        **3️⃣ RUN ANALYSIS**

        Click **▶ RUN ANALYSIS** in the sidebar.
        The engine processes all data in ~30 seconds.
        """)

    st.markdown("---")
    edu_box(f"""
    <strong>💡 B3 UNIVERSE DATABASE</strong><br><br>
    This terminal includes a curated database of <strong>{len(B3_UNIVERSE)} Brazilian stocks</strong>
    across <strong>{len(get_all_sectors())} sectors</strong>: Financial Services, Energy, Utilities,
    Basic Materials, Consumer Defensive, Industrials, Consumer Cyclical, Healthcare,
    Communication Services, Technology, and Real Estate.<br><br>
    The engine combines <code>Random Forest</code> machine learning (with multi-factor features:
    market + fundamentals), <code>Monte Carlo</code> portfolio optimization (10,000 simulations),
    and walk-forward validation to avoid look-ahead bias — institutional-grade methodology.<br><br>
    <strong>Project:</strong> FGV — IA Aplicada ao Mercado Financeiro | Line 3.5 (Generative AI applied to Finance)
    """)
    st.stop()

# ============================================================
# DATA SHORTCUTS
# ============================================================

result = st.session_state.result
prices = st.session_state.prices
fund = st.session_state.fund

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "RANKING", "MARKET", "ML ENGINE", "SIGNALS", "PORTFOLIO"
])

# ============================================================
# TAB 1: RANKING
# ============================================================
with tab1:
    bloomberg_header("EQUITY RANKING", f"COMPOSITE SCORE | PROFILE: {profile.upper()} | N = {len(result)}")

    edu_box("""
    <strong>📚 HOW TO READ THIS SCREEN</strong><br>
    The <strong>Composite Score (0-100)</strong> combines two pillars: <code>Market Score</code> (momentum,
    volatility, drawdown — 50%/30%/20%) and <code>Quality Score</code> (P/VP rank — lower is better).
    Weights adapt to the selected profile. Thresholds: <strong style="color:#00d26a">BUY ≥ 65</strong> |
    <strong style="color:#facc15">HOLD 35-65</strong> | <strong style="color:#ff3b3b">SELL &lt; 35</strong>.
    """)

    # Metrics row
    n_buy = sum("BUY" in str(r) for r in result["recommendation"])
    n_hold = sum("HOLD" in str(r) for r in result["recommendation"])
    n_sell = sum("SELL" in str(r) for r in result["recommendation"])
    avg_score = result["score"].mean()
    top_pick = result.iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("UNIVERSE", len(result))
    c2.metric("BUY", n_buy, delta=f"{n_buy/len(result)*100:.0f}%", delta_color="off")
    c3.metric("HOLD", n_hold, delta=f"{n_hold/len(result)*100:.0f}%", delta_color="off")
    c4.metric("SELL", n_sell, delta=f"{n_sell/len(result)*100:.0f}%", delta_color="off")
    c5.metric("AVG SCORE", f"{avg_score:.1f}")

    st.markdown(f"##### TOP PICK: **{fmt_ticker(top_pick['ticker'])}** | Score: {top_pick['score']:.1f} | {top_pick['recommendation']}")

    # Bar chart
    r = result.sort_values("score", ascending=True)
    colors = ["#00d26a" if "BUY" in rec else "#ff3b3b" if "SELL" in rec else "#facc15"
              for rec in r["recommendation"]]
    fig = go.Figure(go.Bar(
        y=r["ticker"].apply(fmt_ticker),
        x=r["score"],
        orientation="h",
        marker_color=colors,
        marker_line=dict(color="#1e2d3d", width=1),
        text=[f"{s:.1f} | {rec}" for s, rec in zip(r["score"], r["recommendation"])],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=11, color="#e2e8f0"),
    ))
    fig.add_vline(x=50, line_dash="dash", line_color="#64748b", opacity=0.5)
    fig.add_vline(x=65, line_dash="dot", line_color="#00d26a", opacity=0.4)
    fig.add_vline(x=35, line_dash="dot", line_color="#ff3b3b", opacity=0.4)
    fig.update_layout(
        title="COMPOSITE SCORE DISTRIBUTION",
        xaxis_range=[0, 120],
        height=max(350, len(r) * 42),
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Ranking table
    st.markdown("##### 📋 DETAILED RANKING")
    cols = ["rank", "ticker", "nome", "setor", "preco", "P_L", "P_VP",
            "EV_EBITDA", "score", "recommendation"]
    tbl = result[cols].sort_values("rank").copy()
    tbl["ticker"] = tbl["ticker"].apply(fmt_ticker)
    tbl["preco"] = tbl["preco"].apply(fmt_brl)
    tbl["P_L"] = tbl["P_L"].apply(fmt_x)
    tbl["P_VP"] = tbl["P_VP"].apply(fmt_x)
    tbl["EV_EBITDA"] = tbl["EV_EBITDA"].apply(fmt_x)
    tbl["nome"] = tbl["nome"].fillna("-").astype(str)
    tbl["setor"] = tbl["setor"].fillna("-").astype(str)
    tbl.columns = ["#", "TICKER", "NAME", "SECTOR", "PRICE", "P/E", "P/B",
                   "EV/EBITDA", "SCORE", "REC"]
    st.dataframe(tbl, use_container_width=True, hide_index=True)

# ============================================================
# TAB 2: MARKET
# ============================================================
with tab2:
    bloomberg_header("MARKET DASHBOARD", "RISK × RETURN | PRICE EVOLUTION | FUNDAMENTALS")

    edu_box("""
    <strong>📚 RISK-RETURN FRAMEWORK</strong><br>
    The scatter plot shows the <strong>risk-return tradeoff</strong>: x-axis is annualized volatility
    (21-day rolling), y-axis is 6-month momentum. Bubble size represents the composite score.
    Stocks in the <strong>upper-left quadrant</strong> (high momentum, low volatility) are most
    attractive — this is what professional quants call "favorable Sharpe profile".
    """)

    col1, col2 = st.columns(2)
    with col1:
        d = result.dropna(subset=["vol_21", "mom_6m", "score"]).copy()
        fig = go.Figure()
        for rec_type, color in [("BUY", "#00d26a"), ("HOLD", "#facc15"), ("SELL", "#ff3b3b")]:
            sub = d[d["recommendation"] == rec_type]
            if sub.empty: continue
            fig.add_trace(go.Scatter(
                x=sub["vol_21"] * 100,
                y=sub["mom_6m"] * 100,
                mode="markers+text",
                name=rec_type,
                text=sub["ticker"].apply(fmt_ticker),
                textposition="top center",
                textfont=dict(family="JetBrains Mono", size=10, color="#e2e8f0"),
                marker=dict(
                    size=sub["score"] / 2,
                    color=color,
                    line=dict(color="#0a0e17", width=2),
                    opacity=0.8,
                ),
            ))
        fig.add_hline(y=0, line_dash="dash", line_color="#64748b", opacity=0.3)
        fig.update_layout(
            title="RISK × RETURN MATRIX",
            xaxis_title="VOLATILITY 21D (% ann.)",
            yaxis_title="MOMENTUM 6M (%)",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top5 = result.head(5)["ticker"].tolist()
        fig = go.Figure()
        cols = ["#00d26a", "#0ea5e9", "#facc15", "#f97316", "#a855f7"]
        for i, t in enumerate(top5):
            sub = prices[prices["ticker"] == t].dropna(subset=["preco"])
            if sub.empty: continue
            norm = sub["preco"] / sub["preco"].iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=sub["data"], y=norm,
                name=fmt_ticker(t),
                mode="lines",
                line=dict(color=cols[i % len(cols)], width=2),
            ))
        fig.add_hline(y=100, line_dash="dash", line_color="#64748b", opacity=0.4)
        fig.update_layout(
            title="TOP-5 PERFORMANCE (BASE 100)",
            xaxis_title="DATE",
            yaxis_title="CUMULATIVE (%)",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### 📊 FUNDAMENTALS SNAPSHOT")

    edu_box("""
    <strong>📚 KEY MULTIPLES EXPLAINED</strong><br>
    <strong>P/E (Price/Earnings):</strong> price per R$1 of earnings. Lower = cheaper. <br>
    <strong>P/B (Price/Book):</strong> price per R$1 of book value. P/B &lt; 1 suggests undervaluation.<br>
    <strong>EV/EBITDA:</strong> enterprise value over operating profit (ex-D&A). Standard for cross-sector comparison.<br>
    <strong>ROE:</strong> return on equity — how efficiently management uses shareholder capital.
    """)

    fd = fund[["ticker", "nome", "setor", "preco", "marketCap", "P_L", "P_VP",
               "EV_EBITDA", "returnOnEquity", "profitMargins", "revenueGrowth"]].copy()
    fd["ticker"] = fd["ticker"].apply(fmt_ticker)
    fd["preco"] = fd["preco"].apply(fmt_brl)
    fd["marketCap"] = fd["marketCap"].apply(fmt_mcap)
    for c in ["P_L", "P_VP", "EV_EBITDA"]:
        fd[c] = fd[c].apply(fmt_x)
    for c in ["returnOnEquity", "profitMargins", "revenueGrowth"]:
        fd[c] = fd[c].apply(fmt_pct)
    fd["nome"] = fd["nome"].fillna("-").astype(str)
    fd["setor"] = fd["setor"].fillna("-").astype(str)
    fd.columns = ["TICKER", "NAME", "SECTOR", "PRICE", "MKT CAP",
                  "P/E", "P/B", "EV/EBITDA", "ROE", "MARGIN", "GROWTH"]
    st.dataframe(fd, use_container_width=True, hide_index=True)

# ============================================================
# TAB 3: ML ENGINE
# ============================================================
with tab3:
    bloomberg_header("ML ENGINE", "RANDOM FOREST | WALK-FORWARD VALIDATION | CLUSTERING")

    edu_box("""
    <strong>📚 MULTI-FACTOR RANDOM FOREST</strong><br>
    Our Random Forest is trained on <strong>both technical features</strong>
    (vol_21, mom_6m, mom_12m, drawdown) <strong>and fundamental features</strong>
    (P/E, P/B, ROE, profit margin, revenue growth, EBITDA margin, debt/equity).
    All features are z-scored cross-sectionally per date to remove sector bias.
    This combination captures stocks like quality banks (e.g. ITUB4) that have weak
    momentum but strong fundamentals.<br><br>
    <strong>WALK-FORWARD VALIDATION:</strong> We train on years <code>[t-N, t-1]</code>
    and test on year <code>t</code>, rolling forward. This prevents <strong>look-ahead bias</strong>.
    The <strong>Spearman IC</strong> measures rank correlation between predicted and
    realized returns. IC &gt; 0.05 is statistically significant in academic finance.
    """)

    col1, col2 = st.columns(2)
    with col1:
        ml_m = st.session_state.ml_metrics
        if ml_m is not None and not ml_m.empty:
            st.markdown("##### WALK-FORWARD METRICS")
            ml_m_d = ml_m.round(4).copy()
            st.dataframe(ml_m_d, use_container_width=True, hide_index=True)

            fig = go.Figure(go.Bar(
                x=ml_m["year"].astype(str),
                y=ml_m["spearman_ic"],
                marker_color=["#00d26a" if v > 0 else "#ff3b3b" for v in ml_m["spearman_ic"]],
                text=[f"{v:.3f}" for v in ml_m["spearman_ic"]],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11),
            ))
            fig.add_hline(y=0, line_color="#64748b", opacity=0.5)
            fig.update_layout(
                title="SPEARMAN IC BY YEAR",
                yaxis_title="IC",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠ Insufficient data for walk-forward (needs 3+ years)")

    with col2:
        fi = st.session_state.ml_fi
        if fi is not None and not fi.empty:
            st.markdown("##### FEATURE IMPORTANCE")
            fi_mean = fi.mean().sort_values(ascending=True)
            fig = go.Figure(go.Bar(
                y=fi_mean.index,
                x=fi_mean.values,
                orientation="h",
                marker_color="#0ea5e9",
                text=[f"{v:.3f}" for v in fi_mean.values],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=10),
            ))
            fig.update_layout(
                title="AVG IMPORTANCE (across folds)",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)

    preds = st.session_state.ml_preds
    if preds is not None and not preds.empty and "pred_ret_12m" in preds.columns:
        st.markdown("##### 🎯 12-MONTH RETURN FORECAST (Random Forest)")
        pp = preds[["ticker", "pred_ret_12m", "rank_pred"]].dropna().copy()
        pp["ticker"] = pp["ticker"].apply(fmt_ticker)
        pp["pred_ret_12m"] = (pp["pred_ret_12m"] * 100).round(1)
        pp = pp.sort_values("rank_pred")
        fig = go.Figure(go.Bar(
            x=pp["ticker"],
            y=pp["pred_ret_12m"],
            marker_color=["#00d26a" if v > 0 else "#ff3b3b" for v in pp["pred_ret_12m"]],
            text=[f"{v:+.1f}%" for v in pp["pred_ret_12m"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        fig.add_hline(y=0, line_color="#64748b", opacity=0.5)
        fig.update_layout(
            title="PREDICTED 12M FORWARD RETURN (%)",
            yaxis_title="RETURN (%)",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    clust = st.session_state.ml_clusters
    if clust is not None and not clust.empty and "profile" in clust.columns:
        st.markdown("##### 🔍 KMEANS CLUSTERING — RISK PROFILES")

        edu_box("""
        <strong>📚 UNSUPERVISED LEARNING IN PORTFOLIO CONSTRUCTION</strong><br>
        KMeans groups stocks by their risk-return signatures: <code>Defensive</code> (low volatility),
        <code>Growth</code> (high momentum), and <code>High-Risk</code> (high drawdown). This is used
        by hedge funds for <strong>factor neutralization</strong> and diversification across regimes.
        """)

        cl = clust[["ticker", "profile", "vol_21", "mom_6m", "drawdown"]].copy()
        cl["ticker"] = cl["ticker"].apply(fmt_ticker)
        for c in ["vol_21", "mom_6m", "drawdown"]:
            cl[c] = cl[c].apply(fmt_pct_signed)
        cl.columns = ["TICKER", "PROFILE", "VOL 21D", "MOM 6M", "DRAWDOWN"]
        st.dataframe(cl.sort_values("PROFILE"), use_container_width=True, hide_index=True)

# ============================================================
# TAB 4: SIGNALS
# ============================================================
with tab4:
    bloomberg_header("TRADING SIGNALS", "DIRECTIONAL FORECASTS | MA CROSSOVER BACKTEST")

    edu_box("""
    <strong>📚 SIGNAL GENERATION FOR QUANT TRADING</strong><br>
    Two systems run in parallel: (1) a <strong>Random Forest classifier</strong> predicting next-day
    direction (AUC measures discrimination — 0.5 is random, 0.6+ is profitable after costs); and
    (2) a <strong>Moving Average Crossover</strong> strategy (buy when MA-20 &gt; MA-60).
    The cumulative return chart shows whether the systematic strategy beats <strong>buy-and-hold</strong>.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🎯 NEXT-DAY DIRECTION FORECAST")
        pm = st.session_state.pred_metrics
        if pm is not None and not pm.empty:
            pm_d = pm.copy()
            pm_d["ticker"] = pm_d["ticker"].apply(fmt_ticker)
            fig = go.Figure(go.Bar(
                x=pm_d["ticker"],
                y=pm_d["auc"],
                marker_color=["#00d26a" if v > 0.5 else "#ff3b3b" for v in pm_d["auc"]],
                text=[f"{v:.3f}" for v in pm_d["auc"]],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11),
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="#ff3b3b",
                          opacity=0.6, annotation_text="Random (0.5)")
            fig.update_layout(
                title="AUC | DIRECTIONAL MODEL",
                yaxis_title="AUC",
                yaxis=dict(range=[0.3, 0.8], gridcolor="#1e2d3d"),
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "yaxis"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠ Insufficient data")

    with col2:
        st.markdown("##### 📈 MA CROSSOVER BACKTEST")
        ts = st.session_state.trading_summary
        if ts is not None and not ts.empty:
            ts_d = ts.copy()
            ts_d["ticker"] = ts_d["ticker"].apply(fmt_ticker)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="BUY & HOLD", x=ts_d["ticker"], y=ts_d["ret_buyhold"],
                marker_color="#0ea5e9",
            ))
            fig.add_trace(go.Bar(
                name="MA STRATEGY", x=ts_d["ticker"], y=ts_d["ret_estrategia"],
                marker_color="#00d26a",
            ))
            fig.update_layout(
                barmode="group",
                title="CUMULATIVE RETURN: STRATEGY vs B&H (%)",
                yaxis_title="RETURN (%)",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)

    tc = st.session_state.trading_curves
    ts = st.session_state.trading_summary
    if tc is not None and not tc.empty and ts is not None and not ts.empty:
        st.markdown("##### 📊 EQUITY CURVE — INDIVIDUAL ASSET")
        sel = st.selectbox(
            "SELECT TICKER",
            [fmt_ticker(t) for t in ts["ticker"].tolist()],
            label_visibility="collapsed",
        )
        sel_sa = sel + ".SA" if not sel.endswith(".SA") else sel
        sub = tc[tc["ticker"] == sel_sa]
        if not sub.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sub["data"], y=sub["acum_acao"],
                name="BUY & HOLD",
                line=dict(color="#0ea5e9", width=2),
                fill="tonexty",
                fillcolor="rgba(14,165,233,0.05)",
            ))
            fig.add_trace(go.Scatter(
                x=sub["data"], y=sub["acum_est"],
                name="MA STRATEGY",
                line=dict(color="#00d26a", width=2),
            ))
            fig.add_hline(y=1, line_dash="dash", line_color="#64748b", opacity=0.4)
            fig.update_layout(
                title=f"EQUITY CURVE | {sel} | MA(20,60)",
                yaxis_title="CUMULATIVE WEALTH (BASE 1.0)",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 5: PORTFOLIO
# ============================================================
with tab5:
    bloomberg_header("PORTFOLIO OPTIMIZATION", "MONTE CARLO | EFFICIENT FRONTIER | MAX SHARPE")

    edu_box("""
    <strong>📚 MARKOWITZ EFFICIENT FRONTIER</strong><br>
    We simulate <strong>10,000 random portfolios</strong> with weights drawn from a uniform Dirichlet
    distribution. Each point's color represents its <strong>Sharpe Ratio</strong> (return per unit of risk).
    The red star marks the <strong>maximum Sharpe portfolio</strong> — the theoretically optimal
    allocation for a rational investor under the Capital Asset Pricing Model (CAPM) framework.
    Annual figures assume 252 trading days.
    """)

    mc = st.session_state.mc_carteiras
    best = st.session_state.mc_best
    mct = st.session_state.mc_tickers

    if mc is not None and not mc.empty and best:
        col1, col2 = st.columns([3, 2])

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=mc["risco"] * 100, y=mc["retorno"] * 100,
                mode="markers",
                marker=dict(
                    size=4,
                    color=mc["sharpe"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=dict(text="SHARPE", font=dict(size=10))),
                    opacity=0.6,
                ),
                name="Simulations",
                hovertemplate="Risk: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[best["risco"] * 100], y=[best["retorno"] * 100],
                mode="markers",
                marker=dict(
                    size=22, color="#ff3b3b", symbol="star",
                    line=dict(width=2, color="white"),
                ),
                name="MAX SHARPE",
                hovertemplate="<b>OPTIMAL</b><br>Risk: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
            ))
            fig.update_layout(
                title="EFFICIENT FRONTIER | 10,000 SIMULATIONS",
                xaxis_title="ANNUAL RISK (% std dev)",
                yaxis_title="ANNUAL RETURN (%)",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("##### 🎯 OPTIMAL PORTFOLIO")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("SHARPE", f"{best['sharpe']:.2f}")
            mc2.metric("RETURN", f"{best['retorno']*100:.1f}%")
            mc3.metric("RISK", f"{best['risco']*100:.1f}%")

            st.markdown("##### 📊 ALLOCATION (Max Sharpe)")
            alloc = [(fmt_ticker(t), best.get(t, 0) * 100) for t in mct]
            alloc.sort(key=lambda x: -x[1])

            # Pie chart
            fig = go.Figure(go.Pie(
                labels=[a[0] for a in alloc],
                values=[a[1] for a in alloc],
                hole=0.5,
                marker=dict(
                    colors=["#00d26a", "#0ea5e9", "#facc15", "#f97316",
                            "#a855f7", "#ec4899", "#14b8a6", "#ef4444",
                            "#8b5cf6", "#06b6d4"],
                    line=dict(color="#0a0e17", width=2),
                ),
                textinfo="label+percent",
                textfont=dict(family="JetBrains Mono", size=11),
            ))
            fig.update_layout(
                showlegend=False,
                height=380,
                margin=dict(l=0, r=0, t=20, b=20),
                paper_bgcolor="#0a0e17",
                font=dict(color="#e2e8f0"),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### 📋 ALLOCATION TABLE")
        alloc_df = pd.DataFrame(alloc, columns=["TICKER", "WEIGHT (%)"])
        alloc_df["WEIGHT (%)"] = alloc_df["WEIGHT (%)"].apply(lambda x: f"{x:.2f}%")
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)
    else:
        st.info("⚠ Insufficient data for Monte Carlo (needs 2+ tickers)")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748b; font-family:JetBrains Mono; font-size:11px;'>"
    "AI EQUITY ANALYST v5.0 | QUANT RESEARCH TERMINAL | FGV 2026 | "
    f"DATA: Yahoo Finance | LAST UPDATE: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    "</div>",
    unsafe_allow_html=True,
)
