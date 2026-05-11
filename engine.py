"""
engine.py — Motor do Agente de IA com incremento de Asset Management (Seção 3.2)
"""
import warnings, numpy as np, pandas as pd, yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

def safe_div(a, b):
    try:
        a, b = float(a), float(b)
        if np.isnan(a) or np.isnan(b) or b == 0: return np.nan
        return a / b
    except: return np.nan

# --- TOOL 1: PREÇOS ---
def fetch_prices(tickers, start="2021-01-01"):
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series): raw = raw.to_frame(name=tickers[0])
    p = raw.stack(future_stack=True).reset_index()
    p.columns = ["data", "ticker", "preco"]
    p["data"] = pd.to_datetime(p["data"])
    p = p.dropna(subset=["preco"])
    
    metrics = []
    for t in tickers:
        df_t = p[p["ticker"] == t].sort_values("data")
        if len(df_t) < 20: continue
        ret_6m = (df_t["preco"].iloc[-1] / df_t["preco"].iloc[-126] - 1) if len(df_t) > 126 else 0
        vol = df_t["preco"].pct_change().std() * np.sqrt(252)
        dd = (df_t["preco"] / df_t["preco"].cummax() - 1).min()
        metrics.append({"ticker": t, "preco": df_t["preco"].iloc[-1], "mom_6m": ret_6m, "vol_21": vol, "drawdown": dd})
    return p, pd.DataFrame(metrics)

# --- TOOL 2: FUNDAMENTOS ---
def fetch_fundamentals(tickers):
    data = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            inf = tk.info
            data.append({
                "ticker": t, "nome": inf.get("shortName", t), "setor": inf.get("sector", "N/A"),
                "P_L": inf.get("forwardPE"), "P_VP": inf.get("priceToBook"),
                "EV_EBITDA": inf.get("enterpriseToEbitda"), "marketCap": inf.get("marketCap"),
                "div_yield": inf.get("dividendYield", 0)
            })
        except: data.append({"ticker": t, "nome": t, "setor": "N/A"})
    return pd.DataFrame(data)

# --- TOOLS 3, 4, 5 (Simplificadas para o exemplo rodar) ---
def run_nlp(tickers):
    return pd.DataFrame([{"ticker": t, "indice_textual": 65, "sent_label": "Otimista", "sent_score": 0.7} for t in tickers])

def run_valuation(tickers, fund_df):
    return pd.DataFrame([{"ticker": t, "val_score": 70, "impl_price": 50.0, "gap_pct": 0.15} for t in tickers])

def run_ml(prices_df):
    return pd.DataFrame([{"ticker": t, "pred_ret_12m": 0.12, "rank_pred": 1} for t in prices_df["ticker"].unique()])

# --- TOOL 6: SCORING ---
PERFIS = {
    "conservador": {"mercado": 0.2, "val": 0.2, "nlp": 0.1, "qual": 0.5},
    "moderado":    {"mercado": 0.35, "val": 0.25, "nlp": 0.15, "qual": 0.25},
    "agressivo":   {"mercado": 0.5, "val": 0.2, "nlp": 0.2, "qual": 0.1}
}

def run_scoring(price_metrics, fund_df, nlp_df, val_df, ml_preds, perfil="moderado"):
    p = PERFIS[perfil]
    snap = price_metrics.merge(fund_df, on="ticker")
    snap = snap.merge(nlp_df, on="ticker").merge(val_df, on="ticker")
    snap["score"] = np.random.randint(40, 90, len(snap)) # Lógica de score simplificada
    snap["recomendacao"] = snap["score"].apply(lambda x: "Buy" if x > 65 else ("Hold" if x > 45 else "Sell"))
    return snap.sort_values("score", ascending=False).reset_index(drop=True)

# --- NOVA TOOL: ASSET MANAGEMENT (SEÇÃO 3.2) ---
def run_asset_manager_analysis(prices_df):
    df_pivot = prices_df.pivot(index='data', columns='ticker', values='preco').dropna()
    returns = df_pivot.pct_change().dropna()
    corr = returns.corr()
    port_ret = returns.mean(axis=1)
    var_95 = np.percentile(port_ret, 5)
    cvar_95 = port_ret[port_ret <= var_95].mean()
    
    benchmark = returns.mean(axis=1)
    betas = {}
    for col in returns.columns:
        matrix = np.cov(returns[col], benchmark)
        betas[col] = matrix[0,1] / matrix[1,1]
        
    return {"corr": corr, "var": var_95, "cvar": cvar_95, "betas": betas}
