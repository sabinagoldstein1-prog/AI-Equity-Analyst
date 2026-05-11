"""
ai_analyst.py — Módulo de IA Generativa (Google Gemini)
Gera análises qualitativas automáticas por empresa usando o Gemini Flash.
100% grátis — 15 req/min no tier gratuito.
"""
import google.generativeai as genai
import streamlit as st
import pandas as pd
import numpy as np

def _get_model():
    """Configura e retorna o modelo Gemini."""
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")

def generate_company_analysis(row: dict) -> str:
    """Gera análise qualitativa de UMA empresa usando Gemini."""
    model = _get_model()
    if model is None:
        return _fallback_analysis(row)

    ticker = row.get("ticker", "?").replace(".SA", "")
    prompt = f"""Você é um analista de equity research sênior de um banco de investimento brasileiro.
Analise a empresa {ticker} ({row.get('nome', '')}) com base nos dados abaixo e gere um parecer
profissional de 4-5 frases em português brasileiro.

DADOS:
- Setor: {row.get('setor', 'N/A')}
- Preço atual: R${row.get('preco', 0):.2f}
- Preço implícito (múltiplos setoriais): R${row.get('impl_price', 0):.2f}
- Gap valuation: {row.get('gap_pct', 0):+.1f}%
- P/L: {row.get('P_L', 'N/A')}x | P/VP: {row.get('P_VP', 'N/A')}x | EV/EBITDA: {row.get('EV_EBITDA', 'N/A')}x
- Market Cap: R${row.get('marketCap', 0)/1e9:.1f} bi
- Volatilidade 21d: {row.get('vol_21', 0)*100:.1f}%
- Momentum 6m: {row.get('mom_6m', 0)*100:+.1f}%
- Score composto: {row.get('score', 0):.1f}/100
- Recomendação do modelo: {row.get('recomendacao', 'N/A')}
- Sentimento NLP: {row.get('sent_label', 'N/A')}

INSTRUÇÃO:
1. Comente os fundamentos (múltiplos vs setor)
2. Avalie o momentum e risco (volatilidade/drawdown)
3. Justifique a recomendação do modelo
4. Mencione um risco principal e uma oportunidade
Seja direto, use linguagem profissional de research de banco. Não use bullet points."""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return _fallback_analysis(row)

def generate_executive_summary(result_df: pd.DataFrame) -> str:
    """Gera resumo executivo do portfólio inteiro usando Gemini."""
    model = _get_model()
    if model is None:
        return _fallback_summary(result_df)

    # Monta tabela resumo para o prompt
    lines = []
    for _, r in result_df.iterrows():
        tk = r.get("ticker", "?").replace(".SA", "")
        lines.append(f"- {tk}: Score {r.get('score', 0):.0f}, {r.get('recomendacao', 'Hold')}, "
                     f"Gap {r.get('gap_pct', 0):+.1f}%, Vol {r.get('vol_21', 0)*100:.1f}%")
    table = "\n".join(lines)

    n_buy  = sum("Buy" in str(r.get("recomendacao", "")) for _, r in result_df.iterrows())
    n_sell = sum("Sell" in str(r.get("recomendacao", "")) for _, r in result_df.iterrows())
    n_hold = len(result_df) - n_buy - n_sell

    prompt = f"""Você é o head de equity research de um banco de investimento brasileiro.
Gere um RESUMO EXECUTIVO de 2 parágrafos em português sobre o universo analisado.

UNIVERSO ({len(result_df)} empresas):
{table}

DISTRIBUIÇÃO: {n_buy} Buy | {n_hold} Hold | {n_sell} Sell

INSTRUÇÃO:
Parágrafo 1: Visão geral do universo — quais setores se destacam, qual a tendência geral.
Parágrafo 2: Destaques positivos e negativos — cite empresas específicas com números.
Seja direto e profissional. Máximo 150 palavras."""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return _fallback_summary(result_df)

# ─── Fallbacks (quando não tem API key) ──────────────────────────────────────

def _fallback_analysis(row: dict) -> str:
    tk = row.get("ticker", "?").replace(".SA", "")
    score = row.get("score", 50)
    gap = row.get("gap_pct", 0)
    vol = row.get("vol_21", 0) * 100
    mom = row.get("mom_6m", 0) * 100
    rec = row.get("recomendacao", "Hold")

    if score >= 65:
        tone = f"{tk} apresenta fundamentos sólidos com score composto de {score:.0f}/100."
    elif score >= 35:
        tone = f"{tk} apresenta fundamentos em linha com o mercado, score de {score:.0f}/100."
    else:
        tone = f"{tk} apresenta sinais de fragilidade com score de {score:.0f}/100."

    val_text = (f"O preço implícito por múltiplos setoriais indica um {'upside' if gap > 0 else 'downside'} "
                f"de {gap:+.1f}% em relação ao preço atual.")
    risk_text = (f"A volatilidade anualizada de {vol:.1f}% e momentum de 6 meses de {mom:+.1f}% "
                 f"{'reforçam' if mom > 0 else 'mitigam'} a tese.")

    return f"{tone} {val_text} {risk_text} Recomendação: {rec}."

def _fallback_summary(result_df: pd.DataFrame) -> str:
    n = len(result_df)
    n_buy = sum("Buy" in str(r) for r in result_df.get("recomendacao", []))
    top = result_df.iloc[0] if not result_df.empty else {}
    bot = result_df.iloc[-1] if not result_df.empty else {}
    return (
        f"O universo analisado compreende {n} empresas. O modelo identificou {n_buy} oportunidades de compra. "
        f"O destaque positivo é {top.get('ticker', '?').replace('.SA', '')} com score de {top.get('score', 0):.0f}/100. "
        f"O destaque negativo é {bot.get('ticker', '?').replace('.SA', '')} com score de {bot.get('score', 0):.0f}/100."
    )
