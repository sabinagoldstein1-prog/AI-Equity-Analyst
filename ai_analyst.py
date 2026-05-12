"""
ai_analyst.py — Chatbot Financeiro com Google Gemini
"""
import streamlit as st
import pandas as pd
import numpy as np

def get_model():
    try:
        import google.generativeai as genai
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key: return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except: return None

def build_context(result_df, fund_df):
    """Monta contexto com dados das empresas para o Gemini."""
    lines = []
    for _, r in result_df.iterrows():
        tk = r.get("ticker","?").replace(".SA","")
        lines.append(f"- {tk}: Score {r.get('score',0):.0f}, Rec: {r.get('recomendacao','?')}, "
                     f"Preco R${r.get('preco',0):.2f}, P/L {r.get('P_L','N/A')}, "
                     f"P/VP {r.get('P_VP','N/A')}, Vol {r.get('vol_21',0)*100:.1f}%, "
                     f"Mom6m {r.get('mom_6m',0)*100:+.1f}%")
    # Fund extras
    for _, r in fund_df.iterrows():
        tk = r.get("ticker","?").replace(".SA","")
        roe = r.get("returnOnEquity", np.nan)
        mg = r.get("profitMargins", np.nan)
        gr = r.get("revenueGrowth", np.nan)
        if pd.notna(roe) or pd.notna(mg):
            lines.append(f"  {tk} extras: ROE {roe}, Margem {mg}, Crescimento {gr}")
    return "\n".join(lines)

def chat_with_gemini(user_msg, context, history):
    """Envia mensagem para o Gemini com contexto das empresas."""
    model = get_model()
    if model is None:
        return "API Key do Gemini nao configurada. Adicione GEMINI_API_KEY nos Secrets do Streamlit Cloud."
    system = f"""Voce e um analista de equity research senior de um banco de investimento brasileiro.
Responda SEMPRE em portugues brasileiro, de forma profissional e direta.
Use os dados abaixo como base para suas respostas. Se nao tiver informacao suficiente, diga.

DADOS DAS EMPRESAS ANALISADAS:
{context}

REGRAS:
- Seja especifico: cite numeros, tickers e comparacoes
- Use linguagem de research de banco (ex: upside, downside, multiplos, momentum)
- Se perguntarem sobre uma empresa especifica, foque nela
- Se perguntarem opiniao, justifique com dados
- Maximo 200 palavras por resposta"""

    # Build conversation
    messages = [{"role":"user","parts":[system + "\n\nEntendido, estou pronto para responder."]}]
    messages.append({"role":"model","parts":["Entendido. Sou seu analista de equity research. Pode perguntar sobre qualquer empresa do universo analisado."]})
    for h in history:
        messages.append({"role":"user","parts":[h["user"]]})
        messages.append({"role":"model","parts":[h["assistant"]]})
    messages.append({"role":"user","parts":[user_msg]})
    try:
        response = model.generate_content(messages)
        return response.text.strip()
    except Exception as e:
        return f"Erro na API Gemini: {str(e)}"
