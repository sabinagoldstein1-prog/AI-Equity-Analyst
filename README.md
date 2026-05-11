# 🤖 AI Equity Analyst — Projeto Final FGV

**Agente de IA Generativa para Análise Automática de Empresas Brasileiras**

Linha 3.5 — IA Generativa aplicada a Finanças

---

## 🚀 Como fazer o deploy (Streamlit Cloud — 100% grátis)

### Passo 1 — Criar repositório no GitHub
1. Vá em https://github.com/new
2. Nome: `ai-equity-analyst`
3. Marque **Public**
4. Clique em **Create repository**

### Passo 2 — Fazer upload dos arquivos
1. No repositório criado, clique em **"uploading an existing file"**
2. Arraste TODOS estes arquivos:
   - `app.py`
   - `engine.py`
   - `ai_analyst.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
3. Clique em **Commit changes**

### Passo 3 — Deploy no Streamlit Cloud
1. Vá em https://share.streamlit.io
2. Faça login com sua conta GitHub
3. Clique em **New app**
4. Selecione:
   - Repository: `seu-usuario/ai-equity-analyst`
   - Branch: `main`
   - Main file path: `app.py`
5. Clique em **Deploy!**

### Passo 4 — Configurar API Key do Gemini (opcional mas recomendado)
1. No Streamlit Cloud, vá em **Settings** do seu app
2. Clique em **Secrets**
3. Cole:
   ```
   GEMINI_API_KEY = "AIzaSUA_KEY_AQUI"
   ```
4. Clique **Save**
5. O app recarrega automaticamente com IA Generativa ativada

> **Sem a key, o app funciona normalmente** — usa análise heurística em vez do Gemini.

---

## 📁 Estrutura de Arquivos

```
ai-equity-analyst/
├── app.py                    # App principal Streamlit (6 abas)
├── engine.py                 # Motor: 6 tools do agente
├── ai_analyst.py             # Módulo Gemini (IA Generativa)
├── requirements.txt          # Dependências Python
├── .streamlit/
│   └── config.toml           # Tema dark + configuração
└── README.md                 # Este arquivo
```

## 🏗️ Arquitetura

```
Input: tickers + perfil
  │
  ├── TOOL 1: fetch_prices        → vol, momentum, drawdown (Yahoo Finance)
  ├── TOOL 2: fetch_fundamentals  → P/L, P/VP, EV/EBITDA (Yahoo Finance)
  ├── TOOL 3: run_nlp             → sentimento + tópicos risco (léxico)
  ├── TOOL 4: run_valuation       → preço implícito + 5 cenários macro
  ├── TOOL 5: run_ml              → Random Forest walk-forward + clusters
  └── TOOL 6: run_scoring         → score 0-100 + Buy/Hold/Sell
  │
Output: 6 abas interativas + dossiê IA generativa
```

## 📊 6 Abas do App

| Aba | Conteúdo |
|-----|----------|
| 🏆 Ranking | Barras horizontais + tabela com recomendações |
| 📊 Dashboard | Scatter risco×retorno + evolução de preços |
| 📝 NLP | Sentimento + tópicos de risco |
| 💰 Valuation | Múltiplos setoriais + heatmap 5 cenários |
| 🤖 ML | Walk-forward + feature importance + clusters |
| 📋 Dossiê IA | Relatório gerado pelo Gemini por empresa |

## 🔧 Técnicas de Modelagem

1. **Random Forest Regressor** — walk-forward validation, previsão de retorno 12m
2. **Scoring Multi-fator** — ranking percentil cross-sectional com 4 pilares ponderados
3. **KMeans Clustering** — agrupamento por perfil (Defensivo / Crescimento / Risco)
4. **NLP Léxica** — análise de sentimento e tópicos de risco
5. **Valuation por Múltiplos** — P/L, P/VP, EV/EBITDA com medianas setoriais
6. **IA Generativa (Gemini)** — análise qualitativa automática por empresa

---

*FGV — Inteligência Artificial Aplicada ao Mercado Financeiro — 2026*
