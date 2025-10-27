import numpy as np
from scipy.optimize import minimize
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_extras.let_it_rain import rain
import streamlit_authenticator as stauth

# === CONFIGURAZIONE PAGINA ===
st.set_page_config(
    page_title="PortfolioMax",
    page_icon="Money Bag",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === STILE CSS ===
st.markdown("""
    <style>
    .stApp { background-color: #1e1e1e; color: #ffffff; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 10px; padding: 10px 20px; }
    .stButton>button:hover { background-color: #45a049; }
    .stTextInput>div>input { background-color: #2e2e2e; color: #ffffff; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# === CREDENZIALI CON PASSWORD GIA' HASHATA (password123) ===
config = {
    'credentials': {
        'usernames': {
            'testuser': {
                'email': 'test@example.com',
                'name': 'Test User',
                'password': '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW'
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'random_signature_key_2025',
        'name': 'portfoliomax_auth'
    }
    # NESSUN preauthorized → causa errori su Cloud
}

# === AUTENTICAZIONE ===
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# === LOGIN ===
name, authentication_status, username = authenticator.login('Login', 'main')

# === CONTROLLO LOGIN ===
if authentication_status:
    st.success(f"Benvenuto, {name}!")
    authenticator.logout('Logout', 'main')

    # === APP COMPLETA ===
    st.title("PortfolioMax - Ottimizza il Tuo Investimento")
    st.write("Inserisci gli asset e ottieni l’allocazione ottimale.")
    rain(emoji="Money Bag", font_size=20, falling_speed=5, animation_length=1)

    with st.sidebar:
        initial_investment = st.number_input("Importo ($)", 1000.0, 1000000.0, 10000.0, 1000.0)

    assets_input = st.text_input("Asset:", "TSLA, MSFT, GLD")
    assets = [a.strip().upper() for a in assets_input.split(',')]

    if st.button("Calcola", type="primary"):
        with st.spinner("Scarico dati..."):
            try:
                data = yf.download(assets, period='5y')['Close']
                data = data.dropna(axis=1, how='all')
                assets = data.columns.tolist()
                if len(assets) < 2:
                    st.error("Servono almeno 2 asset validi.")
                    st.stop()
            except Exception as e:
                st.error(f"Errore download: {e}")
                st.stop()

        returns = data.pct_change().dropna()
        mu = returns.mean() * 252
        sigma = returns.cov() * 252

        def neg_sharpe(w):
            ret = np.dot(w, mu)
            vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
            return -(ret - 0.02) / vol if vol > 0 else 1e9

        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
        bounds = [(0, 1) for _ in assets]
        res = minimize(neg_sharpe, [1/len(assets)]*len(assets), method='SLSQP', bounds=bounds, constraints=cons)

        if not res.success:
            st.error("Ottimizzazione fallita.")
            st.stop()

        weights = res.x
        alloc = [a for a, w in zip(assets, weights) if w > 0.01]
        w_alloc = [w for w in weights if w > 0.01]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Asset Inseriti")
            for a in assets: st.write(a)
        with col2:
            st.subheader("Asset Allocati")
            for a, w in zip(alloc, w_alloc): st.write(f"{a}: {w:.1%}")

        port_val = initial_investment * (1 + returns.dot(weights)).cumprod()
        bt_return = (port_val.iloc[-1] / initial_investment - 1) * 100
        bt_vol = returns.dot(weights).std() * np.sqrt(252) * 100
        st.write(f"**Rendimento**: {bt_return:.1f}% | **Volatilità**: {bt_vol:.1f}%")

        fig1, ax1 = plt.subplots()
        ax1.pie(w_alloc, labels=alloc, autopct='%1.0f%%')
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(port_val, color='#66b3ff')
        ax2.set_title("Crescita Portfolio")
        st.pyplot(fig2)

        sims = 50
        mc = np.random.normal(mu @ weights / 252, np.sqrt(weights @ sigma @ weights) / np.sqrt(252), (sims, 252))
        mc_val = initial_investment * np.cumprod(1 + mc, axis=1)
        st.write(f"**Monte Carlo - Media finale**: ${np.mean(mc_val[:, -1]):,.0f}")
        fig3, ax3 = plt.subplots()
        ax3.plot(mc_val.T, color='gray', alpha=0.1)
        ax3.plot(np.mean(mc_val, axis=0), color='red')
        st.pyplot(fig3)

elif authentication_status == False:
    st.error('Username o password errati.')
elif authentication_status is None:
    st.warning('Inserisci le credenziali.')