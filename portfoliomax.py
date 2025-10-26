import numpy as np
from scipy.optimize import minimize
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_extras.let_it_rain import rain
import streamlit_authenticator as stauth
import yaml

# === 1. CONFIGURAZIONE PAGINA ===
st.set_page_config(
    page_title="PortfolioMax",
    page_icon="Money Bag",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === 2. STILE CSS ===
st.markdown("""
    <style>
    .stApp { background-color: #1e1e1e; color: #ffffff; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 10px; padding: 10px 20px; }
    .stButton>button:hover { background-color: #45a049; }
    .stTextInput>div>input { background-color: #2e2e2e; color: #ffffff; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# === 3. CREDENZIALI IN CHIARO (Streamlit Cloud richiede hashing al primo login) ===
config = {
    'credentials': {
        'usernames': {
            'testuser': {
                'email': 'test@example.com',
                'name': 'Test User',
                'password': 'password123'  # IN CHIARO!
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'my_secret_key_2025',
        'name': 'portfoliomax_cookie'
    }
    # RIMOSSO preauthorized → non supportato con dizionario
}

# === 4. AUTENTICAZIONE (senza preauthorized) ===
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# === 5. LOGIN ===
name, authentication_status, username = authenticator.login(location='main')

# === CONTROLLO LOGIN ===
if authentication_status == False:
    st.error('Username o password errati.')
    st.stop()

if authentication_status is None:
    st.warning('Inserisci le credenziali.')
    st.stop()

# === UTENTE LOGGATO ===
st.success(f"Benvenuto, {name}!")
authenticator.logout('Logout', 'main')

# === APP ===
st.title("PortfolioMax - Ottimizza il Tuo Investimento")
st.write("""
Benvenuto su **PortfolioMax**, il tuo strumento per ottimizzare un portfolio di investimenti!  
Inserisci gli asset, scegli l'importo e ottieni l'allocazione ottimale.
""")
rain(emoji="Money Bag", font_size=20, falling_speed=5, animation_length=1)

# Sidebar
with st.sidebar:
    st.header("Impostazioni")
    initial_investment = st.number_input("Importo da Investire ($)", min_value=1000.0, max_value=1000000.0, value=10000.0, step=1000.0)
    st.image("https://via.placeholder.com/150", caption="Logo PortfolioMax")

# Input asset
st.write("Inserisci gli asset separati da virgola (es. TSLA, MSFT, GLD)")
assets_input = st.text_input("Asset:", "TSLA, MSFT, GLD", help="Esempi: TSLA, MSFT, GLD, NVDA. Usa SLV per argento.")
assets = [asset.strip() for asset in assets_input.split(',')]

if st.button("Calcola", type="primary"):
    # --- DOWNLOAD DATI ---
    with st.spinner("Scarico dati..."):
        try:
            data = yf.download(assets, period='5y', auto_adjust=False)['Close']
            empty_assets = [a for a in assets if data[a].isna().all()]
            if empty_assets:
                st.warning(f"Asset non validi: {', '.join(empty_assets)}. Proseguo con gli altri.")
                data = data.drop(columns=empty_assets)
                assets = [a for a in assets if a not in empty_assets]
            if len(assets) < 2:
                st.error("Errore: servono almeno 2 asset validi.")
                st.stop()
        except Exception as e:
            st.error(f"Errore download: {e}")
            st.stop()

    # --- RENDIMENTI ---
    returns = data.pct_change().dropna()
    if len(returns) < 2:
        st.error("Dati insufficienti.")
        st.stop()

    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    expected_returns = expected_returns.values
    cov_matrix = cov_matrix.values

    # --- OTTIMIZZAZIONE ---
    def negative_sharpe(w, r, cov):
        ret = np.dot(w, r)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        return -(ret - 0.02) / vol if vol > 0 else np.inf

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(assets)))
    result = minimize(negative_sharpe, [1/len(assets)]*len(assets),
                      args=(expected_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        st.error("Ottimizzazione fallita.")
        st.stop()

    weights = result.x
    allocated_assets = [a for a, w in zip(assets, weights) if w > 0.0001]
    allocated_weights = [w for w in weights if w > 0.0001]

    # --- COLONNE ASSET ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Asset Inseriti")
        for a in assets: st.write(a)
    with col2:
        st.subheader("Asset Allocati")
        for a, w in zip(allocated_assets, allocated_weights):
            st.write(f"{a}: {w:.2%}")

    # --- BACKTEST ---
    def backtest(w, r, inv):
        port_ret = r.dot(w)
        cum = (1 + port_ret).cumprod()
        val = inv * cum
        return {
            'total_return': (val.iloc[-1]/inv - 1)*100,
            'annual_volatility': port_ret.std() * np.sqrt(252) * 100,
            'max_drawdown': ((cum.cummax() - cum)/cum.cummax()).max()*100,
            'sharpe_ratio': (port_ret.mean()*252 - 0.02)/(port_ret.std()*np.sqrt(252)),
            'portfolio_value': val
        }
    bt = backtest(weights, returns, initial_investment)

    st.subheader("Backtesting")
    st.write(f"Rendimento Totale: {bt['total_return']:.2f}%")
    st.write(f"Volatilità: {bt['annual_volatility']:.2f}%")
    st.write(f"Max Drawdown: {bt['max_drawdown']:.2f}%")
    st.write(f"Sharpe: {bt['sharpe_ratio']:.2f}")

    # --- GRAFICO A TORTA ---
    fig1, ax1 = plt.subplots()
    ax1.pie(allocated_weights, autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
            startangle=90, colors=plt.cm.Pastel1.colors)
    ax1.axis('equal')
    plt.legend(allocated_assets, title="Asset Allocati", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    st.pyplot(fig1)

    # --- GRAFICO CRESCITA ---
    fig2, ax2 = plt.subplots()
    ax2.plot(bt['portfolio_value'], color='#66b3ff')
    ax2.set_title('Crescita del Portfolio')
    ax2.set_ylabel('Valore ($)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig2)

    # --- MONTE CARLO ---
    def monte_carlo(w, r, inv, sims=100, years=1):
        mu = np.dot(w, r.mean()) * 252
        sigma = np.sqrt(np.dot(w.T, np.dot(r.cov()*252, w)))
        sim = np.random.normal(mu/252, sigma/np.sqrt(252), (sims, int(252*years)))
        return inv * np.cumprod(1 + sim, axis=1)
    mc = monte_carlo(weights, returns, initial_investment)

    st.subheader("Simulazioni Monte Carlo")
    col3, col4 = st.columns(2)
    with col3:
        st.write(f"Valore Medio: ${np.mean(mc[:, -1]):.2f}")
        st.write(f"5° Percentile: ${np.percentile(mc[:, -1], 5):.2f}")
    with col4:
        st.write(f"95° Percentile: ${np.percentile(mc[:, -1], 95):.2f}")

    fig3, ax3 = plt.subplots()
    ax3.plot(mc.T, color='gray', alpha=0.1)
    ax3.plot(np.mean(mc, axis=0), color='red', lw=2, label='Media')
    ax3.set_title('Simulazioni Monte Carlo')
    ax3.set_ylabel('Valore ($)')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig3)