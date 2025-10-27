import numpy as np
from scipy.optimize import minimize
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_extras.let_it_rain import rain

# === 1. CONFIGURAZIONE PAGINA ===
st.set_page_config(
    page_title="PortfolioMax",
    page_icon="Money Bag",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === 2. STILE CSS SCURO ===
st.markdown("""
    <style>
    .stApp { background-color: #1e1e1e; color: #ffffff; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 10px; padding: 10px 20px; }
    .stButton>button:hover { background-color: #45a049; }
    .stTextInput>div>input { background-color: #2e2e2e; color: #ffffff; border-radius: 5px; }
    .stNumberInput>div>input { background-color: #2e2e2e; color: #ffffff; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# === 3. INIZIALIZZA SESSIONE ===
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# === 4. FUNZIONE LOGIN SEMPLICE (FUNZIONA SU STREAMLIT CLOUD) ===
def login():
    st.markdown("### PortfolioMax - Accesso")
    st.write("Accedi per usare l'app.")
    username = st.text_input("Username", placeholder="es. testuser")
    password = st.text_input("Password", type="password", placeholder="es. password123")
    
    if st.button("Accedi", type="primary"):
        if username == "testuser" and password == "password123":
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Accesso effettuato con successo!")
            st.rerun()
        else:
            st.error("Username o password errati.")

# === 5. FUNZIONE LOGOUT ===
def logout():
    if st.button("Logout", type="secondary"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

# === 6. CONTROLLO ACCESSO ===
if not st.session_state.logged_in:
    login()
    st.stop()

# === 7. UTENTE LOGGATO ===
st.success(f"Benvenuto, {st.session_state.username}!")
logout()

# === 8. APP COMPLETA ===
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
assets_input = st.text_input("Asset:", "TSLA, MSFT, GLD", help="Esempi: TSLA, MSFT, GLD, NVDA, AAPL. Usa SLV per argento.")
assets = [asset.strip().upper() for asset in assets_input.split(',') if asset.strip()]

if st.button("Calcola", type="primary"):
    if len(assets) < 2:
        st.error("Inserisci almeno 2 asset validi.")
        st.stop()

    # --- DOWNLOAD DATI ---
    with st.spinner("Scarico dati da Yahoo Finance..."):
        try:
            data = yf.download(assets, period='5y', auto_adjust=False)['Close']
            # Rimuovi asset senza dati
            empty_assets = [a for a in assets if data[a].isna().all()]
            if empty_assets:
                st.warning(f"Asset non disponibili: {', '.join(empty_assets)}. Proseguo con gli altri.")
                data = data.drop(columns=empty_assets)
                assets = [a for a in assets if a not in empty_assets]
            if len(assets) < 2:
                st.error("Non ci sono abbastanza asset validi per il calcolo.")
                st.stop()
        except Exception as e:
            st.error(f"Errore nel download dei dati: {e}")
            st.stop()

    # --- RENDIMENTI ---
    returns = data.pct_change().dropna()
    if len(returns) < 50:
        st.error("Dati storici insufficienti per un'analisi affidabile.")
        st.stop()

    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # --- FUNZIONE OBIETTIVO: MASSIMIZZARE SHARPE RATIO ---
    def negative_sharpe(weights, returns, cov_matrix):
        portfolio_return = np.dot(weights, returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility if portfolio_volatility > 0 else -1e9
        return -sharpe_ratio

    # --- OTTIMIZZAZIONE ---
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(assets)))
    initial_guess = [1.0 / len(assets)] * len(assets)

    with st.spinner("Ottimizzazione in corso..."):
        result = minimize(
            negative_sharpe,
            initial_guess,
            args=(expected_returns.values, cov_matrix.values),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

    if not result.success:
        st.error("Ottimizzazione fallita. Prova con asset diversi.")
        st.stop()

    weights = result.x
    allocated_assets = [asset for asset, w in zip(assets, weights) if w > 0.001]
    allocated_weights = [w for w in weights if w > 0.001]

    # --- COLONNE ASSET ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Asset Inseriti")
        for a in assets:
            st.write(f"• {a}")
    with col2:
        st.subheader("Asset Allocati (Ottimale)")
        total_alloc = sum(allocated_weights)
        for a, w in zip(allocated_assets, allocated_weights):
            perc = (w / total_alloc) * 100 if total_alloc > 0 else 0
            st.write(f"**{a}**: {perc:.1f}%")

    # --- BACKTEST ---
    portfolio_returns = returns.dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    portfolio_value = initial_investment * cumulative_returns

    total_return = (portfolio_value.iloc[-1] / initial_investment - 1) * 100
    annual_volatility = portfolio_returns.std() * np.sqrt(252) * 100
    max_drawdown = ((cumulative_returns.cummax() - cumulative_returns) / cumulative_returns.cummax()).max() * 100
    sharpe_ratio = (portfolio_returns.mean() * 252 - 0.02) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0

    st.subheader("Backtesting (5 anni)")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Rendimento Totale", f"{total_return:.1f}%")
    with col_b:
        st.metric("Volatilità Annuale", f"{annual_volatility:.1f}%")
    with col_c:
        st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
    with col_d:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    # --- GRAFICO A TORTA ---
    if allocated_weights:
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.pie(allocated_weights, labels=allocated_assets, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
        ax1.axis('equal')
        st.pyplot(fig1)

    # --- GRAFICO CRESCITA ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(portfolio_value.index, portfolio_value, color='#66b3ff', linewidth=2)
    ax2.set_title('Crescita del Portfolio (Backtest 5 anni)')
    ax2.set_xlabel('Data')
    ax2.set_ylabel('Valore ($)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig2)

    # --- MONTE CARLO ---
    def monte_carlo_simulation(weights, returns, initial_investment, simulations=100, years=1):
        mean_daily_return = returns.dot(weights).mean()
        std_daily_return = returns.dot(weights).std()
        sim_days = int(252 * years)
        simulations_array = np.random.normal(mean_daily_return, std_daily_return, (simulations, sim_days))
        portfolio_paths = initial_investment * np.cumprod(1 + simulations_array, axis=1)
        return portfolio_paths

    with st.spinner("Simulazione Monte Carlo..."):
        mc_results = monte_carlo_simulation(weights, returns, initial_investment, simulations=100, years=1)

    st.subheader("Simulazioni Monte Carlo (1 anno)")
    final_values = mc_results[:, -1]
    col3, col4 = st.columns(2)
    with col3:
        st.write(f"**Valore Medio Finale**: ${np.mean(final_values):,.0f}")
        st.write(f"**5° Percentile (Peggiore)**: ${np.percentile(final_values, 5):,.0f}")
    with col4:
        st.write(f"**95° Percentile (Migliore)**: ${np.percentile(final_values, 95):,.0f}")

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(mc_results.T, color='gray', alpha=0.1)
    ax3.plot(np.mean(mc_results, axis=0), color='red', linewidth=2, label='Media')
    ax3.set_title('Simulazioni Monte Carlo - 100 Scenari (1 anno)')
    ax3.set_xlabel('Giorno')
    ax3.set_ylabel('Valore ($)')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig3)

else:
    st.info("Premi **Calcola** per ottimizzare il tuo portfolio.")