import numpy as np
from scipy.optimize import minimize
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

# Installa streamlit-icons se non l'hai fatto (esegui localmente: pip install streamlit-icons)
from streamlit_extras.let_it_rain import rain

# Configurazione pagina con layout moderno
st.set_page_config(
    page_title="PortfolioMax",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Funzione personalizzata per stile scuro con CSS
def set_custom_style():
    st.markdown("""
        <style>
        .stApp {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>div>input {
            background-color: #2e2e2e;
            color: #ffffff;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

set_custom_style()

# Descrizione introduttiva con animazione
st.title("PortfolioMax - Ottimizza il Tuo Investimento 💰")
st.write("""
Benvenuto su **PortfolioMax**, il tuo strumento semplice per ottimizzare un portfolio di investimenti!  
Questa app ti aiuta a decidere come distribuire il tuo denaro tra diversi asset (come azioni o oro) per massimizzare i rendimenti e ridurre i rischi, basandosi su dati reali degli ultimi 5 anni.  
### Come Usarla:
1. **Inserisci gli Asset**: Scrivi i simboli degli asset (es. TSLA, MSFT, GLD) separati da virgole.
2. **Scegli l'Importo**: Nella barra laterale, imposta quanto vuoi investire (da 1.000$ a 1.000.000$).
3. **Calcola**: Clicca il bottone "Calcola" per vedere i risultati.
### Cosa Otterrai:
- **Pesi Ottimali**: La percentuale di ogni asset nel tuo portfolio per il miglior equilibrio rischio/rendimento.
- **Backtesting**: Quanto potrebbe essere cresciuto il tuo investimento negli ultimi 5 anni.
- **Simulazioni Future**: Previsioni sul valore futuro del tuo portfolio con il metodo Monte Carlo.
Prova con TSLA, MSFT e GLD per iniziare, e scopri come ottimizzare i tuoi investimenti in modo facile e veloce!
""")
rain(emoji="💸", font_size=20, falling_speed=5, animation_length=1)  # Animazione leggera

# Sidebar per impostazioni personalizzate
with st.sidebar:
    st.header("Impostazioni")
    initial_investment = st.number_input("Importo da Investire ($)", min_value=1000.0, max_value=1000000.0, value=10000.0, step=1000.0, help="Scegli l'importo per personalizzare i calcoli.")
    st.image("https://via.placeholder.com/150", caption="Logo PortfolioMax", use_column_width=True)  # Placeholder per logo

# Interfaccia Principale
st.write("Inserisci gli asset separati da virgole (es. TSLA, MSFT, GLD) e ottieni l'allocazione ottimale.")

assets_input = st.text_input("Asset:", "TSLA, MSFT, GLD", help="Esempi: TSLA, MSFT, GLD, NVDA. Usa SLV per argento, SI=F per futures.")
assets = [asset.strip() for asset in assets_input.split(',')]

if st.button("Calcola", type="primary"):
    # Scarica dati con gestione asset non validi
    with st.spinner("Sto scaricando dati reali..."):
        try:
            data = yf.download(assets, period='5y', auto_adjust=False)['Close']
            # Controlla asset con dati vuoti
            empty_assets = [asset for asset in assets if data[asset].isna().all()]
            if empty_assets:
                st.warning(f"Asset non validi o senza dati: {', '.join(empty_assets)}. Suggerimenti: Usa SLV per argento, SI=F per futures. Proseguo con gli altri.")
                data = data.dropna(axis=1, how='all')  # Rimuovi colonne vuote
                assets = [asset for asset in assets if asset not in empty_assets]  # Aggiorna lista asset
            if data.empty or data.shape[1] < 2:
                st.error("Errore: Nessun dato scaricato per almeno 2 asset validi. Controlla i simboli.")
                st.stop()
        except Exception as e:
            st.error(f"Errore nel download: {e}")
            st.stop()

    # Calcola rendimenti con validazioni
    returns = data.pct_change(fill_method=None).dropna()
    if len(returns) < 2 or np.any(returns.isna()):
        st.error("Errore: Dati insufficienti o non validi dopo il download. Prova meno asset o controlla i simboli.")
        st.stop()

    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # Validazione dati
    if np.any(np.isnan(expected_returns)) or np.any(np.isnan(cov_matrix)):
        st.error("Errore: Dati contengono valori non validi (NaN). Prova meno asset.")
        st.stop()
    if len(expected_returns) != len(assets) or cov_matrix.shape != (len(assets), len(assets)):
        st.error("Errore: Dimensione dei dati non coincide con il numero di asset.")
        st.stop()

    expected_returns = expected_returns.values
    cov_matrix = cov_matrix.values

    # Funzione per Sharpe negativo
    def negative_sharpe(weights, returns, cov_matrix, risk_free_rate=0.02):
        portfolio_return = np.dot(weights, returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if portfolio_std == 0:  # Evita divisione per zero
            return np.inf
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        return -sharpe_ratio

    # Ottimizzazione
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(assets)))
    initial_guess = np.array([1/len(assets)] * len(assets))

    result = minimize(negative_sharpe, initial_guess, args=(expected_returns, cov_matrix),
                     method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        optimal_weights = result.x
        # Separa asset allocati (peso > 0) e inseriti
        allocated_assets = [asset for asset, weight in zip(assets, optimal_weights) if weight > 0]
        allocated_weights = [weight for weight in optimal_weights if weight > 0]

        # Colonne per asset
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Asset Inseriti 📋")
            for asset in assets:
                st.write(f"{asset}")
        with col2:
            st.subheader("Asset Allocati 📊")
            for asset, weight in zip(allocated_assets, allocated_weights):
                st.write(f"{asset}: {weight:.2%}")

        # Backtesting con validazione
        def backtest_portfolio(optimal_weights, returns_df, initial_investment=10000):
            if not isinstance(returns_df, pd.DataFrame) or returns_df.empty:
                st.error("Errore: Dati di rendimento non validi.")
                st.stop()
            if len(optimal_weights) != len(returns_df.columns):
                st.error(f"Errore: Numero di pesi ({len(optimal_weights)}) non corrisponde al numero di asset ({len(returns_df.columns)}).")
                st.stop()
            portfolio_returns = returns_df.dot(optimal_weights)
            if np.any(np.isnan(portfolio_returns)):
                st.error("Errore: Rendimento del portfolio contiene valori NaN.")
                st.stop()
            cumulative_returns = (1 + portfolio_returns).cumprod()
            portfolio_value = initial_investment * cumulative_returns
            
            total_return = (portfolio_value.iloc[-1] / initial_investment - 1) * 100
            annual_volatility = portfolio_returns.std() * np.sqrt(252) * 100
            max_drawdown = ((cumulative_returns.cummax() - cumulative_returns) / cumulative_returns.cummax()).max() * 100
            sharpe = (portfolio_returns.mean() * 252 - 0.02) / (portfolio_returns.std() * np.sqrt(252))
            
            return {
                'total_return': total_return,
                'annual_volatility': annual_volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe,
                'portfolio_value': portfolio_value
            }

        bt_results = backtest_portfolio(optimal_weights, returns, initial_investment)
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Backtesting 📈")
            st.write(f"Rendimento Totale: {bt_results['total_return']:.2f}%")
            st.write(f"Volatilità Annuale: {bt_results['annual_volatility']:.2f}%")
            st.write(f"Max Drawdown: {bt_results['max_drawdown']:.2f}%")
            st.write(f"Sharpe Ratio: {bt_results['sharpe_ratio']:.2f}")

        # Grafico a torta migliorato (solo asset allocati)
        fig1, ax1 = plt.subplots(figsize=(8, 8) if len(allocated_assets) > 5 else (6, 6))  # Dimensione adattiva
        wedges, texts, autotexts = ax1.pie(allocated_weights, labels=None, autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '', 
                                          startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#c4e1e1'][:len(allocated_assets)])
        ax1.axis('equal')
        plt.setp(autotexts, size=8, weight="bold")  # Testo percentuali più piccole
        plt.legend(wedges, allocated_assets, title="Asset Allocati", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        st.pyplot(fig1)

        # Grafico crescita
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(returns.index, bt_results['portfolio_value'], color='#66b3ff', linewidth=2)
        ax2.set_title('Crescita del Portfolio', fontsize=12, pad=10)
        ax2.set_xlabel('Data', fontsize=10)
        ax2.set_ylabel('Valore ($)', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig2)

        # Monte Carlo
        def monte_carlo_simulation(optimal_weights, returns_df, num_simulations=100, years=1, initial_investment=10000):
            if not isinstance(returns_df, pd.DataFrame) or returns_df.empty:
                st.error("Errore: Dati di rendimento non validi per Monte Carlo.")
                st.stop()
            portfolio_daily_returns = returns_df.mean()
            portfolio_daily_volatility = returns_df.std()
            
            portfolio_return = np.dot(optimal_weights, portfolio_daily_returns) * 252
            portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(returns_df.cov() * 252, optimal_weights))) / np.sqrt(252)
            
            daily_sim_returns = np.random.normal(portfolio_return / 252, portfolio_volatility,
                                               (num_simulations, int(252 * years)))
            future_values = initial_investment * np.cumprod(1 + daily_sim_returns, axis=1)
            
            return future_values

        mc_results = monte_carlo_simulation(optimal_weights, returns, num_simulations=100, years=1, initial_investment=initial_investment)
        st.subheader("Simulazioni Monte Carlo 🎲")
        col5, col6 = st.columns(2)
        with col5:
            st.write(f"Valore Medio Futuro: ${np.mean(mc_results[:, -1]):.2f}")
            st.write(f"Percentile 5% (Peggiore): ${np.percentile(mc_results[:, -1], 5):.2f}")
        with col6:
            st.write(f"Percentile 95% (Miglior): ${np.percentile(mc_results[:, -1], 95):.2f}")

        # Grafico Monte Carlo
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(mc_results.T, color='gray', alpha=0.1)
        ax3.plot(np.mean(mc_results, axis=0), color='red', lw=2, label='Media')
        ax3.set_title('Simulazioni Monte Carlo', fontsize=12, pad=10)
        ax3.set_xlabel('Giorno', fontsize=10)
        ax3.set_ylabel('Valore ($)', fontsize=10)
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig3)
    else:
        st.error(f"Errore nell'ottimizzazione: {result.message}. Controlla i dati o prova meno asset.")