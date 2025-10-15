import numpy as np
from scipy.optimize import minimize
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

# Funzione per Sharpe negativo
def negative_sharpe(weights, returns, cov_matrix, risk_free_rate=0.02):
    portfolio_return = np.dot(weights, returns)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
    return -sharpe_ratio

# Funzione per Backtesting
def backtest_portfolio(optimal_weights, returns_df, initial_investment=10000):
    portfolio_returns = returns_df.dot(optimal_weights)
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

# Funzione per Simulazione Monte Carlo
def monte_carlo_simulation(optimal_weights, returns_df, num_simulations=100, years=1, initial_investment=10000):
    portfolio_daily_returns = returns_df.mean()
    portfolio_daily_volatility = returns_df.std()
    
    portfolio_return = np.dot(optimal_weights, portfolio_daily_returns) * 252
    portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(returns_df.cov() * 252, optimal_weights))) / np.sqrt(252)
    
    daily_sim_returns = np.random.normal(portfolio_return / 252, portfolio_volatility,
                                        (num_simulations, int(252 * years)))
    future_values = initial_investment * np.cumprod(1 + daily_sim_returns, axis=1)
    
    return future_values

# Interfaccia Streamlit
st.title("PortfolioMax - Ottimizzazione Portfolio")
st.write("Inserisci gli asset separati da virgola (es. TSLA, MSFT, GLD) e ottieni l'allocazione ottimale.")

assets_input = st.text_input("Asset:", "TSLA, MSFT, GLD")
assets = [asset.strip() for asset in assets_input.split(',')]

if st.button("Calcola"):
    # Scarica dati
    with st.spinner("Sto scaricando dati reali..."):
        try:
            data = yf.download(assets, period='5y', auto_adjust=False)['Close']
            if data.empty or data.isna().all().all():
                st.error("Errore: Nessun dato scaricato. Controlla i simboli.")
                st.stop()
        except Exception as e:
            st.error(f"Errore nel download: {e}")
            st.stop()

    # Calcola rendimenti
    returns = data.pct_change(fill_method=None).dropna()
    if len(returns) < 2 or np.any(returns.isna()):
        st.error("Errore: Dati insufficienti o non validi.")
        st.stop()

    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    if np.any(np.isnan(expected_returns)) or np.any(np.isnan(cov_matrix)):
        st.error("Errore: Dati non validi.")
        st.stop()

    expected_returns = expected_returns.values
    cov_matrix = cov_matrix.values

    # Ottimizzazione
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(assets)))
    initial_guess = np.array([1/len(assets)] * len(assets))

    result = minimize(negative_sharpe, initial_guess, args=(expected_returns, cov_matrix),
                     method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        optimal_weights = result.x
        st.write("\nPesi Ottimali per il Portfolio (basati su dati reali ultimi 5 anni):")
        for asset, weight in zip(assets, optimal_weights):
            st.write(f"{asset}: {weight:.2%}")

        # Backtesting
        bt_results = backtest_portfolio(optimal_weights, returns)
        st.write("\nRisultati Backtesting (su $10.000 investiti):")
        st.write(f"Rendimento Totale: {bt_results['total_return']:.2f}%")
        st.write(f"Volatilità Annuale: {bt_results['annual_volatility']:.2f}%")
        st.write(f"Max Drawdown: {bt_results['max_drawdown']:.2f}%")
        st.write(f"Sharpe Ratio: {bt_results['sharpe_ratio']:.2f}")

        # Grafico a torta
        fig1, ax1 = plt.subplots()
        ax1.pie(optimal_weights, labels=assets, autopct='%1.2f%%', startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99'])
        ax1.axis('equal')
        st.pyplot(fig1)

        # Grafico crescita
        fig2, ax2 = plt.subplots()
        ax2.plot(returns.index, bt_results['portfolio_value'])
        ax2.set_title('Crescita del Portfolio nel Tempo')
        ax2.set_xlabel('Data')
        ax2.set_ylabel('Valore Portfolio ($)')
        ax2.grid(True)
        st.pyplot(fig2)

        # Monte Carlo
        mc_results = monte_carlo_simulation(optimal_weights, returns, num_simulations=100, years=1)
        st.write("\nRisultati Simulazione Monte Carlo (1 anno, 100 scenari):")
        st.write(f"Valore Medio Futuro: ${np.mean(mc_results[:, -1]):.2f}")
        st.write(f"Percentile 5% (Peggiore Caso): ${np.percentile(mc_results[:, -1], 5):.2f}")
        st.write(f"Percentile 95% (Miglior Caso): ${np.percentile(mc_results[:, -1], 95):.2f}")

        # Grafico Monte Carlo
        fig3, ax3 = plt.subplots()
        ax3.plot(mc_results.T, color='gray', alpha=0.1)
        ax3.plot(np.mean(mc_results, axis=0), color='red', lw=2, label='Media')
        ax3.set_title('Simulazioni Monte Carlo del Portfolio (1 Anno)')
        ax3.set_xlabel('Giorno')
        ax3.set_ylabel('Valore Portfolio ($)')
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig3)
    else:
        st.error("Errore nell'ottimizzazione. Controlla i dati o prova altri asset.")