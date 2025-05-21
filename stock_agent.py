import os
import sys
import time
import pickle
import yfinance as yf
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import pandas as pd
from RandomforestCustom import load_train_data, load_test_data, build_forest, evaluate_forest
from decisionTree import DecisionNode, Leaf
from collections import Counter
import json

# Setup
load_dotenv()
ALPHA_VANTAGE_API_KEY = 'QRKA049YQ6RA2GVY'

CACHE_TTL = 99999999  # 1 hour cache TTL
CACHE_DIR = os.path.join(os.path.dirname(__file__), "yf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Save and Load Model Functions 
def save_model(forest, filename="random_forest_model.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(forest, f)
    print(f"Model saved to {filename}")

def load_model(filename="random_forest_model.pkl"):
    try:
        with open(filename, 'rb') as f:
            forest = pickle.load(f)
        print(f"Model loaded from {filename}")
        return forest
    except FileNotFoundError:
        print(f"No saved model found, training new model.")
        return None

# Caching Helpers 
def cache_load(symbol, data_type):
    path = os.path.join(CACHE_DIR, f"{symbol}_{data_type}.json")
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < CACHE_TTL:
            with open(path, 'r') as f:
                return json.load(f)
    return None

def cache_save(symbol, data_type, data):
    path = os.path.join(CACHE_DIR, f"{symbol}_{data_type}.json")
    with open(path, 'w') as f:
        json.dump(data, f)

# Fetch Stock Data and Ratios
def fetch_stock_data(symbol: str):
    cached = cache_load(symbol, "stock_data")
    if cached is not None:
        print(f"Loaded cached stock data for {symbol}")
        return pd.DataFrame(cached)

    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
        if data.empty:
            print(f"No data returned for {symbol}.")
            return None

        print(f"Successfully fetched data for {symbol}:")
        print(data.head())
        cache_save(symbol, "stock_data", data.to_dict(orient="records"))
        return data

    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def safe_float(x):
    try:
        return float(x) if x not in [None, 'None', ''] else None
    except:
        return None

def fetch_company_overview(symbol: str):
    cached = cache_load(symbol, "company_overview")
    if cached is not None:
        print(f"Loaded cached company overview for {symbol}")
        return cached

    try:
        fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, _ = fd.get_company_overview(symbol=symbol)
        if data.empty:
            print(f"No overview data returned for {symbol}.")
            return None

        row = data.iloc[0]

        # Fetch basic ratios from overview
        pe_ratio = safe_float(row.get('PERatio'))
        eps = safe_float(row.get('EPS'))
        ps_ratio = safe_float(row.get('PriceToSalesRatioTTM'))
        roe = safe_float(row.get('ReturnOnEquityTTM'))
        div_yield = safe_float(row.get('DividendYield'))

        # Fetch NetProfitMargin_ratio from quarterly income statement
        npm = fetch_net_profit_margin(symbol)

        # Fetch current ratio from balance sheet annual
        current_ratio = fetch_current_ratio(symbol)

        # Calculate CAGR from annual income statements
        cagr = calculate_cagr(symbol)

        ratios = {
            "PE_ratio": pe_ratio,
            "EPS_ratio": eps,
            "PS_ratio": ps_ratio,
            "roe_ratio": roe,
            "NetProfitMargin_ratio": npm,
            "current_ratio": current_ratio,
            "div_yeild": div_yield,
            "CAGR": cagr
        }

        cache_save(symbol, "company_overview", ratios)
        print(f"Fetched and cached company overview for {symbol}")
        return ratios

    except Exception as e:
        print(f"Error fetching company overview for {symbol}: {str(e)}")
        return None

def fetch_net_profit_margin(symbol: str):
    cached = cache_load(symbol, "income_statement_quarterly")
    if cached is not None:
        try:
            net_income = safe_float(cached[0].get('netIncome', None))
            total_revenue = safe_float(cached[0].get('totalRevenue', None))
            if net_income is None or total_revenue is None or total_revenue == 0:
                return None
            return net_income / total_revenue
        except:
            return None

    try:
        fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, _ = fd.get_income_statement_quarterly(symbol=symbol)
        if data.empty:
            return None
        row = data.iloc[0]
        net_income = safe_float(row.get('netIncome', None))
        total_revenue = safe_float(row.get('totalRevenue', None))
        if net_income is None or total_revenue is None or total_revenue == 0:
            return None
        npm = net_income / total_revenue
        cache_save(symbol, "income_statement_quarterly", data.head(1).to_dict(orient='records'))
        return npm
    except Exception as e:
        print(f"Error fetching net profit margin for {symbol}: {str(e)}")
        return None

def fetch_current_ratio(symbol: str):
    try:
        fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, _ = fd.get_balance_sheet_annual(symbol=symbol)
        if data.empty:
            return None
        row = data.iloc[0]
        current_assets = safe_float(row.get('totalCurrentAssets'))
        current_liabilities = safe_float(row.get('totalCurrentLiabilities'))
        if current_assets is None or current_liabilities is None or current_liabilities == 0:
            return None
        return current_assets / current_liabilities
    except Exception:
        return None

def calculate_cagr(symbol: str):
    cached = cache_load(symbol, "income_statement_annual")
    if cached is not None:
        try:
            if len(cached) < 2:
                return None
            start = safe_float(cached[-1].get('totalRevenue', None))
            end = safe_float(cached[0].get('totalRevenue', None))
            years = len(cached) - 1
            if start is None or end is None or years <= 0:
                return None
            return (end / start) ** (1 / years) - 1
        except:
            return None

    try:
        fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, _ = fd.get_income_statement_annual(symbol=symbol)
        if data.empty or len(data) < 2:
            return None
        start = safe_float(data.iloc[-1].get('totalRevenue', None))
        end = safe_float(data.iloc[0].get('totalRevenue', None))
        years = len(data) - 1
        if start is None or end is None or years <= 0:
            return None
        cagr = (end / start) ** (1 / years) - 1
        cache_save(symbol, "income_statement_annual", data.to_dict(orient='records'))
        return cagr
    except Exception as e:
        print(f"Error calculating CAGR for {symbol}: {str(e)}")
        return None

def calculate_sharpe_ratio(close_prices: pd.Series, risk_free_rate: float = 0.0):
    try:
        returns = close_prices.pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        if std_return == 0 or pd.isna(std_return):
            return None
        sharpe_ratio = (mean_return - risk_free_rate) / std_return
        return sharpe_ratio
    except Exception:
        return None

# Predict Class for Random Forest 
def predict_class(features, node):
    while isinstance(node, DecisionNode):
        feature_index = node.question.column
        threshold = node.question.value
        if features[feature_index] is None:
            node = node.false_branch  # handle missing gracefully
        elif features[feature_index] >= threshold:
            node = node.true_branch
        else:
            node = node.false_branch
    return max(node.predictions.items(), key=lambda x: x[1])[0]

# Get Stock Prediction 
def get_stock_prediction(symbol: str, forest) -> str:
    print(f"Fetching data for {symbol}…")
    data = fetch_stock_data(symbol)
    if data is None:
        return f"No data for {symbol}"

    financial_data = fetch_company_overview(symbol)
    if financial_data is None:
        return f"Missing financial data for {symbol}"

    sharpe_ratio = calculate_sharpe_ratio(data['4. close'])

    features = [
        sharpe_ratio,
        financial_data['PE_ratio'],
        financial_data['EPS_ratio'],
        financial_data['PS_ratio'],
        financial_data['NetProfitMargin_ratio'],
        financial_data['current_ratio'],
        financial_data['roe_ratio'],
        financial_data['div_yeild'],
        financial_data['CAGR']
    ]

    if None in features:
        return f"Missing required financial ratios for {symbol} to predict."
    

    preds = [predict_class(features, tree) for tree in forest]

    print("Predictions from each tree:", preds) ###
    print("Vote counts:", Counter(preds))   ###

    vote = Counter(preds).most_common(1)[0][0]
    print(f"Majority vote: {vote}")

    print(f"Features for {symbol}: {features}")


    labels = {
        1: "Short Buy",
        2: "Long Term (Capital Gain)",
        3: "Long Buy (Dividend)",
        4: "Don't Buy"
    }

    recommendation = labels.get(vote, f"Unknown Recommendation (vote={vote})")
    return f"The model recommends: {recommendation} for {symbol}."

# Orchestrator 
def analyze_stock(symbol: str):
    forest = load_model("random_forest_model.pkl")

    if forest is None:
        print(f"\nLoading training data for model...")
        train_data = load_train_data("train_refined.csv")
        forest = build_forest(train_data)
        save_model(forest, "random_forest_model.pkl")

    print("\nEvaluating stock prediction using Random Forest...")
    recommendation = get_stock_prediction(symbol, forest)

    print(f"\n=== Recommendation for {symbol} ===\n")
    print(recommendation)

    test_data = load_test_data("test_refined.csv")

    accuracy = evaluate_forest(forest, test_data)
    print(f"\n===============================\n")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Entry Point 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stock_agent.py <SYMBOL>")
        sys.exit(1)
    analyze_stock(sys.argv[1].upper())
    time.sleep(0.5)
