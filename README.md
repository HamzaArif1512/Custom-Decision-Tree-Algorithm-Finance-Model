# Stock Prediction Using Custom Decision Tree and Random Forest Classifier

This project is a stock recommendation system developed as the final project for the **Introduction to Artificial Intelligence** course. It combines custom-built Decision Trees and a handcrafted Random Forest ensemble to predict whether a stock should be bought or not, based on its financial indicators and ratios.

The model fetches real-time data from financial APIs, computes key ratios like Sharpe Ratio, PE, EPS, ROE, and makes predictions across multiple trees for final decision-making.

---

## Project Structure

- `decisionTree.py`: Implements a custom decision tree classifier from scratch, including Gini impurity, pre-pruning, and prediction.
- `RandomforestCustom.py`: Builds a random forest of decision trees with bootstrap sampling, parallelism (`joblib`), and ensemble voting.
- `stock_agent.py`: Full pipeline for fetching financial data (via Alpha Vantage and Yahoo Finance), preprocessing, feature engineering, and final stock prediction.
- `train_refined.csv` & `test_refined.csv`: Training and test data extracted and refined from Kaggle financial datasets.

---

## Features

- End-to-end stock prediction using real financial data
- Custom Decision Tree implementation (no `sklearn`)
- Random Forest built from scratch using parallel processing
- Feature engineering with:
  - PE Ratio
  - EPS
  - PS Ratio
  - ROE
  - Dividend Yield
  - Net Profit Margin
  - Current Ratio
  - CAGR
  - Sharpe Ratio (computed from recent prices)
- Caching of API results to reduce rate limits and reuse data
- Labels:
  - 1 = Short Buy
  - 2 = Long Term (Capital Gain)
  - 3 = Long Buy (Dividend)
  - 4 = Don’t Buy

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HamzaArif1512/Custom-Decision-Tree-Algorithm-Finance-Model.git
   cd Custom-Decision-Tree-Algorithm-Finance-Model
````

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set your [Alpha Vantage API Key](https://www.alphavantage.co/) in a `.env` file:

   ```
   ALPHA_VANTAGE_API_KEY=your_key_here
   ```

---

## Usage

### Training and Evaluation

To train the Random Forest and evaluate it:

```bash
python RandomforestCustom.py
```

### Real-Time Stock Prediction

To get a prediction for a specific stock (e.g., AAPL):

```bash
python stock_agent.py AAPL
```

The model will:

* Download stock prices using `yfinance`
* Fetch financial ratios using Alpha Vantage
* Cache and reuse results
* Predict a class using the ensemble forest

---

## Model Parameters

The forest uses:

* `TOTAL_TREES = 2000`
* `ROWS_PER_TREE = 1500`
* `MAX_DEPTH = 10`
* `MIN_SAMPLES_SPLIT = 30`
* `MIN_SAMPLES_LEAF = 1`
* `MIN_IMPURITY_DECREASE = 0.01`
* `MAX_FEATURES = 1`

These hyperparameters were tuned empirically for generalization.

---

## Sample Output

```bash
Loading training data...
 → 5000 training samples

Building random forest...
 → 2000 trees built

Evaluating stock prediction using Random Forest...
Fetching data for AAPL…
Features for AAPL: [1.2, 28.3, 3.1, 4.5, 0.12, 1.4, 15.2, 0.01, 0.08]
Predictions from each tree: [2, 2, 2, 1, 2, 2, ...]
Vote counts: Counter({2: 1720, 1: 180, 3: 85, 4: 15})
Majority vote: 2

The model recommends: Long Term (Capital Gain) for AAPL.

Model Accuracy: 98.03%
```

---

## Credits

This project was created by:

* **Hamza Arif**
* **Haris Khalid**
* **Raahin Tajuddin**

As part of our final assignment in the **Introduction to Artificial Intelligence** course, we implemented the full pipeline from algorithm to API integration and prediction.
