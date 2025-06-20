# Stock Prediction Using Custom Decision Tree and Random Forest Classifier

This project is a stock recommendation system developed as the final project for the **Introduction to Artificial Intelligence** course. It combines custom-built Decision Trees and a handcrafted Random Forest ensemble to predict whether a stock should be bought or not, based on its financial indicators and ratios.

The model fetches real-time data from financial APIs, computes key ratios like Sharpe Ratio, PE, EPS, ROE, and makes predictions across multiple trees for final decision-making.

---

## Project Structure

- `decisionTree.py`: Custom decision tree classifier from scratch, with Gini impurity, pre-pruning, and prediction.
- `RandomforestCustom.py`: Builds a random forest using bootstrap sampling, parallelism (`joblib`), and majority voting.
- `stock_agent.py`: Full pipeline for data fetching (Alpha Vantage + Yahoo Finance), feature engineering, and prediction.
- `train_refined.csv` & `test_refined.csv`: Refined training/test datasets sourced from Kaggle.

---

## Features

- End-to-end stock prediction using real financial data
- No use of scikit-learn; all algorithms are implemented from scratch
- Parallel tree construction for faster forest training
- Feature engineering includes:
  - PE Ratio
  - EPS
  - PS Ratio
  - ROE
  - Dividend Yield
  - Net Profit Margin
  - Current Ratio
  - CAGR
  - Sharpe Ratio (calculated)
- Caching to handle API rate limits efficiently
- Four prediction labels:
  - `1`: Short Buy  
  - `2`: Long Term (Capital Gain)  
  - `3`: Long Buy (Dividend)  
  - `4`: Don't Buy

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/HamzaArif1512/Custom-Decision-Tree-Algorithm-Finance-Model.git
   cd Custom-Decision-Tree-Algorithm-Finance-Model
````

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Alpha Vantage API key:**

   Create a `.env` file in the root directory and add:

   ```
   ALPHA_VANTAGE_API_KEY=your_key_here
   ```

---

## Usage

### Train and Evaluate the Model

To train the Random Forest and evaluate its accuracy:

```bash
python RandomforestCustom.py
```

### Predict Stock Recommendation

To get a prediction for a specific stock (e.g., AAPL):

```bash
python stock_agent.py AAPL
```

This script will:

* Fetch recent stock prices and financial ratios
* Cache the data for future use
* Predict a recommendation using a trained random forest

---

## Model Parameters

The default hyperparameters used in the forest:

* `TOTAL_TREES = 2000`
* `ROWS_PER_TREE = 1500`
* `MAX_DEPTH = 10`
* `MIN_SAMPLES_SPLIT = 30`
* `MIN_SAMPLES_LEAF = 1`
* `MIN_IMPURITY_DECREASE = 0.01`
* `MAX_FEATURES = 1`

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

Project created by:

* **Hamza Arif**
* **Haris Khalid**
* **Raahin Tajuddin**

This project was developed as part of the final assignment for the **Introduction to Artificial Intelligence** course.
