from typing import List, Any
import pandas as pd
import random
from collections import Counter
from decisionTree import build_tree, predict_class
from joblib import Parallel, delayed
from tqdm import tqdm

# =============================
# Parameters
# =============================

# TOTAL_TREES = 500
# ROWS_PER_TREE = 5000
# MAX_DEPTH = 15 # was 30
# MIN_SAMPLES_SPLIT = 50 # was 200
# MIN_SAMPLES_LEAF = 10 # was 30
# MIN_IMPURITY_DECREASE = 0.0 #was 0.0025
# MAX_FEATURES = 2  # out of 9 ratio features
# 99.06 

# TOTAL_TREES = 1000             # Increase number of trees for more robustness
# ROWS_PER_TREE = 3000           # Reduce the data per tree to encourage variance
# MAX_DEPTH = 10                 # Limit depth to avoid overfitting
# MIN_SAMPLES_SPLIT = 30         # Allow more splits, making the model more sensitive
# MIN_SAMPLES_LEAF = 5           # Ensure smaller leaf nodes for better generalization
# MIN_IMPURITY_DECREASE = 0.01   # Increase impurity decrease threshold to avoid unnecessary splits
# MAX_FEATURES = 1               # Max features per split to increase randomness
# 98.34

# TOTAL_TREES = 2000            # Increase number of trees to enhance diversity
# ROWS_PER_TREE = 1500          # Further reduce data per tree for more randomness
# MAX_DEPTH = 10                # Shallow depth for better generalization (8-12)
# MIN_SAMPLES_SPLIT = 10        # Allow more splits by reducing samples per split
# MIN_SAMPLES_LEAF = 1          # Smaller leaf nodes for better capturing of patterns
# MIN_IMPURITY_DECREASE = 0.1   # Require a larger decrease in impurity for splits
# MAX_FEATURES = 1            # Reduce features considered at each split to 30% for more randomness
# 68.01 2000(4)

TOTAL_TREES = 2000            # Increase number of trees to enhance diversity
ROWS_PER_TREE = 1500          # Further reduce data per tree for more randomness
MAX_DEPTH = 10                # Shallow depth for better generalization (8-12)
MIN_SAMPLES_SPLIT = 30        # Allow more splits by reducing samples per split
MIN_SAMPLES_LEAF = 1          # Smaller leaf nodes for better capturing of patterns
MIN_IMPURITY_DECREASE = 0.01   # Require a larger decrease in impurity for splits
MAX_FEATURES = 1            # Reduce features considered at each split to 30% for more randomness
# 98.03 4.0: 1930, 1.0: 32, 3.0: 29, 2.0: 9

# TOTAL_TREES = 2000            # Increase number of trees to enhance diversity
# ROWS_PER_TREE = 1500          # Further reduce data per tree for more randomness
# MAX_DEPTH = 10                # Shallow depth for better generalization (8-12)
# MIN_SAMPLES_SPLIT = 10        # Allow more splits by reducing samples per split
# MIN_SAMPLES_LEAF = 1          # Smaller leaf nodes for better capturing of patterns
# MIN_IMPURITY_DECREASE = 0.01   # Require a larger decrease in impurity for splits
# MAX_FEATURES = 1           # Reduce features considered at each split to 30% for more randomness
# 98.62 4.0: 1921, 3.0: 36, 1.0: 25, 2.0: 18

# TOTAL_TREES = 5000            # Increase number of trees to enhance diversity
# ROWS_PER_TREE = 1500          # Further reduce data per tree for more randomness
# MAX_DEPTH = 10                # Shallow depth for better generalization (8-12)
# MIN_SAMPLES_SPLIT = 30        # Allow more splits by reducing samples per split
# MIN_SAMPLES_LEAF = 1          # Smaller leaf nodes for better capturing of patterns
# MIN_IMPURITY_DECREASE = 0.01   # Require a larger decrease in impurity for splits
# MAX_FEATURES = 1           # Reduce features considered at each split to 30% for more randomness
# 98.03 4.0: 4832, 3.0: 74, 1.0: 70, 2.0: 24


# Data Loading Functions
def load_train_data(train_file: str) -> List[List[Any]]:
    df = pd.read_csv(train_file, encoding='utf-8-sig', low_memory=False)
    # drop stray Unnamed or blank columns
    df = df.loc[:, ~(df.columns.str.startswith("Unnamed") | (df.columns.str.strip() == ""))]
    # drop metadata if present
    for meta in ('ticker', 'company', 'sector'):
        if meta in df.columns:
            df = df.drop(columns=meta)
    # select the same 9 ratios + label
    features = [
        'Sharpe Ratio', 'PE_ratio', 'EPS_ratio', 'PS_ratio',
        'NetProfitMargin_ratio', 'current_ratio', 'roe_ratio',
        'div_yeild', 'CAGR'
    ]
    label = 'BuyLabel'
    df = df[features + [label]].dropna()
    data = df.values.tolist()
    random.shuffle(data)
    return data

def load_test_data(test_file: str) -> List[List[Any]]:
    df = pd.read_csv(test_file, encoding='utf-8-sig', low_memory=False)
    # drop stray Unnamed or blank columns
    df = df.loc[:, ~(df.columns.str.startswith("Unnamed") | (df.columns.str.strip() == ""))]
    # drop metadata if present
    for meta in ('ticker', 'company', 'sector'):
        if meta in df.columns:
            df = df.drop(columns=meta)
    # select the same 9 ratios + label
    features = [
        'Sharpe Ratio', 'PE_ratio', 'EPS_ratio', 'PS_ratio',
        'NetProfitMargin_ratio', 'current_ratio', 'roe_ratio',
        'div_yeild', 'CAGR'
    ]
    label = 'BuyLabel'
    df = df[features + [label]].dropna()
    return df.values.tolist()


# Forest Builder
def build_forest(train_data: List[List[Any]]) -> List[Any]:
    """Build random forest from training data using parallelism and progress bar"""
    forest = Parallel(n_jobs=-1)(
        delayed(build_tree)(
            random.choices(train_data, k=ROWS_PER_TREE),
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            min_impurity_decrease=MIN_IMPURITY_DECREASE,
            max_features=MAX_FEATURES
        )
        for _ in tqdm(range(TOTAL_TREES), desc="Building Trees")
    )
    return forest


# Evaluation Functions
def majority_vote(predictions: List[Any]) -> Any:
    return Counter(predictions).most_common(1)[0][0]

def evaluate_forest(forest: List[Any], test_data: List[List[Any]]) -> float:
    correct = 0
    total = len(test_data)
    for row in test_data:
        preds = [predict_class(row[:-1], tree) for tree in forest]
        if majority_vote(preds) == row[-1]:
            correct += 1
    accuracy = correct / total if total else 0
    print(f"\nForest Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# Main 
if __name__ == "__main__":
    TRAIN_FILE = "train_refined.csv"
    TEST_FILE  = "test_refined.csv"

    print("Loading training data...")
    train_data = load_train_data(TRAIN_FILE)
    print(f" → {len(train_data)} training samples")

    print("\nBuilding random forest...")
    forest = build_forest(train_data)

    print("\nLoading test data...")
    test_data = load_test_data(TEST_FILE)
    print(f" → {len(test_data)} test samples")

    print("\nEvaluating forest...")
    accuracy = evaluate_forest(forest, test_data)

    print(f"\nProcess completed. Final accuracy: {accuracy:.2%}")
