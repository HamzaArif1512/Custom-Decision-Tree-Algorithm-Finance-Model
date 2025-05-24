from typing import List, Any
import pandas as pd
import random
from collections import Counter
from decisionTree import build_tree, predict_class
from joblib import Parallel, delayed
from tqdm import tqdm

# Parameters

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
