import pandas as pd
import random
from collections import Counter
from decisionTree import build_tree, predict_class

# =============================
# Parameters
# =============================
TOTAL_TREES = 100
ROWS_PER_TREE = 3242
MAX_DEPTH = 7
MIN_SAMPLES_SPLIT = 100
MIN_SAMPLES_LEAF = 50
MIN_IMPURITY_DECREASE = 0.01
MAX_FEATURES = 5  # out of 7 features excluding label

# =============================
# Data Loading Functions
# =============================
def load_train_data(train_file: str) -> List[List[Any]]:
    """Load and prepare training data for forest building"""
    df = pd.read_csv(train_file)
    features = [
        'Sharpe Ratio', 'PE_ratio', 'EPS_ratio', 'NetProfitMargin_ratio',
        'roe_ratio', 'div_yeild', 'CAGR', 'BuyLabel'
    ]
    df = df[features].dropna()
    data = df.values.tolist()
    random.shuffle(data)
    return data

def load_test_data(test_file: str) -> List[List[Any]]:
    """Load and prepare test data for evaluation"""
    df = pd.read_csv(test_file)
    features = [
        'Sharpe Ratio', 'PE_ratio', 'EPS_ratio', 'NetProfitMargin_ratio',
        'roe_ratio', 'div_yeild', 'CAGR', 'BuyLabel'
    ]
    df = df[features].dropna()
    return df.values.tolist()

# =============================
# Forest Builder
# =============================
def build_forest(train_data: List[List[Any]]) -> List[Any]:
    """Build random forest from training data"""
    forest = []
    for i in range(TOTAL_TREES):
        # Get random sample with replacement
        subset = random.choices(train_data, k=ROWS_PER_TREE)
        
        print(f"Building tree {i+1}/{TOTAL_TREES}...")
        tree = build_tree(
            subset,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            min_impurity_decrease=MIN_IMPURITY_DECREASE,
            max_features=MAX_FEATURES
        )
        forest.append(tree)
    return forest

# =============================
# Evaluation Functions
# =============================
def majority_vote(predictions: List[Any]) -> Any:
    """Determine final prediction through majority voting"""
    vote_count = Counter(predictions)
    return vote_count.most_common(1)[0][0]

def evaluate_forest(forest: List[Any], test_data: List[List[Any]]) -> float:
    """Evaluate forest performance on test data"""
    correct = 0
    total = len(test_data)
    for row in test_data:
        predictions = [predict_class(row[:-1], tree) for tree in forest]
        final_prediction = majority_vote(predictions)
        if final_prediction == row[-1]:
            correct += 1
    accuracy = correct / total if total > 0 else 0
    print(f"\nForest Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# =============================
# Main Execution
# =============================
if __name__ == "__main__":
    # File paths - change these to your actual file paths
    TRAIN_FILE = "finance_mod.csv"  # For building the forest
    TEST_FILE = "testdata.csv"    # For evaluation only
    
    print("Loading training data...")
    train_data = load_train_data(TRAIN_FILE)
    
    print("\nBuilding random forest...")
    forest = build_forest(train_data)
    
    print("\nLoading test data...")
    test_data = load_test_data(TEST_FILE)
    
    print("\nEvaluating forest...")
    accuracy = evaluate_forest(forest, test_data)
    
    print(f"\nProcess completed. Final accuracy: {accuracy:.2%}")