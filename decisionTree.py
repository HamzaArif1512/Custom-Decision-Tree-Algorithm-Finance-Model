# decisionTree.py
import pandas as pd
import random
from typing import List, Dict, Any, Tuple, Union

class Question:
    """A question used to partition a dataset."""
    def __init__(self, column: int, value: Any):
        self.column = column
        self.value = value

    def match(self, example: List[Any]) -> bool:
        val = example[self.column]
        if isinstance(val, (int, float)):
            return val >= self.value
        return val == self.value

    def __repr__(self) -> str:
        condition = ">=" if isinstance(self.value, (int, float)) else "=="
        return f"Is column {self.column} {condition} {self.value}?"

class Leaf:
    """A leaf node containing predictions."""
    def __init__(self, rows: List[List[Any]]):
        self.predictions = class_counts(rows)

class DecisionNode:
    """A decision node that asks a question and has two branches."""
    def __init__(self, question: Question, true_branch: Any, false_branch: Any):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

# ========================
# Helper Functions
# ========================
def is_numeric(value: Any) -> bool:
    """Test if a value is numeric."""
    return isinstance(value, (int, float))

def class_counts(rows: List[List[Any]]) -> Dict[Any, int]:
    """Count the number of each class in a dataset."""
    counts = {}
    for row in rows:
        label = row[-1]
        counts[label] = counts.get(label, 0) + 1
    return counts

def partition(rows: List[List[Any]], question: Question) -> Tuple[List[List[Any]], List[List[Any]]]:
    """Partition a dataset based on a question."""
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def gini(rows: List[List[Any]]) -> float:
    """Calculate the Gini impurity for a dataset."""
    counts = class_counts(rows)
    impurity = 1
    for label in counts:
        prob_of_label = counts[label] / len(rows)
        impurity -= prob_of_label ** 2
    return impurity

def info_gain(left: List[List[Any]], right: List[List[Any]], current_uncertainty: float) -> float:
    """Calculate the information gain from a split."""
    p = len(left) / (len(left) + len(right))
    return current_uncertainty - (p * gini(left)) - ((1 - p) * gini(right))

# ========================
# Tree Construction
# ========================
def find_best_split(rows: List[List[Any]], 
                   min_impurity_decrease: float = 0.0, 
                   max_features: int = None) -> Tuple[float, Union[Question, None]]:
    """Find the best question to ask by iterating over every feature/value."""
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # Number of features (excluding label)

    # Limit features if max_features is specified
    features = list(range(n_features))
    if max_features and max_features < n_features:
        features = random.sample(features, max_features)

    for col in features:
        values = {row[col] for row in rows}  # Unique values in the column
        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            
            # Skip if no split occurred
            if not true_rows or not false_rows:
                continue
                
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain and gain >= min_impurity_decrease:
                best_gain, best_question = gain, question

    return best_gain, best_question

def build_tree(rows: List[List[Any]], 
              max_depth: int = 5, 
              min_samples_split: int = 10, 
              min_samples_leaf: int = 5, 
              min_impurity_decrease: float = 0.0, 
              max_features: int = None) -> Union[DecisionNode, Leaf]:
    """Build the decision tree with pre-pruning."""
    gain, question = find_best_split(rows, min_impurity_decrease, max_features)

    # Stopping conditions
    if (gain == 0 or len(rows) < min_samples_split or max_depth <= 0):
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)
    
    # Check leaf size conditions
    if len(true_rows) < min_samples_leaf or len(false_rows) < min_samples_leaf:
        return Leaf(rows)

    # Recursively build the true and false branches
    true_branch = build_tree(true_rows, max_depth-1, min_samples_split, 
                           min_samples_leaf, min_impurity_decrease, max_features)
    false_branch = build_tree(false_rows, max_depth-1, min_samples_split, 
                            min_samples_leaf, min_impurity_decrease, max_features)

    return DecisionNode(question, true_branch, false_branch)

# ========================
# Prediction Functions
# ========================
def classify(row: List[Any], node: Union[DecisionNode, Leaf]) -> Dict[Any, int]:
    """Classify a row using the decision tree."""
    if isinstance(node, Leaf):
        return node.predictions
    branch = node.true_branch if node.question.match(row) else node.false_branch
    return classify(row, branch)

def predict_class(row: List[Any], node: Union[DecisionNode, Leaf]) -> Any:
    """Predict the class for a single row."""
    predictions = classify(row, node)
    return max(predictions.items(), key=lambda x: x[1])[0]

# ========================
# Data Loading (Modified for better importability)
# ========================
def load_data(filename: str, test_size: float = 0.2) -> Tuple[List[List[Any]], List[List[Any]]]:
    """Load and split data into train and test sets."""
    df = pd.read_csv(filename)
    # features = [
    #     'Sharpe Ratio', 'PE_ratio', 'EPS_ratio', 'NetProfitMargin_ratio',
    #     'roe_ratio', 'div_yeild', 'CAGR', 'BuyLabel'
    # ]
    features = [
    'Sharpe Ratio', 'PE_ratio', 'EPS_ratio', 'PS_ratio',
    'NetProfitMargin_ratio', 'current_ratio', 'roe_ratio',
    'div_yeild', 'CAGR', 'BuyLabel'
]
    df = df[features].dropna()
    data = df.values.tolist()
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]

# ========================
# Main Execution (Only runs when file is executed directly)
# ========================

#Uncomment to run separately on its own

if __name__ == "__main__":
    from google.colab import files
    
    # Example usage when run directly
    uploaded = files.upload()
    filename = next(iter(uploaded))
    train_data, test_data = load_data(filename)
    
    # Build and evaluate a single tree
    tree = build_tree(
        train_data,
        max_depth=15, #was 7
        min_samples_split=100,
        min_samples_leaf=20, #was 50
        min_impurity_decrease=0.01,
        max_features=7 #was 5
    )
    
    # Calculate accuracy
    correct = 0
    for row in test_data:
        if predict_class(row[:-1], tree) == row[-1]:
            correct += 1
    print(f"Test Accuracy: {correct / len(test_data):.2%}")