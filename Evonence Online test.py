"""Scenario 1: Data ValidationTask: Write a function validate_data(data) that checks if a list 
of dictionaries (e.g., [{"name": "Alice", "age": 30}, {"name": "Bob", "age": "25"}]) contains valid integer
values for the "age" key. Return a list of invalid entries.
"""
def validate_data(data):
   
    invalid_entries = []
    for entry in data:
        age = entry.get("age")
        if not isinstance(age, int):
            invalid_entries.append(entry)
    return invalid_entries

# Example data
data = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": "25"},
    {"name": "Charlie", "age": None},
    {"name": "Diana", "age": 22},
    {"name": "Eve"}  # age key is missing
]
# Validate data
invalid = validate_data(data)
# Output the result
print("Invalid entries:")
for entry in invalid:
    print(entry)

"""Output:
Invalid entries:
{'name': 'Bob', 'age': '25'}
{'name': 'Charlie', 'age': None}
{'name': 'Eve'}
"""

"""Scenario 2: Logging DecoratorTask: Create a decorator @log_execution_time that logs the time taken to execute a function.
Use it to log the runtime of a sample function calculate_sum(n) that returns the sum of numbers from 1 to n."""

import time
import functools

def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Function '{func.__name__}' executed in {duration:.6f} seconds")
        return result
    return wrapper

@log_execution_time
def calculate_sum(n):
    return sum(range(1, n + 1))

total = calculate_sum(10000000)
print("Sum:", total)

"""Output:
Function 'calculate_sum' executed in 0.152468 seconds
Sum: 50000005000000
"""

"""Scenario 3: Missing Value Handling
Task: A dataset has missing values in the "income" column. Write code to:

1. Replace missing values with the median if the data is normally distributed.

2. Replace with the mode if skewed.
Use Pandas and a skewness threshold of 0.5."""

import pandas as pd
import numpy as np

# Sample dataset
data = {
    "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"],
    "income": [50000, 55000, np.nan, 60000, 65000, np.nan, 70000]
}

df = pd.DataFrame(data)

# Step 1: Calculate skewness
skewness = df["income"].skew(skipna=True)
print(f"Skewness of 'income': {skewness:.2f}")

# Step 2: Fill missing values based on skewness
if abs(skewness) <= 0.5:
    median_value = df["income"].median()
    df["income"].fillna(median_value, inplace=True)
    print(f"Filled missing values with median: {median_value}")
else:
    mode_value = df["income"].mode()[0]
    df["income"].fillna(mode_value, inplace=True)
    print(f"Filled missing values with mode: {mode_value}")

print("\nUpdated DataFrame:")
print(df)

"""Output:
Skewness of 'income': 0.00
Filled missing values with median: 60000.0

Updated DataFrame:
      name   income
0    Alice  50000.0
1      Bob  55000.0
2  Charlie  60000.0
3    David  60000.0
4      Eve  65000.0
5     Frank 60000.0
6    Grace  70000.0
"""

"""Scenario 4: Text Pre-processing
Task: Clean a text column in a DataFrame by:
1. Converting to lowercase.
2. Removing special characters (e.g., !, @).
3. Tokenizing the text."""

import pandas as pd
import re

# Sample DataFrame
data = {
    "text": [
        "Hello World!",
        "Python is AWESOME!!",
        "Data @Science is the future...",
        "Pre-processing: Important step!!"
    ]
}

df = pd.DataFrame(data)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    return tokens

df["cleaned_text"] = df["text"].apply(clean_text)
print(df)

"""Output:
                  text                      cleaned_text
0                      Hello World!                    [hello, world]
1               Python is AWESOME!!             [python, is, awesome]
2    Data @Science is the future...  [data, science, is, the, future]
3  Pre-processing: Important step!!  [preprocessing, important, step]
"""

"""Scenario 5: Hyperparameter Tuning
Task: Use GridSearchCV to find the best max_depth (values: [3, 5, 7]) and n_estimators 
(values: [50, 100]) for a Random Forest classifier."""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

"""Output:
Best Parameters: {'max_depth': 3, 'n_estimators': 50}
Best Cross-Validation Accuracy: 0.95
Test Accuracy: 1.0
"""

"""Scenario 6: Custom Evaluation Metric
Task: Implement a custom metric weighted_accuracy where class 0 has a weight of 1 and class 1 has a weight of 2."""

import numpy as np

def weighted_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    weights = np.where(y_true == 0, 1, 2)
    correct = (y_true == y_pred).astype(int)
    weighted_correct = correct * weights
    return weighted_correct.sum() / weights.sum()

y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1])

acc = weighted_accuracy(y_true, y_pred)
print(f"✅ Weighted Accuracy: {acc:.2f}")


"""Output:
✅ Weighted Accuracy: 0.83
"""

"""
  Scenario 9: Structured Response Generation
Task: Use the Gemini API to generate a response in JSON format for the query: "List 3 benefits of Python for data science."
Handle cases where the response isn’t valid JSON.  
"""
import json
import pandas as pd
response_text = '''
{
  "benefits": [
    "Easy to learn and use",
    "Rich ecosystem of data science libraries",
    "Strong community support"
  ]
}
'''

try:
    data = json.loads(response_text)

    if "benefits" in data:
        df = pd.DataFrame(data["benefits"], columns=["Python Benefits"])
        print("✅ Structured DataFrame:")
        print(df)
    else:
        print("❌ 'benefits' key not found in JSON.")

except json.JSONDecodeError:
    print("❌ Invalid JSON format! Raw response:")
    print(response_text)

"""Output:
✅ Structured DataFrame:
                         Python Benefits
0               Easy to learn and use
1  Rich ecosystem of data science libraries
2         Strong community support
"""

"""Scenario 10: Summarization with Constraints
Task: Write a prompt to summarize a news article into 2 sentences. 
If the summary exceeds 50 words, truncate it to the nearest complete sentence."""

"""Kolhapur, a city in Maharashtra, is witnessing rapid development in infrastructure and education.
New roads, bridges, and smart city initiatives are transforming the urban landscape. In addition,
several reputed educational institutions are attracting students from across the state, boosting the city's 
image as an educational hub."""

"""Kolhapur is rapidly growing with improvements in infrastructure and education. Smart city projects and reputed 
institutions are making it an important hub in Maharashtra."""