


# Decision Tree Classifier: Bank Marketing Dataset

## Overview
This project aims to build and evaluate a **Decision Tree Classifier** to predict whether a customer will subscribe to a term deposit (product) based on their demographic and behavioral data. The dataset used is the **Bank Marketing dataset** from the **UCI Machine Learning Repository**.

The key steps involve:
1. Data loading and preprocessing.
2. Training a decision tree classifier.
3. Model evaluation using metrics like accuracy, classification report, and confusion matrix.
4. Visualizing the decision tree structure.

## Dataset
The dataset used in this project is the **Bank Marketing Dataset**. It can be found at the following link:  
[UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv).

The target variable (`y`) indicates whether the customer subscribed to a term deposit:
- `1`: Yes (Subscribed)
- `0`: No (Did not subscribe)

## Libraries Used
- **pandas**: For data manipulation and preprocessing.
- **scikit-learn**: For building and evaluating the Decision Tree model.
- **matplotlib**: For visualizing the decision tree structure.

## Steps

### Step 1: Importing Necessary Libraries
We start by importing all the necessary Python libraries:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
```

### Step 2: Loading the Dataset
The dataset is loaded directly from the UCI Machine Learning Repository.
```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"
data = pd.read_csv(url, sep=';')
```
- The dataset is in **CSV format**, and we specify the delimiter as a semicolon (`;`).

### Step 3: Data Preprocessing
The dataset contains categorical features, so **one-hot encoding** is used to convert them into a numerical format:
```python
data_encoded = pd.get_dummies(data)

X = data_encoded.drop(columns=['y_yes', 'y_no'])
y = data_encoded['y_yes']
```
- **One-Hot Encoding**: Converts categorical variables into binary columns (0/1).
- **`X`**: Features (input variables), dropping the target columns (`y_yes`, `y_no`).
- **`y`**: Target variable (`y_yes`), representing whether a customer subscribed (`yes = 1`, `no = 0`).

### Step 4: Splitting the Dataset
The data is split into **training** and **testing** sets using a 70-30 split.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
- **70%** for training and **30%** for testing.
- **random_state=42** ensures reproducibility of the results.

### Step 5: Training the Decision Tree Classifier
We instantiate and train the **Decision Tree Classifier**:
```python
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
```
- The decision tree model is trained on the **training set** (`X_train`, `y_train`).

### Step 6: Making Predictions
Once the model is trained, we use it to make predictions on the **test set**:
```python
y_pred = dt_classifier.predict(X_test)
```

### Step 7: Evaluating the Model
The model's performance is evaluated using **accuracy**, **classification report**, and the **confusion matrix**:
```python
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

#### Evaluation Metrics:
- **Accuracy**: Proportion of correct predictions.
- **Classification Report**: Includes precision, recall, F1-score, and support for each class:
  - **Precision**: The proportion of positive identifications that were correct.
  - **Recall**: The proportion of actual positives that were identified correctly.
  - **F1-Score**: The weighted average of precision and recall.
  - **Support**: Number of actual occurrences for each class.
- **Confusion Matrix**: Provides a breakdown of true positives, true negatives, false positives, and false negatives.

### Step 8: Visualizing the Decision Tree
We plot and visualize the decision tree to understand the splits made by the model:
```python
plt.figure(figsize=(20,10))
tree.plot_tree(dt_classifier, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()
```
- **`tree.plot_tree()`** is used to visualize the structure of the decision tree, showing how it splits on different features and classifies data points.

## Results

### Model Performance:
- **Accuracy**: The model achieved an accuracy of **89%**, indicating that 89% of the predictions were correct.
- The **Classification Report** and **Confusion Matrix** revealed that the model performs well in predicting customers who **do not** subscribe but struggles with predicting actual subscribers.

#### Key Observations:
- **Class Imbalance**: The dataset is imbalanced, with far more customers **not subscribing** than subscribing. This leads to a bias in the model toward predicting **no subscription**.
- **Model Precision and Recall**: For predicting subscribers (`class 1`), the model's **precision** and **recall** are both around **0.51**, indicating that it misses many subscribers.

## Improvements:
To improve the performance, especially for predicting subscribers, consider:
- **Handling Class Imbalance**: 
  - Techniques such as **oversampling** the minority class (e.g., using SMOTE) or **undersampling** the majority class.
  - **Class weighting** can also be applied to give more importance to the minority class.
  
- **Hyperparameter Tuning**: Optimize the decision tree's parameters like `max_depth`, `min_samples_split`, etc.
  
- **Using Advanced Algorithms**: Ensemble methods like **Random Forest** or **Gradient Boosting** often outperform basic decision trees.

## Conclusion
This project demonstrates how a decision tree classifier can be used for **binary classification** in a marketing context, predicting customer behavior based on demographic and transactional features. The evaluation metrics provided insight into the model's strengths and limitations, particularly in handling imbalanced data.

The project showcases the importance of:
- **Preprocessing categorical data** through one-hot encoding.
- **Splitting the dataset** into training and testing sets for proper model evaluation.
- **Evaluating models** using multiple metrics beyond just accuracy.
- **Visualizing decision trees** for interpretability.

---

