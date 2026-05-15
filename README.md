<img width="838" height="748" alt="image" src="https://github.com/user-attachments/assets/8309a163-e6f0-44d7-bab1-b4174f3724a4" /># Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. mport the required libraries such as pandas, matplotlib, sklearn model selection, decision tree classifier, and accuracy metrics.
2. mport the required libraries such as pandas, matplotlib, sklearn model selection, decision tree classifier, and accuracy metrics.
3. Divide the dataset into training and testing data using train_test_split(), then create and train the DecisionTreeClassifier model using the training data.
4. Predict the output for test data, calculate the model accuracy using accuracy_score(), and display the decision tree using plot_tree().

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Danaseelan G
RegisterNumber:  212225040053
*/
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
data = pd.read_csv("Employee.csv")
data = pd.get_dummies(data, drop_first=True)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.show()
```

## Output:
<img width="838" height="748" alt="image" src="https://github.com/user-attachments/assets/b699fee9-c4b3-4e43-b07b-eaca0bf509fd" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
