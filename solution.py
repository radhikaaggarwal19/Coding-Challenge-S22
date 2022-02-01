import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

def load_dataset(filename):
  data = pd.read_csv(filename)
  data = pd.get_dummies(data)
  X = data.iloc[:, 2:]
  y = data.iloc[:, 0]
  return X, y, data

X, y, data = load_dataset("/content/mushrooms.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

LR = LogisticRegression()

fittedLR = LR.fit(X_train, y_train)

fittedLRPrediction = LR.predict(X_test)

print("Logistic Regression: " + str(accuracy_score(fittedLRPrediction, y_test) * 100) + "%")