"""
data_modelling.py
"""
import pandas as pd
from data_summary import X, y
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# logit regression
glm = LogisticRegression(random_state=0).fit(X_train, y_train)
glm.predict_proba(X_train)
mean_acc_is = glm.score(X_train, y_train)

yhat = glm.predict(X_test)
mean_acc_oos = glm.score(X_test, y_test)
