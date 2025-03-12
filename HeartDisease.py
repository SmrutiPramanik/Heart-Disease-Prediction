import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
df=pd.read_csv('/content/heart - heart.csv')
df.head()
X=df[['age', 'cp','thalach']]
Y=df['target']
X,Y
model=LogisticRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
model.fit(X_train, Y_train)
Y_train_pred=model.predict(X_train)
Y_test_pred=model.predict(X_test)
y_proba=model.predict_proba(X_test)[:1]
y_proba
def predict_heart_disease():
  age=int(input("Enter age: "))
  cp=int(input("Enter chest pain type (0-3): "))
  thalach=int(input("Enter maximum heart rate achieved: "))

  # Use pd.DataFrame instead of np.DataFrame
  user_data=pd.DataFrame([[age,cp,thalach]], columns=['age', 'cp', 'thalach'])
  prediction=model.predict(user_data)
  result="Heart Disease Present" if prediction[0]==1 else "No Heart Disease"
  print(f"Prediction: {result}")

# Call the function once to start the prediction
predict_heart_disease()

import joblib
joblib.dump(model, 'heart_disease_model.pkl')
