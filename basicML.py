from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

iris = pd.read_csv('data/Iris.csv')
x = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = iris['Species']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=50)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

print(accuracy_score(y_test, y_predict))
print(confusion_matrix(y_test,y_predict))