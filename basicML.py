from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = pd.read_csv('data/Iris.csv')
x = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = iris['Species']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=50) #для iris - 50 достаточно

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

print(accuracy_score(y_test, y_predict))
print(confusion_matrix(y_test,y_predict))

#-----------------------------------
#Матрица ошибок

sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, fmt="d", cmap="hot") #hot выглядит ужасно
plt.xlabel("Ожидаеемый")
plt.ylabel("Предсказанный")
plt.show()

#-----------------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

models = {
    "Лог": LogisticRegression(max_iter=200),
    "Дерево": DecisionTreeClassifier(),
    "Рандомный лес": RandomForestClassifier(),
    "kNN": KNeighborsClassifier(),
    "Градиентный бустинг": GradientBoostingClassifier()
}

for name, clf in models.items():
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    print(f"{name}: {accuracy_score(y_test, preds):.3f}, score: {f1_score(y_test, preds, average='weighted'):.3f}")