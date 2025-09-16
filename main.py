import sqlite3 as sql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

titanic = pd.read_csv('data/train.csv')
print(titanic.info())
print(titanic.head())
print(titanic.describe())
print(titanic.isna().sum())

titanic['Pclass'].hist(bins=30)
plt.savefig('plots/pclass_hist.png')
plt.show()

sb.countplot(x='Survived', data=titanic)
plt.show()

sb.countplot(x='Sex', data=titanic)
plt.show()

sb.countplot(x='Embarked', data=titanic)
plt.show()

sb.boxplot(x='Survived', y='Age', data=titanic)
plt.show()

sb.boxplot(x='Survived', y='Fare', data=titanic)
plt.show()

corr = titanic.corr(numeric_only=True)
sb.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
