import sqlite3 as sql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

titanic = pd.read_csv('train.csv')
print(titanic.info())
print(titanic.head())
print(titanic.describe())
print(titanic.isna().sum())

titanic['Pclass'].hist(bins=30)
plt.show()