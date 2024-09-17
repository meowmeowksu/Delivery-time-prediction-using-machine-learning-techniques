import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

def normal(df):
    for column in df.columns:
        enc = MinMaxScaler()
        enc.fit(df[[column]])
        df[column] = enc.transform(df[[column]])
    return df

df = pd.read_csv('/home/meowksu/delivery/dataset.csv')
print(df.info())
print(df.head(5))

print("Описательная статистика:")
print(df.describe())

print("\nПропущенные значения:")
print(df.isnull().sum())

print("\nДубликаты:")
duplicates = df.duplicated()
print(f"Количество строк с дубликатами: {duplicates.sum()}")

# Обработка пропущенных значений
print(df.isnull().sum())
df.dropna(inplace=True)
print(df.isnull().sum())

# Изменение формата данных на date_time
df['created_at'] = pd.to_datetime(df['created_at'])
df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'])

# Вычисление времени доставки в минутах
df['time_taken(mins)'] = (df['actual_delivery_time'] - df['created_at']).dt.total_seconds() / 60

# Удаление ненужных столбцов
df.drop(['created_at', 'actual_delivery_time'], axis=1, inplace=True)
df.drop(['store_id', 'store_primary_category'], axis=1, inplace=True)

# Применение правила трех сигм для удаления выбросов
threshold = 3
cols_to_check = [col for col in df.columns if df[col].dtype!= 'object']
for col in cols_to_check:
    df = df[np.abs(zscore(df[col])) < threshold]

# Масштабирование данных с помощью StandardScaler
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)

# Разделение данных на признаки (X) и целевую переменную (y)
X = df.drop(columns = ["time_taken(mins)"], axis = 1)
y = df["time_taken(mins)"]

# Удаление нерелевантных столбцов (с корреляцией меньше заданного порога)
threshold = 0.05
correlation_matrix = df.corr()['time_taken(mins)']
irrelevant_columns = correlation_matrix[abs(correlation_matrix) <= threshold].index
df.drop(irrelevant_columns, axis=1)

# Разделение данных на обучающую (70%) и тестовую (30%) выборки
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Обучение и оценка различных моделей машинного обучения
# Линейная регрессия
reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print('___________ linear regression ____________')
print('MAPE:{}'.format(mean_absolute_percentage_error(y_test,y_pred)))
print('r2_score:{}'.format(r2_score(y_test,y_pred)))

# KNN
reg = KNeighborsRegressor(n_neighbors=5)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print('___________ KNN ____________')
print('MAPE:{}'.format(mean_absolute_percentage_error(y_test,y_pred)))
print('r2_score:{}'.format(r2_score(y_test,y_pred)))

# Дерево решений
reg = DecisionTreeRegressor(min_samples_leaf=7,
min_samples_split=7, criterion='squared_error')
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print('___________ Decision tree ____________')
print('MAPE:{}'.format(mean_absolute_percentage_error(y_test,y_pred)))
print('r2_score:{}'.format(r2_score(y_test,y_pred)))

# Случайный лес
reg = RandomForestRegressor(min_samples_leaf=7, min_samples_split=7, criterion='squared_error')
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print('___________ Random forest ____________')
print('MAPE:{}'.format(mean_absolute_percentage_error(y_test,y_pred)))
print('r2_score:{}'.format(r2_score(y_test,y_pred)))

# XGBoost
xgb=XGBRegressor()
xgb.fit(X_train,y_train)
y_pred=xgb.predict(X_test)
print('___________ XGBoost ____________')
print('MAPE:{}'.format(mean_absolute_percentage_error(y_test,y_pred)))
print('r2_score:{}'.format(r2_score(y_test,y_pred)))
