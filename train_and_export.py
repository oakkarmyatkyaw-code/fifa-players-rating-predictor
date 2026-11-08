import pandas as pd
df = pd.read_csv('fifa_players.csv')
df.info()

missing_values = df.isnull().sum()
print(f"missing_values: {missing_values[missing_values >0]}")

cate_summary = pd.DataFrame({
    'Unique_Values': df.select_dtypes(include=['object']).nunique(),
    'Sample_Value': df.select_dtypes(include=['object']).iloc[5]
})
print("\n=== Categorical Columns Analysis ===")
print(cate_summary)

num_summary = pd.DataFrame({
    'Unique_Values': df.select_dtypes(include=['int64', 'float64']).nunique(),
    'Sample_Value': df.select_dtypes(include=['int64', 'float64']).iloc[5]
})
print("\n=== Numerical Columns Analysis ===")
print(num_summary)

continuous_summary = pd.DataFrame({
    'Unique_Values': df.select_dtypes(include=['float64']).nunique(),
    'Sample_Value': df.select_dtypes(include=['float64']).iloc[0]
})
print("\n=== Continuous Columns Analysis ===")
print(continuous_summary)

int_summary = pd.DataFrame({
    'Unique_Values': df.select_dtypes(include=['int64']).nunique(),
    'Sample_Value': df.select_dtypes(include=['int64']).iloc[0]
})
print("\n=== Integer Columns Analysis ===")
print(int_summary)

df['positions'].value_counts()

df = df[~df['positions'].str.contains('GK')]

print(df['positions'].str.contains('GK', na=False).sum())

missing_values = df.isnull().sum()
mv_columns = missing_values[missing_values > 0].index.tolist()
mv_columns = [col for col in mv_columns if col != 'value_euro']
df = df.drop(columns=mv_columns)
df.shape

df = df.dropna(subset=['value_euro'])
df.shape

cate_cols = df.select_dtypes(include = 'object').columns
df = df.drop(columns=cate_cols)
df.shape

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression

y = df['overall_rating']

num_cols = df.select_dtypes(include=['int64', 'float64']).columns

X = df[num_cols.drop('overall_rating')]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=71)

from sklearn.preprocessing import MinMaxScaler


pipeline = Pipeline(steps=[
    ('feature_selection', SelectKBest(score_func=f_regression, k=14)),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
pipeline

pipeline.fit(X_train, y_train)

y_pred =pipeline.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE)     : {mae:.2f}")
print(f"Mean Squared Error (MSE)      : {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared Score (R²)          : {r2:.4f}")

X_selected = pipeline.named_steps['feature_selection'].transform(X_train)
print("Selected Features Shape:", X_selected.shape)
selected_features = X_train.columns[
    pipeline.named_steps['feature_selection'].get_support()
]
pd.DataFrame(X_selected, columns=selected_features, index=X_train.index).head()

df[['potential', 'value_euro', 'skill_moves(1-5)','crossing', 'short_passing' ]].value_counts()

import pickle

with open('Player_rating Predictor.pkl', 'wb') as f:
    pickle.dump(pipeline, f)