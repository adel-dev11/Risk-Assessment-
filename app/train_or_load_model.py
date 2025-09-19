# train_model.py
import pandas as pd
import numpy as np
import os
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import silhouette_score


# ==============================
# 1) Clustering + Classification Model
# ==============================
def train_clustering_model(file_path):
    if file_path.endswith(".csv"):
        df=pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df=pd.read_excel(file_path,engine="openpyxl")
    else:
        raise ValueError("file type are not supported !!!")

    cols_to_drop = ["Link to Bond", "Link to Program", "Link to Framework"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    for col in df.select_dtypes(include="datetime").columns:
        df[col + "_year"] = df[col].dt.year
        df[col + "_month"] = df[col].dt.month
        df[col + "_day"] = df[col].dt.day
        df = df.drop(columns=[col])

    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)

    df["Cluster"] = clusters

    # Random Forest للتصنيف
    X = df.drop(columns=["Cluster"])
    y = df["Cluster"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le, "labelencoder.pkl")
    joblib.dump(kmeans, "kmeans.pkl")

    return accuracy, df, model


def train_or_load_model(data, force_train=True):
    model_path = "model.pkl"

    if os.path.exists(model_path) and not force_train:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(data[["daily_return"]])
    
    score = silhouette_score(data[['daily_return']], model.labels_)
    print("Silhouette Score:", score)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model


# ==============================
# 2) Forecasting Model (ARIMA example)
# ==============================
def train_forecasting_model(df, column="Price", steps=30):
    """
    df: DataFrame فيه بيانات الأسعار (لازم عمود باسم Price أو اللي تحدده)
    steps: عدد الأيام للتنبؤ
    """
    if column not in df.columns:
        raise ValueError(f"العمود {column} مش موجود في البيانات")

    series = df[column].dropna()

    # تدريب ARIMA
    model = ARIMA(series, order=(5, 1, 0))  # ممكن تغير الـ order بعد الـ tuning
    fitted_model = model.fit()

    # التنبؤ
    forecast = fitted_model.forecast(steps=steps)

    # حفظ الموديل
    joblib.dump(fitted_model, "forecast_model.pkl")

    return forecast



if __name__ == "__main__":
    file_path = input("please enter file path : ")

    # تدريب موديل الكلاسترنج
    acc, df, model = train_clustering_model(file_path)
    print("Clustering Model Accuracy:", acc)


    if "Yield" in df.columns:
        forecast = train_forecasting_model(df, column="Yield", steps=30)
        print("Forecasting next 30 days:")
        print(forecast)
