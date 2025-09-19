# 📊 Green Bonds Risk Assessment - Notebook Version
# -------------------------------------------------
# This notebook demonstrates:
# - Data preprocessing
# - Risk metrics calculation
# - Clustering (KMeans)
# - Forecasting (ARIMA)
# - Visualization

# 📌 Step 1: Detect Columns
- `detect_date_column` → Automatically detects the **date column** in the dataset.  
- `detect_price_column` → Automatically detects the **price/value column**.  

  # 📌 Step 2: Preprocess Data
Function: `preprocess_data`  
- Removes NaN & duplicates  
- Sorts by date  
- Converts price to numeric  
- Calculates daily returns  

# 📌 Step 3: Train or Load Model
Function: `train_or_load_model`  
- Uses **KMeans** clustering  
- 3 clusters → Weak, Good, Excellent  

# 📌 Step 4: Calculate Risk Metrics
Function: `calculate_risk_metrics`  
- Expected Return  
- Volatility  
- Max Drawdown  
- Interest Rate Sensitivity  
- Risk Level (Low / Medium / High)  

# 📌 Step 5: Forecast Future (ARIMA)
Function: `forecast_future`  
- Uses ARIMA model  
- Forecasts future prices  
- Evaluates with RMSE  
# 🚀 Final Output
The system allows users to:  
- Upload or manually input bond data  
- Automatically clean and preprocess data  
- Cluster bonds into performance groups  
- Calculate risk metrics  
- Forecast future bond values  
- View results in an interactive dashboard  


from flask import Flask, render_template, request
import pandas as pd
import os
import pickle
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import math
import datetime

app = Flask(__name__)

# ---------- Helper Functions ----------
def detect_date_column(df):
    for col in df.columns:
        name = str(col).lower()
        if any(k in name for k in ["date", "time", "period", "time_period", "year", "تاريخ"]):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.6:
                    return col
            except Exception:
                pass
    return None

def detect_price_column(df):
    for col in df.columns:
        name = str(col).lower()
        if any(k in name for k in ["close", "price", "value", "obs_value", "amount", "issued"]):
            if pd.api.types.is_numeric_dtype(df[col]) or pd.to_numeric(df[col], errors="coerce").notna().mean() > 0.8:
                return col
    numeric_cols = df.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns
    return numeric_cols[0] if len(numeric_cols) > 0 else None

def preprocess_data(data_or_path, save=True, date_col=None, price_col=None):
    if isinstance(data_or_path, pd.DataFrame):
        data = data_or_path.copy()
    elif isinstance(data_or_path, str):
        if data_or_path.endswith(".csv"):
            data = pd.read_csv(data_or_path)
        elif data_or_path.endswith((".xlsx", ".xls")):
            data = pd.read_excel(data_or_path)
        else:
            raise ValueError(" نوع الملف غير مدعوم")
    else:
        raise ValueError(" preprocess_data يستقبل إما DataFrame أو Path")

    data.dropna(how="all", inplace=True)

    if date_col is None:
        date_col = detect_date_column(data)
    if not date_col:
        data["date"] = pd.date_range(start="2024-09-01", periods=len(data), freq="M")
        date_col = "date"

    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data.dropna(subset=[date_col], inplace=True)
    data.drop_duplicates(subset=[date_col], inplace=True)
    data.sort_values(by=date_col, inplace=True)

    if price_col is None:
        price_col = detect_price_column(data)
    if not price_col:
        raise ValueError("لم يتم العثور على عمود قيمة مناسب")

    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")
    data = data.dropna(subset=[price_col])
    data = data[np.isfinite(data[price_col])]

    data["daily_return"] = data[price_col].pct_change()
    data.dropna(subset=["daily_return"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"preprocessed_{timestamp}.csv"
        data.to_csv(output_file, index=False)
        return data, date_col, price_col

    return data, date_col, price_col

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

# ---------- Risk Metrics ----------
def calculate_risk_metrics(df):
    returns = df["daily_return"]

    expected_return = round(np.mean(returns) * 252 * 100, 2)
    volatility = round(np.std(returns) * np.sqrt(252) * 100, 2)
    max_drawdown = round(((df["cum_return"] / df["cum_return"].cummax()) - 1).min() * 100, 2)
    try:
        corr_val = np.corrcoef(returns[1:], returns.shift(1).dropna())[0, 1]
        sensitivity = round(abs(corr_val) * 10, 1)
    except Exception:
        sensitivity = 0.0

    if volatility < 10:
        risk_level = "منخفض المخاطر"
    elif volatility < 20:
        risk_level = "متوسط المخاطر"
    else:
        risk_level = "عالي المخاطر"

    return {
        "العائد المتوقع": f"{expected_return}%",
        "حساسية أسعار الفائدة": sensitivity,
        "أكبر انخفاض (Max Drawdown)": f"{max_drawdown}%",
        "التذبذب (Volatility)": f"{volatility}%",
        "مستوى المخاطرة": risk_level
    }

# ---------- Forecasting (ARIMA) ----------
def forecast_future(df, date_col, price_col, periods=7):
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    series = df[price_col].astype(float).dropna()

    if len(series) < 10:
        return {"labels": [], "values": [], "accuracy_rmse": None}

    model = ARIMA(series, order=(4, 1, 4))
    model_fit = model.fit()

    test_size = max(5, len(series) // 10)
    train, test = series[:-test_size], series[-test_size:]

    model_val = ARIMA(train, order=(4, 1, 4)).fit()
    preds = model_val.forecast(steps=len(test))

    rmse = math.sqrt(mean_squared_error(test, preds))

    forecast = model_fit.forecast(steps=periods)
    last_date = df[date_col].iloc[-1]
    future_dates = pd.date_range(last_date, periods=periods + 1, freq="D")[1:]

    return {
        "labels": future_dates.strftime("%Y-%m-%d").tolist(),
        "values": forecast.tolist(),
        "accuracy_rmse": round(rmse, 2)
    }

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    results, chart_data, prediction, forecast_data, error = None, None, None, None, None

    if request.method == "POST":
        try:
            df = None
            form_type = request.form.get("form_type", None)

            if form_type == "manual":
                country = request.form.get("country")
                sector = request.form.get("sector")
                period = request.form.get("period")
                bond_min = request.form.get("bond_min")
                bond_max = request.form.get("bond_max")

                if not (country and sector and period and bond_min and bond_max):
                    error = "الرجاء تعبئة جميع حقول الإدخال اليدوي"
                    return render_template("index.html", results=None, error=error)

                try:
                    bond_min = float(bond_min)
                    bond_max = float(bond_max)
                except ValueError:
                    error = " قيمة Bond Min/Max غير صحيحة"
                    return render_template("index.html", results=None, error=error)

                filepath = r"C:\Users\adel mohamedll\Desktop\Green Bond Risk Assessment project\Data\Global Sustainable Bonds Data.csv"
                if not os.path.exists(filepath):
                    error = " ملف البيانات المحلي غير موجود (تحقق من المسار)"
                    return render_template("index.html", results=None, error=error)

                df_full = pd.read_csv(filepath)

                cols_lower = {c: c for c in df_full.columns}
                required_lower = ["Issuer location", "Issuer sector", "Bond type", "amount"]
                if not all(r in cols_lower for r in required_lower):
                    missing = [r for r in required_lower if r not in cols_lower]
                    error = f" الأعمدة المطلوبة غير موجودة: {missing}"
                    return render_template("index.html", results=None, error=error)

                loc_col = cols_lower["Issuer location"]
                sector_col = cols_lower["Issuer sector"]
                bondtype_col = cols_lower["Bond type"]
                amount_col = cols_lower["amount"]


                print("المواقع:", df["Issuer location"].unique())
                print("القطاعات:", df["Issuer sector"].unique())
                print("الفترات:", df["Bond type"].unique())
                print("أقل وأكبر مبلغ:", df["Amount"].min(), df["Amount"].max())

                df_full[amount_col] = pd.to_numeric(df_full[amount_col], errors="coerce")
                filtered_df = df[
                (df["Issuer location"].str.strip().str.lower() == country.strip().lower()) &
                (df["Issuer sector"].str.strip().str.lower() == sector.strip().lower()) &
                (df["Bond type"].str.strip().str.lower() == period.strip().lower()) &
                (df["Amount"].between(bond_min, bond_max))
                 ]


                if filtered_df.empty:
                    error = "لا توجد بيانات بعد تطبيق الفلاتر"
                    return render_template("index.html", results=None, error=error)

                df, date_col, price_col = preprocess_data(filtered_df, price_col=amount_col)



            elif form_type == "upload" or ("file" in request.files and request.files["file"].filename != ""):
                file = request.files.get("file")
                if not file or file.filename == "":
                    error = " لم يتم رفع ملف"
                    return render_template("index.html", results=None, error=error)

                filename = secure_filename(file.filename)
                os.makedirs("uploads", exist_ok=True)
                filepath = os.path.join("uploads", filename)
                file.save(filepath)

                df, date_col, price_col = preprocess_data(filepath)

            else:
                error = " لم يتم إدخال بيانات أو رفع ملف (تأكد من اختيار طريقة الإدخال في الواجهة)"
                return render_template("index.html", results=None, error=error)

            if df is len(df) < 5:
                error = " البيانات قليلة جدًا للتنبؤ"
                return render_template("index.html", results=None, error=error)

            df["cum_return"] = (1 + df["daily_return"]).cumprod()

            model = train_or_load_model(df)
            df["cluster"] = model.predict(df[["daily_return"]])

            cluster_means = df.groupby("cluster")["daily_return"].mean().sort_values()

            sorted_idx = list(cluster_means.index)
            mapping = {}
            if len(sorted_idx) >= 3:
                mapping = {
                    sorted_idx[0]: "ضعيف",
                    sorted_idx[1]: "جيد",
                    sorted_idx[2]: "ممتاز"
                }
            else:
                for i, cl in enumerate(sorted_idx):
                    mapping[cl] = ["ضعيف", "جيد", "ممتاز"][i if i < 3 else -1]

            df["final_label"] = df["cluster"].map(mapping)

            prediction = calculate_risk_metrics(df)
            results = df[[date_col, price_col, "daily_return", "final_label"]].tail(10)

            chart_data = {
                "labels": df[date_col].dt.strftime("%Y-%m-%d").tolist(),
                "values": df[price_col].astype(float).tolist()
            }

            forecast_data = forecast_future(df, date_col=date_col, price_col=price_col, periods=7)

        except Exception as e:
            error = f" خطأ أثناء المعالجة: {str(e)}"

    return render_template(
        "index.html",
        results=results,
        chart_data=chart_data,
        prediction=prediction,
        forecast_data=forecast_data,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
