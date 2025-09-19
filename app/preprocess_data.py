import pandas as pd
import numpy as np
import datetime

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
