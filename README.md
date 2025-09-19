# 📊 Green Bonds Risk Assessment

An advanced **Flask web application** for assessing the risk of **Green Bonds** using **Machine Learning** and **Time Series Forecasting**.  

This project provides:
- Automated **data preprocessing**
- **Clustering bonds** into performance categories
- Calculation of **financial risk metrics**
- **Price forecasting** with ARIMA
- Interactive **dashboard visualization**

---

## ✨ Features
- 📂 Upload or input bond data manually  
- 🔄 Automated preprocessing (NaN removal, sorting, returns calculation)  
- 🧩 KMeans clustering (Weak | Good | Excellent)  
- 📈 Risk metrics (Return, Volatility, Max Drawdown, Sensitivity, Risk Level)  
- 🔮 ARIMA forecasting with RMSE evaluation  
- 📊 Beautiful charts & tables in a web dashboard  

---

## 🛠️ Tech Stack
- **Backend**: Flask, Pandas, NumPy, Scikit-learn, Statsmodels  
- **Frontend**: HTML, Bootstrap, Chart.js  
- **ML Models**: KMeans (Clustering), ARIMA (Forecasting)  

---
## 📂 Repository Structure
```
data-warehouse-project/
│
├── data/
├   ├── train       
│   ├── test_data                             
├── app/
├   ├── app.py      
│   ├── preprocess_data.py  
│   ├── train_or_load_model.py      
│   ├── risk_analysis.py
├── templates/
├   ├── index.html
├── README.md                           # Project overview and instructions
├── LICENSE                             # License information for the repository
├── .gitignore                          # Files and directories to be ignored by Git
```
## 🚀 How to Run
```bash
# Clone repo
git clone https://github.com/username/green-bonds-risk.git
cd green-bonds-risk

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py
