# 🔥 Forest Fire Weather Index (FWI) Prediction

A machine learning web app that predicts the **Fire Weather Index (FWI)** using Ridge Regression, trained on the **Algerian Forest Fires** dataset.

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://linearregressionproject.streamlit.app)

---

## 📂 Project Structure

```
├── app.py                  # Streamlit web app
├── application.py          # Flask web app (legacy)
├── models/
│   ├── ridge.pkl           # Trained Ridge Regression model
│   └── scaler.pkl          # StandardScaler
├── Dataset/
│   ├── Algerian_forest_fires_cleaned_dataset.csv
│   └── Algerian_forest_fires_dataset_UPDATE.csv
├── notebooks/
│   ├── EDA And FE Algerian Forest Fires.ipynb
│   └── 3.0-Model Training.ipynb
├── templates/              # Flask HTML templates
├── .streamlit/
│   └── config.toml         # Streamlit theme config
├── requirements.txt
├── Procfile
└── README.md
```

---

## 🧠 Model Details

| Item | Detail |
|------|--------|
| **Algorithm** | Ridge Regression |
| **Preprocessing** | StandardScaler |
| **Dataset** | Algerian Forest Fires (Bejaia & Sidi Bel-Abbes regions) |
| **Target** | Fire Weather Index (FWI) |

### Input Features

| Feature | Description |
|---------|-------------|
| Temperature | Temperature in °C |
| RH | Relative Humidity (%) |
| Ws | Wind Speed (km/h) |
| Rain | Rainfall (mm) |
| FFMC | Fine Fuel Moisture Code |
| DMC | Duff Moisture Code |
| ISI | Initial Spread Index |
| Classes | Fire / Not Fire |
| Region | Bejaia / Sidi Bel-Abbes |

---

## ⚙️ Setup & Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Harshraj112/Linear_regression_project.git
cd Linear_regression_project
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 📊 Notebooks

- **EDA & Feature Engineering** — Exploratory analysis of the Algerian Forest Fires dataset
- **Model Training** — Ridge Regression training with hyperparameter tuning

---

## 🛠️ Tech Stack

- Python, Streamlit, Scikit-learn, Pandas, NumPy
- Ridge Regression with StandardScaler pipeline

---

## 📜 License

Open source — feel free to use and modify.