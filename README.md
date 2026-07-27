# 🚗 Car Price Prediction using Machine Learning

An end-to-end Machine Learning project that predicts the selling price of a car using **Linear Regression**. This project demonstrates data preprocessing, feature engineering, model training, evaluation, and deployment with **Streamlit**.

---

## 📌 Project Overview

The goal of this project is to predict the selling price of a car based on its features such as engine size, horsepower, fuel type, drive wheels, and other specifications.

This project follows a complete Machine Learning workflow:

- Data Collection
- Data Preprocessing
- Feature Encoding
- Train-Test Split
- Model Training
- Model Evaluation
- Model Deployment

---

## 📂 Dataset

**Dataset:** Car Price Assignment Dataset

The dataset contains various car specifications including:

- Car Name
- Fuel Type
- Aspiration
- Car Body
- Drive Wheels
- Engine Size
- Horsepower
- Peak RPM
- Highway MPG
- Price

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Joblib
- Streamlit

---

## 🤖 Machine Learning Model

Model Used:

**Linear Regression**

Evaluation Metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📊 Project Structure

```text
car-price-prediction-ml/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── data/
├── images/
└── models/
```

---

## 🚀 How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
```

### Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📈 Output

The project automatically:

- Trains the model
- Evaluates model performance
- Saves the trained model
- Generates prediction graphs

---

## 🎯 Future Improvements

- Random Forest Regressor
- XGBoost
- Hyperparameter Tuning
- Better Streamlit User Interface

---

## 👨‍💻 Author

Talha
