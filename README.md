# 🏠 Elite Estates | House Price Predictor

**Live Demo**: [Elite Estates Predictor](https://housepricepridiction-cqqleipcmqgbankmbjd5na.streamlit.app/)

Elite Estates is a premium real estate valuation application built with Python, Streamlit, and Scikit-Learn. It provides precision market estimates for residential properties using a trained machine learning regression model.

## 🌟 Features

- **Premium UI/UX**: A state-of-the-art interface featuring custom typography, a modern color palette, and intuitive property spec cards.
- **Interactive Controls**: Users can configure property details using clean dropdown menus for square footage, bedrooms, bathrooms, and more.
- **Real-time Visualization**: Includes a dynamic Plotly scatter chart that benchmarks current estimates against market trends.
- **Accuracy Driven**: Leverages a Linear Regression model with an R² score of ~0.95, ensuring reliable valuations.
- **Persisted State**: Uses serialized model and scaler objects (`.joblib`) for near-instant prediction feedback.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data Visualization**: [Plotly](https://plotly.com/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/)
- **Data Handling**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Model Serialization**: [Joblib](https://joblib.readthedocs.io/)

## 📂 Project Structure

```text
House_Price_prediction/
├── app.py                # Main Streamlit application
├── train_model.py        # Script to train and save the model/scaler
├── House_Price.ipynb      # Original EDA and model development notebook
├── model.joblib          # Trained Linear Regression model
├── scaler.joblib         # Fitted StandardScaler artifact
├── house_price_regression_dataset.csv  # Raw property dataset
└── README.md             # Project documentation
```

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. You will need the following libraries:
```bash
pip install streamlit pandas numpy scikit-learn plotly joblib
```

### 2. Training the Model (Optional)
If you wish to retrain the model or update the scaler:
```bash
python train_model.py
```

### 3. Launching the App
Run the Streamlit server:
```bash
streamlit run app.py
```
The app will open in your default browser at `http://localhost:8501`.


## 🧠 Model & Performance

The core prediction engine is powered by a **Linear Regression** model, selected for its exceptional performance on this dataset.

### Performance Metrics:
- **R² Score**: `0.9475 (~95%)` - Indicates a very high correlation between the features and house prices.
- **Mean Absolute Error (MAE)**: `0.0869`
- **Root Mean Squared Error (RMSE)**: `0.1069`
- **Mean Absolute Percentage Error (MAPE)**: `0.66%`

### Feature Engineering:
- **StandardScaler**: All numerical features (Square Footage, Year Built, etc.) are normalized using `StandardScaler` to ensure the model isn't biased towards larger magnitude numbers.
- **Serialization**: Both the model and scaler are persisted using `joblib` for low-latency predictions in the Streamlit interface.

## 💾 Dataset
The model was trained on a robust dataset featuring:
- `Square_Footage`: Living area in sq ft.
- `Num_Bedrooms`: Total bedroom count.
- `Num_Bathrooms`: Total bathroom count (up to 6).
- `Year_Built`: Property age.
- `Lot_Size`: Land area in acres.
- `Garage_Size`: Vehicle capacity.
- `Neighborhood_Quality`: A curated quality score (1-10).

## 📜 License
This project is for educational and demonstration purposes.

---

