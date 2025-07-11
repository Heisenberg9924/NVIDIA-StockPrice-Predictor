# NVIDIA-StockPrice-Predictor

📂 Project Pipeline
1. 📥 Data Collection
Fetched historical stock data for NVIDIA using the yfinance library.

2. 🧹 Data Preprocessing
Cleaned and prepared the data using pandas.

Created additional features (like moving averages, previous-day returns, etc.) to boost predictive power.

3. 📊 Exploratory Data Analysis (EDA)
Visualized trends and patterns using matplotlib and seaborn.

Explored correlations among features and target variables.

4. 🤖 Model Training
Used Support Vector Regressor (SVR) from scikit-learn.

Tuned hyperparameters and evaluated model performance.

Achieved a low RMSE, and scatter plots confirmed a strong fit.

5. ⚠️ Classification Attempt
Tried using Support Vector Classifier (SVC) to predict price direction (UP/DOWN), but it didn’t yield satisfactory results.

6. 🌐 Web App Deployment
Built an interactive Streamlit app to let users visualize trends and make real-time predictions based on the trained model.

📉 Results
Achieved promising results with SVR for regression.

The scatter plot between predicted vs actual prices shows tight clustering, confirming model reliability.

SVC did not perform well, likely due to noise and complexity in directional movement.

