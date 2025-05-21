OPEN A DATASET:
#import
import pandas as pd
Data=pd.read_csv("/content/Air Quality Dataset.csv")
print(Data)

FILL MISSING VALUES:
#fillna
Data["PT08.S2(NMHC)"] = pd.to_numeric(Data["PT08.S2(NMHC)"], errors='coerce')
Data["NOx(GT)"] = pd.to_numeric(Data["NOx(GT)"], errors='coerce')
Data["NO2(GT)"] = pd.to_numeric(Data["NO2(GT)"], errors='coerce')
Data["PT08.S4(NO2)"] = pd.to_numeric(Data["PT08.S4(NO2)"], errors='coerce')
Data["T"] = pd.to_numeric(Data["T"], errors='coerce')
Data["AH"] = pd.to_numeric(Data["AH"], errors='coerce')
Data["PT08.S2(NMHC)"].fillna(Data["PT08.S2(NMHC)"].mean(),inplace=True)
Data["NOx(GT)"].fillna(Data["NOx(GT)"].median(),inplace=True)
Data["NO2(GT)"].fillna(Data["NO2(GT)"].mean(),inplace=True)
Data["PT08.S4(NO2)"].fillna(Data["PT08.S4(NO2)"].mean(),inplace=True)
Data["T"].fillna(Data["T"].mean(),inplace=True)
Data["AH"].fillna(Data["AH"].mean(),inplace=True)
print(Data)

DATA VISUALIZATION:
#Heatmap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pivot_table = Data.pivot(index='Date', columns='Time')
plt.figure(figsize=(12,4))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu')
plt.title('Air Quality Heatmap')
plt.xlabel('Time')
plt.ylabel('Date')
plt.show()

DATA MODELING:
#model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd 
X = Data[['T']] 
y = Data['AH']   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")



MODEL EVALUATION:
#evaluation
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "KNN Regressor": KNeighborsRegressor()
}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
     results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R² Score": round(r2, 4)
    })
comparison_df = pd.DataFrame(results)
print(comparison_df)

MODEL DEPLOYMENT:
#deployment
import joblib
joblib.dump(model, "model.pkl")
from flask import Flask, request, jsonify
import joblib
import numpy as np
app = Flask(__name__)
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    print("Error: 'model.pkl' not found. Please train and save the model first.")
    model = None 
@app.route('/')
def home():
    return "Air Quality Prediction API (Linear Regression)"
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please check the server setup."}), 500
data = request.get_json()
    try:
        features = np.array([[
            data["PM2.5"], data["PM10"], data["NO2"], data["SO2"],
            data["CO"], data["O3"], data["temperature"], data["humidity"]
        ]])
        
        prediction = model.predict(features)
        return jsonify({"AQI Prediction": round(prediction[0], 2)})
    except KeyError as e:
        return jsonify({"error": f"Missing data in request: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
      app.run(debug=True, port=5000)
 Flask
scikit-learn
from flask import Flask, request, jsonify
import joblib
import numpy as np
app = Flask(__name__)
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    print("Error: 'model.pkl' not found. Please train and save the model first.")
    model = None
@app.route('/')
def home():
    return "Air Quality Prediction API (Linear Regression)"
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please check the server setup."}), 500
    data = request.get_json()
    try:
        features = np.array([[
            data["PM2.5"], data["PM10"], data["NO2"], data["SO2"],
            data["CO"], data["O3"], data["temperature"], data["humidity"]
        ]])
        prediction = model.predict(features)
        return jsonify({"AQI Prediction": round(prediction[0], 2)})
    except KeyError as e:
        return jsonify({"error": f"Missing data in request: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True, port=5000)


