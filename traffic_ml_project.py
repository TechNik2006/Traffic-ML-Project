import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from flask import Flask, request, jsonify
import os

# Ensure data and model directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Step 1: Load or Generate Dataset
data_path = "data/traffic_data.csv"
if not os.path.exists(data_path):
    print("Dataset not found. Generating sample dataset...")
    np.random.seed(42)
    num_samples = 1000
    data = {
        "hour": np.random.randint(0, 24, num_samples),
        "day_of_week": np.random.randint(0, 7, num_samples),
        "temperature": np.random.uniform(10,
35, num_samples),
        "weather_condition": np.random.choice(["Clear", "Cloudy", "Rainy"], num_samples),
        "road_type": np.random.choice(["Highway", "City Road", "Local Street"], num_samples),
        "traffic_volume": np.random.randint(50, 500, num_samples)
    }
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False)
    print("Sample dataset created at", data_path)
else:
    df = pd.read_csv(data_path)

# Step 2: Data Preprocessing
df.dropna(inplace=True)
df = pd.get_dummies(df, drop_first=True)

# Splitting Features and Target
X = df.drop(columns=['traffic_volume'])
y = df['traffic_volume']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train ML Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R2 Score: {r2}')

# Step 5: Visualization
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel('Actual Traffic Volume')
plt.ylabel('Predicted Traffic Volume')
plt.title('Actual vs Predicted Traffic Volume')
plt.show()

# Save the trained model
model_path = 'models/traffic_volume_model.pkl'
joblib.dump(model, model_path)
print(f'Model saved at {model_path}')

# Step 6: Deployment with Flask
app = Flask(__name__)

# Load model at startup
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        return jsonify({'predicted_traffic_volume': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Flask app is running but cannot be executed in this environment.")
