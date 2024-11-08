import mlflow
import lightgbm as lgb
import tensorflow as tf  # Import TensorFlow library

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # Import Joblib library for object serialization

# Set MLFlow environement
mlflow.set_tracking_uri(uri="http://localhost:8080")
mlflow.set_experiment("iris_experiment")  # Set the experiment name for MLflow

mlflow.autolog()

# load iris data
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
