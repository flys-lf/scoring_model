# import pickle
# import pandas as pd
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# # Loading the machine learning model from a pickle file
# model = pickle.load(open("model.pkl", "rb"))

# @app.get('/')
# def index():
#     return {'message': 'ping'}

# # Define a route for making predictions
# @app.route("/predict", methods=["POST"])
# def predict():
#     # Get JSON data from the request
#     json_ = request.json

#     # Convert JSON data into a DataFrame
#     df = pd.DataFrame(json_)

#     # Use the loaded model to make predictions on the DataFrame
#     prediction = model.predict(df)

#     # Return the predictions as a JSON response
#     return jsonify({"Prediction": list(prediction)})

# # Run the Flask app when this script is executed
# if __name__ == "__main__":
#     app.run(debug=True)


#--------------------------------------------------------
# FAST API
import pickle
import uvicorn
from fastapi import FastAPI, Request
import pandas as pd

model = pickle.load(open("model.pkl", "rb"))

app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Cars Recommender ML API'}

@app.post('/predict')
async def predict(request: Request):
    result = await request.json()
    df = pd.DataFrame.from_dict(result)

    # Use the loaded model to make predictions on the DataFrame
    prediction = model.predict(df)
    print(prediction)
    # Return the predictions as a JSON response
    return {"prediction": list(prediction)}
