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
# Vous exposerez votre modèle de prédiction sous forme d’une API qui permet de calculer la probabilité de défaut du client, 
# ainsi que sa classe (accepté ou refusé) en fonction du seuil optimisé d’un point de vue métier.

# FAST API
import pickle
import uvicorn
from fastapi import FastAPI, Request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from preprocessing import preprocessing

model = pickle.load(open("model.pkl", "rb"))
# preprocessing =  pickle.load(open("preprocessing.pkl", "rb"))

app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Credit Scoring API'}

@app.post('/predict')
async def predict(request: Request):
    result = await request.json()
    df = pd.DataFrame.from_dict(result)
    
    # Use the loaded model to make predictions on the DataFrame
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    print(prediction)
    print(probability)

    prediction_df = pd.DataFrame(prediction, columns=['y_pred'])
    probability_df = pd.DataFrame(probability, columns=['proba_classe_0', 'proba_classe_1'])

    # Return the predictions as a JSON response
    return {"prediction": prediction_df, "probability": probability_df}