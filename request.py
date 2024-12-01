import requests
import pandas as pd

# Define the input data as a dictionary
df = pd.read_csv('data/df_final.csv')
df = df.head(2)
df = df[df['TARGET'].notnull()]
feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
data = df[feats]

# Send a POST request to the API with the input data
url = "http://127.0.0.1:8000/predict"

payload = data.to_json()
response = requests.post(url, data= payload)

# Check the HTTP response status code
if response.status_code == 200:
    # Parse and print the JSON response (assuming it contains the prediction)
    result = response.json()
    # print(prediction)
    prediction_df = pd.DataFrame.from_dict(result["prediction"])
    proba_df = pd.DataFrame.from_dict(result["probability"])
    print(prediction_df)
    print(proba_df)
else:
    # Handle the case where the API request failed
    print(f'API Request Failed with Status Code: {response.status_code}')
    print(f'Response Content: {response.text}')