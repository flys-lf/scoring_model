import time

import pandas as pd
import streamlit as st
import requests

from preprocessing import preprocessing
from sklearn.preprocessing import MinMaxScaler

API_URL = "http://127.0.0.1:8000/predict"

def request_prediction(url, data):
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
    return prediction_df, proba_df

@st.cache_resource(max_entries=1, ttl=3600 * 24)
def read_uploaded_file_as_df():
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, nrows= 10)
        with st.spinner('Preprocessing In Progress...'):
            time.sleep(5)
            # Preprocessing input data
            df_processed = preprocessing(df, num_rows=10, debug = True)
            feats = [f for f in df_processed.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

            # Scaling data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_processed[feats])
            df_scaled = pd.DataFrame(scaled_data, columns=list(df_processed[feats]))
            df_scaled_with_id = pd.concat([df_scaled, df_processed['SK_ID_CURR']], axis=1)

        st.success("Preprocessing Done!")
    else :
        df, df_processed, df_scaled_with_id = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    return df, df_processed, df_scaled_with_id
#-------------------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title='Scoring Client',
    page_icon = "🎖️",
    initial_sidebar_state="expanded",
    layout="wide"
)

st.title("Scoring Client 🎖️")

# Préparation des données
with st.expander("Lecture des données Application", expanded=False, icon=":material/database:"):
    uploaded_file = st.file_uploader("Choisissez un fichier", type={"csv"})

df, df_processed, df_scaled_with_id = read_uploaded_file_as_df()
with st.expander("Aperçu Données"):
    st.dataframe(df)
    st.dataframe(df_scaled_with_id)

#-------------------------------------------------------------------------------------------------------------------------------------------
st.title('Credit Scoring Prediction')
# Selection d'un ID Client
liste_clients = list(df_processed['SK_ID_CURR'])
col1, col2 = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
with col1:
    ID_client = st.selectbox("Veuillez sélectionner le numéro de votre client à l'aide du menu déroulant ⬇️", 
                            (liste_clients))
    st.write("Vous avez sélectionné l'identifiant n° :", ID_client)
with col2:
    st.write("")

with st.expander("Fiche Client"):
    infos_client = df.loc[df['SK_ID_CURR']==ID_client]
    st.dataframe(df_processed)

feats = [f for f in df_processed.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
df_scaled_filtered = df_scaled_with_id.loc[df_scaled_with_id['SK_ID_CURR']==ID_client, feats]
st.dataframe(df_scaled_filtered)

# Prédiction
predict_btn = st.button('Prédire')
if predict_btn:
    pred = None
    prediction_df, proba_df = request_prediction(API_URL, data = df_scaled_filtered)
    st.dataframe(prediction_df)
    st.dataframe(proba_df)
    proba = round(proba_df["proba_classe_1"]*100, 2)
    st.write(f"Le client N°{ID_client} a une probabilité de défaut de paiement de : {proba}%")
    # st.write(
    #     'Le prix médian d\'une habitation est de {:.2f}'.format(pred))