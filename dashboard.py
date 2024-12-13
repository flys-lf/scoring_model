import time
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import requests
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import MinMaxScaler
import streamlit.components.v1 as components

from preprocessing import preprocessing

# API_URL = "http://127.0.0.1:8000/predict"
API_URL = "https://apitestscoring-bre0d5dbasdsewhw.francecentral-01.azurewebsites.net/predict"

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
def read_uploaded_file_as_df(num_rows = None):
    df = pd.read_csv(uploaded_file, nrows= num_rows)

    with st.spinner('Preprocessing In Progress...'):
        time.sleep(5)
        # Preprocessing input data
        df_processed = preprocessing(df, num_rows=num_rows, debug = False)
        feats = [f for f in df_processed.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

        # Scaling data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_processed[feats])
        df_scaled = pd.DataFrame(scaled_data, columns=list(df_processed[feats]))
        df_scaled_with_id = pd.concat([df_scaled, df_processed['SK_ID_CURR']], axis=1)

    st.success("Preprocessing Done!")
    return df, df_processed, df_scaled_with_id
#-------------------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title='Scoring Client',
    page_icon = "images/logo.jpg",
    initial_sidebar_state="expanded",
    layout="wide"
)

st.image("images/banner.jpg")
st.title("Scoring Client 🎖️")

st.info("Veuillez uploader les données Application des clients pour lesquelles vous souhaitez prédire une probabilité de défaut de paiement.")
# Préparation des données
with st.expander("Lecture des données Application", expanded=False, icon=":material/database:"):
    uploaded_file = st.file_uploader("Choisissez un fichier", type={"csv"})

if uploaded_file is not None:
    df, df_processed, df_scaled_with_id = read_uploaded_file_as_df(num_rows = None)

    if not df.empty :
        with st.expander("Aperçu Données"):
            st.dataframe(df)
            st.dataframe(df_scaled_with_id)

        #-------------------------------------------------------------------------------------------------------------------------------------------
        st.divider()
        st.header('Prediction Scoring Crédit')

        # Selection d'un ID Client
        liste_clients = list(df_processed['SK_ID_CURR'])
        col1, col2, col3 = st.columns(3)
        with col1:
            id_client = st.selectbox("Veuillez sélectionner le numéro de votre client à l'aide du menu déroulant :",
                                    (liste_clients))
            st.write(f"Vous avez sélectionné l'identifiant N° : **{id_client}**")


        with st.expander("Fiche Client"):
            st.write(f"**Client N° {id_client}**")
            infos_client = df.loc[df['SK_ID_CURR']==id_client, ]
            st.write("**Type Contrat :**", infos_client['NAME_CONTRACT_TYPE'].iloc[0])
            st.write("**Sexe :**", infos_client['CODE_GENDER'].iloc[0])
            st.dataframe(infos_client)

        feats = [f for f in df_processed.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
        df_scaled_filtered = df_scaled_with_id.loc[df_scaled_with_id['SK_ID_CURR']==id_client, feats]
        st.dataframe(df_scaled_filtered)

        # Prédiction
        predict_button = st.button('Prédire')
        if predict_button:
            prediction_df, proba_df = request_prediction(API_URL, data = df_scaled_filtered)

            with st.container(border=True):
                proba = round(proba_df["proba_classe_1"][0]*100, 2)
                prediction = round(prediction_df["y_pred"][0])
                prediction_df["decision"] = np.where(prediction_df.y_pred ==1, "Refusé", "Accordé")
                st.write(f"Le client **N° {id_client}** a une probabilité de défaut de paiement estimé à : **:blue[{proba}%]**")
                decision = prediction_df["decision"][0]
                if decision == "Accordé" :
                    st.success(f"Crédit {decision}")
                elif decision == "Refusé" :
                    st.error(f"Crédit {decision}")

        # ==================================================================================================================================
        st.divider()
        st.header("SHAP VALUE")

        # matlplotlib.figure.Figure
        @st.cache_resource(hash_funcs={plt.figure: lambda _: None})
        def compute_shap_values():
            shap.initjs()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_shap)

            # fig_barplot, ax = plt.subplots(figsize=(15,5))
            # fig_barplot = shap.summary_plot(shap_values, df_shap, plot_type="bar")
            # st.pyplot()

            # fig_shap_plot, ax = plt.subplots(figsize=(15,5))
            # fig_shap_plot = shap.summary_plot(shap_values, df_shap, sort = True)
            # st.pyplot(fig)
            return explainer, shap_values
            # return explainer, shap_values, fig_barplot, fig_shap_plot
        
        def st_shap(plot, height=None):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=height)

        model = pickle.load(open("model.pkl", "rb"))
        feats = [f for f in df_processed.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
        df_shap = df_processed[feats]
        # explainer, shap_values, fig_barplot, fig_shap_plot = compute_shap_values()
        explainer, shap_values = compute_shap_values()

        # Explication Locale --------------------------------------------------------------------------------------------------------------
        st.subheader(f"Explication Locale Client N°{id_client}")
        # récupération de l'index correspondant à l'identifiant du client
        idx_selected = int(df_processed[df_processed['SK_ID_CURR']==id_client].index[0])
        # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
        st_shap(shap.force_plot(explainer.expected_value, shap_values[idx_selected,:], df_shap.iloc[idx_selected,:]))

        # # visualize the training set predictions
        # st_shap(shap.force_plot(explainer.expected_value, shap_values, df_shap), 400)


        col1, col2 = st.columns(2)
        with col1 :
            st.subheader("Explication Globale")
            fig, ax = plt.subplots(figsize=(15,5))
            shap.summary_plot(shap_values, df_shap, plot_type="bar")
            st.pyplot()

        with col2:
            st.subheader("Explication Locale")
            fig = shap.summary_plot(shap_values, df_shap, sort = True)
            st.pyplot(fig)
        

