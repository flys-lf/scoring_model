import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,ConfusionMatrixDisplay, recall_score, f1_score, roc_curve, roc_auc_score, fbeta_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from preprocessing import preprocessing
import re
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
import mlflow
from mlflow.models import infer_signature

import pickle

def get_model_metrics(model, y_test, y_pred, y_prob):
    metrics = {
        "f2_score": fbeta_score(y_test, y_pred, beta=10),
        "accuracy_score": accuracy_score(y_test, y_pred), # Utilisation de accuracy_score
        "score": model.score(X_test, y_test), # Ou utilisation de la méthode score
        "recall": recall_score(y_true=y_test, y_pred=y_pred),        
        "f1_score": f1_score(y_true=y_test, y_pred=y_pred),
        "auc": roc_auc_score(y_test, y_prob),    
    }

    print(f"La méthode accuracy_score donne: {metrics['accuracy_score']}")
    print(f"La méthode score donne: {metrics['score']}")
    print(f"Le recall est de: {metrics['recall']}")
    print(f"Le F1-score est de: {metrics['f1_score']}")
    print(f"L'AUC est de: {metrics['auc']}")
    print(f"Le F2-score est de: {metrics['f2_score']}")

    return metrics

# Import data ----------------------------------------
df = pd.read_csv('data/df_final.csv')

# Clean Data --------------------------------------
df = df[df['TARGET'].notnull()]
Y = df['TARGET']
df_train = df.drop(labels='TARGET', axis=1)
df_train.shape, Y.shape

# Split Train Data : Train, Test, Validation
feats = [f for f in df_train.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
df_train[feats].info()

X_train, X_test, y_train, y_test = train_test_split(df_train[feats], Y, stratify=Y, test_size=0.3, random_state=101)
X_train.shape, X_test.shape

# Scaler -------------------------------------------------
scaler = MinMaxScaler() # Distribution des données n'est pas normale
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LGBM Classifier ---------------------------------------
clf = LGBMClassifier(objective= 'binary', class_weight="balanced").fit(X_train_scaled, y_train)
y_pred__clf = clf.predict(X_test_scaled)
y_prob__clf = clf.predict_proba(X_test_scaled)[:1]
metrics_LGBM = get_model_metrics(clf, y_test, y_pred__clf, y_prob__clf)


# Dump Model --------------------------
# Save to file in the current working directory
pkl_filename = "model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(clf, file)


# Snippet Code for loading model.pkl ------------------------------------
# with open(pkl_filename, 'rb') as file: # Load from file
#     pickle_model = pickle.load(file)

# # Calculate the accuracy score and predict target values
# f2_score = metrics_LGBM['f2_score']
# score = pickle_model.score(X_test_scaled, y_test)
# print("score : {0:.2f} %".format(100 * score))
# print("F2 score : {0:.2f} %".format(100 * f2_score))

# Y_pred = pickle_model.predict(X_test_scaled)
# Y_proba = pickle_model.predict_proba(X_test_scaled)
# print(Y_pred)
# print(Y_proba)

# y_pred_df = pd.DataFrame(Y_pred, columns=['y_pred_test'])
# y_pred_proba_df = pd.DataFrame(Y_proba, columns=['proba_classe_0', 'proba_classe_1'])

# # Récupération du score du client
# y_pred_proba_df = pd.DataFrame(y_pred_proba_df, columns=['proba_classe_0', 'proba_classe_1'])
# y_pred_proba_df = pd.concat([y_pred_proba_df['proba_classe_1'], X_test['SK_ID_CURR']], axis=1)
