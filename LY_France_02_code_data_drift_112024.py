import pandas as pd
import numpy as np
np.float_ = np.float64

import evidently
import pickle
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable
from evidently.metrics import DatasetDriftMetric

# Import data
df_initial = pd.read_csv('data/df_train_clean.csv')
df_new = pd.read_csv('data/df_test_clean.csv')

# Import top 100 features
top_100_features = pickle.load(open("top_100_features.pkl", "rb"))

# Numerical features
# feats = [f for f in df_initial.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
# features = df_initial[feats]

features = df_initial[top_100_features]
numerical_features = features.select_dtypes(include=np.number).columns
column_mapping = ColumnMapping()
column_mapping.numerical_features = numerical_features

# Run data drift report
data_drift_dataset_report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ]
)
data_drift_dataset_report.run(reference_data=df_initial[numerical_features], current_data=df_new[numerical_features])
data_drift_dataset_report

# Save report to html file
# data_drift_dataset_report.save_html("data_drift.html")
data_drift_dataset_report.save_html("LY_France_03_Tableau_HTML_data_drift_evidently_122024.html")