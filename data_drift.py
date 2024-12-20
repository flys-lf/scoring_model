import pandas as pd
import numpy as np
np.float_ = np.float64

import evidently
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable
from evidently.metrics import DatasetDriftMetric

# Import data
df_initial = pd.read_csv('input/application_train.csv')
df_new = pd.read_csv('input/application_test.csv')

# Numerical features
feats = [f for f in df_initial.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
features = df_initial[feats]
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
data_drift_dataset_report.save_html("data_drift.html")