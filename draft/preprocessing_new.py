import warnings
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import gc


PATH_FILE = 'input'

# Function to calculate missing values by column# Funct
def drop_cols_with_missing_values(df, seuil):
    # Total missing values
    mis_val = df.isnull().sum()
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    cols_with_missing_values = [col for col in df.columns if df[col].isnull().mean() > seuil]

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
        "There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.\n"
        "There are " + str(len(cols_with_missing_values)) + f" columns that have more than {seuil*100}% of missing values.\n")


    df = df.drop(labels=cols_with_missing_values, axis=1, inplace=False)

    # Return the dataframe with missing information
    return df, cols_with_missing_values

def split_df_numerical_categorical_columns(df):
    # Split numerical and categorical columns
    numeric_columns = df.select_dtypes(include='number').columns
    categorical_columns = df.select_dtypes(include='object').columns

    data_numeric = df[numeric_columns]
    data_categorical = df[categorical_columns]

    return data_numeric, data_categorical

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    df.columns = df.columns.str.replace(' ', '_')
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def application_train():
    df = pd.read_csv(f'{PATH_FILE}/application_train.csv')

    # NaN values for DAYS_EMPLOYED: 365243 -> nan
    df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = np.nan

    # Delete cols with more than 40% missing values
    df_clean, cols_with_50pct_missing_values = drop_cols_with_missing_values(df, seuil = 0.5)

    # Suppression de la variable cible dans le X-train
    y_train = df_clean['TARGET']
    X_train = df_clean.drop(labels='TARGET', axis=1, inplace=False)
    # X_train.shape, y_train.shape

    # Split numerical and categorical columns
    data_numeric, data_categorical = split_df_numerical_categorical_columns(X_train)

    # Impute missing values for numerical columns with median values
    si = SimpleImputer(missing_values=np.nan, strategy='median')
    data_numeric = pd.DataFrame(si.fit_transform(data_numeric), columns = si.get_feature_names_out())
    # Impute missing values for categorical columns with most frequent values
    si = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    data_categorical = pd.DataFrame(si.fit_transform(data_categorical), columns = si.get_feature_names_out())
    # Join the two masked dataframes back together
    X_train_joined = pd.concat([data_numeric, data_categorical], axis = 1)

    # Categorical features processing
    # Categorical features with Binary encode (0 or 1; two categories)
    # CODE_GENDER : 4 valeurs manquantes pour le genre, on remplace la valeur XNA par F car c'est la valeur la plus fréquente
    X_train_joined['CODE_GENDER'].value_counts()
    X_train_joined['CODE_GENDER'] = X_train_joined['CODE_GENDER'].replace(['XNA'],'F')
    # Categorical features with Binary encode (0 or 1; two categories)
    X_train_joined['CODE_GENDER'], uniques = pd.factorize(X_train_joined['CODE_GENDER'])
    X_train_joined['CODE_GENDER'].value_counts()

    for col_binaire in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        X_train_joined[col_binaire] = X_train_joined[col_binaire].map({'Y': 1, 'N': 0})

    # Categorical features - Reduce dimensionality
    # NAME_TYPE_SUITE : Client accompagné de qui ?
    X_train_joined['NAME_TYPE_SUITE'].value_counts(normalize=True)
    mapping_name_type_suite = {
        'Unaccompanied' : 'Unaccompanied',
        'Family' : 'Family',
        'Spouse, partner' : 'Family',
        'Children' : 'Family',
        'Other_A' : 'Other',
        'Other_B' : 'Other',
        'Group of people' : 'Other'
    }
    X_train_joined['NAME_TYPE_SUITE'] = X_train_joined['NAME_TYPE_SUITE'].map(mapping_name_type_suite)

    # NAME_INCOME_TYPE : Type de revenu du client
    X_train_joined['NAME_INCOME_TYPE'].value_counts(normalize=True)
    mapping_name_income_type = {
        'Working' : 'Working',
        'Commercial associate' : 'Commercial associate',
        'Pensioner' : 'Pensioner or Unemployed',
        'State servant' : 'State servant',
        'Unemployed' : 'Pensioner or Unemployed',
        'Student' : 'Pensioner or Unemployed',
        'Businessman' : 'Other',
        'Maternity leave' : 'Other'
    }
    X_train_joined['NAME_INCOME_TYPE'] = X_train_joined['NAME_INCOME_TYPE'].map(mapping_name_income_type)

    # NAME_EDUCATION_TYPE : Type d'éducation le plus élevé
    X_train_joined['NAME_EDUCATION_TYPE'].value_counts(normalize=True)
    mapping_name_education_type = {
        'Lower secondary' : 'Secondary or Lower',
        'Secondary / secondary special' : 'Secondary or Lower',
        'Higher education' : 'Higher education',
        'Academic degree' : 'Higher education',
        'Incomplete higher' : 'Higher education'
    }

    X_train_joined['NAME_EDUCATION_TYPE'] = X_train_joined['NAME_EDUCATION_TYPE'].map(mapping_name_education_type)

    # NAME_HOUSING_TYPE : Type de logement
    X_train_joined['NAME_HOUSING_TYPE'].value_counts(normalize=True)
    X_train_joined['NAME_HOUSING_TYPE'] = np.where(X_train_joined['NAME_HOUSING_TYPE']!= 'House / apartment', 'Other', 'House / apartment')

    # Drop columns with XNA
    X_train_joined.drop(labels=['ORGANIZATION_TYPE', 'EMERGENCYSTATE_MODE'], axis=1, inplace=True)

    # Numerical features processing
    # Suppression des variables 'FLAG_MOBIL' et 'FLAG_CONT_MOBILE', très peu de lignes à 0
    X_train_joined.drop(labels=['FLAG_MOBIL', 'FLAG_CONT_MOBILE'], axis=1, inplace=True)

    # Discrétisation de certaines variables numériques
    # CNT CHILDREN
    X_train_joined['CNT_CHILDREN'].value_counts(normalize=True)
    X_train_joined['CNT_CHILDREN'] = np.where(
        X_train_joined['CNT_CHILDREN'] == 0, '0',
        np.where(X_train_joined['CNT_CHILDREN'] == 1, '1',
            np.where(X_train_joined['CNT_CHILDREN'] >= 2, '2 or more', X_train_joined['CNT_CHILDREN'])
        )
    )

    # Les variables flag_document : documents fournis par les clients
    # On garde uniquement les variables FLAG_DOCUMENT_3, FLAG_DOCUMENT_6 et FLAG_DOCUMENT_8
    # Suppression des autres variables
    cols_flag_to_drop = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',  
            'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 
            'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 
            'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

    X_train_joined.drop(labels=cols_flag_to_drop, axis=1, inplace=True)
    list_var = ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE'] # Variables presques toutes nulles
    X_train_joined.drop(labels=list_var, axis=1, inplace=True)


    # Ajout de nouvelles features (percentages)
    X_train_joined['DAYS_EMPLOYED_PERC'] = X_train_joined['DAYS_EMPLOYED'] / X_train_joined['DAYS_BIRTH'] # taux de jours employés par rapport à l'âge
    X_train_joined['INCOME_CREDIT_PERC'] = X_train_joined['AMT_INCOME_TOTAL'] / X_train_joined['AMT_CREDIT'] # 
    X_train_joined['INCOME_PER_PERSON'] = X_train_joined['AMT_INCOME_TOTAL'] / X_train_joined['CNT_FAM_MEMBERS'] # revenu par membre du foyer
    X_train_joined['ANNUITY_INCOME_PERC'] = X_train_joined['AMT_ANNUITY'] / X_train_joined['AMT_INCOME_TOTAL'] # part de l'annuité par rapport au salaire total du client
    X_train_joined['PAYMENT_RATE'] = X_train_joined['AMT_ANNUITY'] / X_train_joined['AMT_CREDIT'] # taux de paiement(somme remboursée) par rapport à la somme finale du crédit par année (previous application)

    # Drop colonnes, une parmi les paires de colonnes très corrélées
    cols_to_drop = ['AMT_GOODS_PRICE', 'FLOORSMAX_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'FLOORSMAX_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BEGINEXPLUATATION_MEDI']
    X_train_joined.drop(labels=cols_to_drop, axis=1, inplace=True)

    
    # Categorical features with One-Hot encode
    # X_train_joined, cat_cols = one_hot_encoder(df, nan_as_category= False)

    # Réintroduction TARGET
    X_train_joined['TARGET'] = y_train

    gc.collect()
    return X_train_joined

def previous_applications(df, num_rows = None, nan_as_category = True):
    df_prev = df.copy()
    # Process DAYS columns, Days 365.243 values -> nan
    col_days = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    for col in col_days:
        df_prev.loc[df_prev[col] == 365243, col] = np.nan

    # Delete cols with more than 40% missing values
    df_clean, cols_with_40pct_missing_values = drop_cols_with_missing_values(df_prev, seuil = 0.4)
    
    # Imputation missing values
    # Split numerical and categorical columns
    numeric_columns = df_clean.select_dtypes(include='number').columns
    categorical_columns = df_clean.select_dtypes(include='object').columns

    # Create two DataFrames, one for each data type
    data_numeric = df_clean[numeric_columns]
    data_categorical = df_clean[categorical_columns]

    # Impute missing values for numerical columns with median values
    si = SimpleImputer(missing_values=np.nan, strategy='median')
    data_numeric = pd.DataFrame(si.fit_transform(data_numeric), columns = si.get_feature_names_out())
    # Impute missing values for categorical columns with most frequent values
    si = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    data_categorical = pd.DataFrame(si.fit_transform(data_categorical), columns = si.get_feature_names_out())
    # join the two masked dataframes back together
    df_joined = pd.concat([data_numeric, data_categorical], axis = 1)

    # Variables numériques
    df_joined['DAYS_DECISION'] = round(df_prev['DAYS_DECISION'] / -365, 0)
    df_joined.drop(labels=['SELLERPLACE_AREA', 'NFLAG_LAST_APPL_IN_DAY'], axis=1, inplace=True)

    # Drop colonnes, une parmi les paires de colonnes très corrélées
    cols_to_drop = ['AMT_GOODS_PRICE']
    df_joined.drop(labels=cols_to_drop, axis=1, inplace=True)

    # Variables Catégorielles
    # On conserve les colonnes NAME_CONTRACT_STATUS et NAME_CONTRACT_TYPE, les autres contiennent pas un grand nombre de valeurs 'XNA' ou peu distribution peu discriminante (apporte peu d'informations)
    num_columns = df_joined.select_dtypes(include=[np.number]).columns.to_list()
    cols_cat_keep = ['NAME_CONTRACT_STATUS', 'NAME_CONTRACT_TYPE']
    cols_keep = num_columns + cols_cat_keep
    df_joined = df_joined[cols_keep]

    # Agrégations
    prev = df_prev.copy()
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= False)

    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['mean'],
        'AMT_APPLICATION': ['mean'],
        'AMT_CREDIT': ['mean'],
        'APP_CREDIT_PERC': ['mean', 'var'],
        'HOUR_APPR_PROCESS_START': ['mean'],
        'DAYS_DECISION': ['mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')

    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv(f'{PATH_FILE}/bureau.csv', nrows = num_rows)
    bb = pd.read_csv(f'{PATH_FILE}/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


















