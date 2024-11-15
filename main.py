
import numpy as np
import statsmodels.api as sm

from statsmodels.tools import add_constant

from preprocessing import preprocessing

# import data preprocessed
df = preprocessing()

df_train = df[df['TARGET'].notnull()]
df_train["TARGET"] = df_train["TARGET"].astype('category')
# on définit x et y
y = df_train["TARGET"].cat.codes
# on ne prend que les colonnes quantitatives
x = df_train.select_dtypes(np.number)
# on ajoute une colonne pour la constante
x_stat = add_constant(x)

# on ajuste le modèle
model = sm.Logit(y, x_stat)
result = model.fit()


df_train.select_dtypes('object')
x.isna().sum()

print(x.isnull().any() == False)

x.loc[:, ~(x.isna().any())]
