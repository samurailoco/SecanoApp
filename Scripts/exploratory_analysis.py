import json
import time
import warnings
from datetime import datetime, timedelta
from pprint import pprint
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
pd.set_option('display.max_columns', None)
import numpy as np
import pandas as pd
import requests as req
import swagger_client
from swagger_client.rest import ApiException
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, StackingClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import (
    RFE, RFECV, SelectFromModel, SelectKBest, f_classif
)
from sklearn.preprocessing import (
    LabelEncoder, MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    mean_squared_error, r2_score
)
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import NearMiss

df_total['fecha'] = pd.to_datetime(df_total['fecha'], format='%Y-%m-%d') 
df_total['semana'] = df_total['fecha'].dt.isocalendar().week
df_total['mes'] = df_total['fecha'].dt.month
df_total['anio'] = df_total['fecha'].astype(str).str[:4]
columnas_a_modificar = ['indicativo', 'tmed', 'prec', 'tmin', 'tmax', 'velmedia', 'sol', 'presMax', 'presMin', 'hrMedia', 'dir', 'racha', 'hrMax', 'hrMin', 'semana', 'anio']

for col in columnas_a_modificar:
    df_total[col] = df_total[col].astype(str)
for col in columnas_a_modificar:
    df_total[col] = df_total[col].str.replace(',', '.', regex=False)


columns_convert = [
    'altitud', 'tmed', 'prec', 'tmin', 'tmax', 'velmedia', 'sol', 
    'presMax', 'presMin', 'hrMedia', 'dir', 'racha', 'hrMax', 'hrMin', 'semana', 'anio']

for col in columns_convert:
    df_total[col] = pd.to_numeric(df_total[col], errors='coerce')

df_total['semana_anio'] = df_total['fecha'].dt.strftime('%Y-%W')

df_total[['tmed','prec','tmin','tmax','velmedia']].boxplot()

q1=df_total['tmed'].quantile(0.25)
q3=df_total['tmed'].quantile(0.75)
iqr =q3 - q1
upper=q3 + 1.5*iqr
lower=q1-1.5*iqr
sns.histplot(df_total[(df_total.tmed>lower) & (df_total.tmed<upper)]['tmed'])
plt.show()

df_total.tmed.fillna(df_total[(df_total.tmed>lower) & (df_total.tmed<upper)]['tmed'].median(), inplace=True) 
q1=df_total['tmin'].quantile(0.25)
q3=df_total['tmin'].quantile(0.75)
iqr=q3-q1
upper=q3 + 1.5*iqr
lower=q1-1.5*iqr

df_total[(df_total.tmin>lower) & (df_total.tmin<upper)]['tmed']
sns.histplot(df_total[(df_total.tmin>lower) & (df_total.tmin<upper)]['tmin'])
plt.show()

df_total.tmin.fillna(df_total[(df_total.tmin>lower) & (df_total.tmin<upper)]['tmin'].mean(), inplace=True) 
q1=df_total['tmax'].quantile(0.25)
q3=df_total['tmax'].quantile(0.75)
iqr=q3-q1
upper=q3 + 1.5*iqr
lower=q1-1.5*iqr

sns.histplot(df_total[(df_total.tmax>lower) & (df_total.tmax<upper)]['tmax'])
plt.show()

df_total.tmax.fillna(df_total[(df_total.tmax>lower) & (df_total.tmax<upper)]['tmax'].median(), inplace=True)
q1=df_total['prec'].quantile(0.25)
q3=df_total['prec'].quantile(0.75)
iqr=q3-q1
upper=q3 + 3*iqr
lower=q1-3*iqr

sns.histplot(df_total[(df_total.prec>lower) & (df_total.prec<upper)]['prec'], kde=True)
plt.show()

df_total.prec.fillna(df_total[(df_total.prec>lower) & (df_total.prec<upper)]['prec'].median(), inplace=True)  
q1=df_total['velmedia'].quantile(0.25)
q3=df_total['velmedia'].quantile(0.75)
iqr=q3-q1
upper=q3 + 1.5*iqr
lower=q1-1.5*iqr

sns.histplot(df_total[(df_total.velmedia>lower) & (df_total.velmedia<upper)]['velmedia'], kde=True)
plt.show()

df_total.velmedia.fillna(df_total[(df_total.velmedia>lower) & (df_total.velmedia<upper)]['velmedia'].median(), inplace=True)
df_total[['presMax', 'presMin', 'hrMedia', 'racha', 'hrMax', 'hrMin']].boxplot()

q1=df_total['presMax'].quantile(0.25)
q3=df_total['presMax'].quantile(0.75)
iqr=q3-q1
upper=q3 + 1.5*iqr
lower=q1-1.5*iqr

sns.histplot(df_total[(df_total.presMax>lower) & (df_total.presMax<upper)]['presMax'], kde=True)
plt.show()

df_total['semana_anio'] = df_total['fecha'].dt.strftime('%Y-%W')

def ajustar_presion_por_altitud(presion_referencia, altitud_referencia, altitud_actual):
    if pd.isna(presion_referencia) or pd.isna(altitud_referencia):
        return np.nan
    return presion_referencia * ((1 - (altitud_actual / 44330)) / (1 - (altitud_referencia / 44330))) ** 5.255

presion_promedio_estaciones = df_total.groupby(['semana_anio', 'indicativo'])[['presMax', 'presMin', 'altitud']].mean()

for i, row in df_total[df_total['presMax'].isna()].iterrows():
    estacion = row['indicativo']
    semana1 = row['semana_anio']
    altitud_actual = row['altitud']

    if (semana1, estacion) in presion_promedio_estaciones.index:
        datos_referencia = presion_promedio_estaciones.loc[(semana1, estacion)]
        presion_referencia = datos_referencia['presMax']
        altitud_referencia = datos_referencia['altitud']

        df_total.at[i, 'presMax'] = ajustar_presion_por_altitud(presion_referencia, altitud_referencia, altitud_actual)

for i, row in df_total[df_total['presMin'].isna()].iterrows():
    estacion = row['indicativo']
    semana1 = row['semana_anio']
    altitud_actual = row['altitud']

    if (semana1, estacion) in presion_promedio_estaciones.index:
        datos_referencia = presion_promedio_estaciones.loc[(semana1, estacion)]
        presion_referencia = datos_referencia['presMin']
        altitud_referencia = datos_referencia['altitud']

        df_total.at[i, 'presMin'] = ajustar_presion_por_altitud(presion_referencia, altitud_referencia, altitud_actual)

df_total['presMax'] = df_total['presMax'].interpolate(method='linear', limit_direction='forward', axis=0)
df_total['presMin'] = df_total['presMin'].interpolate(method='linear', limit_direction='forward', axis=0)

df_total['racha'] = df_total['racha'].interpolate(method='linear', limit_direction='forward', axis=0)
df_total['racha']=df_total['racha'].fillna(3.60)rm('mean')

prom_mes__hrMin = df_total.groupby(['indicativo', 'mes'])['hrMin'].transform('mean')

df_total['hrMax'].fillna(prom_mes__hrMax, inplace=True)
df_total['hrMin'].fillna(prom_mes__hrMin, inplace=True)
df_total['hrMax'] = df_total['hrMax'].interpolate(method='linear', limit_direction='forward', axis=0)
df_total['hrMin'] = df_total['hrMin'].interpolate(method='linear', limit_direction='forward', axis=0)

prec_anual = df_total.groupby(['indicativo', 'anio'])['prec'].sum().reset_index()
prec_anual.rename(columns={'prec': 'prec_anual'}, inplace=True)

df_total = df_total.merge(prec_anual, on=['indicativo', 'anio'], how='left')
df_total['hrMedia'] = df_total['hrMedia'].fillna((df_total['hrMin'] + df_total['hrMax']) / 2)
df_total[['tmed','tmax', 'tmin', 'prec', 'mes']].boxplot()
q1=df_total['prec'].quantile(0.25)
q3=df_total['prec'].quantile(0.75)
iqr =q3 - q1
upper=q3 +3*iqr
lower=q1-3*iqr
robust = RobustScaler()
df_total['prec_robust'] = robust.fit_transform(df_total[['prec']])
df_total['prec_log'] = np.log1p(df_total.prec)
sns.histplot(df_total.prec_log, bins=50)
plt.show()
df_total[['prec_robust' , 'prec', 'prec_log']].boxplot()
df_total['tmin_log'] = np.log1p(df_total.tmin)
robust = RobustScaler()
df_total['tmin_robust'] = robust.fit_transform(df_total[['tmin']])
df_total[['tmin', 'tmin_log','tmin_robust']].boxplot()
df_total['tmax_log'] = np.log1p(df_total.tmax)

robust = RobustScaler()
df_total['tmax_robust'] = robust.fit_transform(df_total[['tmax']])

df_total[['tmax', 'tmax_log','tmax_robust']].boxplot()
df_total['tmed_log'] = np.log1p(df_total.tmed)

robust = RobustScaler()
df_total['tmed_robust'] = robust.fit_transform(df_total[['tmed']])
df_total[['tmed', 'tmed_log','tmed_robust']].boxplot()
df_total[[ 'presMax', 'presMin', 'prec_anual' ]].boxplot()
df_total['prec_anual_log'] = np.log1p(df_total.prec_anual)

robust = RobustScaler()
df_total['prec_anual_robust'] = robust.fit_transform(df_total[['prec_anual']])
df_total[['prec_anual_log', 'prec_anual_robust' ]].boxplot()

from sklearn.preprocessing import PowerTransformer
pt_yeojohnson = PowerTransformer(method='yeo-johnson')
df_total['prec_anual_yeo'] = pt_yeojohnson.fit_transform(df_total[['prec_anual_robust']])
df_total[['prec_anual_log', 'prec_anual_robust','prec_anual_yeo' ]].boxplot()

sns.histplot(df_total['prec_anual_log'], kde=True)
sns.histplot(df_total['prec_anual_robust'], kde=True)
sns.histplot(df_total['prec_anual_yeo'], kde=True)
plt.show()

df_total[['velmedia', 'hrMin', 'hrMedia','hrMax', 'racha' ]].boxplot()
df_total['hrMax_log'] = np.log1p(df_total.hrMax)
robust = RobustScaler()
df_total['hrMax_robust'] = robust.fit_transform(df_total[['hrMax']])
pt_boxcox = PowerTransformer(method='box-cox')
df_total['hrMax_box'] = pt_boxcox.fit_transform(df_total[['hrMax_log']])
df_total[['hrMax_robust', 'hrMax_log', 'hrMax_box' ]].boxplot()

df_total['hrMin_log'] = np.log1p(df_total.hrMin)

robust = RobustScaler()
df_total['hrMin_robust'] = robust.fit_transform(df_total[['hrMin']])
pt_yeojohnson = PowerTransformer(method='yeo-johnson')
df_total['hrMin_yeo'] = pt_yeojohnson.fit_transform(df_total[['hrMin_robust']])

pt_boxcox = PowerTransformer(method='box-cox')
df_total['hrMin_box'] = pt_boxcox.fit_transform(df_total[['hrMin_log']])

df_total[['hrMin_robust', 'hrMin_log', 'hrMin_box', 'hrMin_yeo', 'hrMin' ]].boxplot()
df_total['velmedia_log'] = np.log1p(df_total.velmedia)

robust = RobustScaler()
df_total['velmedia_robust'] = robust.fit_transform(df_total[['velmedia']])

pt_yeojohnson = PowerTransformer(method='yeo-johnson')
df_total['velmedia_yeo'] = pt_yeojohnson.fit_transform(df_total[['velmedia']])
df_total[['velmedia_robust', 'velmedia_log', 'velmedia_yeo' ]].boxplot()
pt_yeojohnson = PowerTransformer(method='yeo-johnson')
df_total['velmedia_yeo2'] = pt_yeojohnson.fit_transform(df_total[['velmedia_log']])
df_total[[ 'velmedia_log', 'velmedia_yeo2' ]].boxplot()
from sklearn.preprocessing import PowerTransformer

df_total['racha_log'] = np.log1p(df_total.racha)

robust = RobustScaler()
df_total['racha_robust'] = robust.fit_transform(df_total[['racha']])

pt_yeojohnson = PowerTransformer(method='yeo-johnson')
df_total['racha_yeo'] = pt_yeojohnson.fit_transform(df_total[['racha']])
df_total[['racha_robust', 'racha_log', 'racha_yeo' ]].boxplot()

pt_yeojohnson = PowerTransformer(method='yeo-johnson')
df_total['racha_yeo2'] = pt_yeojohnson.fit_transform(df_total[['racha_log']])
df_total[[ 'racha_log', 'racha_yeo2' ]].boxplot()

le = LabelEncoder()
df_total['nombre_encoded'] = le.fit_transform(df_total['nombre'])
df_total['altitud_cat']=df_total['altitud'].astype('category')
altitudcat = pd.get_dummies(df_total.altitud_cat, drop_first=False)
df_total = pd.concat([df_total, altitudcat], axis=1)
df_total = df_total.drop(columns=['horatmin', 'horatmax','horaPresMax',  'horaPresMin', 'horaracha','horaHrMax', 'horaHrMin', 'dir'])
df_total['tmax_log']=df_total['tmax_log'].fillna(df_total['tmax_log'].median())
df_total['tmin_log']=df_total['tmin_log'].fillna(df_total['tmin_log'].mean())
df_total['tmed_log']=df_total['tmed_log'].fillna(df_total['tmed_log'].median())

df_total.info()