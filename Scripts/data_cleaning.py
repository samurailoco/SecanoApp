import pandas as pd
import json
import time
import warnings
from datetime import datetime, timedelta
from pprint import pprint
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
pd.set_option('display.max_columns', None)
import numpy as np
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

df_total=pd.read_csv("/mnt/c/Users/danie/OneDrive/Desktop/Proyectos/proyecto_final/Data/raw/datos_climatologicos.csv")


#creo nuevas columnas que me ayudaran a ver mejor el df y hacer comparaciones
df_total['fecha'] = pd.to_datetime(df_total['fecha'], format='%Y-%m-%d')

#columna 'semana' 
df_total['semana'] = df_total['fecha'].dt.isocalendar().week

#columna 'mes' 
df_total['mes'] = df_total['fecha'].dt.month

#columna 'anio' 
df_total['anio'] = df_total['fecha'].astype(str).str[:4]
# reemplazo comas por puntos
columnas_a_modificar = ['indicativo', 'tmed', 'prec', 'tmin', 'tmax', 'velmedia', 'sol', 'presMax', 'presMin', 'hrMedia', 'dir', 'racha', 'hrMax', 'hrMin', 'semana', 'anio']

for col in columnas_a_modificar:
    df_total[col] = df_total[col].astype(str)
for col in columnas_a_modificar:
    df_total[col] = df_total[col].str.replace(',', '.', regex=False)


columns_convert = [
    'altitud', 'tmed', 'prec', 'tmin', 'tmax', 'velmedia', 'sol', 
    'presMax', 'presMin', 'hrMedia', 'dir', 'racha', 'hrMax', 'hrMin', 'semana', 'anio']
# convertir las columnas a numericas
for col in columns_convert:
    df_total[col] = pd.to_numeric(df_total[col], errors='coerce')

df_total['semana_anio'] = df_total['fecha'].dt.strftime('%Y-%W') #creo otra columna 'semana-anio' qiuzas facilite ver cosas
#empoezamos con la tmed, tiene pocos outliers
q1=df_total['tmed'].quantile(0.25)
q3=df_total['tmed'].quantile(0.75)
iqr =q3 - q1
upper=q3 + 1.5*iqr
lower=q1-1.5*iqr
#rellenamos con la mediana porque la distribucion en el histograma no es nada normal
df_total.tmed.fillna(df_total[(df_total.tmed>lower) & (df_total.tmed<upper)]['tmed'].median(), inplace=True)
q1=df_total['tmin'].quantile(0.25)
q3=df_total['tmin'].quantile(0.75)
iqr=q3-q1
upper=q3 + 1.5*iqr
lower=q1-1.5*iqr
#despues de ver que la distribucion en tmin es bastante normal procedemos a rellenar con la media
df_total.tmin.fillna(df_total[(df_total.tmin>lower) & (df_total.tmin<upper)]['tmin'].mean(), inplace=True)  
#analisis de tmax
q1=df_total['tmax'].quantile(0.25)
q3=df_total['tmax'].quantile(0.75)
iqr=q3-q1
upper=q3 + 1.5*iqr
lower=q1-1.5*iqr
#mismo criterio con tmax, usamos mediana porque su distribucion no es normal
df_total.tmax.fillna(df_total[(df_total.tmax>lower) & (df_total.tmax<upper)]['tmax'].median(), inplace=True)
df_total.prec.fillna(df_total[(df_total.prec>lower) & (df_total.prec<upper)]['prec'].median(), inplace=True)  #aqui uso la mediana
df_total.velmedia.fillna(df_total[(df_total.velmedia>lower) & (df_total.velmedia<upper)]['velmedia'].median(), inplace=True)
#hay una formula que se usa en meteorologia para conocer la presion
# Funcion para estimar la presion basada en la altitud y otra presion de referencia
def ajustar_presion_por_altitud(presion_referencia, altitud_referencia, altitud_actual):
    if pd.isna(presion_referencia) or pd.isna(altitud_referencia):
        return np.nan
    return presion_referencia * ((1 - (altitud_actual / 44330)) / (1 - (altitud_referencia / 44330))) ** 5.255

# Calcular la presion promedio por semana y estación (donde haya datos)
presion_promedio_estaciones = df_total.groupby(['semana_anio', 'indicativo'])[['presMax', 'presMin', 'altitud']].mean()

# Rellenar presMax
for i, row in df_total[df_total['presMax'].isna()].iterrows():
    estacion = row['indicativo']
    semana1 = row['semana_anio']
    altitud_actual = row['altitud']

    # Obtener la presión promedio y altitud de la estación para esa semana
    if (semana1, estacion) in presion_promedio_estaciones.index:
        datos_referencia = presion_promedio_estaciones.loc[(semana1, estacion)]
        presion_referencia = datos_referencia['presMax']
        altitud_referencia = datos_referencia['altitud']

        # Ajustar la presión por la diferencia de altitud
        df_total.at[i, 'presMax'] = ajustar_presion_por_altitud(presion_referencia, altitud_referencia, altitud_actual)

# Rellenar presMin de manera similar
for i, row in df_total[df_total['presMin'].isna()].iterrows():
    estacion = row['indicativo']
    semana1 = row['semana_anio']
    altitud_actual = row['altitud']

    if (semana1, estacion) in presion_promedio_estaciones.index:
        datos_referencia = presion_promedio_estaciones.loc[(semana1, estacion)]
        presion_referencia = datos_referencia['presMin']
        altitud_referencia = datos_referencia['altitud']

        df_total.at[i, 'presMin'] = ajustar_presion_por_altitud(presion_referencia, altitud_referencia, altitud_actual)
#con esta funcion logro rellenar solo unos poco null (1700 aprox) asi qeu recurro a la  interpolacion lineal
#2 parte para rellenar nan de presMin y presMax, esta vez  interpolar valores faltantes de forma lineal por tiempo
df_total['presMax'] = df_total['presMax'].interpolate(method='linear', limit_direction='forward', axis=0)
df_total['presMin'] = df_total['presMin'].interpolate(method='linear', limit_direction='forward', axis=0)
# ahora pasamos a rellenar los nan de 'racha', como tenemos 77% de datos podemos empezar con
#interpolacion lineal en los registros que falte un dato   
df_total['racha'] = df_total['racha'].interpolate(method='linear', limit_direction='forward', axis=0)
df_total['racha']=df_total['racha'].fillna(3.60) #solo queda 1 solo nan y se rellena con el promedio de ese dia los otros años
#para las hr usamos el promedio de esa misma estacion, ese mismo mes
prom_mes__hrMax = df_total.groupby(['indicativo', 'mes'])['hrMax'].transform('mean')

# Media mensual por estación para hrMin
prom_mes__hrMin = df_total.groupby(['indicativo', 'mes'])['hrMin'].transform('mean')

# Rellenar nulos restantes con las medias mensuales por estación
df_total['hrMax'].fillna(prom_mes__hrMax, inplace=True)
df_total['hrMin'].fillna(prom_mes__hrMin, inplace=True)
df_total['hrMax'] = df_total['hrMax'].interpolate(method='linear', limit_direction='forward', axis=0)
df_total['hrMin'] = df_total['hrMin'].interpolate(method='linear', limit_direction='forward', axis=0)
# ya con todos los nan rellenos, creamos una nueva columna donde indique la precipitacion
#acumulada durante todo el año para usarla luego en las condciones del cultivo 
#calcular la suma anual de precipitacion por estacion
prec_anual = df_total.groupby(['indicativo', 'anio'])['prec'].sum().reset_index()
prec_anual.rename(columns={'prec': 'prec_anual'}, inplace=True)

# unir la columna de precipitacion anual al DataFrame original
df_total = df_total.merge(prec_anual, on=['indicativo', 'anio'], how='left')
robust = RobustScaler()
pt_yeojohnson = PowerTransformer(method='yeo-johnson')
pt_boxcox = PowerTransformer(method='box-cox')
df_total['hrMedia'] = df_total['hrMedia'].fillna((df_total['hrMin'] + df_total['hrMax']) / 2)
df_total['prec_log'] = np.log1p(df_total.prec)
df_total['tmin_robust'] = robust.fit_transform(df_total[['tmin']])
df_total['tmax_robust'] = robust.fit_transform(df_total[['tmax']])
df_total['tmed_robust'] = robust.fit_transform(df_total[['tmed']])
df_total['prec_anual_robust'] = robust.fit_transform(df_total[['prec_anual']])
df_total['prec_anual_yeo'] = pt_yeojohnson.fit_transform(df_total[['prec_anual_robust']])
df_total['hrMax_log'] = np.log1p(df_total.hrMax)
df_total['hrMax_box'] = pt_boxcox.fit_transform(df_total[['hrMax_log']])
df_total['hrMin_robust'] = robust.fit_transform(df_total[['hrMin']])
df_total['velmedia_log'] = np.log1p(df_total.velmedia)
df_total['racha_log'] = np.log1p(df_total.racha)

#hago un label encoder a 'nombre' para enumerar cada estacion.
le = LabelEncoder()

# Aplicar LabelEncoder a la columna 'nombre'
df_total['nombre_encoded'] = le.fit_transform(df_total['nombre'])
df_total['altitud_cat']=df_total['altitud'].astype('category')
#hago un dumie a la columna altitud
altitudcat = pd.get_dummies(df_total.altitud_cat, drop_first=False)
df_total = pd.concat([df_total, altitudcat], axis=1)
#elimino columnas que me son innecesarias
df_total = df_total.drop(columns=['horatmin', 'horatmax','horaPresMax',  'horaPresMin', 'horaracha','horaHrMax', 'horaHrMin', 'dir'])
df_total.columns = df_total.columns.astype(str)

df_train = df_total.dropna(subset=['sol'])
#estrategia 1 para rellenar sol
#modelo LGBMRegressor con C2 y n_estimators=500, learning_rate=0.05, max_depth=5
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import train_test_split


df_nan = df_total[df_total['sol'].isnull()]


X = df_total[['altitud', 'tmed', 'prec', 'tmin', 'tmax', 'velmedia', 'presMax', 'presMin', 
              'hrMedia', 'hrMax', 'hrMin', 'semana', 'anio', 'mes', 'prec_anual', 'tmax_robust',
                'tmin_robust', 'tmed_robust', 'prec_log',
               'prec_anual_yeo', 'hrMax_box', 
             'hrMin_robust', 'velmedia_log',
                'racha_log', 
              '533', '540', '594', '605', '609', '620', '665', '667', '672', '690', '740', '763', 
              '884', '890', '924', '1004', '1030', '1159', '1450', '1532', '1893', 'nombre_encoded']]

y = df_total['sol']

#(80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
X_train_clean = X_train.dropna()
y_train_clean = y_train.loc[X_train_clean.index]



# LGBMRegressor
model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)


callbacks = [early_stopping(stopping_rounds=50)]


model.fit(X_train_clean, y_train_clean, 
          eval_set=[(X_test, y_test)], 
          eval_metric='l2', 
          callbacks=callbacks)


X_nan = df_nan[X_train_clean.columns]


predicciones_sol = model.predict(X_nan)
predicciones_sol = np.maximum(predicciones_sol, 0)  #(con esto quizas puedo obtener valores mayores a cero)


df_total['sol_full'] = df_total['sol'].copy()  
df_total.loc[df_total['sol'].isnull(), 'sol_full'] = predicciones_sol  
#este segundo modelo no sirve porque al haber muchos datos cercanos a cero inevitablemente tiene a la negatividad... sirve si los valores de sol fueran mas altos



df_nan = df_total[df_total['sol'].isnull()]


columnas_deseadas = ['altitud', 'tmed', 'prec', 'tmin', 'tmax', 'velmedia', 'presMax', 'presMin', 
                      'hrMedia', 'hrMax', 'hrMin', 'semana', 'anio', 'mes', 'prec_anual', 'tmax_robust',
                      'prec_robust', 'prec_log', 'tmin_robust', 'tmed_robust', 'prec_anual_log', 
                      'prec_anual_robust', 'prec_anual_yeo', 'hrMax_log', 'hrMax_robust', 'hrMax_box', 
                      'hrMin_log', 'hrMin_robust', 'hrMin_yeo', 'hrMin_box', 'velmedia_log', 'velmedia_robust', 
                      'velmedia_yeo', 'velmedia_yeo2', 'racha_log', 'racha_robust', 'racha_yeo', 'racha_yeo2',
                      '533', '540', '594', '605', '609', '620', '665', '667', '672', '690', '740', '763', 
                      '884', '890', '924', '1004', '1030', '1159', '1450', '1532', '1893', 'nombre_encoded']


columnas_existentes = [col for col in columnas_deseadas if col in df_total.columns]
X = df_total[columnas_existentes].copy()
y = np.log1p(df_total['sol']).fillna(0)  # Aplicar transformación logarítmica evitando NaN

#80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train_clean = X_train.dropna()
y_train_clean = y_train.loc[X_train_clean.index]


X_test_clean = X_test.dropna()
y_test_clean = y_test.loc[X_test_clean.index]

# # Entrenar el modelo
model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)
callbacks = [early_stopping(stopping_rounds=50)]

model.fit(X_train_clean, y_train_clean, 
           eval_set=[(X_test_clean, y_test_clean)],  # Evaluación en datos limpios
           eval_metric='l2', 
           callbacks=callbacks)

# # Predecir valores faltantes de 'sol' con el modelo
X_nan = df_nan[X.columns]  # Asegurar que se usen las mismas columnas del modelo
predicciones_sol_log = model.predict(X_nan)
predicciones_sol = np.maximum(predicciones_sol, 0)

# # Aplicar la transformación inversa (expm1) para regresar a la escala original
predicciones_sol = np.expm1(predicciones_sol_log)  # exp(sol_log) - 1
predicciones_sol = np.maximum(predicciones_sol, 0)
# # Crear nueva columna 'sol_full' con los valores originales de 'sol'
df_total['sol_full2'] = df_total['sol'].copy()

# # Reemplazar NaN en 'sol_full' con las predicciones
df_total.loc[df_total['sol'].isnull(), 'sol_full2'] = predicciones_sol
#la 3 estrategia tampocpo funciona por la misma razon que la estrategia 2
df_nan = df_total[df_total['sol'].isnull()]


X = df_total[['altitud', 'tmed', 'prec', 'tmin', 'tmax', 'velmedia', 'presMax', 'presMin', 
              'hrMedia', 'hrMax', 'hrMin', 'semana', 'anio', 'mes', 'prec_anual', 'tmax_robust',
                'tmin_robust', 'tmed_robust', 'prec_log',
               'prec_anual_yeo', 'hrMax_box', 
             'hrMin_robust', 'velmedia_log',
                'racha_log', 
              '533', '540', '594', '605', '609', '620', '665', '667', '672', '690', '740', '763', 
              '884', '890', '924', '1004', '1030', '1159', '1450', '1532', '1893', 'nombre_encoded']]

y = df_total['sol']

#(80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
X_train_clean = X_train.dropna()
y_train_clean = y_train.loc[X_train_clean.index]



# LGBMRegressor
model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, min_child_weight=15) #con min_child_weight = 1 no funciona, con 10 tampoco


callbacks = [early_stopping(stopping_rounds=50)]


model.fit(X_train_clean, y_train_clean, 
          eval_set=[(X_test, y_test)], 
          eval_metric='l2', 
          callbacks=callbacks)


X_nan = df_nan[X_train_clean.columns]


predicciones_sol = model.predict(X_nan)
predicciones_sol = np.maximum(predicciones_sol, 0)

df_total['sol_full3'] = df_total['sol'].copy()  
df_total.loc[df_total['sol'].isnull(), 'sol_full3'] = predicciones_sol  
#4 estrategia
df_nan = df_total[df_total['sol'].isnull()]


X = df_total[['altitud', 'tmed', 'prec', 'tmin', 'tmax', 'velmedia', 'presMax', 'presMin', 
              'hrMedia', 'hrMax', 'hrMin', 'semana', 'anio', 'mes', 'prec_anual', 'tmax_robust',
                'tmin_robust', 'tmed_robust', 'prec_log',
               'prec_anual_yeo', 'hrMax_box', 
             'hrMin_robust', 'velmedia_log',
                'racha_log', 
              '533', '540', '594', '605', '609', '620', '665', '667', '672', '690', '740', '763', 
              '884', '890', '924', '1004', '1030', '1159', '1450', '1532', '1893', 'nombre_encoded']]

y = df_total['sol']

#(80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
X_train_clean = X_train.dropna()
y_train_clean = y_train.loc[X_train_clean.index]

def custom_loss(y_true, y_pred):
    error = (y_true - y_pred) ** 2  # MSE básico
    penalizacion = np.where(y_pred < 0, 100, 0)  # Penaliza predicciones negativas
    return 'custom_loss', np.mean(error + penalizacion), False

# LGBMRegressor
model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=5) 


callbacks = [early_stopping(stopping_rounds=50)]


model.fit(X_train_clean, y_train_clean, 
          eval_set=[(X_test, y_test)], 
          eval_metric='l2', 
          callbacks=callbacks)


X_nan = df_nan[X_train_clean.columns]


predicciones_sol = model.predict(X_nan)
predicciones_sol = np.maximum(predicciones_sol, 0)

df_total['sol_full4'] = df_total['sol'].copy()  
df_total.loc[df_total['sol'].isnull(), 'sol_full4'] = predicciones_sol  
#5 estrategia
df_nan = df_total[df_total['sol'].isnull()]


X = df_total[['altitud', 'tmed', 'prec', 'tmin', 'tmax', 'velmedia', 'presMax', 'presMin', 
              'hrMedia', 'hrMax', 'hrMin', 'semana', 'anio', 'mes', 'prec_anual', 'tmax_robust',
                'tmin_robust', 'tmed_robust', 'prec_log',
               'prec_anual_yeo', 'hrMax_box', 
             'hrMin_robust', 'velmedia_log',
                'racha_log', 
              '533', '540', '594', '605', '609', '620', '665', '667', '672', '690', '740', '763', 
              '884', '890', '924', '1004', '1030', '1159', '1450', '1532', '1893', 'nombre_encoded']]

y = df_total['sol']

#(80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
X_train_clean = X_train.dropna()
y_train_clean = y_train.loc[X_train_clean.index]



# LGBMRegressor
model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, objective='poisson')


callbacks = [early_stopping(stopping_rounds=50)]


model.fit(X_train_clean, y_train_clean, 
          eval_set=[(X_test, y_test)], 
          eval_metric='l2', 
          callbacks=callbacks)


X_nan = df_nan[X_train_clean.columns]


predicciones_sol = model.predict(X_nan)

df_total['sol_full5'] = df_total['sol'].copy()  
df_total.loc[df_total['sol'].isnull(), 'sol_full5'] = predicciones_sol
df_solfull =df_total[['sol_full','sol_full2', 'sol_full3', 'sol_full4', 'sol_full5']].copy()
df_solfull['sol_def']= df_solfull.apply(lambda x: x.mean(),axis=1) #hago una columna nuevo con el promedio de todas las columnas 
df_total['sol_def']=df_solfull['sol_def']
df_total.drop(columns=['sol_full','sol_full2','sol_full3','sol_full4', 'sol_full5'], inplace=True)
#agrupar por semana y calcular promedios y acumulaciones cereal (trigo y cebada)

#agrupar por 'semana_anio' y agregar las columnas necesarias, manteniendo la columna 'mes'
semana_df = df_total.groupby('semana_anio').agg({
    'tmed': 'mean',        # Tmedia semanal
    'prec': 'sum',         # Prec acumulada semanal
    'hrMedia': 'mean',     # Hr media semanal
    'velmedia': 'mean',    # Velmedia semanal
    'tmin': 'min',         # Tminsemanal
    'tmax': 'max',         # Tmax semanal
    'sol_def': 'sum',          # Suma total de horas de sol semanal
    'prec_anual': 'sum',   # Precipitacion anual total
    'mes': 'first'         # Mantener el mes del primer registro de cada semana (chatgpt)
}).reset_index()


semana_df['sol_diario'] = semana_df['sol_def'] / 7  # promedio diario de horas de sol


def definir_siembra_cereales(row):
 
    if row['mes'] not in [9, 10, 11, 12]:
        return 0  
    if not (15 <= row['tmed'] <= 20):  
        return 0
    if row['hrMedia'] <= 60:  
        return 0
    if row['velmedia'] >= 15:  
        return 0
    if row['prec_anual'] <= 550:
        return 0
    if row['sol_diario'] <= 10:  
        return 0
    return 1  


semana_df['cereales'] = semana_df.apply(definir_siembra_cereales, axis=1)

# unir la info al df_total 
df_total = df_total.merge(
    semana_df[['semana_anio', 'cereales']],
    on='semana_anio',
    how='left'
)

#ahora lo definimos para legumbres  (lentejas y garbanzos)
semana_df = df_total.groupby(['indicativo', 'semana_anio']).agg({
    'tmed': 'mean',
    'hrMedia': 'mean',            
    'tmin': 'min',          
    'tmax': 'max',          
    'prec_anual': 'first',  
    'sol_def': 'sum',           
    'mes': 'first'          
}).reset_index()

# promedio diario de horas de sol
semana_df['sol_diario'] = semana_df['sol_def'] / 7  

# definir la funcion para determinar si es apta para la siembra de legumbres
def definir_siembra_legumbres(row):
    # filtrar por los meses de febrero (2), marzo (3), abril (4) y mayo (5)
    #porque ya paso el frio fuerte que daña las legumbres
    if row['mes'] not in [2, 3, 4, 5, 6]:
        return 0  # Si no es uno de esos meses, no es apto para sembrar
    
    
    if not (10 <= row['tmed'] <= 15):
        return 0
    
    if row['prec_anual'] <= 400:
        return 0
    
    if row['hrMedia'] <= 50:  
        return 0

    if row['tmin'] <= 0:
        return 0

    if row['tmax'] > 10:
   
        if row['sol_diario'] > 6:
            return 1 
    return 0   


semana_df['legumbres'] = semana_df.apply(definir_siembra_legumbres, axis=1)


df_total = df_total.merge(
    semana_df[['indicativo', 'semana_anio', 'legumbres']],
    on=['indicativo', 'semana_anio'],
    how='left'
)

pt_yeojohnson = PowerTransformer(method='yeo-johnson')
df_total['soldef_yeo'] = pt_yeojohnson.fit_transform(df_total[['sol_def']])
df_total['fecha'] = pd.to_datetime(df_total['fecha'])

#corrijo los tipos de datos de las columnas
df_total['altitud_cat'] = df_total['altitud_cat'].astype('category')

for col in df_total.select_dtypes(include=['int64', 'float64']).columns:
    df_total[col] = pd.to_numeric(df_total[col], downcast='integer') if df_total[col].dtype == 'int64' else pd.to_numeric(df_total[col], downcast='float')
df_total.to_csv("datos_climatologicos_limpios", index=False)