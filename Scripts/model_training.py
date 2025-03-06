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

pd.set_option('display.max_columns', None)  
%matplotlib inline

#df_total=pd.read_csv("/mnt/c/Users/danie/OneDrive/Desktop/Proyectos/proyecto_final/Notebooks/datos_climatologicos_limpios.csv")
#aqui definimos las fechas futuras para cereales
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Definir el rango de fechas
start_date = '2025-01-01'
end_date = '2026-12-31'
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Crear un DataFrame con las fechas
df_prediccion = pd.DataFrame(date_range, columns=['fecha'])

# Extraer características de la fecha
df_prediccion['anio'] = df_prediccion['fecha'].dt.year
df_prediccion['mes'] = df_prediccion['fecha'].dt.month
df_prediccion['semana'] = df_prediccion['fecha'].dt.isocalendar().week


df_prediccion_fecha = df_prediccion[['fecha']].copy()

columnas_X = ['altitud', 'prec', 'semana', 'anio', 'mes', 'racha_log', 'nombre_encoded', '620', '667', 'tmax_robust', 'tmed_robust', 'prec_anual_yeo', 'sol_def']

# Extraer X 
X = df_total[columnas_X]
y = df_total['cereales']

# 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Aplicar SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Entrenar el modelo XGBoost
modelo = xgb.XGBClassifier(scale_pos_weight=len(y_train_res) / sum(y_train_res == 1), random_state=42)
modelo.fit(X_train_res, y_train_res)


for col in columnas_X:
    if col not in df_prediccion.columns:
        df_prediccion[col] = 0  # Rellenar con ceros o valores más adecuados si los tienes

# Ordenar las columnas en el mismo orden que en X_train
df_prediccion = df_prediccion[columnas_X]

# Escalar los datos
df_prediccion_scaled = scaler.transform(df_prediccion)

# Hacer predicción
df_prediccion['cereales'] = modelo.predict(df_prediccion_scaled)

# Agregar la columna 'nombre' desde df_total (sin repetir estaciones)
df_nombres = df_total[['nombre']].drop_duplicates().reset_index(drop=True)

# Crear todas las combinaciones posibles de 'nombre' con fechas del rango
df_nombres['key'] = 1
df_prediccion['key'] = 1
df_secano = df_prediccion.merge(df_nombres, on='key').drop(columns=['key'])

# Seleccionar solo las columnas finales
df_secano = df_secano[['nombre', 'anio', 'mes', 'semana', 'cereales']]





#legumbres 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import optuna



#  1. Preparación de datos históricos
columnas_numericas = ['altitud', 'prec', 'presMax', 'presMin', 'hrMedia', 'semana', 'anio', 'mes',
                      'racha_log', 'nombre_encoded', '533', '540', '594', '605', '609', '620', 
                      '665', '667', '672', '690', '740', '763', '884', '890', '924', '1004', 
                      '1030', '1159', '1450', '1532', '1893', 'prec_log', 'tmin_robust', 
                      'tmax_robust', 'tmed_robust', 'hrMax_box', 'hrMin_robust', 'velmedia_log', 
                      'prec_anual_yeo', 'sol_def', 'soldef_yeo']

X = df_total[columnas_numericas]
y = df_total['legumbres']

# División train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Balanceo con SMOTE focalizado
smote = SMOTE(sampling_strategy=0.5, k_neighbors=3, random_state=42)  # Relación 2:1
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)


# 3. Definición de la función objetivo para Optuna
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': trial.suggest_int('scale_pos_weight', 5, 30),  # Peso para clase 1
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 10.0),
        'n_jobs': -1  # Usar todos los hilos
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test_scaled)
    f1_score_1 = classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']
    return f1_score_1

# 4. Optimización con Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # 50 iteraciones para una buena búsqueda

#  5. Entrenamiento del modelo final con los mejores parámetros
best_params = study.best_params
best_params['objective'] = 'binary'
best_params['metric'] = 'binary_logloss'
best_params['boosting_type'] = 'gbdt'
best_params['n_jobs'] = -1

lgbm_model = lgb.LGBMClassifier(**best_params)
lgbm_model.fit(X_train_balanced, y_train_balanced)

# Evaluación
y_pred = lgbm_model.fit(X_train_balanced, y_train_balanced).predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)


# 6. Importancia de características
feature_importance = pd.DataFrame({
    'feature': columnas_numericas,
    'importance': lgbm_model.feature_importances_
}).sort_values('importance', ascending=False)


#  7. Preparación de df_secano para predicción
df_secano_pred = df_secano[df_secano['anio'].isin([2025, 2026])].copy()

# Calcular promedios de características meteorológicas por 'nombre' desde df_total
df_caracteristicas = df_total.groupby('nombre')[columnas_numericas].mean().reset_index()

# Unir con df_secano_pred
df_secano_pred = df_secano_pred.merge(df_caracteristicas, on='nombre', how='left')

# Asegurar que todas las columnas de columnas_numericas estén presentes
for col in columnas_numericas:
    if col not in df_secano_pred.columns:
        df_secano_pred[col] = 0
    if col in ['semana', 'anio', 'mes'] and col in df_secano.columns:
        df_secano_pred[col] = df_secano[df_secano['anio'].isin([2025, 2026])][col]

# Preparar X_futuro y escalar
X_futuro = df_secano_pred[columnas_numericas]
X_futuro_scaled = scaler.transform(X_futuro)

# 8. Predicción con el modelo LightGBM
predicciones = lgbm_model.predict(X_futuro_scaled)

# Agregar predicciones a df_secano como 'legumbres2'
df_secano.loc[df_secano['anio'].isin([2025, 2026]), 'legumbres'] = predicciones


#df_secano.to_csv("df_secano_final.csv", index=False)