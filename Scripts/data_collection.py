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


pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

configuration = swagger_client.Configuration()
configuration.api_key['api_key'] = 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJpcGhvbmVkZWRhbmllbEBnbWFpbC5jb20iLCJqdGkiOiJjZThiMzM4YS1mM2NhLTRkNDgtOTQ3Zi00NTUzMmZkNTNiY2QiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTczODUwMjQwNiwidXNlcklkIjoiY2U4YjMzOGEtZjNjYS00ZDQ4LTk0N2YtNDU1MzJmZDUzYmNkIiwicm9sZSI6IiJ9.-50vwaqb_nrSxCFkTqrg0MQt5dppbNAU2yDSmJEFjfU'

api_instance = swagger_client.ValoresClimatologicosApi(swagger_client.ApiClient(configuration))
fecha_ini_str = '2014-07-13T00:00:00UTC'  
fecha_fin_str = '2015-01-11T23:59:59UTC' 
idema = '3200,3195,3196,3129,3191E,3170Y,3268C,3100B,3182Y,3110C,3191E,3126Y,3194Y,3266A,2462,3104Y,3338,3330Y,3125Y,3111D,3229Y,3343Y' # estaciones meteorologicas de la comunidad de Madrid

try:
    api_response = api_instance.climatologas_diarias_(fecha_ini_str, fecha_fin_str, idema)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling ValoresClimatologicosApi->climatologas_diarias_: %s\n" % e)

configuration = swagger_client.Configuration()
configuration.api_key['api_key'] = 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJpcGhvbmVkZWRhbmllbEBnbWFpbC5jb20iLCJqdGkiOiJjZThiMzM4YS1mM2NhLTRkNDgtOTQ3Zi00NTUzMmZkNTNiY2QiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTczODUwMjQwNiwidXNlcklkIjoiY2U4YjMzOGEtZjNjYS00ZDQ4LTk0N2YtNDU1MzJmZDUzYmNkIiwicm9sZSI6IiJ9.-50vwaqb_nrSxCFkTqrg0MQt5dppbNAU2yDSmJEFjfU'
api_instance = swagger_client.ValoresClimatologicosApi(swagger_client.ApiClient(configuration))


idema = '3200,3195,3196,3129,3191E,3170Y,3268C,3100B,3182Y,3110C,3191E,3126Y,3194Y,3266A,2462,3104Y,3338,3330Y,3125Y,3111D,3229Y,3343Y'  #estaciones meteorologicas
fecha_inicio = datetime(2000, 1, 1)  
fecha_fin_total = datetime(2025, 1, 31)  


df_total = pd.DataFrame()


while fecha_inicio < fecha_fin_total:
    fecha_fin = fecha_inicio + timedelta(days=182)  
    if fecha_fin > fecha_fin_total:
        fecha_fin = fecha_fin_total 


    fecha_ini_str = fecha_inicio.strftime('%Y-%m-%dT00:00:00UTC')
    fecha_fin_str = fecha_fin.strftime('%Y-%m-%dT23:59:59UTC')

    try:
   
        api_response = api_instance.climatologas_diarias_(fecha_ini_str, fecha_fin_str, idema)
        url = api_response.datos
        response = req.get(url)
        r = response.json()

        df = pd.DataFrame(r)

       
        df_total = pd.concat([df_total, df], ignore_index=True)

        
        df_total.to_csv("datos_climatologicos.csv", index=False)

        print(f"Datos de {fecha_ini_str} a {fecha_fin_str} agregados correctamente.")

    except Exception as e:
        print(f"Error obteniendo datos de {fecha_ini_str} a {fecha_fin_str}: {e}")

    fecha_inicio = fecha_fin + timedelta(days=1)

