# Aplicación para predecir el momento óptimo de siembra de secano en Madrid

![Aplicación](https://eos.com/wp-content/uploads/2023/05/wheat-rows-on-a-field.jpg.webp)

## Descripción del proyecto
Este proyecto utiliza Machine Learning para predecir el momento óptimo de siembra de cereales (trigo y cebada) y legumbres (lentejas y garbanzos) en cultivos de secano en la Comunidad de Madrid.

## Motivación
Este proyecto surge de mi interés por combinar mis conocimientos previos con nuevas habilidades en programación y análisis de datos. Mi objetivo es contribuir a la solución de uno de los principales desafíos que enfrentan los agricultores: la elección de la fecha ideal para sembrar cultivos de secano. Además, este proyecto me ha permitido fortalecer mis competencias, expandir mis horizontes y abrirme paso profesionalmente en el mundo de la programación y el análisis de datos.

## Resumen
Para obtener los datos meteorológicos necesarios, se utilizó la API de la Agencia Estatal de Meteorología (AEMET) en su sección de Datos Abiertos. Se realizaron consultas en períodos de seis meses mediante Python y Visual Studio Code (VS Code).

Los datos recopilados se almacenaron en un DataFrame, facilitando su limpieza, procesamiento, visualización y análisis. Los valores nulos (NaN) fueron tratados mediante diferentes técnicas estadísticas, como media, mediana, moda, interpolación lineal e incluso regresión lineal.

Tras la limpieza, cada semana fue identificada como apta o no apta para la siembra utilizando un lenguaje binario (0 y 1), lo que permitió desarrollar un modelo de predicción para futuras fechas óptimas de siembra.

## Registros históricos
Los datos fueron obtenidos de la Agencia Estatal de Meteorología, accediendo a su API (gratuita por tres meses) y descargando la información de todas las estaciones meteorológicas de la Comunidad de Madrid de los últimos 25 años. Estos registros se almacenaron en un DataFrame de Pandas (`df_total`) para su posterior limpieza y procesamiento.

## Limpieza de datos
La limpieza de datos se realizó con Python, eliminando columnas innecesarias y ajustando tipos de datos (fechas, números, textos y booleanos). Para completar registros incompletos, se emplearon métodos estadísticos como media, mediana, moda e interpolación lineal. También se aplicaron ecuaciones meteorológicas para la imputación de valores. Se utilizaron gráficos como boxplots e histogramas para identificar y corregir valores atípicos, evitando una menor precisión en las predicciones.

## Escalado de datos
Para la corrección de valores atípicos, se aplicaron los siguientes métodos de reescalado:
- Transformación logarítmica
- RobustScaler
- StandardScaler
- Yeo-Johnson
- Box-Cox

## Creación de nuevas columnas
Se emplearon `LabelEncoder` y `get_dummies` de Pandas para generar columnas adicionales que mejoraron el proceso de predicción.

## Modelos predictivos
Se evaluaron diferentes métodos de regresión y clasificación para la predicción de fechas óptimas de siembra.

### Modelos de regresión evaluados:
- XGBRegressor
- RandomForestRegressor
- LinearRegression
- SVR
- GradientBoostingRegressor
- CatBoostRegressor
- LGBMRegressor
- MLPRegressor

El modelo seleccionado para predecir valores de la columna `sol` (fotoperiodo diario) fue **LGBMRegressor** con `n_estimators=500`, `learning_rate=0.05` y `max_depth=5`, logrando un **93% de R2**.

### Modelos de clasificación evaluados:
- RandomForestClassifier
- LogisticRegression
- XGBClassifier
- KNeighborsClassifier
- SVC
- GradientBoostingClassifier
- DecisionTreeClassifier
- GaussianNB

Para la predicción de la variable `legumbres`, se utilizó un modelo de **StackingClassifier** combinando **RandomForestClassifier** y **XGBClassifier**. Para `cereales`, se seleccionó **XGBClassifier**.

## Interfaz
La aplicación cuenta con una interfaz desarrollada en **Tkinter**, permitiendo seleccionar el año y tipo de cultivo para visualizar las semanas óptimas de siembra y la necesidad de riego asistido.

## Estructura del proyecto
```bash
proyecto/
│
├── data/
│   ├── raw/                # Datos originales sin procesar
│   ├── processed/          # Datos después de la limpieza y transformación
│
├── notebooks/
│   ├── 01_data_collection.ipynb       # Recolección de datos desde la API
│   ├── 02_data_cleaning.ipynb         # Limpieza y tratamiento de datos
│   ├── 03_exploratory_analysis.ipynb  # Análisis exploratorio de datos
│   └── 04_model_training.ipynb        # Entrenamiento de modelos predictivos
│
├── scripts/
│   ├── data_collection.py    # Script para recolección de datos
│   ├── data_cleaning.py      # Script para limpieza de datos
│   ├── exploratory_analysis.py # Script para análisis exploratorio
│   └── model_training.py     # Script para entrenamiento de modelos
│
├── models/
│   └── saved_models/         # Modelos entrenados guardados
│
├── README.md                 # Documentación del proyecto
```

## Créditos
Agradezco al portal [Datos abiertos](https://opendata.aemet.es/centrodedescargas/inicio) de la [Agencia Estatal de Meteorología](https://www.aemet.es/es/datos_abiertos) por facilitar el acceso a los datos meteorológicos, fundamentales para este proyecto.

## Lenguaje y librerías utilizadas
Este proyecto fue desarrollado en **Python** utilizando **Visual Studio Code**.

### 📚 Librerías principales:
- **Manipulación de datos:** `numpy`, `pandas`
- **Visualización:** `matplotlib`, `seaborn`
- **Machine Learning:** `xgboost`, `lightgbm`, `catboost`, `sklearn`
- **Balanceo de datos:** `imblearn`
- **Interfaz:** `tkinter`

Con este conjunto de herramientas, se logró desarrollar un modelo eficiente para predecir las fechas óptimas de siembra en cultivos de secano en la Comunidad de Madrid.

