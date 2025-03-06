# -*- coding: utf-8 -*-

import sett

import os
import numpy as np
import pywt  # Para realizar transformadas Wavelet, utiles para eliminacion de ruido en señales.
import pandas as pd # Para manipulacion y analisis de datos estructurados, como DataFrames.
import requests # Para realizar solicitudes HTTP, como descargar datos o interactuar con APIs.
import datetime  # Para manejar y operar con fechas y tiempos.
from datetime import datetime, timedelta  # Operaciones adicionales con fechas y tiempos.
import json
import joblib


def fetch_data_in_chunks(symbol, interval, start_date, end_date, api_key, max_retries=3, retry_delay=5):
    """
    Obtiene datos de series temporales en intervalos definidos desde la API TwelveData.

    Args:
        symbol (str): Simbolo del par de divisas (e.g., 'EUR/USD').
        interval (str): Intervalo de tiempo (e.g., '1day', '1h').
        start_date (datetime): Fecha de inicio de los datos.
        end_date (datetime): Fecha de fin de los datos.
        api_key (str): Clave de API de TwelveData.
        max_retries (int): Numero maximo de reintentos en caso de fallo.
        retry_delay (int): Tiempo de espera entre reintentos (en segundos).

    Returns:
        pd.DataFrame: DataFrame con los datos obtenidos.
    """
    BASE_URL = "https://api.twelvedata.com/time_series"
    all_data = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=500), end_date)
        params = {
            'symbol': symbol,
            'interval': interval,
            'apikey': api_key,
            'start_date': current_start.strftime('%Y-%m-%d'),
            'end_date': current_end.strftime('%Y-%m-%d'),
        }

        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(BASE_URL, params=params, timeout=10)
                response.raise_for_status()  # Lanzar error para codigos de estado >= 400

                data = response.json()
                if "values" in data:
                    all_data.extend(data["values"])

                break

            except requests.exceptions.RequestException as e:
                retries += 1
                print(f"Error de red o API (intento {retries}/{max_retries}): {e}")
                sleep(retry_delay)

        if retries == max_retries:
            print(f"Error persistente al obtener datos para el rango: {current_start.strftime('%Y-%m-%d')} - {current_end.strftime('%Y-%m-%d')}")

        current_start = current_end + timedelta(days=1)

    if not all_data:
        print("No se obtuvieron datos.")
        return pd.DataFrame()

    # Crear un DataFrame a partir de los datos obtenidos
    try:
        df = pd.DataFrame(all_data)

        # Validar y procesar las columnas del DataFrame
        if 'datetime' not in df.columns:
            raise KeyError("La columna 'datetime' no existe en los datos obtenidos.")

        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime')

        # Convertir columnas numericas a tipo float
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = df[col].astype(float)
            else:
                raise KeyError(f"La columna '{col}' no existe en los datos obtenidos.")

        # Manejo de datos faltantes
        if df.isnull().values.any():
            print("Advertencia: Se encontraron valores nulos en los datos. Eliminando...")
            df = df.dropna()

    except Exception as e:
        print(f"Error procesando los datos obtenidos: {e}")
        return pd.DataFrame()

    return df
    
def wavelet_denoising(data, wavelet='db4', level=None, threshold_method='soft'):
    """
    Aplica suavizado con Wavelet Denoising a las columnas numericas de un DataFrame,
    calcula metricas de validacion y genera graficos para evidenciar los resultados.

    Args:
        data (pd.DataFrame): DataFrame con los datos originales.
        wavelet (str): Nombre del wavelet a utilizar (por defecto, 'db4').
        level (int): Nivel de descomposicion (por defecto, se determina automaticamente).
        threshold_method (str): Metodo de umbralizacion ('soft' o 'hard').
        output_path (str): Ruta para guardar el archivo CSV con los datos suavizados (opcional).

    Returns:
        pd.DataFrame: DataFrame con las columnas suavizadas.
    """
    try:
        # Copiar datos originales
        df = data.copy()

        # Seleccionar columnas numericas para el suavizado
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        # Diccionario para guardar metricas
        metrics = {}

        for col in numeric_columns:
            # Descomposicion wavelet
            signal = df[col].values
            coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)

            # Determinar umbral basado en la desviacion estandar de los coeficientes de detalle
            sigma = np.std(coeffs[-1])
            threshold = sigma * np.sqrt(2 * np.log(len(signal)))

            # Aplicar umbralizacion (soft o hard)
            coeffs_thresholded = [pywt.threshold(c, threshold, mode=threshold_method) if i > 0 else c for i, c in enumerate(coeffs)]

            # Reconstruccion de la señal suavizada
            smoothed_signal = pywt.waverec(coeffs_thresholded, wavelet=wavelet)

            # Limitar la longitud para evitar problemas de borde
            smoothed_signal = smoothed_signal[:len(df)]

            # Asignar la columna suavizada al DataFrame
            df[f"{col}_smoothed"] = smoothed_signal

            # Calcular metricas de validacion
            mae = np.mean(np.abs(signal - smoothed_signal))  # Mean Absolute Error
            mse = np.mean((signal - smoothed_signal)**2)     # Mean Squared Error
            metrics[col] = {"MAE": mae, "MSE": mse}

        return df

    except Exception as e:
        print(f"Error en el suavizado con wavelet: {e}")
        return None
        
def calcular_indicadores_tecnicos(data, usar_suavizada=False, columna_suavizada="close_smoothed"):
    """
    Calcula indicadores tecnicos sobre datos suavizados, genera etiquetas, calcula metricas de validacion y guarda el resultado en un archivo CSV.

    Args:
        data (pd.DataFrame): DataFrame con columnas suavizadas y originales.
        usar_suavizada (bool): Si True, utiliza las columnas suavizadas para los calculos.
        columna_suavizada (str): Nombre de la columna de precios de cierre suavizada.
        output_path (str): Ruta para guardar el archivo CSV consolidado con indicadores, etiquetas y metricas.

    Returns:
        pd.DataFrame: DataFrame con indicadores tecnicos, etiquetas y metricas de calidad.
    """
    try:
        # Copiar datos originales
        df = data.copy()

        # Validar columnas necesarias
        if usar_suavizada:
            required_columns = ['high_smoothed', 'low_smoothed', columna_suavizada]
        else:
            required_columns = ['high', 'low', 'close']

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Las siguientes columnas necesarias faltan en los datos: {missing_columns}")

        # Seleccionar columnas de precios
        high_col = 'high_smoothed' if usar_suavizada else 'high'
        low_col = 'low_smoothed' if usar_suavizada else 'low'
        close_col = columna_suavizada if usar_suavizada else 'close'

        # ---------------- Calcular Indicadores Tecnicos ----------------
        # RSI
        window_rsi = 14
        delta = df[close_col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window_rsi).mean()
        avg_loss = loss.rolling(window=window_rsi).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = df[high_col] - df[low_col]
        high_close = (df[high_col] - df[close_col]).abs()
        low_close = (df[low_col] - df[close_col]).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()

        # ROC
        window_roc = 14
        df['ROC'] = df[close_col].diff(window_roc) / df[close_col].shift(window_roc) * 100

        # Williams %R
        window_wr = 14
        highest_high = df[high_col].rolling(window=window_wr).max()
        lowest_low = df[low_col].rolling(window=window_wr).min()
        df['WR'] = (highest_high - df[close_col]) / (highest_high - lowest_low + 1e-10) * -100

        # ---------------- Definicion de Etiquetas ----------------
        # Etiquetas basadas en cambio porcentual
        df['target'] = df[close_col].pct_change().shift(-1)  # Cambio porcentual en el precio suavizado

        df['target'] = pd.cut(
            df['target'],
            bins=[-np.inf, -0.001, 0.001, np.inf],  # Baja (< -0.1%), Rango (-0.1% a 0.1%), Sube (> 0.1%)
            labels=['Baja', 'Rango', 'Sube']
        )

        # ---------------- Calculo de Metricas de Validacion ----------------
        metrics = {
            'RSI': {'mean': df['RSI'].mean(), 'std': df['RSI'].std(), 'min': df['RSI'].min(), 'max': df['RSI'].max()},
            'ATR': {'mean': df['ATR'].mean(), 'std': df['ATR'].std(), 'min': df['ATR'].min(), 'max': df['ATR'].max()},
            'ROC': {'mean': df['ROC'].mean(), 'std': df['ROC'].std(), 'min': df['ROC'].min(), 'max': df['ROC'].max()},
            'WR': {'mean': df['WR'].mean(), 'std': df['WR'].std(), 'min': df['WR'].min(), 'max': df['WR'].max()},
        }

        # Metrica de distribucion de etiquetas
        label_distribution = df['target'].value_counts(normalize=True)

        return df

    except Exception as e:
        print(f"Error al calcular indicadores tecnicos: {e}")
        return None
        
def integrar_analisis_tecnico_lightgbm(data, modelo_path, label_encoder_path, fecha_dada, time_steps):
    """
    Integra el analisis tecnico con el modelo LightGBM para predecir el movimiento del mercado
    del dia siguiente basado en una fecha especifica, incluyendo el porcentaje de probabilidad,
    y guarda la integracion como un modelo empaquetado para reutilizacion futura.

    Args:
        data_path (str): Ruta al archivo CSV con los datos y analisis tecnico.
        modelo_path (str): Ruta al archivo del modelo LightGBM guardado.
        label_encoder_path (str): Ruta al archivo del codificador de etiquetas.
        fecha_dada (str): Fecha base para realizar la prediccion (formato YYYY-MM-DD).

    Returns:
        dict: Diccionario con la fecha siguiente, prediccion del movimiento (Sube, Baja, Rango) y la probabilidad asociada.
    """
    try:
        # Asegurarse de que la fecha este en formato datetime
        data['datetime'] = pd.to_datetime(data['datetime'])

        # Filtrar los datos hasta la fecha dada
        data_filtrada = data.sort_values(by='datetime')

        if data_filtrada.empty:
            raise ValueError("No hay datos disponibles para la fecha dada o anteriores.")

        # Seleccionar el ultimo registro como entrada para el modelo
        ultimo_dato = data_filtrada.tail(time_steps)

        # Cargar el modelo LightGBM
        modelo = joblib.load(modelo_path)

        # Cargar el codificador de etiquetas
        label_encoder = joblib.load(label_encoder_path)

        # Seleccionar las caracteristicas para la prediccion
        features = ['close_smoothed', 'RSI', 'ATR', 'ROC', 'WR']
        X = np.array(ultimo_dato[features].values)
        X_input = [X.reshape(-1)]

        # Realizar la prediccion
        y_pred_probs = modelo.predict(X_input, num_iteration=modelo.best_iteration)[0]

        y_pred = np.argmax(y_pred_probs)

        # Decodificar la etiqueta predicha
        etiqueta_predicha = label_encoder.inverse_transform([y_pred])[0]

        # Calcular el porcentaje de probabilidad de la prediccion
        probabilidad = y_pred_probs[y_pred]

        # Calcular la fecha siguiente
        fecha_siguiente = pd.to_datetime(fecha_dada) + timedelta(days=1)

        # Crear la estructura JSON
        recomendacion = "Compra" if etiqueta_predicha == "Sube" else "Venta" if etiqueta_predicha == "Baja" else "Mantener"
        justificacion = (
            f"La prediccion para el dia siguiente muestra un movimiento {'positivo' if etiqueta_predicha == 'Sube' else 'negativo' if etiqueta_predicha == 'Baja' else 'neutral'} "
            f"con una probabilidad estimada del {probabilidad:.2f}. Esta recomendacion se basa en los indicadores tecnicos (RSI, ATR, ROC, WR) "
            "y el analisis historico del modelo."
        )

        json_resultado = {
            "fecha_de_analisis": fecha_dada,
            "recomendacion": recomendacion,
            "probabilidad": round(probabilidad, 2),
            "justificacion": justificacion
        }

        return json_resultado

    except Exception as e:
        print(f"Error en la integracion: {e}")
        return None
        
def response_analisis_tecnico(fecha_de_dato_predecir, paridad):
    """
    Devuelve la respuesta en base a una fecha solicitada y la paridad de cambio.

    Args:
        fecha_de_dato_predecir (str): fecha en fotmato yyyy-mm-dd.
        paridad (str): paridad en formato ejemplp EUR/USD.

    Returns:
        dict: Diccionario con fecha_de_analisis, recomendacion, probabilidad, justificacion.
    """
    symbol = paridad
    interval = "1day"
    time_steps = 30
    max_retries_ = 3
    retry_delay_ = 5

    fecha_de_dato_predecir = pd.to_datetime(fecha_de_dato_predecir)

    start_date = fecha_de_dato_predecir - timedelta(days=200)
    end_date = fecha_de_dato_predecir# - timedelta(days=1)

    # Carga de Data del Api Time Series
    df = fetch_data_in_chunks(symbol, interval, start_date, end_date, api_key = sett.api_key_time_series, max_retries = max_retries_, retry_delay = retry_delay_)
    df = wavelet_denoising(df, wavelet='db4', level=None, threshold_method='soft')
    df = df_con_indicadores = calcular_indicadores_tecnicos(df,
                                                            usar_suavizada=True, # Trabajamos con datos suavizados
                                                            columna_suavizada='close_smoothed' # Columna de cierre suavizada
                                                           )

    resultado = integrar_analisis_tecnico_lightgbm(df,
                                                   modelo_path = sett.path_model + "/" + sett.classification_model,
                                                   label_encoder_path = sett.path_model + "/" + sett.classification_label,
                                                   fecha_dada = fecha_de_dato_predecir,
                                                   time_steps = time_steps
                                                  )

    return resultado