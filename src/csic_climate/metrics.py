import os
import smtplib
from email.message import EmailMessage
from typing import Dict, Any, List, Union, Optional
import pandas as pd
import numpy as np

def calculate_climate_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula los indicadores climáticos principales a partir de un DataFrame filtrado.
    
    Args:
        df: DataFrame filtrado con datos climáticos.
        
    Returns:
        Dict[str, Any]: Diccionario con las métricas calculadas.
    """
    if df.empty:
        return {
            "temp_media": 0.0,
            "anomalia_media": 0.0,
            "precipitacion_total": 0.0,
            "meses_sequia_severa": 0,
            "dias_totales_ola_calor": 0
        }
        
    temp_media = round(df["temperatura_media_c"].mean(), 2)
    anomalia_media = round(df["anomalia_termica_c"].mean(), 2)
    precipitacion_total = round(df["precipitacion_mm"].sum(), 1)
    
    # Se considera sequía severa cuando el índice SPEI es menor a -1.5 (Norma IPE-CSIC)
    meses_sequia = int((df["indice_spei_sequia"] < -1.5).sum())
    tot_olas_calor = int(df["dias_ola_calor"].sum())
    
    return {
        "temp_media": temp_media,
        "anomalia_media": anomalia_media,
        "precipitacion_total": precipitacion_total,
        "meses_sequia_severa": meses_sequia,
        "dias_totales_ola_calor": tot_olas_calor
    }

def calculate_decadal_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa los datos por década y calcula el promedio de anomalía térmica y sequía.
    """
    df_copy = df.copy()
    df_copy["decada"] = (df_copy["year"] // 10) * 10

    
    summary = df_copy.groupby("decada").agg({
        "temperatura_media_c": "mean",
        "anomalia_termica_c": "mean",
        "precipitacion_mm": "mean",
        "indice_spei_sequia": "mean",
        "dias_ola_calor": "sum"
    }).reset_index()
    
    summary = summary.round(2)
    return summary

# ==============================================================================
# ESPACIO DE TRABAJO PARA EL GRUPO 1 (BACKEND / CIENCIA DE DATOS)
# ==============================================================================
# El Grupo 1 añadirá aquí sus nuevas funciones analíticas avanzadas, por ejemplo:
# - calculate_warming_rate_per_decade(df)
# - detect_extreme_climate_events(df)
# ==============================================================================


def get_hottest_and_coldest_year(
    df: pd.DataFrame,
    start_year: int = 1961,
    end_year: int = 2024,
    comunidades: Union[List[str], str] = "all",
) -> Dict[str, Any]:
    """
    Identifica el año más cálido y el año más frío dentro de un periodo temporal,
    evaluados en base al promedio anual de la anomalía térmica.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con registros climáticos. Debe contener obligatoriamente
        las siguientes columnas:
        - 'year' (int): Año de la observación.
        - 'comunidad_autonoma' (str): Nombre de la comunidad autónoma.
        - 'anomalia_termica_c' (float): Anomalía térmica respecto al periodo base (°C).
    start_year : int, default=1961
        Año límite inferior del rango temporal (inclusivo: df['year'] >= start_year).
    end_year : int, default=2024
        Año límite superior del rango temporal (inclusivo: df['year'] <= end_year).
    comunidades : Union[List[str], str], default="all"
        Filtro espacial de comunidades autónomas:
        - Si es "all": Se consideran todas las comunidades presentes en el DataFrame.
        - Si es List[str]: Se filtran únicamente los registros pertenecientes
          a los nombres de comunidad incluidos en la lista (coincidencia exacta).

    Returns
    -------
    Dict[str, Any]
        Diccionario con las estadísticas del periodo y comunidades seleccionadas:
        {
            "hottest_year": int,    # Año con la mayor anomalía media anual.
            "hottest_value": float, # Valor de la anomalía media más alta (redondeado a 2 decimales).
            "coldest_year": int,    # Año con la menor anomalía media anual.
            "coldest_value": float  # Valor de la anomalía media más baja (redondeado a 2 decimales).
        }

    Raises
    ------
    ValueError
        Si el DataFrame resultante tras aplicar los filtros de años y/o
        comunidades se encuentra vacío (ej. rango temporal sin datos o comunidades inexistentes).
    KeyError
        Si el DataFrame de entrada no contiene alguna de las columnas requeridas:
        'year', 'comunidad_autonoma' o 'anomalia_termica_c'.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'year': [2000, 2000, 2001, 2001],
    ...     'comunidad_autonoma': ['Andalucía', 'Aragón', 'Andalucía', 'Aragón'],
    ...     'anomalia_termica_c': [1.2, 0.8, -0.4, -0.2]
    ... })
    >>> get_hottest_and_coldest_year(data, start_year=2000, end_year=2001)
    {'hottest_year': 2000, 'hottest_value': 1.0, 'coldest_year': 2001, 'coldest_value': -0.3}

    >>> # Filtrando por comunidad específica:
    >>> get_hottest_and_coldest_year(data, comunidades=['Andalucía'])
    {'hottest_year': 2000, 'hottest_value': 1.2, 'coldest_year': 2001, 'coldest_value': -0.4}
    """
    # 1. Filtrar por rango de años
    df_filtered = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()

    # 2. Filtrar por comunidades autónomas
    if isinstance(comunidades, list):
        df_filtered = df_filtered[
            df_filtered["comunidad_autonoma"].isin(comunidades)
        ]

    # 3. Validar que queden datos
    if df_filtered.empty:
        raise ValueError(
            f"No hay datos disponibles para el rango [{start_year}, {end_year}] "
            f"con las comunidades seleccionadas."
        )

    # 4. Calcular la anomalía media por año
    anomalia_anual = (
        df_filtered.groupby("year")["anomalia_termica_c"]
        .mean()
    )

    # 5. Identificar año más cálido y más frío
    hottest_year = int(anomalia_anual.idxmax())
    coldest_year = int(anomalia_anual.idxmin())

    return {
        "hottest_year": hottest_year,
        "hottest_value": round(float(anomalia_anual[hottest_year]), 2),
        "coldest_year": coldest_year,
        "coldest_value": round(float(anomalia_anual[coldest_year]), 2),
    }
