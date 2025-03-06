# -*- coding: utf-8 -*-

import sett

def response_analisis_sentimiento(fecha_de_analisis, paridad):
    """
    Devuelve la respuesta en base a una fecha solicitada y la paridad de cambio.

    Args:
        fecha_de_dato_predecir (str): fecha en fotmato yyyy-mm-dd.
        paridad (str): paridad en formato ejemplp EUR/USD.

    Returns:
        dict: Diccionario con fecha_de_analisis, recomendacion, probabilidad, justificacion.
    """
    
    probabilidad = 0.95

    justificacion = f"La prediccion de analisis de sentimiento respecto a la paridad '{paridad}' para la fecha de analisis '{fecha_de_analisis}' "\
                    f"es de sentimiento 'negativo' con una probabilidad estimada del {probabilidad:.2f}. Esta recomendacion se obtiene despues "\
                    "de anizar 1000 comentarios respecto a ese tema."

    resultado = {"fecha_de_analisis": fecha_de_analisis,
                 "sentimiento": "negativo",
                 "probabilidad": probabilidad,
                 "justificacion": justificacion
                }
    
    return resultado