import numpy as np
import matplotlib.pyplot as plt
from FuncionesYModulosORIGINAL import *
import math
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from uncertainties import ufloat
import pandas as pd


import os

# Cambiar el directorio de trabajo al del script actual
os.chdir(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update(plt.rcParamsDefault)
#plt.rcParams["figure.figsize"] = [8.0, 5.50]
plt.rcParams["figure.autolayout"] = True
#plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = "serif"

# Definimos los umbrales a probar
umbrales = [-100, -150, -200, -300, -400, -500]

# Archivos de entrada (solo 950V y 1000V)
files_BC = ['Caracterización2y3(950V).txt', 'Caracterización2y3(1000V).txt']
files_AD = ['Caracterización1y4(950V).txt', 'Caracterización1y4(1000V).txt']

def calcular_tasas(files, umbrales, canales_variable='BC'):
    print(f"\nTasas para canales {canales_variable}:\n")

    for i, file in enumerate(files):
        event_data, t, *_ = read(file)

        for u in umbrales:
            if canales_variable == 'BC':
                m_umbral, t_umbral = nuevo_umbral(event_data, t, -200, -200, u, u)
            elif canales_variable == 'AD':
                m_umbral, t_umbral = nuevo_umbral(event_data, t, -200, -200, u, u)
            else:
                raise ValueError("canales_variable debe ser 'BC' o 'AD'")

            eliminadas = len(event_data) - len(m_umbral)
            porcentaje = 100 * eliminadas / len(event_data)
            print(f"El número de coincidencias eliminadas aplicando el umbral {u} mV fueron {eliminadas} de un total de {len(event_data)} → {porcentaje:.2f} %")

            if len(m_umbral) > 0:
                intervalo_tiempo = t_umbral[-1] - t_umbral[0]
                if intervalo_tiempo > 0:
                    tasa_val, incertidumbre = tasa(len(m_umbral), t_umbral[-1], t_umbral[0])
                    print(f"Archivo {files[i]} | Umbral {u} mV → Tasa: {tasa_val:.3f} ± {incertidumbre:.3f} Hz\n")
                else:
                     print(f"Archivo {files[i]} | Umbral {u} mV → Tasa: Intervalo temporal nulo\n")
            else:
                print(f"Archivo {files[i]} | Umbral {u} mV → Tasa: Insuficientes coincidencias tras el umbral\n")

# Llama a la función para los dos casos:
calcular_tasas(files_BC, umbrales, canales_variable='BC')
calcular_tasas(files_AD, umbrales, canales_variable='AD')
