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

# Rutas de los archivos:
# Tipo 1: Configuración inicial (sin tapar)
files_tipo1 = ['TriggerA.txt', 'TriggerAB.txt', 'TriggerABC.txt']
# Tipo 2: Tapando todo
files_tipo2 = ['tapando_detectores_y_fotomultiplicadores_A.txt', 'tapando_detectores_y_fotomultiplicadores_AB.txt', 'tapando_detectores_y_fotomultiplicadores_ABC.txt']
# Tipo 3: Tapando detectores
files_tipo3=['tapando_detectores_A.txt','tapando_detectores_AB.txt','tapando_detectores_ABC.txt']
# Tipo 4: Tapando fotomultiplicadores
files_tipo4 = ['tapando_fotomultiplicador_A.txt', 'tapando_fotomultiplicador_AB.txt', 'tapando_fotomultiplicador_ABC.txt']

def flujo_muones_por_configuracion(files):
    event_data_A, tA, time_initA, timeA, triggersA = read(files[0])
    event_data_AB, tAB, time_initAB, timeAB, triggersAB = read(files[1])
    event_data_ABC, tABC, time_initABC, timeABC, triggersABC = read(files[2])

    # Supresión de ceros y umbrales:
    event_data_A = supresion(event_data_A, tA, -80)
    event_data_AB = supresion(event_data_AB, tAB, -80)
    event_data_ABC = supresion(event_data_ABC, tABC, -80)

    event_data_A, tA = nuevo_umbral(event_data_A, tA, 0, 0, 0, 0)
    event_data_AB, tAB = nuevo_umbral(event_data_AB, tAB, 0, 0, 0, 0)
    event_data_ABC, tABC = nuevo_umbral(event_data_ABC, tABC, 0, 0, 0, 0)

    # Flujos pre correcciones:
    flujo_A = flujo_muones(event_data_A, tA)
    flujo_AB = flujo_muones(event_data_AB, tAB)
    flujo_ABC = flujo_muones(event_data_ABC, tABC)

    # Eficiencias intrínsecas calculadas en el apartado anterior
    ef_i_A = ufloat(0.996, 0.040)
    ef_i_B = ufloat(0.988, 0.057)
    ef_i_C = ufloat(0.988, 0.057)

    # Calculamos las combinaciones de eficiencias intrínsecas
    ef_i_AB = ef_i_A * ef_i_B
    ef_i_ABC = ef_i_A * ef_i_B * ef_i_C

    # Eficiencias geométricas calculadas mediante la simulación MonteCarlo
    ef_g_A = ufloat(0.3200, 0.0010)
    ef_g_AB = ufloat(0.8720, 0.0015)
    ef_g_ABC = ufloat(0.7300, 0.0012)

    # Eficiencias totales para cada tipo de trigger
    ef_t_A = ef_i_A * ef_g_A
    ef_t_AB = ef_i_AB * ef_g_AB
    ef_t_ABC = ef_i_ABC * ef_g_ABC

    print("\n--- EFICIENCIAS ---")
    print("Eficiencia total A:", ef_t_A)
    print("Eficiencia total AB:", ef_t_AB)
    print("Eficiencia total ABC:", ef_t_ABC)

    # Flujos corregidos:
    print("\n--- FLUJOS CORREGIDOS ---")
    flujo_A_corr = flujo_A / ef_t_A
    flujo_AB_corr = flujo_AB / ef_t_AB
    flujo_ABC_corr = flujo_ABC / ef_t_ABC

    print("Flujo corregido A:", flujo_A_corr)
    print("Flujo corregido AB:", flujo_AB_corr)
    print("Flujo corregido ABC:", flujo_ABC_corr)

# Procesar los archivos de cada tipo de montaje
print("\n--- TIPO 1 (Configuración inicial) ---")
flujo_muones_por_configuracion(files_tipo1)

print("\n--- TIPO 2 (Tapando con una manta todo el montaje) ---")
flujo_muones_por_configuracion(files_tipo2)

print("\n--- TIPO 3 (Solo tapando los detectores) ---")
flujo_muones_por_configuracion(files_tipo3)

print("\n--- TIPO 4 (Solo tapando los fotomultiplicadores) ---")
flujo_muones_por_configuracion(files_tipo4)