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

event_data_A, tA, time_initA, timeA, triggersA = read('TriggerA.txt')
event_data_AB, tAB, time_initAB, timeAB, triggersAB = read('TriggerAB.txt')
event_data_ABC, tABC, time_initABC, timeABC, triggersABC = read('TriggerABC.txt')

event_data_A = supresion(event_data_A, tA, -80)
event_data_AB = supresion(event_data_AB, tAB, -80)
event_data_ABC = supresion(event_data_ABC, tABC, -80)

event_data_A, tA = nuevo_umbral(event_data_A, tA, -150, 0, 0, 0)
event_data_AB, tAB = nuevo_umbral(event_data_AB, tAB, -150, -150, 0, 0)
event_data_ABC, tABC = nuevo_umbral(event_data_ABC, tABC, -150, -150, -150, 0)

# Calculamos los flujos pre correcciones:
flujo_A = flujo_muones(event_data_A, tA)
flujo_AB = flujo_muones(event_data_AB, tAB)
flujo_ABC = flujo_muones(event_data_ABC, tABC)

# Añadimos las eficiencias intrínsecas calculadas en el apartado anterior:
ef_i_A = ufloat(0.998, 0.046)
ef_i_B = ufloat(0.992, 0.044)
ef_i_C = ufloat(0.917, 0.044)

# Calculamos las que necesitamos combinar:
ef_i_A = ef_i_A
ef_i_AB = ef_i_A * ef_i_B
ef_i_ABC = ef_i_A * ef_i_B * ef_i_C

# Añadimos las eficiencias geométricas calculadas mediante la simulación MonteCarlo:
ef_g_A = ufloat(0.930, 0.0010)
ef_g_AB = ufloat(0.910, 0.001)
ef_g_ABC = ufloat(0.885, 0.001)

# Calculamos las eficiencias totales para cada tipo de trigger:
ef_t_A = ef_i_A * ef_g_A
ef_t_AB = ef_i_AB * ef_g_AB
ef_t_ABC = ef_i_ABC * ef_g_ABC

print("\n--- EFICIENCIAS ---")
print("Eficiencia total A:", ef_t_A)
print("Eficiencia total AB:", ef_t_AB)
print("Eficiencia total ABC:", ef_t_ABC)

# Flujos corregidos
print("\n--- FLUJOS CORREGIDOS ---")
flujo_A_corr = flujo_A / ef_t_A
flujo_AB_corr = flujo_AB / ef_t_AB
flujo_ABC_corr = flujo_ABC / ef_t_ABC

print("Flujo corregido A:", flujo_A_corr)
print("Flujo corregido AB:", flujo_AB_corr)
print("Flujo corregido ABC:", flujo_ABC_corr)