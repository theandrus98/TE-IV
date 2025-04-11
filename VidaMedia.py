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

files_noches = ['Noche3.txt','Noche2.txt']

umbral=150

event_data_1, t1, time_init1, time1, triggers1 = read(files_noches[0])
event_data_2, t2, time_init2, time2, triggers2 = read(files_noches[1])


event_data_1 = supresion(event_data_1, t1, -50)
event_data_2=supresion(event_data_2, t2, -50)

event_data_1 ,t=nuevo_umbral(event_data_1 ,t1, -umbral,-umbral,-umbral,None)
event_data_2 ,t2=nuevo_umbral(event_data_2, t2, -umbral,-umbral,-umbral,None)


# Prueba con el canal 4
"""
canal = 4
tiempo_minimo = 100
print('\n')
print('Canal: ', canal)
print('Umbral: ', umbral)
print('\n')

tiempo_electron, n_candidatos = busqueda_segundo_minimo(event_data, canal, tiempo_minimo, -umbral)

print('El número de segundos picos durante la primera noche fue de: ', n_candidatos)

tiempo_electron_2, n_candidatos_2 = busqueda_segundo_minimo(event_data_2, canal)"""

tiempo_electron_total = []
tiempo_electron_total_3 = []
tiempo_electron_total_4 = []

canales = [2,3,4]
tiempo_minimo = 100

for i in range(len(canales)):
    print('\n')
    print('Canal: ', canales[i])
    
    tiempo_electron, n_candidatos = busqueda_segundo_minimo(event_data_1, canales[i], tiempo_minimo, -umbral)
    print('El número de segundos picos durante la primera noche fue: ', n_candidatos)
    for j in range(len(tiempo_electron)):
        tiempo_electron_total.append(tiempo_electron[j])
        if i==1: # Si el canal es el 3
            tiempo_electron_total_3.append(tiempo_electron[j])
        if i==2: # Si el canal es el 4
            tiempo_electron_total_4.append(tiempo_electron[j])

    tiempo_electron_2, n_candidatos_2 = busqueda_segundo_minimo(event_data_2, canales[i], tiempo_minimo, -umbral)
    print('El número de segundos picos durante la segunda noche fue: ', n_candidatos_2)
    for k in range(len(tiempo_electron_2)):
        tiempo_electron_total.append(tiempo_electron_2[k])
        if i==1:
            tiempo_electron_total_3.append(tiempo_electron_2[k])
        if i==2:
            tiempo_electron_total_4.append(tiempo_electron_2[k])
    

plt.rcParams["figure.figsize"] = [7.0, 5.50]


# --- Ajuste automático del número de bins ---
N = len(tiempo_electron_total)

if N < 5:
    print("Muy pocos eventos, el histograma puede no ser representativo.")
    
n_bins = max(4, min(20, math.ceil(math.log2(N) + 1)))  # Limita el número de bins entre 4 y 20


lim = max(tiempo_electron_total)
bin_range = (0, lim)

# Histograma de los tiempos
hist, bordes = np.histogram(tiempo_electron_total, bins=n_bins, range=bin_range)
ancho_bins = bordes[1] - bordes[0]
x = bordes[:-1] + ancho_bins / 2  # Centros de los bins

# Desviaciones típicas para el ajuste ponderado
sigma = 1 / np.sqrt(hist)
# Para evitar divisiones por cero en bins vacíos:
sigma[hist == 0] = np.inf  # Así esos puntos no afectan al ajuste

# Ajuste de la exponencial
popt, pcov = curve_fit(exponencial_tau, x, hist, sigma=sigma, bounds=([0, 1000], [250, 4000]))
ajuste = exponencial_tau(x, *popt)

# Cálculo de errores y chi-cuadrado
perr = np.sqrt(np.diag(pcov))
s_N, s_tau = perr
chi2, g_l, chi2_r = chi_axuste(hist, ajuste, 2)

# --- Resultados del ajuste ---
print('\n')
print('n_bins:', n_bins)
print('τ estimado =', popt[1])
print('Error en τ =', s_tau)
print('χ² reducido =', chi2_r)

# --- Gráfico del ajuste exponencial ---
x2 = np.linspace(x[0], x[-1], 100)
plt.grid(alpha=0.5)
plt.hist(tiempo_electron_total, bins=n_bins, range=bin_range, alpha=0.6, rwidth=1, edgecolor='black', color='navy')
plt.plot(x2, exponencial_tau(x2, *popt), '--', color='rebeccapurple', label=r'$N_0 e^{-t/ \tau}$')
plt.xlabel('t (ns)')
plt.ylabel('Frecuencia')
plt.legend()
plt.savefig('vida_media_muon.png', dpi=300)
plt.show()

# --- Comparativa entre canal C y D (detectores intermedios) ---
n_bins_3 = 15
n_bins_4 = 8
rango_comun = (0, 14260)  # Puedes ajustar este valor si cambia el rango de tus datos

plt.hist(tiempo_electron_total_4, bins=n_bins_4, range=rango_comun, alpha=0.9, rwidth=0.85, color='cornflowerblue', label='Detector D')
plt.hist(tiempo_electron_total_3, bins=n_bins_3, range=rango_comun, alpha=0.7, rwidth=0.85, color='lightskyblue', label='Detector C')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('t (ns)')
plt.ylabel('Frecuencia')
plt.legend()
plt.savefig('comparativa_C_y_D.png', dpi=300)
plt.show()