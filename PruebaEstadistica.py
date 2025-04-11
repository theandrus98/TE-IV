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

umbral = -150

event_data, t, time_init, time, triggers = read('Noche2.txt')
event_data = supresion(event_data,t, -50)
event_data, t = nuevo_umbral(event_data, t, umbral, umbral, umbral, None)
plt.rcParams["figure.figsize"] = [8.0, 5.50]

delta = 200 # Tamaño del intervalo temporal en segundos

cuentas_por_intervalo = numero_cuentas(t, delta)

print('\n')
print('Intervalo de tempo', delta)
print('\n')

maximo_cuentas_intervalo = int(max(cuentas_por_intervalo)) # para ajustar los bins
numero_bins = maximo_cuentas_intervalo
histograma, bordes = np.histogram(cuentas_por_intervalo, bins = numero_bins, range=(0, maximo_cuentas_intervalo))
# Histograma de frecuencias (cuantas veces aparece cada número de cuentas en los intervalos)

print('Reagrupamos los bins para que el número de cuentas sea significativo en todos los casos. ')

cuentas = bordes[:-1]
media = media_(cuentas, histograma)
histograma_nuevo, inicio_agrupado, fin_agrupado = preparar_chi2(histograma)

nombre = 'tabla_chi.xlsx'
nombre2 = 'tabla_chi_sinagrupar.xlsx'

g_libertad, chi_Poisson, chi_Gauss, p_value, p_value_Gauss, chiv_Poisson_no, chiv_Gauss_no = tabla_chi(cuentas, histograma, nombre, inicio_agrupado, fin_agrupado)
g_libertad, chi_Poisson, chi_Gauss, p_value, p_value_Gauss, chiv_Poisson, chiv_Gauss = tabla_chi(cuentas, histograma, nombre2, 0, 0)

x = cuentas + 1/2 # ajustamos por que la posición del bin se define en el centro del intervalo
f = interp1d(x, chiv_Poisson, kind = 'cubic') # interpolación cúbica entre los valores de x y los valores de Chi^2
f2 = interp1d(x, chiv_Gauss, kind = 'cubic') # igual para gauss

x_new = np.linspace(x[0], x[-1], 100) # genera 100 puntos equidistantes en el rango de x para una interpolación más suave
y_new = f(x_new) # Calcula los valores interpolados de χ² para Poisson en los puntos x_new
y_new_2 = f2(x_new) # igual para gauss

errors = np.sqrt(histograma) # desviación estándar de las frecuencias observadas en el histograma

plt.figure()
plt.clf()
plt.hist(cuentas_por_intervalo, bins = numero_bins, range=(0, maximo_cuentas_intervalo), alpha = 0.65, rwidth = 1, color = 'lightskyblue', edgecolor = 'black')
plt.errorbar(x, histograma, yerr=errors, fmt='o', color='lightskyblue', linewidth=2,label='Puntos experimentais')
plt.plot(x_new,y_new,'--',color='darkslateblue')
plt.plot(x_new,y_new_2,'--',color='mediumpurple')
plt.plot(x,chiv_Poisson,'o',label='Poisson',color='darkslateblue')
plt.plot(x,chiv_Gauss,'o',label='Gauss',color='mediumpurple')

plt.title('dt ='+str(delta)+'s')
plt.grid(axis='y',alpha=0.75)
plt.xlabel('Número de cuentas')
plt.xticks(bordes)
plt.ylabel('Frecuencia')
plt.legend(fontsize=12,loc='upper right')
plt.savefig('pruebaestadística.png', dpi=300)
plt.show()