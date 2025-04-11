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

event_data, t, time_init, time, triggers = read('Noche3.txt')

umbral = -150

event_data = supresion(event_data, t, -80)
event_data, t = nuevo_umbral(event_data, t, umbral, umbral, umbral, None)

dt = np.diff(t)/1e6
media = np.mean(dt)

#numero_bins = math.ceil(math.log2(len(dt))) + 1 # Usamos Sturges para calcular el número razonable de bins de un histograma
## print('El número de bins calculados por Sturges es: ', numero_bins)

# Número de bins con la regla de Sturges
numero_bins_sturges = math.ceil(math.log2(len(dt)) + 1)
print('Número de bins calculado por Sturges:', numero_bins_sturges)
numero_bins = 30

lim = 30 # Descartamos los tiempos entre eventos que superen 25s
dt2 = []

for i in range(len(dt)): # Calculamos la nueva media con los tiempos filtrados
    if dt[i] < lim:
        dt2.append(dt[i])
media2 = np.mean(dt2)

bin_range = (0, lim) # rango de los valores que representaremos
histograma, bordes = np.histogram(dt2, bins = numero_bins, range = bin_range)

histograma_normalizado = histograma/sum(histograma)

ancho_bins = bordes[1] - bordes[0] # Calculamos el punto medio de cada bin
x = bordes[:-1] + ancho_bins/2

x2 = np.linspace(x[0], x[-1], 100) 

nombre = 'tabla_chi_exp.xlsx'
g_libertad, chi_exp, p_value, xiOi_tot, Eiv_exp = tabla_chi_exp(x, histograma, media2, nombre)

errors = np.sqrt(histograma) # errores estadístico (raiz de Oi)
f = interp1d(x, Eiv_exp, kind = 'cubic')
x_new = np.linspace(x[0], x[-1], 100)
y_new = f(x_new)

print('Media: ', media)

plt.grid(alpha = 0.75)
plt.hist(dt, bins = numero_bins, range = bin_range, alpha = 0.8, color = 'lightskyblue', edgecolor = 'darkslategrey')
plt.errorbar(x, histograma, yerr=errors, fmt = 'o', markersize = 5, color = 'lightskyblue', linewidth = 2, label = 'Puntos experimentales')
plt.plot(x, Eiv_exp,'o', markersize=5, color = 'rebeccapurple',label='Exponencial')
plt.plot(x_new, y_new,'--',color = 'rebeccapurple')
plt.legend(loc='upper right')
plt.xlabel('t(s)')
plt.ylabel('Frecuencia')
plt.savefig('ExponencialPruebaFigura.png', dpi=300)
plt.show()