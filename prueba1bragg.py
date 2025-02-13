#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:43:19 2025

@author: andres
"""

# Presión: p = 281 mbar

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


# Leemos el documento del Palmtop y definimos el tiempo de la medida

datos=np.loadtxt('prueba txt 281mbar.mca')

t = 385 #s

#Creamos una variable vacía para saber cuantos canales tenemos

canal = len(datos)       #no sería mejor usar len(datos) y ya está?
cs = datos/t #Número de cuentas por segundo
x = np.arange(canal) #array con valores desde 0 hasta canal-1, es decir, índice de cada canal

'''
# Para comprobar la lectura correcta de los datos representamos
# Representamos los datos obtenidos hasta ahora sin tratar

plt.figure()
plt.plot(x, cs, '.', markersize=5)
plt.rc('xtick', labelsize=20) #Tamaño del label
plt.rc('ytick', labelsize=20)
plt.xlabel('Canal',fontsize=30)
plt.ylabel('Cuentas por segundo', fontsize=25)
plt.grid(True)
plt.title('p = 281mbar', fontsize=25)
plt.xlim(0, 8200)
plt.ylim(0,0.25)
'''

### REBBINING ###

# Propósito:
# Los datos de desintegraciones nucleares suelen tener fluctuaciones estadísticas debido a la naturaleza aleatoria del proceso.
# En cada canal del espectrómetro, el número de cuentas puede variar significativamente debido al ruido estadístico.
# Al agrupar los datos en bloques de X canales, reducimos la influencia de las fluctuaciones aleatorias, además de suavizar la representación visualmente.
# Vamos a escoger intervalos con anchura de X canales

i = 0
datos2 = np.zeros(canal) #creamos un array de ceros con la longitud de datos
N = 4
for j in cs:
    if i < canal-N:
        media = (cs[i]+cs[i+1]+cs[i+2]+cs[i+3])
        datos2[i+2] = media
        i += 4
    else:
        break

datos3 = []
x2 = []
for j in range(len(datos2)):  #Recorremos datos2 y guardamos solo los valores distintos de 0 en datos3 (valores de cuentas por segundo después del rebbining)
    if datos2[j] != 0:
        datos3.append(datos2[j])
        x2.append(x[j])  #x2 almacena los valores correspondientes de x de los valores de cuentas por segundo que hemos guardado en datos3

# Graficamos los datos procesados en el Rebbining

plt.figure()
plt.plot(x2, datos3, '.', color='steelblue')
plt.xlabel('Canal')
plt.ylabel('Cuentas por segundo')
plt.title('p = 25mbar')
plt.grid(True)

### AJUSTE ###

x_data = np.array(x2)
y_data = np.array(datos3)

# Seleccionamos un rango específico de canales (5000-6500, por ejemplo) ya que es la región que nos interesa
# De la estadística detrás de la selección de un rango u otro no tengo npi

inicio = np.searchsorted(x_data, 5700, side='left') #np.searchsorted busca el primer índice donde 4800 podría insertarse sin alterar el orden 
final = np.searchsorted(x_data, 6500, side='right') # Ver su función y sintaxis a parte (ChatGPT)

#Básicamente inicio y final nos dan el primer y último canal de interés de la región que nos importa
x_data = x_data[inicio:final] # Selecciona todos los los valores de x_data que están entre el índice inicio y final
y_data = y_data[inicio:final] # Igual para y_data.

bar = np.zeros(len(y_data)) #Creamos un array bar para almacenar las incertidumbres de cada valor de y_data

for i in range(len(bar)):
    bar[i] = np.sqrt(y_data[i]*t)/t #Incertidumbre en cada canal [Raíz(número de cuentas*t) / t]

# Definimos la función que va a describir el modelo no lineal al que ajustaremos:
def ajuste(x, c1, x1, s1, a, b):     #El primer término no es más que un pico Gaussiano (para modelar la distribución de picos en el espectro)
    e1 = c1*np.exp(-(x-x1)**2/(2*s1**2))
    return e1 + a + b*x  # El segundo término es un término lineal que intenta modelar el fondo lineal de ruido (tampoco se la estadística detrás de esta elección)

# Empleamos curve_fit() para realizar el ajuste:

# Creo que al ser un ajuste no lineal tenemos que introducir manualmente valores de los parámetros de los que depende el ajuste que acabamos de definir.
parametros_iniciales = (0.02, 6000, 18, 0.08, -0.00002) # No sé de dd se supone que salen estos valores.
parametros_optimizados, covarianza = curve_fit(ajuste, x_data, y_data, p0=parametros_iniciales, maxfev=100000) #maxfev aumenta el número de iteraciones para asegurar la convergencia
incertidumbre = np.sqrt(np.diag(covarianza)) # Extraemos las incertidumbres de los parámetros a partir de la matriz de covarianza.

# Graficamos el ajuste:

plt.figure()
plt.errorbar(x_data, y_data, yerr=bar, fmt='.', color='steelblue', label='Datos experimentales')
plt.plot(x_data, ajuste(x_data, *parametros_optimizados), color='brown', label='Ajuste no lineal')
plt.xlabel('Canales')
plt.ylabel('Cuentas por segundo')
plt.grid(True)

# Agregar el texto (Valor del canal medio y su incertidumbre)
canal_medio = parametros_optimizados[1]  # El canal medio está en el segundo parámetro de la optimización
incertidumbre = incertidumbre[1]  # Incertidumbre del canal medio

'''
print('Valor del canal medio: ', parametros_optimizados[1], '+/-', incertidumbre[1])
'''

# Para que nos aparezca el valor del canal medio en la propia gráfica en lugar de en un simple print:
plt.text(0.95, 0.95, f'Canal medio: {canal_medio:.2f} ± {incertidumbre:.2f}', # El texto se coloca en una posición específica (por ejemplo, en el punto más alto del gráfico)
         transform=plt.gca().transAxes, ha='right', va='top', fontsize=8, color='black')
plt.show()
