import numpy as np
import random as rd
import matplotlib.pyplot as plt
from numba import jit # librería para acelerar la ejecución de código en Python, especialmente en cálculos numéricos

import os

# Cambiar el directorio de trabajo al del script actual
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Definimos el método Monte Carlo modificado para probar diferentes disparos

@jit(nopython=True)  # JIT es un decorador que optimiza el código


def MC_multiple_triggers(N, L, d, trigger_type):
    # N: número de simulaciones (muones generados)
    # L: tamaño de la placa del detector (cuadrada) (cm)
    # d: distancia entre cada detector (cm)
    # trigger_type: tipo de disparo; decidimos si el disparo inicial se realiza en el detector A, B, C o D

    cuentas = [0, 0, 0, 0] # Inicializamos las cuentas para cada detector

    theta_max = np.deg2rad(81.6) # Aquí metemos el ángulo máximo de cada configuración
    cos_theta_max = np.cos(theta_max)
    print('Theta max: ', theta_max)

    for i in range(N): # Para cada muón simulado:
        # Generamos ángulos phi y theta aleatorios para simular la dirección de los muones
        phi = 2 * np.pi * rd.random() # Dirección azimutal aleatoria

        r = rd.random()
        cos_theta = 1 - r * (1 - cos_theta_max)
        theta = np.arccos(cos_theta) # Ángulo zenital aleatorio pero teniendo en cuenta el ángulo máximo para cada trigger

        # Definimos las posiciones de impacto aleatorias en el primer detector:
        x1 = rd.uniform(0, L) # Coordenada x en el detector 1
        y1 = rd.uniform(0,L) # Coordenada y en el detector 1

        # Calculamos la trayectoria de las partículas en función de la dirección aleatoria
        x2 = x1 - d * np.tan(theta) * np.cos(phi)
        y2 = y1 - d * np.tan(theta) * np.sin(phi)
        
        x3 = x1 - 2 * d * np.tan(theta) * np.cos(phi)
        y3 = y1 - 2 * d * np.tan(theta) * np.sin(phi)
        
        x4 = x1 - 3 * d * np.tan(theta) * np.cos(phi)
        y4 = y1 - 3 * d * np.tan(theta) * np.sin(phi)

        # Creamos una lista de todas las posiciones (x, y) de la partícula
        x = [x1, x2, x3, x4]
        y = [y1, y2, y3, y4]

        # Dependiendo del tipo de trigger, se verifican las condiciones:

        if trigger_type == 1: # Trigger en A
            if 0 < x1 < L and 0 < y1 < L: # Verificamos si pasa por cada detector (coordenadas x e y dentro de los límites)
                cuentas[0] += 1
                if 0 < x2 < L and 0 < y2 < L:
                    cuentas[1] += 1
                if 0 < x3 < L and 0 < y3 < L:
                    cuentas[2] += 1
                if 0 < x4 < L and 0 < y4 < L:
                    cuentas[3] += 1
        elif trigger_type == 2: # Trigger en A y en B
            if 0 < x1 < L and 0 < y1 < L:
                if 0 < x2 < L and 0 < y2 < L:
                    cuentas[0] += 1 # El número de coincidencias en A y B será igual dado el requerimiento
                    cuentas[1] += 1
                    if 0 < x3 < L and 0 < y3 < L:
                        cuentas[2] += 1
                    if 0 < x4 < L and 0 < y4 < L:
                        cuentas[3] += 1
        elif trigger_type == 3: # Trigger en A, en B y en C
            if 0 < x1 < L and 0 < y1 < L:
                if 0 < x2 < L and 0 < y2 < L:
                    if 0 < x3 < L and 0 < y3 < L:
                        cuentas[0] += 1 # El número de coincidencias en A, B y C será igual dado el requerimiento
                        cuentas[1] += 1
                        cuentas[2] += 1
                        if 0 < x4 < L and 0 < y4 <L:
                            cuentas[3] += 1
    return cuentas

# Variables de entrada:

N = 1259740 # Número de muones a simular
L = 20.1 # (cm) Longitud del lado del detector
d = 0.92 # (cm) Distancias entre los detectores
trigger_type = 3 # 1: Coincidencias en A ; 2: Coincidencias en A y B ; 3: Coincidencias en A, B y C

# Llamada a la función de la simulación:
cuentas = MC_multiple_triggers(N, L, d, trigger_type)

# Print de los resultados:
print(f"Resultados para trigger {trigger_type}:")
print(f"Detector A: {cuentas[0]}")
print(f"Detector B: {cuentas[1]}")
print(f"Detector C: {cuentas[2]}")
print(f"Detector D: {cuentas[3]}")