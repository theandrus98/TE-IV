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

# Leemos
event_data, t, time_init, time, triggers = read('Trigger12y3(VemosEl4).txt')
event_data = supresion(event_data, t, -80)
event_data, t = nuevo_umbral(event_data, t, 0, 0, 0, 0) # -150 por defecto en A, B y C cuando se midió
min_amp, min_t = minimos_amplitud(event_data)

umbral = -250 # Umbral a aplicar al canal a caracterizar
min_amp_D = min_amp[3]  # AHORA CAMBIA. El detector D es el que vamos a caracterizar
total_eventos = len(event_data)

eficiencia = eficiencia_intrinseca_detector(min_amp_D, total_eventos, umbral)
print(f"Eficiencia intrínseca del detector caracterizado: {eficiencia.n:.5f} ± {eficiencia.s:.5f}")
print('Total eventos: ', total_eventos)