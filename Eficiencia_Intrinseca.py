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


event_data, t, time_init, time, triggers = read('Trigger2y4(VemosEl3).txt')
event_data = supresion(event_data, t, -80)
event_data, t = nuevo_umbral(event_data, t, 0, 0, 0, 0) # -150 por defecto en A y B cuando se midió
min_amp, min_t = minimos_amplitud(event_data)

umbral = -120  # Umbral a aplicar al canal a caracterizar
min_amp_C = min_amp[2]  # canal C
total_eventos = len(event_data)

eficiencia = eficiencia_intrinseca_detector(min_amp_C, total_eventos, umbral)
print(f"Eficiencia intrínseca del detector caracterizado: {eficiencia.n:.5f} ± {eficiencia.s:.5f}")
print('Total eventos: ', total_eventos)