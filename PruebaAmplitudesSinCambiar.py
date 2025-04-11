# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 13:42:58 2025

@author: thean
"""
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

files_BC=['Caracterización2y3(800V).txt','Caracterización2y3(850V).txt','Caracterización2y3(900V).txt','Caracterización2y3(950V).txt', 'Caracterización2y3(1000V).txt']
files_BC.reverse()
labels=['HV=800 V', 'HV=850 V','HV=900 V','HV=950 V','HV=1000 V']
labels.reverse()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
alphas=[1,0.8,0.4,0.4, 0.6]
histogramas_amplitudes(files_BC,labels,colors,alphas)

files_AD=['Caracterización1y4(800V).txt','Caracterización1y4(850V).txt','Caracterización1y4(900V).txt','Caracterización1y4(950V).txt', 'Caracterización1y4(1000V).txt']
files_AD.reverse()
labels=['HV=800 V', 'HV=850 V','HV=900 V','HV=950 V','HV=1000 V']
labels.reverse()
colors2 = ['#17becf', '#bcbd22', '#e377c2', '#8c564b', '#7f7f7f']
alphas=[1,0.8,0.4,0.4, 0.6]
histogramas_amplitudes(files_AD,labels,colors2,alphas)