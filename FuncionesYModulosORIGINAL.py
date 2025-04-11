# -*- coding: utf-8 -*-
"""
Created on Thu Apr 03 12:29:27 2025

@author: Andrés
"""

import numpy as np
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import stats as stats
from uncertainties import ufloat
import pandas as pd

#### FUNCIÓN LECTURA ####

def read(file):
    file = open(file,"r")
    triggers = 0  #contador del número de coincidencias
    event_data = [] #lista de matrices con los datos de cada coincidencia (Fila 0 -> tiempo en ns ; Filas 1-4 -> Datos canales)
    t = [] #tiempos absolutos
    time_ns = [] ; cA = [] ; cB = [] ; cC = [] ; cD = [] ; event = []

    for l in file:
        linea = l.strip() #para eliminar espacios en blanco
        if len(linea) == 0:
            continue #si la linea está vacía continuamos (saltos entre coincidencias)
        if len(linea.split()) == 1: # TIEMPO DE REFERENCIA
            if triggers == 0:
                time_init = int(linea) #guardamos la linea como time_init si la linea tiene solo un elemento (tiempo de referencia)
            else: #si no tiene un solo elemento:
                event = np.stack((time_ns, cA, cB, cC, cD))
                event_data.append(event) #almacenamos la información en event_data
                time_ns = [] ; cA = [] ; cB = [] ; cC = [] ; cD = [] ; event = [] #reiniciamos las listas temporales
            triggers = triggers + 1 #aumentamos el contador de coincidencias
            time = int(linea) #convertimos el texto a un entero
            t.append(time) #lista con los tiempos de cada trigger

        else: # LECTURA SEÑALES
            str_dt_ns , str_amplitud_cA, str_amplitud_cB, str_amplitud_cC, str_amplitud_cD = linea.split(" ")
            # dividimos la linea en 5 valores separandolos por espacios y los asignamos a las variables
            #dt_ns es el tiempo transcurrido en ns desde el trigger
            time_ns.append(int(str_dt_ns)) #Convertimos a enteros y guardamos en listas
            cA.append(int(str_amplitud_cA))
            cB.append(int(str_amplitud_cB))
            cC.append(int(str_amplitud_cC))
            cD.append(int(str_amplitud_cD))
    event = np.stack((time_ns, cA, cB, cC, cD)) #organizamos todo en la matriz event (cada fila representa una medida en un instante de tiempo)
    event_data.append(event) #añadimos todas las matrices de eventos a la matriz event_data
    # cada elemento de event_data es una matriz con los datos de un evento
    return event_data, t, time_init, time, triggers

def print_info_archivo(file):
    event_data, t, time_init, time, triggers = read(file)      
    tiempo_s = (time-time_init)/1000000   #Calculamos el tiempo total de la toma de datos y lo pasamos de us a s
    print('Archivo: ', file) #nombre archivo analizado
    print('Tiempo inicial: {} µs'.format(time_init))
    print('Tiempo final: {} µs'.format(time))
    print('Tiempo total en segundos: {:.2f}'.format(tiempo_s))
    print('Número total de coincidencias: {}'.format(triggers))
    print('Número de coincidencias por segundo: {.2f}'.format(triggers/tiempo_s))

#### FUNCIÓN SUPRESIÓN DE CEROS ####

def supresion(event_data, t, supresion): #queremos que nos devuelva la misma matriz pero que elimine -
    # - todas las FILAS con todos los voltajes menores que lo que metamos como supresion
    n = len(event_data)
    event_data_nueva = [] #matriz nueva para guardar los datos post supresión
    indices_valores_suprimidos = [] #guardar índices valores eliminados
    with open("temp.txt", "w") as file:
        for i in range(n): #vamos trigger a trigger
            time_ns = event_data[i][0]
            cA = event_data[i][1]
            cB = event_data[i][2]
            cC = event_data[i][3]
            cD = event_data[i][4]
            indices_valores_suprimidos = []
            for j in range(len(cA)): #suprimimos los ceros
                if (cA[j] < supresion) or (cA[j] < supresion) or (cA[j] < supresion) or (cA[j] < supresion):
                    continue #verificamos para cada valor si es menor que el umbral de supresion
                    # resaltar que nos interesan los valores menores ya que son negativos
                else:
                    indices_valores_suprimidos.append(j) #si son mayores guardamos el índice de esa fila
            time_ns = np.delete(time_ns, indices_valores_suprimidos) #eliminamos los valores con esos índices
            cA = np.delete(cA, indices_valores_suprimidos)
            cB = np.delete(cB, indices_valores_suprimidos)
            cC = np.delete(cC, indices_valores_suprimidos)
            cD = np.delete(cD, indices_valores_suprimidos)
            event = np.stack((time_ns, cA, cB, cC, cD))
            file.write(str(t[i])) #tiempo asociado al trigger i, t (lista con los tiempos de referencia cada trigger)
            file.write('\n') #salto de linea
            np.savetxt(file, np.transpose(event), fmt='%d')
            file.write('\n')
            event_data_nueva.append(event)
    return event_data_nueva

#### FUNCIÓN NUEVO UMBRAL ####

def nuevo_umbral(event_data, t, umbral_A, umbral_B, umbral_C, umbral_D):
    # queremos que devuelva las matrices event_data y t pero eliminando todos los TRIGGERS con valores > umbral
    # umbral_D: None/valor do umbral
    event_data_nueva = [] #para guardar los triggers no eliminados
    t_nuevo = [] # lo mismo para los t no eliminados
    eliminaciones = 0 # contador eliminaciones
    for i in range(len(event_data)): #recorremos todos los triggers
        time_ns = event_data[i][0]
        cA = event_data[i][1]
        cB = event_data[i][2]
        cC = event_data[i][3]
        cD = event_data[i][4]
        for j in range(len(time_ns)):
            if time_ns[j] == 40: # estamos en el punto en el que se ha dado la coincidencia
                if umbral_D == None: # sin aplicar umbral en D
                    if (cA[j] > umbral_A) or (cB[j] > umbral_B) or (cC[j] > umbral_C): # si alguna de las amplitudes supera el umbral
                        eliminaciones += 1 #sumamos 1 al contador de eliminaciones y no la añadimos a la nueva matriz de eventos
                    else:
                        event_data_nueva.append(event_data[i]) # si no se da la condición guardamos el trigger
                        t_nuevo.append(t[i]) # y su tiempo
                else: # aplicando también umbral en D
                    if (cA[j] > umbral_A) or (cB[j] > umbral_B) or (cC[j] > umbral_C) or (cD[j] > umbral_D):
                        eliminaciones += 1
                    else:
                        event_data_nueva.append(event_data[i])
                        t_nuevo.append(t[i])
    porcentaje_eliminaciones = (eliminaciones/len(event_data))*100 # porcentaje de triggers eliminados
    print('El número de coincidencias eliminadas aplicando el nuevo umbral fueron ', eliminaciones, 'de un total de', len(event_data), 'lo que supone un ', porcentaje_eliminaciones, '%')
    return event_data_nueva, t_nuevo

#### FUNCIÓN CÁLCULO MÍNIMOS CON PARÁBOLA ####

def minimo_parabola(x1, y1, x2, y2, x3, y3): # búsqueda del vértice de una parábola definida por 3 puntos
    a = ((y2 - y1)*(x3 - x1) - (y3 - y1)*(x2 - x1)) / ((x1 - x2)*(x3**2 - x1**2) - (x1 - x3))*(x2**2 - x1**2)
    b = ((y2 - y1) - a * (x2**2 - x1**2)) / (x2 - x1)
    c = y1 - a*x1**2 -b*x1
    # Vértice:
    xv = -b / (2*a)
    yv = a * xv**2 + b*xv + c
    return xv, yv   # coordenadas del mínimo

#### FUNCIÓN BÚSQUEDA MÍNIMOS DE AMPLITUD ####

def minimos_amplitud(event_data):
    # queremos los valores minimos de amplitud y sus tiempos para cada uno de los canales

    minA = [] ; minB = [] ; minC= [] ; minD = [] # inicializamos las listas donde guardaremos los valores
    minA_t = [] ; minB_t = [] ; minC_t = [] ; minD_t = []

    for i in range(len(event_data)): # recorremos los triggers
        time_ns = event_data[i][0]
        cA = event_data[i][1]
        cB = event_data[i][2]
        cC = event_data[i][3]
        cD = event_data[i][4]

        for j, canal in enumerate([cA, cB, cC, cD]): # recorremos los 4 canales (j=0 (A), j=1 (B), j=2 (C), j=3 (D))
            indice_min = np.argmin(canal) # para encontral el índice donde la amplitud es minima en cada canal
            # ahora, dependiendo de la posición del valor mínimo:
            if indice_min == 0: # si se da en el primer valor
                tiempo_min, amp_min = time_ns[indice_min], canal[indice_min] # tomamos el valor mínimo en ese punto sin hacer nada más
            elif indice_min == len(canal)-1: # si se da en el último valor
                tiempo_min, amp_min = time_ns[indice_min], canal[indice_min]
            else: # si está en el medio empleamos la función para el cálculo del vértice de la parábola con 3 puntos
                tiempo_min, amp_min = minimo_parabola(time_ns[indice_min-1], canal[indice_min-1], time_ns[indice_min], canal[indice_min], time_ns[indice_min+1], canal[indice_min+1])
            # Almacenamos los valores mínimos según el canal (j)
            if j == 0:
                minA.append(amp_min)
                minA_t.append(tiempo_min)
            elif j == 1:
                minB.append(amp_min)
                minB_t.append(tiempo_min)
            elif j == 2:
                minC.append(amp_min)
                minC_t.append(tiempo_min)
            elif j == 3:
                minD.append(amp_min)
                minD_t.append(tiempo_min)
    minimos_amp = np.stack((minA, minB, minC, minD))  # Cada fila de minimos_amp tendrá las amplitudes mínimas de los 4 canales
    minimos_tiempos = np.stack((minA_t, minB_t, minC_t, minD_t)) # Cada fila de minimos_tiempos tiene los tiempos correspondientes a las amp minimas
    return minimos_amp, minimos_tiempos

#### HISTOGRAMAS DE LAS AMPLITUDES MÍNIMAS ####

def histogramas_amplitudes(files, labels, colors, alphas):
    event_data_amp = []
    for file in files:
        event_data, t, time_init, time, triggers = read(file)
        print('Número de coincidencias', len(event_data))
        event_data_nueva = supresion(event_data, t, -80) # mantenemos amplitudes por debajo de ...
        event_data_nueva_2, t_nuevo_2 = nuevo_umbral(event_data_nueva, t, 0, 0, 0, 0) # umbral_A, umbral_B, umbral_C, umbral_D
        minimos_amp, minimos_tiempos = minimos_amplitud(event_data_nueva_2)
        print ('El número de mínimos es', (len(minimos_amp[0])))
        event_data_amp.append(minimos_amp)
        n_canales = len(event_data_amp[0])

    plt.clf()  # Creamos la figura
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize = (10,8))
    
    ax1.grid(axis = 'y', alpha = 0.75)
    ax2.grid(axis = 'y', alpha = 0.75)
    ax3.grid(axis = 'y', alpha = 0.75)
    ax4.grid(axis = 'y', alpha = 0.75)

    for i in range(len(event_data_amp)):  # Histogramas
        cA = event_data_amp[i][0]  # Canal A
        cB = event_data_amp[i][1]  # Canal B              #### REVISAR ÍNDICES ####
        cC = event_data_amp[i][2]  # Canal C
        cD = event_data_amp[i][3]  # Canal D
        n_canales = 30
        bin_range = (-2000, 0)
        hist1, bins1, _ = ax1.hist(cA, bins = n_canales, range = bin_range, color = colors[i], alpha = alphas[i], rwidth = 1, label = labels[i], edgecolor = 'black')
        hist2, bins2, _ = ax2.hist(cB, bins = n_canales, range = bin_range, color = colors[i], alpha = alphas[i], rwidth = 1, label = labels[i], edgecolor = 'black')
        hist3, bins3, _ = ax3.hist(cC, bins = n_canales, range = bin_range, color = colors[i], alpha = alphas[i], rwidth = 1, label = labels[i], edgecolor = 'black')
        hist4, bins4, _ = ax4.hist(cD, bins = n_canales, range = bin_range, color = colors[i], alpha = alphas[i], rwidth = 1, label = labels[i], edgecolor = 'black')

        bin_centers = (bins1[:-1] + bins1[1:]) / 2
        bin_width = bins1[1] - bins1[0]
        dot_x = bin_centers

        errors1 = np.sqrt(hist1)
        errors2 = np.sqrt(hist2)
        errors3 = np.sqrt(hist3)
        errors4 = np.sqrt(hist4)

    ax1.set_xlabel('Amplitud (mV)')  #### ETIQUETAS ####
    ax1.set_ylabel('Frecuencia')
    ax2.set_xlabel('Amplitud (mV)')
    ax2.set_ylabel('Frecuencia')
    ax3.set_xlabel('Amplitud (mV)')
    ax3.set_ylabel('Frecuencia')
    ax4.set_xlabel('Amplitud (mV)')
    ax4.set_ylabel('Frecuencia')

    ax1.legend(loc='best',fontsize='x-small')   #### LEYENDA ####
    ax2.legend(loc='best',fontsize='x-small')
    ax3.legend(loc='best',fontsize='x-small')
    ax4.legend(loc='best',fontsize='x-small')

    ax1.set_title('Canal A')  #### TÍTULOS ####
    ax2.set_title('Canal B')
    ax3.set_title('Canal C')
    ax4.set_title('Canal D')

    plt.xlim(-2000,0) # límites eje
    plt.savefig('hist_amplitudes.png', dpi=300)
    plt.show()

def tasa(cuentas, t, t0): # t: tiempo final del intervalo de medida ; t0 :tiempo inicial (microsegundos)
    time = (t - t0)/1000000 #pasamos a segundos
    tasa = cuentas/time
    u_cuentas=(np.sqrt(cuentas)) # Asumimos que la incertidumbre es la de una distribucion de Poisson
    u_t=0 # Consideramos incertidumbre nula en el tiempo, será despreciable comparada con la de las cuentas
    u_tasa=np.sqrt((cuentas*u_t/time)**2+(u_cuentas/time)**2) # Propagando
    return tasa,u_tasa # En Hz

def numero_cuentas(t, delta): # Dividimos en intervalos y calculamos el número de cuentas que caen en cada uno
    time = (t[-1] - t[0]) / 1000000 # Tiempo en segundos entre el primer y último evento
    numero_intervalos = int (time/delta) +1 # Número de intervalos de longitud delta que caben en ese tiempo total
    cuentas_por_intervalo = np.zeros(numero_intervalos) # Inicializamos un array de ceros que contendrá el número de cuentas en cada intervalo temporal
    for tiempo_evento in t: # Recorremos los tiempos de evento, calculamos a que intervalo temporal pertenece ese evento y sumamos 1 al contador correspondiente
        indice_intervalo = int((tiempo_evento-t[0])/1e6/delta)
        cuentas_por_intervalo[indice_intervalo] += 1
    return cuentas_por_intervalo

def media_(x, frecuencia): # Calculamos la media ponderada
    frecuencia = list(frecuencia)
    numerador=0 ; denominador=0
    for i in range (len(frecuencia)):
        numerador+=(frecuencia[i]*x[i]) ; denominador+=frecuencia[i]
    return numerador/denominador

def Poisson(n, lamb):
    n = int(n)
    P = ((lamb**n)/math.factorial(n))*np.exp(-lamb)
    return P

def Gauss(n, lamb):
    G = (1/(2*np.pi*lamb)**0.5)*np.exp((-(n-lamb)**2)/(2*lamb))
    return G


def preparar_chi2(x):
    inicio_agrupado = 0
    fin_agrupado = 0
    
    # Agrupación por la izquierda:
    while x[0] < 5: # Repetimos mientras el primer bin tenga menos de 5 cuentas
        inicio_agrupado += 1
        x[1] += x[0] # Sumamos el contenido del primer bin al segundo
        x = np.delete(x, 0) # Se elimina el primer bin
    
    # Agrupación por la derecha:
    while x[len(x)-1] < 5:
        fin_agrupado += 1
        x[len(x)-2] += x[len(x)-1] # Suma el último bin al anterior
        x = np.delete(x, len(x)-1) # Elimina el último bin
    return(x, inicio_agrupado, fin_agrupado)

def tabla_chi(cuentas, frecuencia, nombre, inicio_agrupado, fin_agrupado): # nombre: nombre del archivo .xlsx donde se guardarán los datos
    media = media_(cuentas, frecuencia) # media ponderada

    numero_bins = len(frecuencia) 
    numero_de_cuentas = sum(frecuencia) # total de eventos observados (suma de las frecuencias de todos los bins)

    # Inicializamos listas vacias que iremos llenando

    xiOi = []    # Oi (frecuencia observada) y xi (valor de las cuentas en el bin i)
    Eiv_Poisson = [] # Frecuencias esperadas de la distribución de Poisson para cada bin
    Eiv_Gauss = [] # Frecuencias esperadas de la distribución de Gauss para cada bin

    chiv_Poisson = [] # Lista para almacenar los valores de Chi para Poisson en cada bin
    chiv_Gauss = [] # Lista para almacenar los valores de Chi para Gauss en cada bin

    g_libertad = numero_bins -2

    for i in range(len(frecuencia)):
        xiOi.append(cuentas[i]*frecuencia[i])
        Ei_Poisson = numero_de_cuentas*Poisson(cuentas[i], media) # para cada bin
        Ei_Gauss = numero_de_cuentas*Gauss(cuentas[i], media)
        Eiv_Poisson.append(Ei_Poisson) # las agregamos a las listas
        Eiv_Gauss.append(Ei_Gauss)

    # Reagrupación de Bins con pocas cuentas:

    for i in range (inicio_agrupado): # se elimina el primer bien en cada iteración y sus cuentas y frecuencias se agregan al siguiente
        cuentas = np.delete(cuentas, 0)
        frecuencia[1] += frecuencia[0]
        frecuencia = np.delete(frecuencia, 0)
        Eiv_Poisson[1] += Eiv_Poisson[0]
        Eiv_Poisson = np.delete(Eiv_Poisson, 0)
        xiOi[1] += xiOi[0]
        xiOi = np.delete(xiOi, 0)
        Eiv_Gauss[1] += Eiv_Gauss[0]
        Eiv_Gauss = np.delete(Eiv_Gauss, 0)
    
    for i in range(fin_agrupado): # igual pero para los últimos bins
        cuentas = np.delete(cuentas, len(cuentas)-1)
        xiOi[len(xiOi)-2] += xiOi[len(xiOi)-1]
        xiOi = np.delete(xiOi, len(xiOi)-1)
        frecuencia[len(frecuencia)-2] += frecuencia[len(frecuencia)-1]
        frecuencia = np.delete(frecuencia, len(frecuencia)-1)
        Eiv_Poisson[len(Eiv_Poisson)-2] += Eiv_Poisson[len(Eiv_Poisson)-1]
        Eiv_Poisson = np.delete(Eiv_Poisson, len(Eiv_Poisson)-1)
        Eiv_Gauss[len(Eiv_Gauss)-2] += Eiv_Gauss[len(Eiv_Gauss)-1]
        Eiv_Gauss = np.delete(Eiv_Gauss, len(Eiv_Gauss)-1)
    
    # Calculamos el valor de Chi^2 después de la reagrupación:

    for i in range(len(frecuencia)):
        chi_Poisson = ((frecuencia[i] - Eiv_Poisson[i]) ** 2) / Eiv_Poisson[i]
        chi_Gauss = ((frecuencia[i] - Eiv_Gauss[i]) ** 2) / Eiv_Gauss[i]
        chiv_Poisson.append(chi_Poisson)
        chiv_Gauss.append(chi_Gauss)
        
        #Calculamos para cada bin Chi^2 tanto para Poisson como para Gauss y los agregamos a las listas

    media_post_agrupacion = media_(cuentas, frecuencia)

    data = {'Cuentas': cuentas,
            'Oi': frecuencia,
            'xiOi': xiOi,
            'Ei_Poiss':Eiv_Poisson,
            'Ei_Gauss':Eiv_Gauss,
            'chi_poiss':chiv_Poisson,
            'chi_gauss':chiv_Gauss
            }
    data_frame = pd.DataFrame(data)
    data_frame.to_excel(nombre, index=False)
    chi_Poisson = sum(chiv_Poisson)
    chi_Gauss = sum(chiv_Gauss)
    p_value = 1- stats.chi2.cdf(chi_Poisson, g_libertad) # Usamos la función chi2.cdf de SciPy
    p_value_Gauss = 1 - stats.chi2.cdf(chi_Gauss, g_libertad) # Nos da la prob acumulada de obtener un valor de Chi2 menor que el calculado para los gl

    print('\n')
    print('A media do histograma é', media)
    print('\n')

    print('Valores para la distribución de Poisson')
    print('chi_squared =', chi_Poisson)
    print('Grados de liberdad:',g_libertad)
    print('chi_reducido=',chi_Poisson/g_libertad)
    print('1-alpha = ', p_value)

    print('Valores para la distribución de Gauss')
    print('chi_squared =', chi_Gauss)
    print('Grados de liberdad:',g_libertad)
    print('chi_reducido=',chi_Gauss/g_libertad)
    print('1-alpha = ', p_value_Gauss)

    return g_libertad, chi_Poisson, chi_Gauss, p_value, p_value_Gauss, Eiv_Poisson, Eiv_Gauss

def Exp(r,t):
    E = r*np.exp(-r*t)
    return E

def tabla_chi_exp(x, histograma, media2, nombre): # x vector con los puntos medios de los bins del histograma
    media = media2

    numero_bins = len(histograma)
    numero_de_cuentas = sum(histograma)

    xiOi = []
    Eiv_exp = []
    chiv_exp = []

    g_libertad = numero_bins - 2
    r = 1/media # parámetro de la exponencial

    for i in range(len(x)): # Para cada bin:
        xiOi.append(x[i] * histograma[i])
        Ei_exp = numero_de_cuentas * Exp(r, x[i]) # Valor esperado distribución exponencial
        Eiv_exp.append(Ei_exp) # Lo guardamos en Eiv_exp
        chi_exp = ((histograma[i] - Ei_exp)**2) / Ei_exp # Contribución al estadístico
        chiv_exp.append(chi_exp)
    data = {
        'xi': x,
        'Oi': histograma,
        'xiOi': xiOi,
        'Ei_exp': Eiv_exp,
        'chi_exp': chiv_exp,
    }
    
    df = pd.DataFrame(data)
    df.to_excel(nombre, index=False)  
    chi_exp = sum(chiv_exp) # Sumamos todos los chi (valor total del test)
    xiOi_tot = sum(xiOi) 
    p_value = 1-stats.chi2.cdf(chi_exp, g_libertad) # Calculamos p

    print('La media del histograma es: ', media)
    print('\n')
        
    print('Valores para la distribución exponencial')
    print('chi_squared = ', chi_exp)
    print('Grados de liberdad: ', g_libertad)
    print('chi_reducido= ',chi_exp/g_libertad)
    print('1-alpha = ', p_value)
    
    return g_libertad, chi_exp, p_value, xiOi_tot, Eiv_exp

def eficiencia_intrinseca_detector(minimos_amp_C, total_eventos, umbral):
    """
    minimos_amp_C: lista o array de valores de amplitud mínima en el canal C (varía en función de cual estemos caracterizando)
    Devuelve la eficiencia intrínseca con su incertidumbre.
    """
    eventos_detectados = np.sum(minimos_amp_C <= umbral) # Cuantos de esos eventos fueron detectados también por el detector caracterizado, según un criterio de umbral

    eficiencia = ufloat(eventos_detectados, np.sqrt(eventos_detectados)) / \
                 ufloat(total_eventos, np.sqrt(total_eventos))

    return eficiencia

def flujo_muones(event_data, t, L = ufloat(20.1,0.1)):

    # Calculamos el flujo de muones a partir de los datos de eventos y tiempos

    A = L**2 # Área efectiva en cm**2
    triggers = ufloat(len(event_data), np.sqrt(len(event_data))) # Número de eventos y su error estadístico
    tiempo_s = (t[-1] - t[0]) / 1e6 # Tiempo total en segundos

    tasa = (triggers/(A * tiempo_s)) * 60 # Tasa en cm**-2 min**-1 (J)

    print('Número de cuentas: ', triggers)
    print(f'Diferencia de tiempos (s): {tiempo_s:.6f}')
    print('Área: ', A)
    print('Tasa (cm^{-2} min^{-1}): ', tasa)

    return tasa

def busqueda_segundo_minimo(event_data, canal, tiempo_minimo, amplitud):
    tiempo_electron = []
    tiempo = 0
    n_candidatos = 0
    n_triggers = len(event_data)
    
    for i in range(len(event_data)): # Recorremos los eventos
        nanosegundos = event_data[i][0] # Lista de tiempos (pulsos registrados)
        for j in range(len(nanosegundos)): # Recorremos cada pulso del evento actual
            if nanosegundos[j] > tiempo_minimo: # Filtramos con el tiempo mínimo (solo consideramos pulsos que ocurren más tarde)
                if event_data[i][canal][j] < amplitud: # Buscamos si la amplitud en ese canal es menor que el umbral amplitud
                    tiempo = nanosegundos[j] # Guardamos el tiempo
                    amplitud = event_data[i][canal][j] # Guardamos la amplitud
                    if tiempo > 0:
                        tiempo_electron.append(tiempo) # Si el tiempo es válido lo guardamos
                        n_candidatos = n_candidatos + 1
                    tiempo = 0 # Reset
    return tiempo_electron, n_candidatos

def chi_axuste(hist,axuste,ligaduras):
    diferencia = hist - axuste
    chi2 = np.sum(diferencia**2 / np.sqrt(hist + 1e-6))
    n_bins = len(hist)
    g_l = n_bins - ligaduras
    chi2_r=chi2/g_l
    print('chi_squared =', chi2)
    print('Graos de liberdade:',g_l)
    print('chi_squared reducido=', chi2_r)
    print('Percentil do text X2 = ', 1-stats.chi2.cdf(chi2, g_l))
    return chi2,g_l,chi2_r

def exponencial_tau(x,N,tau):
    return N*np.exp(-x/tau)
