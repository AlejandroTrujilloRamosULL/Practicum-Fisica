# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 09:22:55 2025

@author: aleja
"""

# Implementamos código para hacer un estudio cualitativo de los samples
# y cómo se comportan y dependen gamma y el dome-seeing según el seeing y
# la escala externa

# Este código fue una alternativa para solucionar el problema de la degeneración
# presente en el código y cálculo de valores de referencia del seeingy dome-seeing
# mediante un entendimiento cualitativo de como se comportaban gamma y dome-seeing
# variando los valores de seeing y escala externa, pudiendo así tener una idea
# de qué rangos considerar para los priors y constringir aún más los valores
# para poder intentar solventar el problema de la degeneración

# Importamos paquetes
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from statistics import mean, stdev

# Definimos una función donde se accede al "path" de los archivos
# uno por uno y se van extrayendo los datos de anchura a media altura 
# (en eje x e y) y se calcula la magnitud total mediante la media cuadrática
def deal_with(fits_filepath):
    with fits.open(fits_filepath) as hdul:
        bin_table = hdul[1]
        data = bin_table.data
        fwhm_x = np.zeros(len(data)) 
        fwhm_y = np.zeros(len(data))
        fwhm = np.zeros(len(data))  
        for i in range(len(data)):
            fwhm_x[i] = data[i][6]*0.2 # 0.2 arcsec/pixel
            fwhm_y[i] = data[i][7]*0.2 # 0.2 arcsec/pixel
            fwhm[i] = np.sqrt((fwhm_x[i])*(fwhm_y[i])) # mag.total
        return fwhm

# Almacenamos la longitud de onda en un array aparte 
# En file_path escriba el "path" de su archivo donde tiene los datos de 
# fwhm y longitud de onda (cámbielo a su propia ruta del archivo)
file_path = "C:\\path\\to\\your\\file.fits"
with fits.open(file_path) as hdul:
    bin_table = hdul[1]
    data = bin_table.data
    wave = np.zeros(len(data))
    for j in range(len(data)):
        wave[j] = data[j][10]  

# Definimos una lista con todos los archivos de las estrellas del 
# campo y extraemos y almacenamos los datos de fwhm en files_list

# En list_of_filespath se escribe la ruta del conjunto de archivos para el 
# campo, en este caso usamos 4 estrellas (cámbielo a su propia ruta del conjunto de archivos)
list_of_filespath = ["C:\\path\\to\\your\\files"]
files_list = [deal_with(filename) for filename in list_of_filespath]

# totalsortedwave es un lista de arrays que almacena todos los valores de fwhm
# de una longitud de onda en específico en cada uno de los arrays dedicados
# a almacenar los datos de fwhm a un valor de longitud de onda en concreto
def totalsortedwave(files_list):
    fwhm_totalsortedwave = [np.zeros(len(files_list)) for i in range(len(wave))]
    for j in range(len(wave)):
        for i in range(len(files_list)):
            fwhm_totalsortedwave[j][i] = files_list[i][j]
    return fwhm_totalsortedwave

# Calculamos la media y desviación estándar de fwhm 
# de 4 de las estrellas para cada longitud de onda ordenada
def average(files_list):
    mean_fwhm = np.zeros(len(wave))
    for j in range(len(wave)):
        mean_fwhm[j] = mean(totalsortedwave(files_list)[j]) 
    return mean_fwhm

def standev(files_list):
    stdv_fwhm = np.zeros(len(wave))
    for j in range(len(wave)):
        stdv_fwhm[j] = stdev(totalsortedwave(files_list)[j])
    return stdv_fwhm

# Escogemos nuestra longitud de onda de referencia lam0
lam0 = 7000 #Angstroms

# Definimos el dome-seeing
def dome_seeing(lam, gamma, Cdome):
    dome = Cdome*(lam/lam0)**((gamma - 4)/(gamma - 2))
    return dome

# Vamos a obtener una idea de la relación de los parámetros para distintos 
# valores de seeing y escala externa
size = 100
C0_values = np.random.uniform(0.3, 0.76, size)
L0_values = np.random.uniform(5, 25, size)

# Definimos la ecuación de "dome-seeing", teniendo en cuenta Tokovinin y 
# la ecuación de calidad de imagen, donde supondremos valores de seeing y 
# escala externa fijos para poder graficar la expresión del "dome-seeing" y 
# calcular gamma y Cdome (valor de referencia del "dome-seeing")
def dome_seeing_calc(lam, C0, L0, iq):
    dome_seeing_values = [np.zeros(len(lam)) for i in range(size)]
    for j in range(size):
        for i in range(len(lam)):
            epsilon0 = C0[j]*((lam[i]/lam0)**(-1/5))
            r0 = (0.976*lam[i]*10**(-10))/(epsilon0*(np.pi/(180.*3600.)))
            arg = 1 - 2.183*(r0/L0[j])**(0.356)
            clipped_arg = np.clip(arg, 0, None)
            tok = epsilon0*np.sqrt(clipped_arg)
            dome_seeing_values[j][i] = np.sqrt((iq[i])**2 - (tok)**2)
    return dome_seeing_values

# Extraemos los valores de dome-seeing
dome_values = [np.zeros(len(wave)) for i in range(size)]
for i in range(len(wave)):
    for j in range(size):
        dome_values[j][i] = dome_seeing_calc(wave, C0_values, L0_values, average(files_list))[j][i]
    
# Almacenamos los distintos valores de gamma y dome-seeing
params = [np.zeros(2) for i in range(size)]
params_cov = [np.zeros(4) for i in range(size)]
for i in range(size):
    params[i], params_cov[i] = curve_fit(dome_seeing, wave, dome_values[i],
                                         bounds=([2.1, 0.1], [6, 0.8]))

# Almacenamos los valores de gamma y del seeing
gamma_values = np.zeros(size)
Cdome_values = np.zeros(size)
for i in range(size):
    gamma_values[i] = params[i][0]
    Cdome_values[i] = params[i][1]
    
# Plotting
# Creamos un mapa de parámetros donde obtenderemos una intuición de los 
# valores de los parámetros para distintos valores de L0 y seeing, de manera que
# consigamos discernir un área con significado físico y valores razonables 
# para ambos parámetros y ayudar a discutir los resultados obtenidos mediante MCMC
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

# Scatter plots
sc1 = ax[0].scatter(C0_values, L0_values, c=gamma_values, cmap="viridis")
fig.colorbar(sc1, ax=ax[0])
ax[0].set_ylabel("$\mathcal{L}_{0}$ (m)")
ax[0].set_xlabel("$ϵ_{0}$ (arcsec)")
ax[0].set_title("Exponente  $\gamma$")
ax[0].grid()

sc2 = ax[1].scatter(C0_values, L0_values, c=Cdome_values, cmap="viridis")
fig.colorbar(sc2, ax=ax[1])
ax[1].set_ylabel("$\mathcal{L}_{0}$ (m)")
ax[1].set_xlabel("$ϵ_{0}$ (arcsec)")
ax[1].set_title("$ϵ_{dome}$ (arcsec)")
ax[1].grid()

plt.tight_layout()
plt.show()

 

