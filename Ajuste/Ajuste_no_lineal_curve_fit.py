# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 18:02:22 2025

@author: aleja
"""

# Código donde se extraen todos los datos de los archivos del campo, en
# este caso HD90177a, y se computa la media y desviación estándar de los datos 
# de las diferentes estrellas del campo
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
from statistics import mean, stdev

# Definimos una función donde se accede al "path" de los archivos
# uno por uno y se van extrayendo los datos de anchura a media altura 
# (en eje x e y) y se calcula la magnitud total mediante la media geométrica
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
file_path = "C:\\path\\to\\your\\file.fits"
with fits.open(file_path) as hdul:
    bin_table = hdul[1]
    data = bin_table.data
    wave = np.zeros(len(data))
    for j in range(len(data)):
        wave[j] = data[j][10]  

# Definimos una lista con todos los archivos de las estrellas del campo
# y le aplicamos la función que nos extraen los datos de cada
# uno mediante una "lista bucle"
# files_list extrae y almacena los datos de fwhm de los archivos
list_of_filespath = ["C:\\path\\to\\your\\file.fits",
                     "C:\\path\\to\\your\\file.fits",
                     "C:\\path\\to\\your\\file.fits",
                     "C:\\path\\to\\your\\file.fits"]
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

# Media y desviación estándar de las medidas de fwhm de las estrellas
# Calculamos la media de fwhm de las estrellas para cada longitud de onda ordenada
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

# Escogemos una longitud de referencia para que el parámetro ajustado sea
# el "seeing" epsilon0 con dependencia lambda**(-1/5)
lam_ref = wave[1] # lambda = 5000 A (en este caso)

# Definimos la ecuación predictiva de Tokovinin con la que se hace el ajuste
def Tokovinin(lam, epsilon0, L0):
    epsilon0_lambda = epsilon0*((lam/lam_ref)**(-1/5))
    r0 = (0.976*lam*10**(-10))/(epsilon0_lambda*(np.pi/(180*3600)))
    return epsilon0_lambda*np.sqrt(1 - 2.183*(r0/L0)**(0.356)) 

# Sacamos el valor de los parámetros
param, param_cov = curve_fit(Tokovinin, wave, average(files_list), 
                             p0=[0.7, 15], sigma=standev(files_list))

# Cifras significativas de los errores y std
decimals = 3
decimals_err = 3
std = np.sqrt(np.diag(param_cov))

# La parte final será representar los resultados, pero antes definimos la 
# curva de Tokovinin con el valor de los parámetros óptimos que se calcularon
# mediante curve_fit
opt_tokovinin = Tokovinin(wave, param[0], param[1])

# Para cerrar automáticamente las figuras y que se actualicen 
plt.close("all")

#Plotting de los cuatro estrellas (cuatro archivos)
plt.plot(wave, files_list[0], "o", color="red", label="S11")
plt.plot(wave, files_list[1], "o", color="blue", label="S1")
plt.plot(wave, files_list[2], "o", color="green", label="S2")
plt.plot(wave, files_list[3], "o", color="magenta", label="S3")
plt.plot(wave, opt_tokovinin, color="purple", linewidth=2, label="Predicción Tokovinin $ϵ_{LE}$")
plt.errorbar(wave, average(files_list), standev(files_list), 
            linestyle="None", marker="s",
            color = "black", capsize=3)
plt.xlabel("$\lambda$ ($\AA$)", fontsize="14")
plt.ylabel("FWHM (arcsec)", fontsize="14")
plt.legend(loc="upper right")
plt.title("Ajuste HD90177a", fontsize="18",)
plt.text(5160, 0.508, r"$ϵ_{0} = $" + f"{param[0]:.{decimals}} $\pm$ {std[0]:.{decimals_err}}" "\n" 
         r"$\mathcal{L}_{0} = $" + f"{param[1]:.{decimals}} $\pm$ {std[1]:.{decimals_err}}", fontsize="16", 
         bbox=dict(facecolor="white"))
plt.grid()
plt.tight_layout()