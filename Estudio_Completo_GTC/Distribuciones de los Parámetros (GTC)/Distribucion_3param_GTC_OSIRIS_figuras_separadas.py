# -*- coding: utf-8 -*-
"""
Created on Sat May 10 18:02:57 2025

@author: aleja
"""

# Código para graficar la distribución de parámetros a partir de las tablas
# de valores de seeing, escala externa y gamma

# Importamos paquetes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits

# Definimos el path del archivo, defina el suyo en fits_filespath
fits_filepath = "C:\\path\\to\\your\\file.fits"

# Abrimos el archivo fits con los datos
with fits.open(fits_filepath) as hdul:
    print("Información del FITS")
    hdul.info() # Información del FITS
    primary_hdu = hdul[0] # Header primario del FITS 
    bin_table = hdul[1] # Tabla binaria
    bin_header = bin_table.header # Extraemos el header de hdul[1]
    data = bin_table.data # Extraemos los datos de hdul[1]
    print("")
    print(primary_hdu.header)
    print("")
    print("Aquí está el Header de la Tabla Binaria:")
    print(bin_header)
    print("")
    print("Filas de la Tabla Binaria:")
    print(data)
    
# Definimos arrays para elmacenar los datos de seeing y escala externa
seeing = np.zeros(len(data))
external_scale = np.zeros(len(data))
gamma = np.zeros(len(data))

# Bucle para almacenar los valores
for i in range(len(data)):
    seeing[i] = data[i][1]
    external_scale[i] = data[i][3]
    gamma[i] = data[i][5]
    
# Clipping valores mayores de 100 de la escala externa para el histograma
# y almacenarlos en un bin en específico
external_scale_clipped = np.clip(external_scale, a_min=None, a_max=105)
    
# Cerramos el plot
plt.close("all")    
    
# Creamos los histogramas y los plots para los parámetros, vamos a crear
# una figura para la distribución de cada parámetro
  
# Definimos intervalos (bins) en específico por si queremos usar unos intervalos
# específicos en el plotting
seeing_bins = np.arange(min(seeing), max(seeing), 0.1)
external_scale_bins = np.arange(min(external_scale_clipped), 110, 5)
gamma_bins = np.arange(min(gamma), 4.1, 0.1)

# Definimos los "labels" del eje x para ambos parámetros
seeing_ticks = np.arange(min(seeing) - 0.1, max(seeing) + 0.1, 0.1)
external_scale_ticks = np.arange(0, 105, 5)
gamma_ticks = np.arange(min(gamma), 4.1, 0.1)

# Histrograma con binding específico (plotting) para los 3 parámetros
plt.figure(figsize=(16, 8))
sns.histplot(seeing, bins=seeing_bins, kde=True, color="slateblue", edgecolor="black")
plt.xlabel("Seeing $ϵ_{0}$ (arcsec)", fontsize=14)
plt.xlim(0.5, max(seeing) + 0.1)
plt.xticks(seeing_ticks)
plt.tick_params(labelsize=12)
plt.ylabel("Frecuencia", fontsize=14)
plt.ylim(None, 25)
plt.title("Distribución Seeing", fontsize=16, fontstyle="italic")
plt.grid()
plt.tight_layout()

plt.figure(figsize=(16, 8))
sns.histplot(external_scale_clipped, bins=external_scale_bins, kde=True, color="slateblue", edgecolor="black")
plt.xlabel("Escala Externa $\mathcal{L}_{0}$ (metros)", fontsize=14)
plt.xlim(0, 110)
plt.xticks(external_scale_ticks)
plt.tick_params(labelsize=12)
plt.ylabel("Frecuencia", fontsize=14)
plt.ylim(None, 100)
plt.title("Distribución Escala Externa", fontsize=16, fontstyle="italic")
plt.text(103.5, -2.8, "($L_{0}$ > 100)", fontsize=11)
plt.grid()
plt.tight_layout()

plt.figure(figsize=(16, 8))
sns.histplot(gamma, bins=gamma_bins, kde=True, color="slateblue", edgecolor="black")
plt.xlabel("Exponente Gamma $\gamma$", fontsize=14)
plt.xlim(3, 4)
plt.xticks(gamma_ticks)
plt.tick_params(labelsize=12)
plt.ylabel("Frecuencia", fontsize=14)
plt.title("Distribución Exponente Gamma", fontsize=16, fontstyle="italic")
plt.grid()
plt.tight_layout()