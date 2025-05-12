# -*- coding: utf-8 -*-
"""
Created on Fri May  2 13:03:42 2025

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
    
# Creamos los histogramas y los plots para los parámetros
(fig, (ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(24, 12), dpi=120)
  
# Definimos intervalos (bins) en específico por si queremos usar unos intervalos
# específicos en el plotting
seeing_bins = np.arange(min(seeing), max(seeing), 0.1)
external_scale_bins = np.arange(min(external_scale_clipped), 110, 5)
gamma_bins = np.arange(min(gamma), 4.1, 0.1)

# Definimos los "labels" del eje x para ambos parámetros
seeing_ticks = np.arange(min(seeing), max(seeing), 0.3)
external_scale_ticks = np.arange(0, 105, 5)
gamma_ticks = np.arange(min(gamma), 4.1, 0.1)

# Histrograma con binding específico (plotting) para los 3 parámetros
sns.histplot(seeing, bins=seeing_bins, kde=True, ax=ax1, color="slateblue", edgecolor="black")
ax1.set_xlabel("Seeing $ϵ_{0}$ (arcsec)", fontsize=12)
ax1.set_xticks(seeing_ticks)
ax1.tick_params(labelsize=9)
ax1.set_ylabel("Frecuencia", fontsize=12)
ax1.set_title("Distribución Seeing", fontsize=14)
ax1.grid()

sns.histplot(external_scale_clipped, bins=external_scale_bins, kde=True, ax=ax2, color="slateblue", edgecolor="black")
ax2.set_xlabel("Escala Externa $\mathcal{L}_{0}$ (metros)", fontsize=12)
ax2.set_xticks(external_scale_ticks)
ax2.tick_params(labelsize=9)
ax2.set_ylabel("Frecuencia", fontsize=12)
ax2.set_title("Distribución Escala Externa", fontsize=14)
ax2.text(103.5, -2.8, "($L_{0}$ > 100)", fontsize=9)
ax2.grid()

sns.histplot(gamma, bins=gamma_bins, kde=True, ax=ax3, color="slateblue", edgecolor="black")
ax3.set_xlabel("Exponente Gamma $\gamma$", fontsize=12)
ax3.set_xticks(gamma_ticks)
ax3.tick_params(labelsize=10)
ax3.set_ylabel("Frecuencia", fontsize=12)
ax3.set_title("Distribución Exponente Gamma", fontsize=14)
ax3.grid()

plt.suptitle("Distribución de Parámetros GTC (Seeing + Dome-Seeing)", fontsize=16, y=1.05, color="black", fontstyle="oblique")
plt.tight_layout()