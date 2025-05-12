# -*- coding: utf-8 -*-
"""
Created on Thu May  8 14:44:09 2025

@author: aleja
"""

# Implementamos código prueba para obtener los valores de seeing y escala 
# externa y formatearlos en una tabla según datos de GTC OSIRIS
# AVISO 
# (En las variables donde se definen las rutas de los archivos de donde
# se extraen los datos, cambie la ruta a la suya de su propio archivo de su 
# ordenador, estas variables que aparecen en el código son file_path y 
# list_of_filespath)

# Importamos los paquetes
import numpy as np
import emcee 
import os
from scipy.optimize import curve_fit
from astropy.io import fits
from statistics import median
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Definimos una función donde se accede al "path" de los archivos
# uno por uno y se van extrayendo los datos de anchura a media altura y su error
def deal_with(fits_filepath):
    with fits.open(fits_filepath) as hdul:
        bin_table = hdul[1]
        data = bin_table.data
        fwhm = np.zeros(len(data)) 
        fwhm_err = np.zeros(len(data))
        for i in range(len(data)):
            fwhm[i] = data[i][0] # mag.total
            fwhm_err[i] = data[i][1] # error
        return fwhm, fwhm_err
    

# Almacenamos la longitud de onda en un array aparte
# En file_path escriba el "path" de su archivo donde tiene los datos de 
# fwhm y longitud de onda (cámbielo a su propia ruta del archivo)
file_path = "C:\\path\\to\\your\\file.fits"
with fits.open(file_path) as hdul:
    bin_table = hdul[1]
    data = bin_table.data
    wave = np.zeros(len(data))
    for j in range(len(data)):
        wave[j] = data[j][2]  

# Definimos una lista con todos los archivos de las estrellas del 
# campo y extraemos los datos de cada uno 

# En list_of_filespath se escribe la ruta del conjunto de archivos para el 
# campo (cámbielo a su propia ruta del conjunto de archivos)
# Extraemos todos los archivos y sacamos sus datos de fwhm y fwhm_err
folder = "C:\\path\\to\\your\\folder\\of\\files"
list_of_filespath = os.listdir(folder)
files_list = [deal_with(os.path.join(folder, filename)) for filename in list_of_filespath]

# Se define totalsortedwave para agrupar todas las longitudes de onda para un 
# valor de lambda determinado de todas las estrellas en un array,
# en orden de longitud de onda menor a la mayor
# Tenemos una lista de arrays donde hay 44 arrays (uno para cada longitud de 
# onda determinado) y en cada array hay 4 elementos (longitudes de onda de cada
# estrella)
def totalsortedwave(files_list):
    fwhm_totalsortedwave = [np.zeros(len(files_list)) for i in range(len(wave))]
    for j in range(len(wave)):
        for i in range(len(files_list)):
            fwhm_totalsortedwave[j][i] = files_list[i][j][0]
    return fwhm_totalsortedwave

# Escogemos una longitud de referencia para que el parámetro ajustado sea
# el "seeing" epsilon0 con dependencia lambda**(-1/5)
lam_ref = wave[1] # lambda = 5000 A (en este caso)
n_param = 2 #número de parámetros

# Definimos la ecuación predictiva de Tokovinin
def Tokovinin(lam, epsilon0, L0): 
    epsilon0_lambda = epsilon0*((lam/lam_ref)**(-1/5))
    r0 = (0.976*lam*10**(-10))/(epsilon0_lambda*(np.pi/(180.*3600.)))
    arg = 1 - 2.183*(r0/L0)**(0.356)
    clipped_arg = np.clip(arg, 0, None)
    return epsilon0_lambda*np.sqrt(clipped_arg)
                   
# Sacamos el valor de los parámetros
param = [np.zeros(n_param) for i in range(len(list_of_filespath))]
param_cov = [np.zeros(n_param*n_param) for i in range(len(list_of_filespath))]
for i in range(len(list_of_filespath)):
    param[i], param_cov[i]  = curve_fit(Tokovinin, wave, files_list[i][0], 
                                  p0=[2, 20], #bounds=([0.3, 2], [4, 50]), 
                                  sigma=files_list[i][1])

# Cifras significativas de los errores y std para graficar después
decimals = 3
decimals_err = 3

# Bucle para almacenar la desviación estándar de los parámetros
std = [np.zeros(n_param) for i in range(len(list_of_filespath))]
for i in range(len(list_of_filespath)):
    std[i] = np.sqrt(np.diag(param_cov[i])) 
    
print("")
print("Parámetros y errores calculados")

# Aquí definimos el método de Markov Chain Monte Carlo para conocer la 
# distribución de los parámetros
def Tokovinin_mcmc_model(theta, x):
    epsilon_0, L_0 = theta
    epsilon_0_lambda = epsilon_0*((x/lam_ref)**(-1/5))
    r0 = (0.976*x*10**(-10))/(epsilon_0_lambda*(np.pi/(180*3600)))
    arg = 1 - 2.183*(r0/L_0)**(0.356)
    clipped_arg = np.clip(arg, 0, None)
    return epsilon_0_lambda*np.sqrt(clipped_arg)

# Definimos las funciones que serán necesarias para implementar el algoritmo
# Markov Chain MonteCarlo

# La función lnlike() que cuantifica cómo de bueno es el modelo para ajustarse
# a los datos para un set de parámetros concretos, suponemos un distribución
# normal Gaussiana para los errores (tomando logaritmo)
def lnlike(theta, x, y, y_err):
    lnlikeness = -0.5*np.sum(((y - Tokovinin_mcmc_model(theta, x))**2/(y_err)**2) 
                             + np.log(2*np.pi*y_err**2))                         
    return lnlikeness 

# La función lnprior() que impone límites a los valores de los parámetros que
# se quieren optimizar (conocemos del ajuste por mínimos cuadrados 
# valores orientativos)
# Si se cumplen los límites para los parámetros la función resulta en 0, 
# si no se cumple, resulta en -infinito y se indica al "walker" a 
# iterar de nuevo
def lnprior(theta):
    epsilon_0, L_0 = theta
    if 0.3 <= epsilon_0 <= 5 and 2 <= L_0 <= 100:
        return 0.0 
    else:
        return -np.inf
    
# Definimos la última funcion lnprob(), que escoge de lnprior() los distintos 
# resultados, 0 o -infinito, y compara la finitud de los valores, 
# donde cuando sea cero otorga el resultado de lnlike() y cuando sea -infinito 
# resulta en -infinito y el sampler lo descarta
def lnprob(theta, x, y, y_err):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, y_err)

# Definimos ahora para el sampling el número de walkers "nwalkers", además del
# número de iteraciones (niter) y valores iniciales (initial), donde estos 
# últimos son perturbados con un error aleatorio de distribución normal 
# (np.random.randn(ndim)) del tamaño de la dimensión del espacio de parámetros 
# (ndim=2)

# Esto nos permite tener un conjunto de sets de parámetros de seeing y escala 
# externa perturbados alrededor de los valores optimizados ajustados anteriormente
# Cambie "nwalkers", "initial" "niter" a su gusto
nwalkers = 200
niter = 6000
initial = np.array([2, 25])
ndim = len(initial)
#p0 = [initial + 10**(-2)*np.random.randn(ndim) 
      #for i in range(nwalkers)]
      
# Diferentes perturbaciones para los parámetros
epsilon0_p0 = np.zeros(nwalkers)
L0_p0 = np.zeros(nwalkers)
for i in range(nwalkers):
    epsilon0_p0[i] = initial[0] + np.random.randn(1) 
    L0_p0[i] = initial[1] + 10*np.random.randn(1) 
    
# Coleccionamos los valores perturbados para cada parámetro en p0
p0 = [np.zeros(ndim) for i in range(nwalkers)]
for j in range(nwalkers):
    p0[j][0] = epsilon0_p0[j]
    p0[j][1] = L0_p0[j]

# Importamos el paquete emcee donde ejecutaremos el MonteCarlo y extraeremos
# los resultados de este
# Iteramos para conseguir un sample único para cada estrella 
pos = [np.zeros(nwalkers) for i in range(len(list_of_filespath))]
prob = [np.zeros(nwalkers) for i in range(len(list_of_filespath))]
state = [np.zeros(5) for i in range(len(list_of_filespath))]
samples = [np.zeros(nwalkers*niter) for i in range(len(list_of_filespath))]
for i in range(len(list_of_filespath)):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                                    args=(wave, files_list[i][0], files_list[i][1]))
    pos0, prob0, state0 = sampler.run_mcmc(p0, niter)
    pos[i], prob[i], state[i] = pos0, prob0, state0 
    samples0 = sampler.flatchain
    samples[i] = samples0
    
# Extraemos los parámetros de mejor ajuste (más probables) para cada estrella
# Para ello, definimos un bucle que extraiga de manera iterativa estos 
# parámetros para cada una de las estrellas
best_fit_param = [np.zeros(n_param) for i in range(len(list_of_filespath))]
for i in range(len(list_of_filespath)):
    best_fit_param[i] = samples[i][np.argmax(sampler.flatlnprobability)]
    
print("")
print("Samples calculados")
    
# Coleccionamos los valores de epsilon0 y L0 por separado para cada 
# iteración y cada walker
coll_epsilon0 = [np.zeros(nwalkers*niter) for i in range(len(list_of_filespath))]
coll_L0 = [np.zeros(nwalkers*niter) for i in range(len(list_of_filespath))]
coll_gamma = [np.zeros(nwalkers*niter) for i in range(len(list_of_filespath))]
for i in range(len(list_of_filespath)):
    for j in range(nwalkers*niter):
        coll_epsilon0[i][j] = samples[i][j][0]
        coll_L0[i][j] = samples[i][j][1]
        
# Extraemos la media y mediana para cada estrella
median_coll_epsilon0 = [np.zeros(nwalkers*niter) for i in range(len(list_of_filespath))]
median_coll_L0 = [np.zeros(nwalkers*niter) for i in range(len(list_of_filespath))]
for i in range(len(list_of_filespath)):
    median_coll_epsilon0[i] = median(coll_epsilon0[i])
    median_coll_L0[i] = median(coll_L0[i])
    
print("")
print("Todos los datos coleccionados")
    
# Esta parte del código sirve para crear una tabla archivo.fits
# Usamos los paquetes Columns y BinTableHDU para crear la tabla binaria y el 
# header de la tabla

# Definimos datos de la tabla además de los valores ya coleccionados
# file_names tiene los nombrse de los archivos
file_names = list_of_filespath
units = ["arcsec", "metros"]

# Coleccionamos los valores correspondientes para las columnas de las tablas
seeing_values = [p[0] for p in param]
L0_values = [p[1] for p in param]
seeing_error = [p[0] for p in std]
L0_error = [p[1] for p in std]
seeing_map = [p[0] for p in best_fit_param]
L0_map = [p[1] for p in best_fit_param]

# Definimos el formato de las columnas y lo que almacenamos
columns = [fits.Column(name="NOMBRE_ARCHIVO", format="200A", array=file_names),
           fits.Column(name="SEEING", format="E", array=seeing_values),
           fits.Column(name="SEEING_ERROR", format="E", array=seeing_error),
           fits.Column(name="ESCALA_EXTERNA", format="E", array=L0_values),
           fits.Column(name="ESCALA_EXTERNA_ERROR", format="E", array=L0_error),
           fits.Column(name="SEEING_MEDIANA", format="E", array=median_coll_epsilon0),
           fits.Column(name="ESCALA_EXTERNA_MEDIANA", format="E", array=median_coll_L0),
           fits.Column(name="SEEING_MAP", format="E", array=seeing_map),
           fits.Column(name="ESCALA_EXTERNA_MAP", format="E", array=L0_map)]

# Creamos el header
hdu = fits.BinTableHDU.from_columns(columns)

# Añadimos los datos y definimos el header 
header = hdu.header
header["DATE"]="2025-05-01"
header["SEEING_UNIDADES"]=units[0]
header["ESCALA_EXTERNA_UNIDADES"]=units[1]

# Escribimos el archivo fits y guardamos
hdu.writeto("R1000B_E_SEEING_ESCALA_EXTERNA_ALL_DATA.fits", overwrite=True)
print("TABLA TERMINADA")    