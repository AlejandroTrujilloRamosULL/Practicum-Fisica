# -*- coding: utf-8 -*-
"""
Created on Fri May  2 12:13:55 2025

@author: aleja
"""

# Implementamos código prueba para obtener los valores de seeing, escala 
# externa y gamma para formatearlos en una tabla según datos de GTC OSIRIS
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
# uno por uno y se van extrayendo los datos de anchura a media altura 
# (en eje x e y) y se calcula la magnitud total mediante la media cuadrática
def deal_with(fits_filepath):
    with fits.open(fits_filepath) as hdul:
        bin_table = hdul[1]
        data = bin_table.data
        fwhm = np.zeros(len(data)) 
        fwhm_err = np.zeros(len(data))
        for i in range(len(data)):
            fwhm[i] = data[i][0] # mag.total
            fwhm_err[i] = data[i][1]
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

# En folder escribimos la ruta de la carpeta donde tenemos los archivos
# list_of_filespath accede a los archivos que se encuentran dentro de la carpeta
# files_list extrae los datos de fwhm y fwhm_err de los archivos
folder = "C:\\path\\to\\your\\folder\\of\\files"
list_of_filespath = os.listdir(folder)
files_list = [deal_with(os.path.join(folder, filename)) for filename in list_of_filespath]

# Se define totalsortedwave para agrupar todas las longitudes de onda para un 
# valor de lambda determinado de todas las estrellas en un array
# en orden de longitud de onda menor a la mayor
def totalsortedwave(files_list):
    fwhm_totalsortedwave = [np.zeros(len(files_list)) for i in range(len(wave))]
    for j in range(len(wave)):
        for i in range(len(files_list)):
            fwhm_totalsortedwave[j][i] = files_list[i][j][0]
    return fwhm_totalsortedwave

# Escogemos una longitud de referencia para que el parámetro ajustado sea
# el "seeing" epsilon0 con dependencia lambda**(-1/5)
lam_ref = wave[1] # lambda = 3450 A (en este caso)
epsilon_dome = 0.2 # suponemos un dome-seeing de referencia
n_param = 3 # número de parámetros

# Definimos la ecuación predictiva de Tokovinin
def Tokovinin(lam, epsilon0, L0, gamma): 
    epsilon0_lambda = epsilon0*((lam/lam_ref)**(-1/5))
    epsilon_dome_lambda = epsilon_dome*((lam/lam_ref)**((gamma - 4)/(gamma - 2)))
    r0 = (0.976*lam*10**(-10))/(epsilon0_lambda*(np.pi/(180.*3600.)))
    arg = 1 - 2.183*(r0/L0)**(0.356)
    clipped_arg = np.clip(arg, 0, None)
    tok = epsilon0*np.sqrt(clipped_arg)
    return np.sqrt((tok)**2 + (epsilon_dome_lambda)**2)
                   
# Sacamos el valor de los parámetros
param = [np.zeros(n_param) for i in range(len(list_of_filespath))]
param_cov = [np.zeros(n_param*n_param) for i in range(len(list_of_filespath))]
for i in range(len(list_of_filespath)):
    param[i], param_cov[i]  = curve_fit(Tokovinin, wave, files_list[i][0], 
                                  p0=[0.6, 20, 3.5], bounds=([0.3, 2, 3], [5, 1e20, 4]), 
                                  sigma=files_list[i][1])

# Aquí definimos la curva de Tokovinin con el valor de los parámetros óptimos
opt_tokovinin = [np.zeros(len(wave)) for i in range(len(list_of_filespath))]
for i in range(len(list_of_filespath)):
    opt_tokovinin[i] = Tokovinin(wave, param[i][0], param[i][1], param[i][2])

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
    epsilon_0, L_0, gamma = theta
    epsilon_0_lambda = epsilon_0*((x/lam_ref)**(-1/5))
    epsilon_dome_lambda = epsilon_dome*((x/lam_ref)**((gamma - 4)/(gamma - 2)))
    r0 = (0.976*x*10**(-10))/(epsilon_0_lambda*(np.pi/(180*3600)))
    arg = 1 - 2.183*(r0/L_0)**(0.356)
    clipped_arg = np.clip(arg, 0, None)
    tok = epsilon_0*np.sqrt(clipped_arg)
    return np.sqrt((tok)**2 + (epsilon_dome_lambda)**2)

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
    epsilon_0, L_0, gamma = theta
    if 0.3 <= epsilon_0 <= 5 and 2 <= L_0 <= 100 and 3 <= gamma <= 4:
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
initial = np.array([2, 30, 3.5])
ndim = len(initial)
      
# Diferentes perturbaciones para los parámetros
epsilon0_p0 = np.zeros(nwalkers)
L0_p0 = np.zeros(nwalkers)
gamma_p0 = np.zeros(nwalkers)
for i in range(nwalkers):
    epsilon0_p0[i] = initial[0] + 10**(-1)*np.random.randn(1) 
    L0_p0[i] = initial[1] + 6*np.random.randn(1) 
    gamma_p0[i] = initial[2] + 10**(-1)*np.random.randn(1)
    
# Coleccionamos los valores perturbados para cada parámetro en p0
p0 = [np.zeros(ndim) for i in range(nwalkers)]
for j in range(nwalkers):
    p0[j][0] = epsilon0_p0[j]
    p0[j][1] = L0_p0[j]
    p0[j][2] = gamma_p0[j]

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
    
# Coleccionamos los valores de epsilon0, L0 y gamma por separado para cada 
# iteración y cada walker
coll_epsilon0 = [np.zeros(nwalkers*niter) for i in range(len(list_of_filespath))]
coll_L0 = [np.zeros(nwalkers*niter) for i in range(len(list_of_filespath))]
coll_gamma = [np.zeros(nwalkers*niter) for i in range(len(list_of_filespath))]
for i in range(len(list_of_filespath)):
    for j in range(nwalkers*niter):
        coll_epsilon0[i][j] = samples[i][j][0]
        coll_L0[i][j] = samples[i][j][1]
        coll_gamma[i][j] = samples[i][j][2]
        
# Extraemos la media y mediana para cada estrella
median_coll_epsilon0 = [np.zeros(nwalkers*niter) for i in range(len(list_of_filespath))]
median_coll_L0 = [np.zeros(nwalkers*niter) for i in range(len(list_of_filespath))]
median_coll_gamma = [np.zeros(nwalkers*niter) for i in range(len(list_of_filespath))]
for i in range(len(list_of_filespath)):
    median_coll_epsilon0[i] = median(coll_epsilon0[i])
    median_coll_L0[i] = median(coll_L0[i])
    median_coll_gamma[i] = median(coll_gamma[i])
    
print("")
print("Todos los datos coleccionados")
    
# Esta parte del código sirve para crear una tabla archivo.fits
# Usamos los paquetes Columns y BinTableHDU para crear la tabla binaria y el 
# header de la tabla

# Definimos datos de la tabla además de los valores ya coleccionados
# files_names contiene los nombres de los archivos
file_names = list_of_filespath
units = ["arcsec", "metros", "adimensional"]

# Coleccionamos los valores correspondientes para las columnas
seeing_values = [p[0] for p in param]
L0_values = [p[1] for p in param]
gamma_values = [p[2] for p in param]
seeing_error = [p[0] for p in std]
L0_error = [p[1] for p in std]
gamma_error = [p[2] for p in std]
seeing_map = [p[0] for p in best_fit_param]
L0_map = [p[1] for p in best_fit_param]
gamma_map = [p[2] for p in best_fit_param]

# Definimos el formato de las columnas y lo que almacenamos
columns = [fits.Column(name="FILE_NAME", format="200A", array=file_names),
           fits.Column(name="SEEING", format="E", array=seeing_values),
           fits.Column(name="SEEING_ERROR", format="E", array=seeing_error),
           fits.Column(name="EXTERNAL_SCALE", format="E", array=L0_values),
           fits.Column(name="EXTERNAL_SCALE_ERROR", format="E", array=L0_error),
           fits.Column(name="GAMMA", format="E", array=gamma_values),
           fits.Column(name="GAMMA_ERROR", format="E", array=gamma_error),
           fits.Column(name="SEEING_MEDIAN", format="E", array=median_coll_epsilon0),
           fits.Column(name="EXTERNAL_SCALE_MEDIAN", format="E", array=median_coll_L0),
           fits.Column(name="GAMMA_MEDIAN", format="E", array=median_coll_gamma),
           fits.Column(name="SEEING_MAP", format="E", array=seeing_map),
           fits.Column(name="EXTERNAL_SCALE_MAP", format="E", array=L0_map),
           fits.Column(name="GAMMA_MAP", format="E", array=gamma_map)]

# Creamos el header
hdu = fits.BinTableHDU.from_columns(columns)

# Añadimos los datos y definimos el header 
header = hdu.header
header["DATE"]="2025-05-01"
header["SEEING_UNIDADES"]=units[0]
header["ESCALA_EXTERNA_UNIDADES"]=units[1]
header["EXPONENTE_GAMMA_UNIDADES"]=units[2]

# Escribimos el archivo fits y guardamos
hdu.writeto("R1000B_F_SEEING_ESCALA_EXTERNA_GAMMA_ALL_DATA.fits", overwrite=True)
print("TABLA COMPILADA")    