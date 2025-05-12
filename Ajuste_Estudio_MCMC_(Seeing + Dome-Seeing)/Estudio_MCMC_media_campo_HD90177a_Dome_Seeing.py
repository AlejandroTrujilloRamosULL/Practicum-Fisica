# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 13:38:10 2025

@author: aleja
"""

# Implementamos código para obtener resultados sobre la constricción de los 
# parámetros y sus valores óptimos mediante Markov Chain Monte Carlo, este 
# código es para las 4 estrellas que se consideran de HD90177a
# También, en este código consideramos la contribución del seeing de cúpula
# dome-seeing para le media del campo
# AVISO (En las variables donde se definen las rutas de los archivos de donde
# se extraen los datos, cambie la ruta a la suya de su propio archivo de su 
# ordenador, estas variables que aparecen en el código son file_path y 
# list_of_filespath)

# Importamos los paquetes
import numpy as np
import matplotlib.pyplot as plt
import emcee 
import corner
from scipy.optimize import curve_fit
from astropy.io import fits
from statistics import mean, stdev, median
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Definimos una función donde se accede al "path" de los archivos
# uno por uno y se van extrayendo los datos de anchura a media altura 
# (en eje x e y) y la magnitud total
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
# fwhm y longitud de onda 
# (cámbielo para su propia ruta del archivo)
file_path = "C:\\path\\to\\your\\file.fits"
with fits.open(file_path) as hdul:
    bin_table = hdul[1]
    data = bin_table.data
    wave = np.zeros(len(data))
    for j in range(len(data)):
        wave[j] = data[j][10]  

# Definimos una lista con todos los archivos de las estrellas del 
# campo HD90177a y extraemos y almacenamos los datos de fwhm en files_list
# En list_of_filespath se escribe la ruta del conjunto de archivos para el 
# campo (cámbielo a su propia ruta del conjunto de archivos)
list_of_filespath = ["C:\\path\\to\\your\\file.fits",
                     "C:\\path\\to\\your\\file.fits",
                     "C:\\path\\to\\your\\file.fits",
                     "C:\\path\\to\\your\\file.fits"]
files_list = [deal_with(filename) for filename in list_of_filespath]

# Se define totalsortedwave como lista de arrays para agrupar todos los valores
# de fwhm para un valor de longitud de onda concreto en cada array en orden
# de longitud de onda menor a la mayor
# Tenemos una lista de arrays donde hay 44 arrays (uno para cada longitud de 
# onda determinado) y en cada array hay 4 elementos (longitudes de onda de cada
# estrella)
def totalsortedwave(files_list):
    fwhm_totalsortedwave = [np.zeros(len(files_list)) for i in range(len(wave))]
    for j in range(len(wave)):
        for i in range(len(files_list)):
            fwhm_totalsortedwave[j][i] = files_list[i][j]
    return fwhm_totalsortedwave

# Media y desviación estándar de las medidas de fwhm
# de las 4 estrellas consideradas de HD90177a
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

# Escogemos una longitud de referencia para que el parámetro ajustado sea
# el "seeing" epsilon0 con dependencia lambda**(-1/5)
lam_ref = wave[21] # lambda = 7000 A (en este caso)
epsilon_dome = 0.25 # 0.25 arcsec dome-seeing para lambda = 7000 A

# Definimos la ecuación predictiva de Tokovinin, añadiendo ahora la 
# contribución de dome-seeing
def Tokovinin_dome(lam, epsilon0, L0, gamma): 
    epsilon0_lambda = epsilon0*((lam/lam_ref)**(-1/5))
    epsilon_dome_lambda = epsilon_dome*((lam/lam_ref)**((gamma - 4)/(gamma - 2)))
    r0 = (0.976*lam*10**(-10))/(epsilon0_lambda*(np.pi/(180.*3600.)))
    arg = 1 - 2.183*(r0/L0)**(0.356)
    clipped_arg = np.clip(arg, 0, None)
    tok = epsilon0*np.sqrt(clipped_arg)
    return np.sqrt((tok)**2 + (epsilon_dome_lambda)**2)
                   
# Sacamos el valor de los parámetros
param, param_cov  = curve_fit(Tokovinin_dome, wave, average(files_list), 
                              p0=[0.7, 12, 3.5], #bounds=([0.4, 2, 3], [0.85, 20, 4]), 
                              sigma=standev(files_list))

# Aquí definimos la curva de Tokovinin con el valor de los parámetros óptimos
opt_tokovinin = Tokovinin_dome(wave, param[0], param[1], param[2])

# Cifras significativas de los errores y std para graficar después
decimals = 3
decimals_err = 3
std = np.sqrt(np.diag(param_cov)) 

# Aquí definimos el método de Markov Chain Monte Carlo para conocer la 
# distribución de los parámetros, prevenimos soluciones imaginarias con np.clip
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
# Markov Chain Monte Carlo

# La función lnlike() que cuantifica cómo de bueno es el modelo para ajustarse
# a los datos para un set de parámetros concretos, suponemos un distribución
# normal Gaussiana para los errores (tomando logaritmo)
def lnlike(theta, x, y, y_err):
    lnlikeness = -0.5*np.sum(((y - Tokovinin_mcmc_model(theta, x))**2/(y_err)**2) 
                             + np.log(2*np.pi*y_err**2))                         
    return lnlikeness 

# La función lnprior() que impone límites a los valores de los parámetros que
# se quieren optimizar (conocemos del ajuste por mínimos cuadrados no lineal
# valores orientativos)
# Si se cumplen los límites para los parámetros la función resulta en 0, 
# si no se cumple, resulta en -infinito y se indica al "walker" a 
# iterar de nuevo
def lnprior(theta):
    epsilon_0, L_0, gamma = theta
    if 0.4 <= epsilon_0 <= 0.85 and 2 <= L_0 <= 25 and 3 <= gamma <= 4:
        return 0.0 
    else:
        return -np.inf
    
# Definimos la última funcion lnprob(), que escoge de lnprior() los distintos 
# resultados, 0 o -infinito, y compara la finitud de los valores, 
# donde cuando sea cero otorga el resultado de lnlike() y cuando sea -infinito 
# resulta en -infinito
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
niter = 5000
initial = np.array([0.7, 15, 3.5])
ndim = len(initial)
#p0 = [initial + 10**(-2)*np.random.randn(ndim) 
      #for i in range(nwalkers)]

# Diferentes perturbaciones para los parámetros
epsilon0_p0 = np.zeros(nwalkers)
L0_p0 = np.zeros(nwalkers)
gamma_p0 = np.zeros(nwalkers)
for i in range(nwalkers):
    epsilon0_p0[i] = initial[0] + 10**(-1)*np.random.randn(1) 
    L0_p0[i] = initial[1] + 4*np.random.randn(1) 
    gamma_p0[i] = initial[2] + 10**(-1)*np.random.randn(1)
    
# Coleccionamos los valores perturbados para cada parámetro en p0
p0 = [np.zeros(ndim) for i in range(nwalkers)]
for j in range(nwalkers):
    p0[j][0] = epsilon0_p0[j]
    p0[j][1] = L0_p0[j]
    p0[j][2] = gamma_p0[j]

# Importamos el paquete emcee donde ejecutaremos el MonteCarlo y extraeremos
# los resultados de este a partir de los valores medios de fwhm y las desviaciones
# estándar de las fwhm de las 4 estrellas consideradas
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                                args=(wave, average(files_list), standev(files_list)))
# Dejamos que itere un poco para calentar el sampler y reseteamos
print("Ejecutando burn-in")
pos, prob, state = sampler.run_mcmc(p0, 500)
sampler.reset()
# Ejecutamos el sampler final y sacamos la posición actual de los walkers (pos), 
# su probabilidad, y los random states 
print("Ejecutando sample final")
pos, prob, state = sampler.run_mcmc(p0, niter)

# Para cerrar automáticamente las figuras y que se actualicen 
plt.close("all")

# Creamos un array con un único eje con .flatchain de la cadena de valores
# de los parámetros, obteniendo los samples
# También podemos graficar los parámetros más "probables" que interpretamos 
# que son aquellos en el sample que muestran el "mejor ajuste" (best_fit_param)
samples = sampler.flatchain
best_fit_param = samples[np.argmax(sampler.flatlnprobability)]

# Coleccionamos los valores de epsilon0 y L0 por separado para cada iteración
# y cada walker
coll_epsilon0 = np.zeros(nwalkers*niter)
coll_L0 = np.zeros(nwalkers*niter)
coll_gamma = np.zeros(nwalkers*niter)
for j in range(nwalkers*niter):
    coll_epsilon0[j] = samples[j][0]
    coll_L0[j] = samples[j][1]
    coll_gamma[j] = samples[j][2]
mean_coll_epsilon0 = mean(coll_epsilon0)
mean_coll_L0 = mean(coll_L0)
mean_coll_gamma = mean(coll_gamma)

# También coleccionamos los valores de las medianas de los parámetros
median_coll_epsilon0 = median(coll_epsilon0)
median_coll_L0 = median(coll_L0)
median_coll_gamma = median(coll_gamma)

# Printeamos los valores de los parámetros según curve_fit
print("")
print("Valores de los parámetros según curve_fit: ")
print("epsilon0 = " + f"{param[0]}")
print("L0 = " + f"{param[1]}")
print("gamma = " + f"{param[2]}")
# Printeamos los valores de Mejor Ajuste según MCMC de los parámetros
print("")
print("Valores de Mejor Ajuste de los parámetros según MCMC: ")
print("epsilon0 = " + f"{best_fit_param[0]}")
print("L0 = " + f"{best_fit_param[1]}")
print("gamma = " + f"{best_fit_param[2]}")
# También las medianas de los parámetros
print("")
print("Medianas de los parámetros según MCMC: ")
print("epsilon0_median = " + f"{median(coll_epsilon0)}")
print("L0_median = " + f"{median(coll_L0)}")
print("gamma_median = " + f"{median(coll_gamma)}")

#Plotting
plt.plot(wave, files_list[0], "o", color="orange", label="S11")
plt.plot(wave, files_list[1], "o", color="blue", label="S1")
plt.plot(wave, files_list[2], "o", color="green", label="S2")
plt.plot(wave, files_list[3], "o", color="magenta", label="S3")
plt.plot(wave, opt_tokovinin, color="purple", linewidth=2,label="Predicción Tokovinin $(ϵ_{LE})^{2}$ + Dome-Seeing $(ϵ_{dome})^{2}$")
plt.plot(wave, Tokovinin_dome(wave, best_fit_param[0], best_fit_param[1], best_fit_param[2]),
         "--", color="red", linewidth=2, label="MCMC MAP")
plt.errorbar(wave, average(files_list), standev(files_list), 
            linestyle="None", marker="s",
            color = "black", capsize=3)
plt.xlabel("$\lambda$ ($\AA$)", fontsize="14")
plt.ylabel("FWHM (arcsec)", fontsize="14")
plt.legend(loc="upper right")
plt.title("Ajuste HD90177a", fontsize="18")
plt.text(5160, 0.504, r"$ϵ_{0}$ =" + f"{param[0]:.{decimals}} $\pm$ {std[0]:.{decimals_err}}" "\n" 
         r"$\mathcal{L}_{0} = $" + f"{param[1]:.{decimals}} $\pm$ {std[1]:.{decimals_err}}" "\n"
         r"$\gamma = $" + f"{param[2]:.{decimals}} $\pm$ {std[2]:.{decimals_err}}",
         fontsize="16", bbox=dict(facecolor="white"))
plt.grid()
plt.tight_layout()

# Graficamos mediante el paquete corner.py la distribución de los parámetros 
# para obtener una intuición de su correlación y su incertidumbre, también 
# indicamos los cuantiles (0.16 y 0.84 que nos dan un intervalo de 1 sigma de 
# error, un 68% de confianza de los valores, y 0.5 que indica la mediana)

# Además, añadimos líneas verticales y horizontales en las gráficas que nos 
# indican la media de los valores usados por el sample de epsilon0 y L0
# y tambien los mejores parámetros (más probables)
labels = [r"Seeing ($ϵ_{0}$)", r"Escala Externa ($\mathcal{L}_{0}$)", r"Exponente Dome-Seeing ($\gamma$)"]
fig_postdistr = corner.corner(samples, show_titles=True, labels=labels,
                              quantiles=[0.16, 0.5, 0.84])
axes = np.array(fig_postdistr.axes).reshape(ndim, ndim)

# Coleccionamos los valores de epsilon0 y L0 por separado para cada iteración
# y cada walker
coll_epsilon0 = np.zeros(nwalkers*niter)
coll_L0 = np.zeros(nwalkers*niter)
coll_gamma = np.zeros(nwalkers*niter)
for j in range(nwalkers*niter):
    coll_epsilon0[j] = samples[j][0]
    coll_L0[j] = samples[j][1]
    coll_gamma[j] = samples[j][2]
mean_coll_epsilon0 = mean(coll_epsilon0)
mean_coll_L0 = mean(coll_L0)
mean_coll_gamma = mean(coll_gamma)

# Dibujamos líneas verticales para los histogramas (diagonales)
for i in range(ndim):
    ax = axes[i, i]
    ax.axvline(mean_coll_epsilon0, color="blue")
    ax.axvline(mean_coll_L0, color="blue")
    ax.axvline(mean_coll_gamma, color="blue")
    ax.axvline(best_fit_param[0], color="red")
    ax.axvline(best_fit_param[1], color="red")
    ax.axvline(best_fit_param[2], color="red")

# Dibujamos líneas horizontales y verticales para el plot de la correlación
# de parámetros
for i in range(ndim):
    for j in range(i):  
        ax = axes[i, j]
        ax.axvline(mean_coll_epsilon0, color="blue")
        ax.axvline(mean_coll_L0, color="blue")
        ax.axvline(mean_coll_gamma, color="blue")
        ax.axhline(mean_coll_epsilon0, color="blue")
        ax.axhline(mean_coll_L0, color="blue")
        ax.axhline(mean_coll_gamma, color="blue")
        ax.plot(mean_coll_epsilon0, mean_coll_L0, "sb")
        ax.plot(mean_coll_epsilon0, mean_coll_gamma, "sb")
        ax.plot(mean_coll_L0, mean_coll_gamma, "sb")
        ax.axvline(best_fit_param[0], color="red")
        ax.axvline(best_fit_param[1], color="red")
        ax.axvline(best_fit_param[2], color="red")
        ax.axhline(best_fit_param[0], color="red")
        ax.axhline(best_fit_param[1], color="red")
        ax.axhline(best_fit_param[2], color="red")
        ax.plot(best_fit_param[0], best_fit_param[1], "sr")
        ax.plot(best_fit_param[0], best_fit_param[2], "sr")
        ax.plot(best_fit_param[1], best_fit_param[2], "sr")

# Leyendas del corner plot
legend_elem = [Patch(facecolor="blue", label="Medias de " r"$ϵ_{0}$" 
                     " , " "$\mathcal{L}_{0}$" " , " "$\gamma$"), 
               Patch(facecolor="red", label="Valores de Mejor Ajuste de " r"$ϵ_{0}$" 
                     " , " "$\mathcal{L}_{0}$" " , " "$\gamma$"),
               Line2D([0], [0], color="black", linestyle="--", 
                      label="Cuantiles [16 %, 50 %, 84 %]")]
plt.legend(handles=legend_elem, loc="upper left")