# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 09:16:20 2025

@author: aleja
"""

# Código donde se ajustan los datos de OSIRIS y se hace un estudio mediante 
# Markov Chain Monte Carlo

# Importamos los paquetes
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from scipy.optimize import curve_fit
from astropy.io import fits
from statistics import mean, median
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
        data_full = bin_table.data
        data = data_full
        fwhm = np.zeros(len(data))  
        for i in range(len(data)):
            fwhm[i] = data[i][0] # mag.total
        return fwhm

# Almacenamos la longitud de onda en un array aparte 
# al igual que el error de fwhm
file_path = "C:\\path\\to\\your\\file.fits"
with fits.open(file_path) as hdul:
    bin_table = hdul[1]
    data_full = bin_table.data
    data = data_full
    wave = np.zeros(len(data))
    for j in range(len(data)):
        wave[j] = data[j][2]   
    fwhm_err = np.zeros(len(data))        
    for j in range(len(data)):
        fwhm_err[j] = data[j][1]

# Definimos una lista con todos los archivos de las estrellas del 
# campo HD90177a y le aplicamos la función que nos extraen los datos de cada
# uno mediante la "lista bucle"
list_of_filespath = ["C:\\path\\to\\your\\file.fits"]
files_list = [deal_with(filename) for filename in list_of_filespath]

# Para agrupar todas las columnas (acumular los datos) de una 
# misma longitud de onda
def totalsortedwave(files_list):
    fwhm_totalsortedwave = [np.zeros(len(files_list)) for i in range(len(wave))]
    for j in range(len(wave)):
        for i in range(len(files_list)):
            fwhm_totalsortedwave[j][i] = files_list[i][j]
    return fwhm_totalsortedwave

# Escogemos una longitud de onda de referencia para que el parámetro ajustado 
# sea el "seeing" epsilon0 con dependencia lambda**(-1/5)
lam_ref = wave[1] # lambda = 5123 A (en este caso)
nparam = 2 # definimos el número de parámetros

# Definimos la ecuación predictiva de Tokovinin
def Tokovinin(lam, epsilon0, L0):
    epsilon0_lambda = epsilon0*((lam/lam_ref)**(-1/5))
    r0 = (0.976*lam*10**(-10))/(epsilon0_lambda*(np.pi/(180.*3600.)))
    return epsilon0_lambda*np.sqrt(1 - 2.183*(r0/L0)**(0.356))
                  
# Sacamos el valor de los parámetros de manera iterativa para 
# extraerlos para cada estrella y graficarlo

# Extraemos los parámetros y covarianzas 
param, param_cov = curve_fit(Tokovinin, wave, files_list[0], p0=[0.7, 12],
                             sigma=fwhm_err)

# Cifras significativas de los errores y std
decimals = 3
decimals_err = 3
std = np.sqrt(np.diag(param_cov))

# La parte final será representar los resultados, pero antes definimos la 
# curva de Tokovinin con el valor de los parámetros óptimos
def optimal_tokovinin(param_epsilon0, param_L0):
    return Tokovinin(wave, param_epsilon0, param_L0)

# Aquí definimos e introducimos el algoritmo Markov Chain MonteCarlo para 
# estudiar el sample de valores de los parámetros de "seeing" y "escala externa"
# para la estrella

# Aquí definimos el método de Markov Chain Monte Carlo para conocer la 
# distribución de los parámetros
def Tokovinin_mcmc_model(theta, x):
    epsilon_0, L_0 = theta
    epsilon_0_lambda = epsilon_0*((x/lam_ref)**(-1/5))
    r0 = (0.976*x*10**(-10))/(epsilon_0_lambda*(np.pi/(180*3600)))
    return epsilon_0_lambda*np.sqrt(1 - 2.183*(r0/L_0)**(0.356))

# Definimos las funciones que serán necesarias para implementar el algoritmo
# Markov Chain MonteCarlo

# La función lnlike() que cuantifica cómo de bueno es el modelo para ajustarse
# a los datos para un set de parámetros concretos, suponemos un distribución
# Gaussiana (normal) para los errores
def lnlike(theta, x, y, y_err):
    lnlikeness = -0.5*np.sum(((y - Tokovinin_mcmc_model(theta, x))**2/(y_err)**2) 
                             + np.log(2*np.pi*y_err**2))                         
    return lnlikeness

# La función lnprior() que impone límites a los valores de los parámetros que
# se quieren optimizar (conocemos del ajuste por mínimos cuadrados 
# valores orientativos)
# Si se cumplen los límites la función resulta en 0, si no se cumple, resulta 
# -infinito y se indica al "walker" a iterar de nuevo
def lnprior(theta):
    epsilon_0, L_0 = theta
    if 0.4 <= epsilon_0 <= 0.9 and 5 <= L_0 <= 25:
        return 0.0 
    else:
        return -np.inf
    
# Definimos la última funcion lnprob(), que escoge de lnprior() los distintos 
# resultados 0 o -infinito y compara la finitud de los valores, 
# donde cuando sea cero otorga el resultado de lnlike() y cuando sea -infinito 
# resulta en -infinito
def lnprob(theta, x, y, y_err):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, y_err)

# Definimos ahora para el sampling el número de walkers "nwalkers", de 
# iteraciones "niter" además de la dimensión del espacio de parámetros 
# (2 en este caso), valores iniciales y estos últimos perturbados con un error 
# aleatorio de distribución normal del tamaño de la dimensión del espacio de 
# parámetros

# Esto nos permite tener un conjunto de sets de parámetros de seeing y escala 
# externa perturbados alrededor de los valores optimizados ajustados anteriormente
nwalkers = 300
niter = 2000
initial = np.array([0.7, 12])
ndim = len(initial)

# Diferentes perturbaciones para los parámetros
epsilon0_p0 = np.zeros(nwalkers)
L0_p0 = np.zeros(nwalkers)
#gamma_p0 = np.zeros(nwalkers)
for i in range(nwalkers):
    epsilon0_p0[i] = initial[0] + 10**(-1)*np.random.randn(1) 
    L0_p0[i] = initial[1] + 4*np.random.randn(1) 
    #gamma_p0[i] = initial[2] + 10**(-1)*np.random.randn(1)
    
# Coleccionamos los valores perturbados para cada parámetro en p0
p0 = [np.zeros(ndim) for i in range(nwalkers)]
for j in range(nwalkers):
    p0[j][0] = epsilon0_p0[j]
    p0[j][1] = L0_p0[j]
    #p0[j][2] = gamma_p0[j]

# Importamos el paquete emcee donde ejecutaremos el MonteCarlo y extraeremos
# Importamos el paquete emcee donde ejecutaremos el MonteCarlo y extraeremos
# los resultados de este a partir de los valores medios de fwhm y las desviaciones
# estándar de las fwhm de las 6 estrellas consideradas
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                                args=(wave, files_list[0], fwhm_err))
# Dejamos que itere un poco para calentar el sample y reseteamos
print("Ejecutando burn-in")
pos, prob, state = sampler.run_mcmc(p0, 500)
sampler.reset()
# Ejecutamos el sample final y sacamos la posición actual de los walkers (pos), 
# su probabilidad, y los random states 
print("Ejecutando sample final")
pos, prob, state = sampler.run_mcmc(p0, niter)

# Extraemos los parámetros de mejor ajuste (más probables) para cada estrella
# Para ello, definimos un bucle que extraiga de manera iterativa estos 
# parámetros para cada una de las estrellas
samples = sampler.flatchain
best_fit_param = samples[np.argmax(sampler.flatlnprobability)]

# Creamos un array con un único eje con .flatchain, donde podemos graficar los 
# parámetros más "probables" que interpretamos que son aquellos en el sample 
# que muestran el "mejor ajuste"
               
# Para cerrar automáticamente las figuras y que se actualicen 
plt.close("all")

# Coleccionamos los valores de epsilon0, L0 y gamma por separado para cada 
# iteración y cada walker
coll_epsilon0 = np.zeros(nwalkers*niter)
coll_L0 = np.zeros(nwalkers*niter)
coll_gamma = np.zeros(nwalkers*niter)
for j in range(nwalkers*niter):
    coll_epsilon0[j] = samples[j][0]
    coll_L0[j] = samples[j][1]
mean_coll_epsilon0 = mean(coll_epsilon0)
mean_coll_L0 = mean(coll_L0)

# Podemos también extraer la mediana de los datos
median_coll_epsilon0 = median(coll_epsilon0)
median_coll_L0 = median(coll_L0)

# Printeamos los valores de los parámetros según curve_fit
print("")
print("Valores de los parámetros según curve_fit: ")
print("epsilon0 = " + f"{param[0]}")
print("L0 = " + f"{param[1]}")
# Printeamos los valores de Mejor Ajuste según MCMC de los parámetros
print("")
print("Valores de Mejor Ajuste de los parámetros según MCMC: ")
print("epsilon0 = " + f"{best_fit_param[0]}")
print("L0 = " + f"{best_fit_param[1]}")
# También las medianas de los parámetros
print("")
print("Medianas de los parámetros según MCMC: ")
print("epsilon0_median = " + f"{median(coll_epsilon0)}")
print("L0_median = " + f"{median(coll_L0)}")

# Plotting
plt.plot(wave, files_list[0], "o", color="blue", label="S3")
plt.plot(wave, optimal_tokovinin(param[0], param[1]), 
        color="purple", linewidth=2, label="Predicción Tokovinin $ϵ_{LE}$")
plt.plot(wave, Tokovinin(wave, best_fit_param[0], 
        best_fit_param[1]), "--",
        color="red", linewidth=3, label="MCMC Mejor Ajuste")
#ax1.errorbar(wave, average(files_list), standev(files_list), 
            #linestyle="None", marker="s", markersize=4,
            #color = "black", capsize=3)
plt.xlabel("$\lambda$ ($\AA$)", fontsize="14")
plt.ylabel("FWHM (arcsec)", fontsize="14")
plt.ylim(0.5, 1.5)
plt.legend(loc="upper right")
plt.title("OSIRIS", fontsize="18",)
plt.text(3700, 0.685, r"$ϵ_{0}$ =" + f"{param[0]:.{decimals}} $\pm$ {std[0]:.{decimals_err}}" "\n" 
         r"$\mathcal{L}_{0} = $" + f"{param[1]:.{decimals}} $\pm$ {std[1]:.{decimals_err}}", fontsize="16", 
         bbox=dict(facecolor="white"))
plt.grid()
plt.tight_layout()

# Graficamos mediante el paquete corner.py la distribución de los parámetros 
# para obtener una intuición de su correlación y su incertidumbre, también 
# indicamos los cuantiles (0.16 y 0.84 que nos dan un intervalo de 1 sigma de 
# error, un 68% de confianza de los valores, y 0.5 que indica la mediana)

# Además, añadimos líneas verticales y horizontales en las gráficas que nos 
# indican la media de los valores usados por el sample de epsilon0 y L0
# y tambien los mejores parámetros (más probables)
labels = [r"Seeing ($ϵ_{0}$)", r"Escala Externa ($\mathcal{L}_{0}$)"]
fig_postdistr = corner.corner(samples, show_titles=True, labels=labels,
                              quantiles=[0.16, 0.5, 0.84])
axes = np.array(fig_postdistr.axes).reshape(ndim, ndim)

# Dibujamos líneas verticales para los histogramas (diagonales)
for i in range(ndim):
    ax = axes[i, i]
    ax.axvline(mean_coll_epsilon0, color="blue")
    ax.axvline(mean_coll_L0, color="blue")
    ax.axvline(best_fit_param[0], color="red")
    ax.axvline(best_fit_param[1], color="red")

# Dibujamos líneas horizontales y verticales para el plot de la correlación
# de parámetros
for i in range(ndim):
    for j in range(i):  
        ax = axes[i, j]
        ax.axvline(mean_coll_epsilon0, color="blue")
        ax.axvline(mean_coll_L0, color="blue")
        ax.axhline(mean_coll_epsilon0, color="blue")
        ax.axhline(mean_coll_L0, color="blue")
        ax.plot(mean_coll_epsilon0, mean_coll_L0, "sb")
        ax.axvline(best_fit_param[0], color="red")
        ax.axvline(best_fit_param[1], color="red")
        ax.axhline(best_fit_param[0], color="red")
        ax.axhline(best_fit_param[1], color="red")
        ax.plot(best_fit_param[0], best_fit_param[1], "sr")

# Leyendas del corner plot
legend_elem = [Patch(facecolor="blue", label="Medias de " r"$ϵ_{0}$" 
                     " y " "$\mathcal{L}_{0}$"), 
               Patch(facecolor="red", label="Valores de Mejor Ajuste de " r"$ϵ_{0}$" 
                                    " y " "$\mathcal{L}_{0}$"),
               Line2D([0], [0], color="black", linestyle="--", 
                      label="Cuantiles [16 %, 50 %, 84 %]")]
plt.legend(handles=legend_elem, loc="best")