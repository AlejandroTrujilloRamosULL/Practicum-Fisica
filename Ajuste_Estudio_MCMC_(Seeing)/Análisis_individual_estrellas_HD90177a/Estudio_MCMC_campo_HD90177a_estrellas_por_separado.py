# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 10:03:48 2025

@author: aleja
"""

# Código donde se extraen todos los datos de los archivos del campo,
# en este caso HD90177a y se computa la media y desviación estándar de 
# los datos de las diferentes estrellas del campo
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
file_path = "C:\\path\\to\\your\\file.fits"
with fits.open(file_path) as hdul:
    bin_table = hdul[1]
    data = bin_table.data 
    wave = np.zeros(len(data))
    for j in range(len(data)):
        wave[j] = data[j][10]   

# Definimos una lista con todos los archivos de las estrellas del 
# campo en list_of_filespath y extraemos y almacenamos los valores de fwhm
# en files_list
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

# Media y desviación estándar de las medidas de fwhm
# Calculamos la media y desviación estándar de fwhm de las 4 esrtrellas 
# consideradas para cada longitud de onda
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

# Escogemos una longitud de onda de referencia para que el parámetro ajustado 
# sea el "seeing" epsilon0 con dependencia lambda**(-1/5)
lam_ref = wave[1] # lambda = 5000 A (en este caso)
nparam = 2 # número de parámetros a ajustar

# Definimos la ecuación predictiva de Tokovinin
def Tokovinin(lam, epsilon0, L0):
    epsilon0_lambda = epsilon0*((lam/lam_ref)**(-1/5))
    r0 = (0.976*lam*10**(-10))/(epsilon0_lambda*(np.pi/(180.*3600.)))
    return epsilon0_lambda*np.sqrt(1 - 2.183*(r0/L0)**(0.356))
                  
# Sacamos el valor de los parámetros de manera iterativa para 
# extraerlos para cada estrella y graficarlo
# Definimos primeramente el número de parámetros que serán optimizados 
def param(list_of_filespath, n_param):
    parameters = [np.zeros(n_param) for i in range(len(list_of_filespath))]
    parameters_cov = [np.zeros(len(list_of_filespath)) for i in range(len(list_of_filespath))]
    for i in range(len(list_of_filespath)):
        parameters[i], parameters_cov[i] = curve_fit(Tokovinin, wave, files_list[i], 
                                                    p0=[0.7, 15], #bounds=([0.5, 10], [0.9, 30])
                                                    )
    return parameters, parameters_cov

# Cifras significativas de los errores y std
decimals = 3
decimals_err = 3

# Desviación estándar calculada de manera iterativa para cada estrella
def stand_param(param, n_param):
    std = [np.zeros(1) for i in range(len(list_of_filespath))]
    for j in range(len(list_of_filespath)):
        for i in range(n_param):
            std[j] = np.sqrt(np.diag(param(list_of_filespath, n_param)[1][j])) # square root of the variances (diagonals)
    return std 

# La parte final será representar los resultados, pero antes definimos la 
# curva de Tokovinin con el valor de los parámetros óptimos
def optimal_tokovinin(param_epsilon0, param_L0):
    return Tokovinin(wave, param_epsilon0, param_L0)

# Calculamos la media de los parámetros obtenidos para las estrellas
def col_param(list_of_filespath, n_param):
    col_parameters = [np.zeros(len(list_of_filespath)) for i in range(n_param)]
    for i in range(n_param):
        for j in range(len(list_of_filespath)):
            col_parameters[i][j] = param(list_of_filespath, n_param)[0][j][i]
    return col_parameters

# Aquí definimos e introducimos el algoritmo Markov Chain MonteCarlo para 
# estudiar el sample de valores de los parámetros de "seeing" y "escala externa"
# para cada estrella

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
    if 0.4 <= epsilon_0 <= 0.9 and 10 <= L_0 <= 35:
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
nwalkers = 200
niter = 6000
initial = np.array([0.7, 20])
ndim = len(initial)
      
# Diferentes perturbaciones para los parámetros
epsilon0_p0 = np.zeros(nwalkers)
L0_p0 = np.zeros(nwalkers)
for i in range(nwalkers):
    epsilon0_p0[i] = initial[0] + 10**(-1)*np.random.randn(1) 
    L0_p0[i] = initial[1] + 4*np.random.randn(1) 
    
# Coleccionamos los valores perturbados para cada parámetro en p0
p0 = [np.zeros(ndim) for i in range(nwalkers)]
for j in range(nwalkers):
    p0[j][0] = epsilon0_p0[j]
    p0[j][1] = L0_p0[j]

# Importamos el paquete emcee donde ejecutaremos el MonteCarlo y extraeremos
# los resultados de este
# Iteramos para conseguir un sample único para cada estrella 
pos = [np.zeros(nwalkers) for j in range(len(list_of_filespath))]
prob = [np.zeros(nwalkers) for j in range(len(list_of_filespath))]
state = [np.zeros(5) for j in range(len(list_of_filespath))]
samples = [np.zeros(nwalkers*niter) for j in range(len(list_of_filespath))]
for i in range(len(list_of_filespath)):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                                    args=(wave, files_list[i], standev(files_list)))
    pos0, prob0, state0 = sampler.run_mcmc(p0, niter)
    pos[i], prob[i], state[i] = pos0, prob0, state0 
    samples0 = sampler.flatchain
    samples[i] = samples0
    
# Extraemos los parámetros de mejor ajuste (más probables) para cada estrella
# Para ello, definimos un bucle que extraiga de manera iterativa estos 
# parámetros para cada una de las estrellas
best_fit_param = [np.zeros(2) for j in range(len(list_of_filespath))]
for i in range(len(list_of_filespath)):
    best_fit_param[i] = samples[i][np.argmax(sampler.flatlnprobability)]

# Coleccionamos los valores de epsilon0 y L0 por separado para cada iteración
# y cada walker
coll_epsilon0 = [np.zeros(nwalkers*niter) for i in range(len(list_of_filespath))]
coll_L0 = [np.zeros(nwalkers*niter) for i in range(len(list_of_filespath))]
for i in range(len(list_of_filespath)):
    for j in range(nwalkers*niter):
        coll_epsilon0[i][j] = samples[i][j][0]
        coll_L0[i][j] = samples[i][j][1]

# Calculamos la media de los valores de seeing y escala externa para 
# cada estrella, además de su mediana
mean_coll_epsilon0 = np.zeros(len(list_of_filespath))
mean_coll_L0 = np.zeros(len(list_of_filespath))
median_coll_epsilon0 = np.zeros(len(list_of_filespath))
median_coll_L0 = np.zeros(len(list_of_filespath))
for i in range(len(list_of_filespath)):
    mean_coll_epsilon0[i] = mean(coll_epsilon0[i])
    mean_coll_L0[i] = mean(coll_L0[i])
    median_coll_epsilon0[i] = median(coll_epsilon0[i])
    median_coll_L0[i] = median(coll_L0[i])
    
# Printeamos en la consola los valores de los parámetros según curve_fit
print("")
print("Valores de los parámetros según curve_fit: ")
for i in range(len(list_of_filespath)):
    print("epsilon0 = " + f"{param(list_of_filespath, nparam)[0][i][0]}")
for i in range(len(list_of_filespath)):
    print("L0 = " + f"{param(list_of_filespath, nparam)[0][i][1]}")
# Printeamos también los valores de Mejor Ajuste según MCMC
print("")
print("Valores de los parámetros más probables según MCMC: ")
for i in range(len(list_of_filespath)):
    print("epsilon0 = " + f"{best_fit_param[i][0]}")
for i in range(len(list_of_filespath)):
    print("L0 = " + f"{best_fit_param[i][1]}")
# También añadimos los valores de las medianas de los parámetros
print("")
print("Valores de las medianas de los parámetros: ")
for i in range(len(list_of_filespath)):
    print("epsilon0 = " + f"{median_coll_epsilon0[i]}")
for i in range(len(list_of_filespath)):
    print("L0 = " + f"{median_coll_L0[i]}")
           
# Para cerrar automáticamente las figuras y que se actualicen 
plt.close("all")

# Definimos un entorno con subfiguras donde graficar para las 
# diferentes estrellas
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,10))

# Plotting
ax1.plot(wave, files_list[0], "o", color="orange", label="S11")
ax1.plot(wave, optimal_tokovinin(param(list_of_filespath, nparam)[0][0][0], 
        param(list_of_filespath, nparam)[0][0][1]), color="purple", 
        linewidth=2,label="Predicción Tokovinin $ϵ_{LE}$")
ax1.plot(wave, Tokovinin(wave, best_fit_param[0][0], 
        best_fit_param[0][1]), "--",
        color="red", linewidth=3, label="MCMC MAP")
#ax1.errorbar(wave, average(files_list), standev(files_list), 
            #linestyle="None", marker="s", markersize=4,
            #color = "black", capsize=3)
ax1.set_xlabel("$\lambda$ ($\AA$)", fontsize="14")
ax1.set_ylabel("FWHM (arcsec)", fontsize="14")
ax1.legend(loc="upper right")
ax1.set_title("HD90177a (S11)", fontsize="18",)
ax1.text(4950, 0.51, r"$ϵ_{0}$ =" + f"{param(list_of_filespath, nparam)[0][0][0]:.{decimals}} $\pm$ {stand_param(param, nparam)[0][0]:.{decimals_err}}" "\n" 
         r"$\mathcal{L}_{0} = $" + f"{param(list_of_filespath, nparam)[0][0][1]:.{decimals}} $\pm$ {stand_param(param, nparam)[0][1]:.{decimals_err}}", fontsize="16", 
         bbox=dict(facecolor="white"))
ax1.grid()
plt.tight_layout()

ax2.plot(wave, files_list[1], "o", color="blue", label="S1")
ax2.plot(wave, optimal_tokovinin(param(list_of_filespath, nparam)[0][1][0], 
        param(list_of_filespath, nparam)[0][1][1]), color="purple", 
        linewidth=2,label="Predicción Tokovinin $ϵ_{LE}$")
ax2.plot(wave, Tokovinin(wave, best_fit_param[1][0], 
        best_fit_param[1][1]), "--",
        color="red", linewidth=3, label="MCMC MAP")
#ax2.errorbar(wave, average(files_list), standev(files_list), 
            #linestyle="None", marker="s", markersize=4,
            #color = "black", capsize=3)
ax2.set_xlabel("$\lambda$ ($\AA$)", fontsize="14")
ax2.set_ylabel("FWHM (arcsec)", fontsize="14")
ax2.legend(loc="upper right")
ax2.set_title("HD90177a (S1)", fontsize="18",)
ax2.text(4950, 0.51, r"$ϵ_{0}$ =" + f"{param(list_of_filespath, nparam)[0][1][0]:.{decimals}} $\pm$ {stand_param(param, nparam)[1][0]:.{decimals_err}}" "\n" 
         r"$\mathcal{L}_{0} = $" + f"{param(list_of_filespath, nparam)[0][1][1]:.{decimals}} $\pm$ {stand_param(param, nparam)[1][1]:.{decimals_err}}", fontsize="16", 
         bbox=dict(facecolor="white"))
ax2.grid()
plt.tight_layout()

ax3.plot(wave, files_list[2], "o", color="green", label="S2")
ax3.plot(wave, optimal_tokovinin(param(list_of_filespath, nparam)[0][2][0], 
        param(list_of_filespath, nparam)[0][2][1]), color="purple", 
        linewidth=2,label="Predicción Tokovinin $ϵ_{LE}$")
ax3.plot(wave, Tokovinin(wave, best_fit_param[2][0], 
        best_fit_param[2][1]), "--",
        color="red", linewidth=3, label="MCMC MAP")
#ax3.errorbar(wave, average(files_list), standev(files_list), 
            #linestyle="None", marker="s", markersize=4,
            #color = "black", capsize=3)
ax3.set_xlabel("$\lambda$ ($\AA$)", fontsize="14")
ax3.set_ylabel("FWHM (arcsec)", fontsize="14")
ax3.legend(loc="upper right")
ax3.set_title("HD90177a (S2)", fontsize="18",)
ax3.text(4950, 0.515, r"$ϵ_{0}$ =" + f"{param(list_of_filespath, nparam)[0][2][0]:.{decimals}} $\pm$ {stand_param(param, nparam)[2][0]:.{decimals_err}}" "\n" 
         r"$\mathcal{L}_{0} = $" + f"{param(list_of_filespath, nparam)[0][2][1]:.{decimals}} $\pm$ {stand_param(param, nparam)[2][1]:.{decimals_err}}", fontsize="16", 
         bbox=dict(facecolor="white"))
ax3.grid()
plt.tight_layout()

ax4.plot(wave, files_list[3], "o", color="magenta", label="S3")
ax4.plot(wave, optimal_tokovinin(param(list_of_filespath, nparam)[0][3][0], 
        param(list_of_filespath, nparam)[0][3][1]), color="purple", 
        linewidth=2,label="Predicción Tokovinin $ϵ_{LE}$")
ax4.plot(wave, Tokovinin(wave, best_fit_param[3][0], 
        best_fit_param[3][1]), "--",
        color="red", linewidth=3, label="MCMC MAP")
#ax4.errorbar(wave, average(files_list), standev(files_list), 
            #linestyle="None", marker="s", markersize=4,
            #color = "black", capsize=3)
ax4.set_xlabel("$\lambda$ ($\AA$)", fontsize="14")
ax4.set_ylabel("FWHM (arcsec)", fontsize="14")
ax4.legend(loc="upper right")
ax4.set_title("HD90177a (S3)", fontsize="18",)
ax4.text(4950, 0.54, r"$ϵ_{0} = $" + f"{param(list_of_filespath, nparam)[0][3][0]:.{decimals}} $\pm$ {stand_param(param, nparam)[3][0]:.{decimals_err}}" "\n" 
         r"$\mathcal{L}_{0} = $" + f"{param(list_of_filespath, nparam)[0][3][1]:.{decimals}} $\pm$ {stand_param(param, nparam)[3][1]:.{decimals_err}}", fontsize="16", 
         bbox=dict(facecolor="white"))
ax4.grid()
plt.tight_layout()

# Graficamos mediante el paquete corner.py la distribución de los parámetros 
# para obtener una intuición de su correlación y su incertidumbre, también 
# indicamos los cuantiles (0.16 y 0.84 que nos dan un intervalo de 1 sigma de 
# error, un 68% de confianza de los valores, y 0.5 que indica la mediana)

# Además, añadimos líneas verticales y horizontales en las gráficas que nos 
# indican la media (azul) de los valores usados por el sample de epsilon0 y L0
# y tambien los mejores parámetros (más probables) (rojo)

# Coleccionamos los valores de epsilon0 y L0 por separado para cada iteración
# y cada walker
coll_epsilon0 = [np.zeros(nwalkers*niter) for i in range(len(list_of_filespath))]
coll_L0 = [np.zeros(nwalkers*niter) for i in range(len(list_of_filespath))]
for i in range(len(list_of_filespath)):
    for j in range(nwalkers*niter):
        coll_epsilon0[i][j] = samples[i][j][0]
        coll_L0[i][j] = samples[i][j][1]

# Calculamos la media de los valores de seeing y escala externa para 
# cada estrella
mean_coll_epsilon0 = np.zeros(len(list_of_filespath))
mean_coll_L0 = np.zeros(len(list_of_filespath))
for i in range(len(list_of_filespath)):
    mean_coll_epsilon0[i] = mean(coll_epsilon0[i])
    mean_coll_L0[i] = mean(coll_L0[i])

# Creamos las varias figuras del corner plot mediante iteración
labels = [r"Seeing ($ϵ_{0}$)", r"Escala Externa ($\mathcal{L}_{0}$)"]
for i in range(len(list_of_filespath)):
    cfig = corner.corner(samples[i], show_titles=True, labels=labels,
                              quantiles=[0.16, 0.5, 0.84])
    axes = np.array(cfig.axes).reshape(ndim, ndim)
    legend_elem = [Patch(facecolor="blue", label="Medias de " r"$ϵ_{0}$" 
                         " y " "$\mathcal{L}_{0}$"), 
                   Patch(facecolor="red", label="Valores de Mejor Ajuste de " r"$ϵ_{0}$" 
                                        " y " "$\mathcal{L}_{0}$"),
                   Line2D([0], [0], color="black", linestyle="--", 
                          label="Cuantiles [16 %, 50 %, 84 %]")]
    plt.legend(handles=legend_elem, loc="best")
    for j in range(ndim):
        ax = axes[j, j]
        ax.axvline(mean_coll_epsilon0[i], color="blue")
        ax.axvline(mean_coll_L0[i], color="blue")
        ax.axvline(best_fit_param[i][0], color="red")
        ax.axvline(best_fit_param[i][1], color="red")
    for k in range(ndim):
        for l in range(k):
            ax = axes[k, l]
            ax.axvline(mean_coll_epsilon0[i], color="blue")
            ax.axvline(mean_coll_L0[i], color="blue")
            ax.axhline(mean_coll_epsilon0[i], color="blue")
            ax.axhline(mean_coll_L0[i], color="blue")
            ax.plot(mean_coll_epsilon0[i], mean_coll_L0[i], "sb")
            ax.axvline(best_fit_param[i][0], color="red")
            ax.axvline(best_fit_param[i][1], color="red")
            ax.axhline(best_fit_param[i][0], color="red")
            ax.axhline(best_fit_param[i][1], color="red")
            ax.plot(best_fit_param[i][0], best_fit_param[i][1], "sr") 
