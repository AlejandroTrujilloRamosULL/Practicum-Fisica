# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 11:27:11 2025

@author: aleja
"""

# Implementamos código para obtener resultados sobre la constricción de los 
# parámetros y sus valores óptimos mediante Markov Chain Monte Carlo, este 
# código es para las 4 estrellas que se consideran de HD90177a.
# Este código se centra principalmente en obtener una idea de la constricción
# de los valores de seeing y dome-seeing de referencia para medidas y 
# observaciones propias, sin tener que depender de valores de referencia 
# externos. Realización de ajuste y posterior análisis de los parámetros 
# mediante MCMC.
# AVISO 
# (En las variables donde se definen las rutas de los archivos de donde
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
# campo, en este caso usamos 4 estrellas (cámbielo a su propia ruta 
# del conjunto de archivos)
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

# Definimos un lambda de referencia para calcular nuestro valor de seeing y 
# dome-seeing para un valor concreto de longitud de onda a partir de nuestros 
# datos y así no tener que depender de valores de referencia de otros lugares 
# de observación
lam_ref = 7000 # Angstroms

# Definimos la ecuación predictiva de Tokovinin, donde mediante np.clip() 
# despreciamos los valores que resulten en un número imaginario debido
# a ser el argumento negativo dentro de la raíz cuadrada, considerando
# la contribución del dome-seeing
def Tokovinin(lam, C0, L0, gamma, Cdome): 
    epsilon0 = C0*((lam/lam_ref)**(-1/5))
    r0 = (0.976*lam*10**(-10))/(epsilon0*(np.pi/(180.*3600.)))
    epsilon_dome = Cdome*((lam/lam_ref)**((gamma - 4)/(gamma - 2)))
    arg = 1 - 2.183*(r0/L0)**(0.356)
    clipped_arg = np.clip(arg, 0, None)
    tok = epsilon0*np.sqrt(clipped_arg)
    return np.sqrt((tok)**2 + (epsilon_dome)**2)
                   
# Sacamos el valor de los parámetros
param, param_cov  = curve_fit(Tokovinin, wave, average(files_list), 
                              p0=[0.6, 10, 3.5, 0.3], #bounds=([0.3, 10, 3, 0.2], [0.9, 25, 4, 0.6]), 
                              sigma=standev(files_list))

# Aquí definimos la curva de Tokovinin con el valor de los parámetros óptimos
opt_tokovinin = Tokovinin(wave, param[0], param[1], param[2], param[3])

# Cifras significativas de los errores y std para graficar después
decimals = 3
decimals_err = 3
std = np.sqrt(np.diag(param_cov)) 

# Definimos función de "seeing" y de "dome-seeing" para graficar las curvas

#Seeing
def epsilon0(lam, C0_param):
    return C0_param*((lam/lam_ref)**(-1/5))

# Dome-seeing
def epsilon_dome(lam, gamma_param, Cdome_param):
    alpha = (gamma_param - 4)/(gamma_param - 2)
    return Cdome_param*((lam/lam_ref)**(alpha))

# Definimos aquí un método de Markov Chain Monte Carlo para poder estudiar 
# la relación y convergencia de los parámetros 

# Aquí definimos el método de Markov Chain Monte Carlo para conocer la 
# distribución de los parámetros
def Tokovinin_mcmc_model(theta, x):
    C0, L_0, gamma, Cdome = theta
    epsilon_0_lambda = C0*((x/lam_ref)**(-1/5))
    epsilon_dome_lambda = Cdome*((x/lam_ref)**((gamma - 4)/(gamma - 2)))
    r0 = (0.976*x*10**(-10))/(epsilon_0_lambda*(np.pi/(180*3600)))
    arg = 1 - 2.183*(r0/L_0)**(0.356)
    clipped_arg = np.clip(arg, 0, None)
    tok = (epsilon_0_lambda*np.sqrt(clipped_arg))
    return np.sqrt((tok)**2 + (epsilon_dome_lambda)**2)

# Definimos las funciones que serán necesarias para implementar el algoritmo
# Markov Chain Monte Carlo

# La función lnlike() que cuantifica cómo de bueno es el modelo para ajustarse
# a los datos para un set de parámetros concretos, suponemos una distribución
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
    C0, L_0, gamma, Cdome = theta
    if 0.3 <= C0 <= 0.85 and 2 <= L_0 <= 25 and 3 <= gamma <= 4 and 0.15 <= Cdome <= 0.55:
        return 0.0 
    else:
        return -np.inf

# Definimos la última funcion lnprob(), que escoge de lnprior() los distintos 
# resultados, 0 o -infinito, y compara la finitud de los valores, 
# donde cuando sea cero otorga el resultado de lnlike() y cuando sea -infinito 
# resulta en -infinito y el samnpler lo descarta
def lnprob(theta, x, y, y_err):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, y_err)

# Definimos ahora para el sampling el número de walkers "nwalkers", además del
# número de iteraciones (niter) y valores iniciales (initial), donde estos 
# últimos son perturbados con un error aleatorio de distribución normal 
# (np.random.randn(ndim)) del tamaño de la dimensión del espacio de parámetros 
# (ndim) (ndim=4 en este caso)

# Esto nos permite tener un conjunto de sets de parámetros perturbados 
# alrededor de los valores optimizados ajustados anteriormente
# Cambie "nwalkers", "initial" y "niter" a su gusto
nwalkers = 200
niter = 50000
initial = np.array([0.6, 13, 3.6, 0.3])
ndim = len(initial)

# Diferentes perturbaciones para los parámetros
C0_p0 = np.zeros(nwalkers)
L0_p0 = np.zeros(nwalkers)
gamma_p0 = np.zeros(nwalkers)
Cdome_p0 = np.zeros(nwalkers)
for i in range(nwalkers):
    C0_p0[i] = initial[0] + 10**(-1)*np.random.randn(1) 
    L0_p0[i] = initial[1] + 4*np.random.randn(1) 
    gamma_p0[i] = initial[2] + 10**(-1)*np.random.randn(1)
    Cdome_p0[i] = initial[3] + 10**(-1)*np.random.randn(1)

# Coleccionamos los valores perturbados para cada parámetro en p0
p0 = [np.zeros(ndim) for i in range(nwalkers)]
for j in range(nwalkers):
    p0[j][0] = C0_p0[j]
    p0[j][1] = L0_p0[j]
    p0[j][2] = gamma_p0[j]
    p0[j][3] = Cdome_p0[j]

# Importamos el paquete emcee donde ejecutaremos el MonteCarlo y extraeremos
# los resultados de este a partir de los valores medios de fwhm y las desviaciones
# estándar de las fwhm de las 4 estrellas consideradas
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                                args=(wave, average(files_list), standev(files_list)))
# Dejamos que itere un poco para calentar el sample y reseteamos
print("")
print("Ejecutando burn-in")
pos, prob, state = sampler.run_mcmc(p0, 500)
sampler.reset()
# Ejecutamos el sample final y sacamos la posición actual de los walkers (pos), 
# su probabilidad (prob), y los random states (state) 
print("Ejecutando sample final")
pos, prob, state = sampler.run_mcmc(p0, niter)

# Creamos un array con un único eje con .flatchain de la cadena de valores
# de los parámetros, obteniendo los samples
# También podemos graficar los parámetros más "probables" que interpretamos 
# que son aquellos en el sample que muestran el "mejor ajuste" (best_fit_param)
samples = sampler.flatchain
best_fit_param = samples[np.argmax(sampler.flatlnprobability)]

# Extraemos los valores de los parámetros por separado del sample para cada 
# iteración y cada walker
coll_C0 = np.zeros(nwalkers*niter)
coll_L0 = np.zeros(nwalkers*niter)
coll_gamma = np.zeros(nwalkers*niter)
coll_Cdome = np.zeros(nwalkers*niter)
for j in range(nwalkers*niter):
    coll_C0[j] = samples[j][0]
    coll_L0[j] = samples[j][1]
    coll_gamma[j] = samples[j][2]
    coll_Cdome[j] = samples[j][3]
mean_coll_C0 = mean(coll_C0)
mean_coll_L0 = mean(coll_L0)
mean_coll_gamma = mean(coll_gamma)
mean_coll_Cdome = mean(coll_Cdome)

# También coleccionamos los valores de las medianas de los parámetros
median_coll_C0 = median(coll_C0)
median_coll_L0 = median(coll_L0)
median_coll_gamma = median(coll_gamma)
median_coll_Cdome = median(coll_Cdome)

# Printeamos los valores de los parámetros según curve_fit
print("")
print("Valores de los parámetros según curve_fit: ")
print("C0 = " + f"{param[0]}")
print("L0 = " + f"{param[1]}")
print("gamma = " + f"{param[2]}")
print("Cdome = " + f"{param[3]}")
# Printeamos los valores de Mejor Ajuste según MCMC de los parámetros
print("")
print("Valores más probables de los parámetros según MCMC: ")
print("C0 = " + f"{best_fit_param[0]}")
print("L0 = " + f"{best_fit_param[1]}")
print("gamma = " + f"{best_fit_param[2]}")
print("Cdome = " + f"{best_fit_param[3]}")
# También las medianas de los parámetros
print("")
print("Medianas de los parámetros según MCMC: ")
print("C0_median = " + f"{median(coll_C0)}")
print("L0_median = " + f"{median(coll_L0)}")
print("gamma_median = " + f"{median(coll_gamma)}")
print("Cdome_median = " + f"{median(coll_Cdome)}")

# Para cerrar automáticamente las figuras y que se actualicen 
plt.close("all")

# Plotting
plt.plot(wave, files_list[0], "o", color="orange", label="S11")
plt.plot(wave, files_list[1], "o", color="blue", label="S1")
plt.plot(wave, files_list[2], "o", color="green", label="S2")
plt.plot(wave, files_list[3], "o", color="magenta", label="S3")
plt.plot(wave, opt_tokovinin, color="purple", linewidth=3.5,label="Predicción IQ ")
plt.plot(wave, Tokovinin(wave, best_fit_param[0], best_fit_param[1], best_fit_param[2], best_fit_param[3]),
         "-.", color="red", linewidth=2.5, label="MCMC MAP")
#plt.plot(wave, Tokovinin(wave, median_coll_C0, median_coll_L0, median_coll_gamma, median_coll_Cdome),
         #"--", color="gray", linewidth=2.5, label="Mediana")
plt.plot(wave, epsilon0(wave, median(coll_C0)), "--", color="darkslategray",  linewidth=2.5,
         label="Seeing $ϵ_{0}(\lambda_{ref} = 7000$ $\AA$)")
plt.plot(wave, epsilon_dome(wave, median(coll_gamma), median(coll_Cdome)), "--", color="cornflowerblue", 
         linewidth=2.5, label="Dome-Seeing $ϵ_{dome}(\lambda_{ref} = 7000$ $\AA$)")
plt.errorbar(wave, average(files_list), standev(files_list), 
            linestyle="None", marker="s",
            color = "black", capsize=3)
plt.xlabel("$\lambda$ ($\AA$)", fontsize="14")
plt.ylabel("IQ (arcsec)", fontsize="14")
plt.xlim(4800, 9300)
plt.ylim(0.25, 0.8)
plt.legend(loc="upper right")
plt.title("HD90177a", fontsize="18")
#plt.text(5160, 0.418, r"$C_{0} = $" + f"{param[0]:.{decimals}} $\pm$ {std[0]:.{decimals_err}}" "\n" 
         #r"$\mathcal{L}_{0} = $" + f"{param[1]:.{decimals}} $\pm$ {std[1]:.{decimals_err}}" "\n"
         #r"$\gamma = $" + f"{param[2]:.{decimals}} $\pm$ {std[2]:.{decimals_err}}" "\n"
         #r"$C_{dome} = $" + f"{param[3]:.{decimals}} $\pm$ {std[3]:.{decimals_err}}",
         #fontsize="16", 
         #bbox=dict(facecolor="white"))
plt.grid()
plt.tight_layout()

# Graficamos mediante el paquete corner.py la distribución de los parámetros 
# para obtener una intuición de su correlación y su incertidumbre, también 
# indicamos los cuantiles (0.16 y 0.84 que nos dan un intervalo de 1 sigma de 
# error, un 68% de confianza de los valores, y 0.5 que indica la mediana)

# Además, añadimos líneas verticales y horizontales en las gráficas que nos 
# indican los mejores parámetros (más probables)
labels = [r"[$ϵ_{0}(\lambda_{ref} = 7000$ $\AA$)]", r"Escala Externa [$\mathcal{L}_{0}$]", 
          r"Exponente Dome-Seeing [$\gamma$]", r"[$ϵ_{dome}(\lambda_{ref} = 7000$ $\AA$)]"]
fig_postdistr = corner.corner(samples, show_titles=True, labels=labels,
                              quantiles=[0.16, 0.5, 0.84])
axes = np.array(fig_postdistr.axes).reshape(ndim, ndim)

# Dibujamos líneas verticales para los histogramas (diagonales) de los parámetros
# sobre los valores de mejor ajuste, en rojo
for i in range(ndim):
    ax = axes[i, i]
    #ax.axvline(mean_coll_C0, color="blue")
    ax.axvline(best_fit_param[i], color="red")
        
# Dibujamos líneas verticales y horizontales en las distribuciones posteriores
# 2D y cuadrados que indican el valor de mejor ajuste, en rojo
for j in range(ndim-1):
    for i in range(j+1, ndim):
        ax = axes[i, j]
        ax.axvline(best_fit_param[j], color="red")
        ax.axhline(best_fit_param[i], color="red")
        ax.plot(best_fit_param[j], best_fit_param[i], "sr")

# Leyendas del corner plot
legend_elem = [Patch(facecolor="red", label="Valores de Mejor Ajuste"),
               Line2D([0], [0], color="black", linestyle="--", 
                      label="Cuantiles [16 %, 50 %, 84 %]")]
plt.legend(handles=legend_elem, loc="upper right")