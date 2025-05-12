# Practicum-Fisica
Repositorio de las prácticas realizadas en la asignatura de Prácticas Externas (ULL) en el IAC

**Descripción General**

En este Repositorio se muestran los trabajos realizados en las prácticas que se llevaron a cabo en el **Instituto de Astrofísica de Canarias (IAC)** que conciernen el análisis de la turbulencia atmosférica para los Observatorios de Paranal (Chile) y GTC en el Roque de los Muchachos (La Palma). Para ello se han considerado datos obtenidos de ambos Observatorios, para Paranal se ha analizado el campo HD90177, en una de sus ventanas de observación, HD90177a, mientras que para GTC se han utilizado múltiples archivos de estrellas. Todos los datos que se han utilizado y más se encuentran en la carpeta **Datos** del repositorio. Este proyecto permite cuantificar la influencia atmosférica, mediante el modelaje de esta para ambos lugares de observación, en la obtención de datos de observación y poder inferir su calidad. 

**Proyecto**

Este proyecto engloba un análisis de parámetros físicos que cuantifican la turbulencia atmosférica como son el "seeing", la "escala externa" o el "seeing" de cúpula ("dome-seeing") mediante medidas de la anchura a media altura (FWHM), que cuantifica la calidad de imagen, obtenidas a partir de ajustar los perfiles de intensidad de las estrellas. De esta manera, se adjuntan estudios en el que se considera únicamente el "seeing" atmosférico, y otros en los que se añade además la contribución del "seeing" de cúpula. Para ello, se consideraron métodos de ajuste de mínimos cuadrados no lineales (mediante la ecuación de Tokovinin que se define en los códigos), para posteriormente reforzar el análisis mediante un método Bayesiano del tipo Monte Carlo, más concretamente el Markov Chain Monte Carlo (MCMC), para obtener las distribuciones posteriores de los parámetros (corner plots). Varios códigos y ejemplos se muestran a este respecto para los diferentes casos mencionados. Se añaden además códigos de un estudio para la obtención valores de referencia del "seeing" y "dome-seeing" a una longitud de onda determinada, añadiendo un "scatter plot" que permite visualizar la dependencia de los diferentes parámetros entre ellos. Los ejemplos que se muestran están referidos al campo de estrellas HD90177a, donde se han considerado las estrellas S1, S2, S3 y S11 por su buena relación señal-ruido.

En el caso de GTC se incluyen códigos de creación de tablas para ambos casos "seeing" (2 parámetros) y "seeing" + "dome-seeing" (3 parámetros), y también códigos para graficar y obtener las distribuciones (histogramas) de los parámetros con ejemplos.

**Consideraciones**

Para este proyecto es fundamental tener una versión de Python que permita la utilización de los paquetes numpy, matplotlib, emcee y seaborn, estos son los paquetes necesarios para poder ejecutar todos los códigos del repositorio, graficar y hacer los análisis. 
