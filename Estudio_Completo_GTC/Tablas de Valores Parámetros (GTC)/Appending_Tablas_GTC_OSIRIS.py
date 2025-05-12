# -*- coding: utf-8 -*-
"""
Created on Thu May  8 11:01:02 2025

@author: aleja
"""

# Código para juntar las tablas de datos con los valores de los parámetros de 
# GTC OSIRIS en una sola tabla

# Importamos paquetes
from astropy.io import fits
from astropy.table import Table, vstack

# Archivos de las tablas que se quieren juntar, se crea la nueva tabla 
# en new_table
table_fits_1 = Table.read("C:\\path\\to\\your\\file", format="fits")
table_fits_2 = Table.read("C:\\path\\to\\your\\file", format="fits")
table_fits_3 = Table.read("C:\\path\\to\\your\\file", format="fits")
table_fits_4 = Table.read("C:\\path\\to\\your\\file", format="fits")
table_fits_5 = Table.read("C:\\path\\to\\your\\file", format="fits")
table_fits_6 = Table.read("C:\\path\\to\\your\\file", format="fits" )                    
new_table = vstack([table_fits_1, table_fits_2, table_fits_3, table_fits_4,
                    table_fits_5, table_fits_6])
new_table.write("NEW_TABLE_NAME", format="fits", overwrite=True) 

# Abrimos el archivo y mostramos (mediante print) la tabla
with fits.open("NEW_TABLE_NAME") as hdul:
    hdul.info()
    bin_table = hdul[1]
    bin_header = bin_table.header
    data = bin_table.data
    print(bin_header)
    print(data)