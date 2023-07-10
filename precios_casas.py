# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 19:42:26 2023

@author: Jose Antonio
"""

#%% Imports
import pandas as pd
import numpy as np
import os
from sys import path
from tqdm import tqdm
from pandas_profiling import ProfileReport
import requests
import tarfile
from matplotlib import pyplot as plt
import seaborn as sns
import scienceplots
import csv
import utm
from scipy.interpolate import LinearNDInterpolator, interp2d, RectSphereBivariateSpline, griddata, RectBivariateSpline, interpn, RegularGridInterpolator
import xarray as xr

plt.style.use('science')


#%% Funciones

def savefig(plot_dir, name):
    plt.savefig(os.path.join(plot_dir, name), bbox_inches="tight", dpi=180)

#%% Data load
URL = "https://mymldatasets.s3.eu-de.cloud-object-storage.appdomain.cloud/housing.tgz"
PATH = "housing.tgz"

def getData(url=URL, path=PATH):
  r = requests.get(url)
  with open(path, 'wb') as f:
    f.write(r.content)
  housing_tgz = tarfile.open(path)
  housing_tgz.extractall()
  housing_tgz.close()
  
getData()  
  
PATH = "housing.csv"

def loadData(path=PATH):
  return pd.read_csv(path)

data = loadData()
data.info() #Para ver lo que pesa


#%% EDA
description = data.describe()

# report = ProfileReport(data)
# report.to_file(output_file='output.html')

nans = pd.isnull(data) 
conteo = nans.sum() #Hay 207 nans en data en total_bedrooms, vamos a quitarlos (también se podrían rellenar con la media)

idx = [i for i,e in enumerate(nans.total_bedrooms) if e == True] #Obtengo los índices
data.drop(idx, axis=0, inplace=True)


#%% Matriz de correlacion
plt.close()
data_for_corr = data.iloc[:, :len(data.columns)-1]
matrix = data_for_corr.corr()
sns.heatmap(matrix, annot=True)

#%% Quito las más correladas y vuelvo a ver la matriz de correlación 
data.drop(['total_bedrooms', 'total_rooms', 'population', 'median_income'], axis=1, inplace=True)
data.reset_index(drop=True, inplace=True)
plt.close()
data_for_corr = data.iloc[:, :len(data.columns)-1]
matrix = data_for_corr.corr()
sns.heatmap(matrix, annot=True)

#%% Algunos plots
plt.close()
data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.5,
          s=data["households"]/100, label="households", c="median_house_value", cmap=plt.get_cmap("jet"),
          colorbar=True)
plt.title("Posición de las casas junto con los households y el precio medio")


#%%
""" TODO: Para plots mas chulos y mejor entendimiento podría estar bien conseguir 
    una bty de la zona (POR ENTENDER LA CERCANÍA LA MAR Y LA ALTITUD DE LOS BARRIOS)"""


min_lat, max_lat, min_lon, max_lon = min(data.latitude), max(data.latitude), min(data.longitude), max(data.longitude) #Para descargarla de gebco
ruta_bty = os.path.join(os.getcwd(), 'bty', 'gebco_2023_n41.95_s32.54_w-124.35_e-108.31.asc')

with open(ruta_bty, 'r') as bty:
    bty_iter = csv.reader(bty, delimiter = ' ')
    bty_data = [data for data in bty_iter]

d = {'ncols': int(bty_data[0][len(bty_data[0])-1]),
     'nrows': int(bty_data[1][len(bty_data[1])-1]),
     'xllcorner': float(bty_data[2][len(bty_data[2])-1]),
     'yllcorner': float(bty_data[3][len(bty_data[3])-1]),
     'cellsize': float(bty_data[4][len(bty_data[4])-1]),
     'nodata_value': float(bty_data[5][len(bty_data[5])-1])
     }

bty = np.zeros((d['nrows'], d['ncols']))
for i in range(d['nrows']):
    bty[i,:] = [-float(j) for j in bty_data[6+i][1:]]
    
lon = np.array([d['xllcorner'] + d['cellsize']*i for i in range(d['ncols'])]) # Esto se puede con arange
lat = np.array([d['yllcorner'] + d['cellsize']*i for i in range(d['nrows'])])


def haversine(origin, dest, radians = False):
    '''
    Calcula la distancia entre un punto "origen" y punto(s) "destino" en km 
    modelizando la Tierra como una esfera de radio 6371 km. Utiliza el modelo
    de Haversine.
    origin : [lat0, lon0]
    dest :   [lat1, lon1]
    '''    
    R = 6371.0
    
    if not radians:
        lat1 = np.radians(origin[0])
        lon1 = np.radians(origin[1])
        lat2 = np.radians(dest[0])
        lon2 = np.radians(dest[1])
        
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R*c

# Distancias de la región considerada
ladox = haversine((lat[lat.size//2], lon[0]), (lat[lat.size//2], lon[-1]))
ladoy = haversine((lat[0], lon[lon.size//2]), (lat[-1], lon[lon.size//2]))
print('El área de batimetría abarca unos {:.2f} km de lado en X, mientras que '
      'en latitud abarca unos {:.2f} km.'.format(ladox, ladoy))

"""
Voy a probar con ~500 m de resolución en un principio
"""
#%% Procesado bty
dx = dy = 500     # Distancia en metros de celda
nx, ny = int(np.round(ladox*1e3 / dx)), int(np.round(ladoy*1e3 / dy))

# Mallado en lat/lon
x, y = np.linspace(lon[0], lon[-1], nx), np.linspace(lat[0], lat[-1], ny)
xg, yg = np.meshgrid(x,y)

# Mallado uniforme en UTM (es un mallado que abarca algo más del área original)
aux = utm.from_latlon(lat[0], lon[0])
x1,y1 = aux[0], aux[1]
aux = utm.from_latlon(lat[0], lon[-1])
x2,y2 = aux[0], aux[1]
aux = utm.from_latlon(lat[-1], lon[-1])
x3,y3 = aux[0], aux[1]
aux = utm.from_latlon(lat[-1], lon[0])
x4,y4 = aux[0], aux[1]
xleft, xright = min(x1,x2,x3,x4), max(x1,x2,x3,x4)
ybot, ytop    = min(y1,y2,y3,y4), max(y1,y2,y3,y4)
xutm = np.linspace(xleft, xright, nx)
yutm = np.linspace(ybot, ytop, ny)
# np.save(os.path.join(root, "resultados", "xUTM_MA.npy"), xutm)
# np.save(os.path.join(root, "resultados", "yUTM_MA.npy"), yutm)
xgm, ygm = np.meshgrid(xutm, yutm)   # MALLADO FINAL

xdiff = np.diff(xutm)[0]  # Resolucion en x (debe ser muy cercano a dx)
ydiff = np.diff(yutm)[1]  # Resolucion en y (debe ser muy cercano a dy)

bty_xutm, bty_yutm  = utm.from_latlon(*np.meshgrid(lat,lon))[:2]
bty_xutm, bty_ytum = bty_xutm.T, bty_yutm.T
# Debemos desenrollar para el griddata
bty_xutm_r = bty_xutm.ravel(order='F')
bty_yutm_r = bty_yutm.ravel(order='C')
# Interpolamos al mallado estándar
bty_interpolado = griddata(np.concatenate((bty_yutm_r[:,None], bty_xutm_r[:,None]), axis = 1),
                           bty.ravel(order = 'F'), (ygm, xgm), method = 'nearest')
# Objeto interpolante para los transectos
bty_interpolator = RectBivariateSpline(yutm, xutm, bty_interpolado)
bbound2 = [xleft, ybot, xright, ytop]

#%% Plotteo bty

""" Revisar que está pasando con el plot, que no tiene sentido que salgan casas tan lejos
    si el cuadro se ha dibujado con los datos de las casas"""

lati, long = data.latitude.values, data.longitude.values

coords_utm = [utm.from_latlon(par[0], par[1])[:2] for par in zip(lati, long)] #Hay que hacerlo así iterando, con la serie entera fallaría
coords_utm = np.asarray(coords_utm)
coords_utm_plot = coords_utm * 1e-3

savePlot = False
fig, ax0 = plt.subplots(figsize=(10,10))
im = ax0.imshow(bty_interpolado, extent= [xleft/1000, (xright+xdiff)/1000,
                ybot/1000, (ytop+ydiff)/500], cmap='rainbow_r',
                origin='lower', vmin=0, interpolation='gaussian')
ax0.contourf(xgm/1000, ygm/1000, bty_interpolado, [-np.inf, -0.1], colors = 'gray')
ax0.contour(xgm/1000, ygm/1000, bty_interpolado, [-0.1], linewidths = 1, linestyles = 'solid', colors = 'k')
# ax0.scatter(utmsrc[0]*1e-3, utmsrc[1]*1e-3, marker='x', c = 'k', s = 50)
ax0.set_xlabel(r'$x$ [km]')
ax0.set_ylabel(r'$y$ [km]')
# ax0.scatter(x=coords_utm_plot[:,0], y=coords_utm_plot[:,1])
# ax0.setxlim()
fig.colorbar(im, label = 'Depth [m]')
# if savePlot:
#     savefig(fig, "bty")

#%% Plot casas junto a bty (meter también el tamaño de las casas dependiente del precio)
root = os.getcwd()
plot_dir = os.path.join(root, 'plots')

plt.close()
lati, long = data.latitude.values, data.longitude.values
btys_folder = 'GEBCO_10_Jul_2023_0c0e83e4a836'
da = xr.open_dataset(os.path.join(root, 'bty', btys_folder,'gebco_2023_n41.95_s32.54_w-124.35_e-108.31.nc'))

da.elevation.plot(cmap="jet", figsize=(8,6))
im = plt.scatter(long, lati, c=data["median_house_value"], s=data["households"]/50, label = "households", alpha=0.3, 
            cmap=plt.cm.get_cmap('twilight'))
plt.colorbar(label="Median house value")

plt.legend()
plt.title("Casas con households y bty")
plt.show()
savefig(plot_dir, "Relacion_bty_households")


#%% Voy a extraer la altimetría para cada punto (casa/barrio)
""" Al final no lo voy a hacer porque se aprecia que las casas
    mas caras no se encuentran en zonas de mayor altitud, y, además,
    todas las casas están en casi el nivel del mar por ser un dataset de una
    zona costera. Comento y no borro porque me parece interesante como extraer los
    features de un nc"""

# bty = da.elevation.values #Matríz de shape (len(lat_bty), len(lon_bty))
# lat_bty = list(da.lat.values)
# lon_bty = list(da.lon.values)
# coords_bty = [[par[0], par[1]] for par in zip(lat_bty, lon_bty)]
# coords_bty = np.asarray(coords_bty)

# lati, long = data.latitude.values, data.longitude.values
# coords_casas = [[par[0], par[1]] for par in zip(lati, long)]
# coords_casas = np.asarray(coords_casas)

#%% Histogramas

cont = 0
nfeatures = len(data.columns)
fig, ax = plt.subplots(ncols=int(nfeatures/2), nrows=int(nfeatures/2))
for i in range(int(nfeatures/2)):
    if i != int((nfeatures/2)-1):
        ax[i,cont].hist(data[data.columns[i]])
    else:
        cont= cont+1
        





