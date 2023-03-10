import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
import math

import seaborn as sns
# https://seaborn.pydata.org/tutorial/color_palettes.html
# https://holypython.com/python-visualization-tutorial/colors-with-python/

########### Fonte igual ao LaTeX ###### <https://matplotlib.org/stable/tutorials/text/usetex.html> ######
#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "serif",
#    "font.serif": ["Computer Modern Roman"]})
#########################################################################################################

# Leitura de entrada dos dados

transit_depth_tl_46_ff_0 = np.genfromtxt("output_transit_depth(trans_lat-46graus,f_spot=0.0%).txt", delimiter=",")
transit_depth_tl_46_ff_4 = np.genfromtxt("output_transit_depth(trans_lat-46graus,f_spot=4%).txt", delimiter=",")
transit_depth_tl_46_ff_8 = np.genfromtxt("output_transit_depth(trans_lat-46graus,f_spot=8%).txt", delimiter=",")
epsilon_Rackham_ff_4 = np.genfromtxt("output_epsilon_Rackham(trans_lat-46graus,f_spot=4%).txt", delimiter=",")
epsilon_Rackham_ff_8 = np.genfromtxt("output_epsilon_Rackham(trans_lat-46graus,f_spot=8%).txt", delimiter=",")
wavelengths = np.genfromtxt("output_wavelengths.txt", delimiter=",")


########## Ajuste polinomial ###############################
#deg = 2
#z1 = np.polyfit(wavelengths, epsilon_Rackham, deg)
#y1 = np.poly1d(z1)
#z2 = np.polyfit(wavelengths, transit_depth, deg)
#y2 = np.poly1d(z2)
############################################################

fig = plt.figure()
graph = 2
if (graph == 1):

    graph1 = fig.add_subplot(1, 1, 1)
    graph1.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    graph1.set_ylabel('Contamination Factor ($\epsilon$)', fontsize=22, fontweight="bold", labelpad=10) # labelpad é a distância entre o título e o eixo
    graph1.set_xlabel('R$_{spot}$/R$_{star}$', fontsize=25, fontweight="bold", labelpad=10)
    #graph1.plot(wavelengths, epsilon,  'o', linestyle='none', markersize=7, color='red', label='$\epsilon$')
    graph1.plot(wavelengths, epsilon_Rackham_ff_4,  'o', linestyle='none', markersize=5, color='blue')
    graph1.plot(wavelengths, epsilon_Rackham_ff_8,  'o', linestyle='none', markersize=5, color='red')
    graph1.plot(wavelengths, epsilon_Rackham_ff_4,  '-', linewidth=3, color='blue', label='$\epsilon_{R}$ - Trans. Lat. = 46$^{\circ}$; f$_{spot}=4\%$')
    graph1.plot(wavelengths, epsilon_Rackham_ff_8,  '-', linewidth=3, color='red', label='$\epsilon_{R}$ - Trans. Lat. = 46$^{\circ}$; f$_{spot}=8\%$')
    graph1.tick_params(axis="x", direction="in", labelsize=12)
    graph1.tick_params(axis="y", direction="in", labelsize=12)
    plt.subplots_adjust(top=0.9)
    #plt.plot(wavelengths, y1(wavelengths), "-", color='red') # ajuste polinomial


elif (graph == 2):
    palette = sns.color_palette("mako", 3)
    graph2 = fig.add_subplot(1, 1, 1)
    graph2.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    #graph2.set_ylabel('D$_{\mathrm{unnoc}}$ -- D$_{\mathrm{phot}}$ [ppm]', fontsize=25, fontweight="bold") # labelpad é a distância entre o título e o eixo
    graph2.set_ylabel('Transit Depth [ppm]', fontsize=22, fontweight="bold", labelpad=10) # labelpad é a distância entre o título e o eixo
    graph2.set_xlabel('Wavelength (nm)', fontsize=22, fontweight="bold", labelpad=10)
    graph2.plot(wavelengths, transit_depth_tl_46_ff_0,  'o', linestyle='none', markersize=5, color=palette[0])
    graph2.plot(wavelengths, transit_depth_tl_46_ff_0, '-', color=palette[0], linewidth=3, label='Trans. Lat. = 46$^{\circ}$; f$_{spot}=0\%$')
    graph2.plot(wavelengths, transit_depth_tl_46_ff_4,  'o', linestyle='none', markersize=5, color=palette[1])
    graph2.plot(wavelengths, transit_depth_tl_46_ff_4, '-', color=palette[1], linewidth=3, label='Trans. Lat. = 46$^{\circ}$; f$_{spot}=4\%$')
    graph2.plot(wavelengths, transit_depth_tl_46_ff_8,  'o', linestyle='none', markersize=5, color=palette[2])
    graph2.plot(wavelengths, transit_depth_tl_46_ff_8, '-', color=palette[2], linewidth=3, label='Trans. Lat. = 46$^{\circ}$; f$_{spot}=8\%$')
    graph2.tick_params(axis="x", direction="in", labelsize=12)
    graph2.tick_params(axis="y", direction="in", labelsize=12)
    plt.subplots_adjust(top=0.9)
    #plt.xlabel('Wavelength (nm)', labelpad=15)
    #plt.plot(wavelengths, y2(wavelengths), "-", color='red') # ajuste polinomial



legend = plt.legend(prop={'size': 20})

plt.tick_params(axis="x", direction="in", labelsize=15)
plt.tick_params(axis="y", direction="in", labelsize=15)

plt.show()