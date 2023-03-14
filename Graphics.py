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





########## Ajuste polinomial ###############################
#deg = 2
#z1 = np.polyfit(wavelengths, epsilon_Rackham, deg)
#y1 = np.poly1d(z1)
#z2 = np.polyfit(wavelengths, transit_depth, deg)
#y2 = np.poly1d(z2)
############################################################

fig = plt.figure()

graph = 1

if (graph == 1):

    # Leitura de entrada dos dados

    epsilon_Rackham_ff_2 = np.genfromtxt("output_epsilon_Rackham(trans_lat-46graus,f_spot=2%).txt", delimiter=",")
    epsilon_Rackham_ff_4 = np.genfromtxt("output_epsilon_Rackham(trans_lat-46graus,f_spot=4%).txt", delimiter=",")
    epsilon_Rackham_ff_6 = np.genfromtxt("output_epsilon_Rackham(trans_lat-46graus,f_spot=6%).txt", delimiter=",")
    epsilon_Rackham_ff_8 = np.genfromtxt("output_epsilon_Rackham(trans_lat-46graus,f_spot=8%).txt", delimiter=",")
    epsilon_Rackham_ff_10 = np.genfromtxt("output_epsilon_Rackham(trans_lat-46graus,f_spot=10%).txt", delimiter=",")
    wavelengths = np.genfromtxt("output_wavelengths.txt", delimiter=",")

    table_epsilon_ourWork = np.genfromtxt('output_epsilon (our work).txt', delimiter=",", usecols=(6, 7, 8, 9, 10),
                                          skip_header=1)
    table_epsilon_ourWork = np.transpose(table_epsilon_ourWork)

    palette = sns.color_palette("mako", 5)
    graph1 = fig.add_subplot(1, 1, 1)
    graph1.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    graph1.set_ylabel('Contamination Factor ($\epsilon$)', fontsize=22, fontweight="bold", labelpad=10) # labelpad é a distância entre o título e o eixo
    graph1.set_xlabel('Wavelength (nm)', fontsize=22, fontweight="bold", labelpad=10)
    #graph1.plot(wavelengths, epsilon,  'o', linestyle='none', markersize=7, color='red', label='$\epsilon$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_10,  'o', linestyle='none', markersize=5, color=palette[4])
    graph1.plot(wavelengths, epsilon_Rackham_ff_10,  '-', linewidth=3, color=palette[4], label='$\epsilon_{R}$ - Trans. Lat. = 46$^{\circ}$; f$_{spot}=10\%$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_8,  'o', linestyle='none', markersize=5, color=palette[3])
    graph1.plot(wavelengths, epsilon_Rackham_ff_8,  '-', linewidth=3, color=palette[3], label='$\epsilon_{R}$ - Trans. Lat. = 46$^{\circ}$; f$_{spot}=8\%$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_6,  'o', linestyle='none', markersize=5, color=palette[2])
    graph1.plot(wavelengths, epsilon_Rackham_ff_6,  '-', linewidth=3, color=palette[2], label='$\epsilon_{R}$ - Trans. Lat. = 46$^{\circ}$; f$_{spot}=6\%$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_4,  'o', linestyle='none', markersize=5, color=palette[1])
    graph1.plot(wavelengths, epsilon_Rackham_ff_4,  '-', linewidth=3, color=palette[1], label='$\epsilon_{R}$ - Trans. Lat. = 46$^{\circ}$; f$_{spot}=4\%$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_2,  'o', linestyle='none', markersize=5, color=palette[0])
    graph1.plot(wavelengths, epsilon_Rackham_ff_2,  '-', linewidth=3, color=palette[0], label='$\epsilon_{R}$ - Trans. Lat. = 46$^{\circ}$; f$_{spot}=2\%$')

    #graph1.plot(wavelengths, table_epsilon_ourWork[0], '.', linestyle='none', markersize=5, color=palette[0])
    graph1.plot(wavelengths, table_epsilon_ourWork[4], '--', linewidth=3, color=palette[4],
                label='$\epsilon$ - Trans. Lat. = 46$^{\circ}$; f$_{spot}=10\%$')
    graph1.plot(wavelengths, table_epsilon_ourWork[3], '--', linewidth=3, color=palette[3],
                label='$\epsilon$ - Trans. Lat. = 46$^{\circ}$; f$_{spot}=8\%$')
    graph1.plot(wavelengths, table_epsilon_ourWork[2], '--', linewidth=3, color=palette[2],
                label='$\epsilon$ - Trans. Lat. = 46$^{\circ}$; f$_{spot}=6\%$')
    graph1.plot(wavelengths, table_epsilon_ourWork[1], '--', linewidth=3, color=palette[1],
                label='$\epsilon$ - Trans. Lat. = 46$^{\circ}$; f$_{spot}=4\%$')
    graph1.plot(wavelengths, table_epsilon_ourWork[0], '--', linewidth=3, color=palette[0],
                label='$\epsilon$ - Trans. Lat. = 46$^{\circ}$; f$_{spot}=2\%$')

    graph1.tick_params(axis="x", direction="in", labelsize=12)
    graph1.tick_params(axis="y", direction="in", labelsize=12)
    plt.subplots_adjust(top=0.9)
    #plt.plot(wavelengths, y1(wavelengths), "-", color='red') # ajuste polinomial

elif (graph == 2):

    # Leitura de entrada dos dados

    transit_depth_tl_46_ff_0 = np.genfromtxt("output_transit_depth(trans_lat-46graus,f_spot=0.0%).txt", delimiter=",")
    transit_depth_tl_46_ff_2 = np.genfromtxt("output_transit_depth(trans_lat-46graus,f_spot=2%).txt", delimiter=",")
    transit_depth_tl_46_ff_4 = np.genfromtxt("output_transit_depth(trans_lat-46graus,f_spot=4%).txt", delimiter=",")
    transit_depth_tl_46_ff_6 = np.genfromtxt("output_transit_depth(trans_lat-46graus,f_spot=6%).txt", delimiter=",")
    transit_depth_tl_46_ff_8 = np.genfromtxt("output_transit_depth(trans_lat-46graus,f_spot=8%).txt", delimiter=",")
    transit_depth_tl_46_ff_10 = np.genfromtxt("output_transit_depth(trans_lat-46graus,f_spot=10%).txt", delimiter=",")
    wavelengths = np.genfromtxt("output_wavelengths.txt", delimiter=",")

    palette = sns.color_palette("mako", 6)
    graph2 = fig.add_subplot(1, 1, 1)
    graph2.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    #graph2.set_ylabel('D$_{\mathrm{unnoc}}$ -- D$_{\mathrm{phot}}$ [ppm]', fontsize=25, fontweight="bold") # labelpad é a distância entre o título e o eixo
    graph2.set_ylabel('Transit Depth [ppm]', fontsize=22, fontweight="bold", labelpad=10) # labelpad é a distância entre o título e o eixo
    graph2.set_xlabel('Wavelength (nm)', fontsize=22, fontweight="bold", labelpad=10)
    graph2.plot(wavelengths, transit_depth_tl_46_ff_10,  'o', linestyle='none', markersize=5, color=palette[5])
    graph2.plot(wavelengths, transit_depth_tl_46_ff_10, '-', color=palette[5], linewidth=3, label='Trans. Lat. = 46$^{\circ}$; f$_{spot}=10\%$')
    graph2.plot(wavelengths, transit_depth_tl_46_ff_8,  'o', linestyle='none', markersize=5, color=palette[4])
    graph2.plot(wavelengths, transit_depth_tl_46_ff_8, '-', color=palette[4], linewidth=3, label='Trans. Lat. = 46$^{\circ}$; f$_{spot}=8\%$')
    graph2.plot(wavelengths, transit_depth_tl_46_ff_6,  'o', linestyle='none', markersize=5, color=palette[3])
    graph2.plot(wavelengths, transit_depth_tl_46_ff_6, '-', color=palette[3], linewidth=3, label='Trans. Lat. = 46$^{\circ}$; f$_{spot}=6\%$')
    graph2.plot(wavelengths, transit_depth_tl_46_ff_4,  'o', linestyle='none', markersize=5, color=palette[2])
    graph2.plot(wavelengths, transit_depth_tl_46_ff_4, '-', color=palette[2], linewidth=3, label='Trans. Lat. = 46$^{\circ}$; f$_{spot}=6\%$')
    graph2.plot(wavelengths, transit_depth_tl_46_ff_2,  'o', linestyle='none', markersize=5, color=palette[1])
    graph2.plot(wavelengths, transit_depth_tl_46_ff_2, '-', color=palette[1], linewidth=3, label='Trans. Lat. = 46$^{\circ}$; f$_{spot}=2\%$')
    graph2.plot(wavelengths, transit_depth_tl_46_ff_0,  'o', linestyle='none', markersize=5, color=palette[0])
    graph2.plot(wavelengths, transit_depth_tl_46_ff_0, '-', color=palette[0], linewidth=3, label='Trans. Lat. = 46$^{\circ}$; photosphere')


    graph2.tick_params(axis="x", direction="in", labelsize=12)
    graph2.tick_params(axis="y", direction="in", labelsize=12)
    plt.subplots_adjust(top=0.9)
    #plt.xlabel('Wavelength (nm)', labelpad=15)
    #plt.plot(wavelengths, y2(wavelengths), "-", color='red') # ajuste polinomial



legend = plt.legend(prop={'size': 10})

plt.tick_params(axis="x", direction="in", labelsize=15)
plt.tick_params(axis="y", direction="in", labelsize=15)

plt.show()