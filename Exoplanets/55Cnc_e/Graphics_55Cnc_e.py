import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
import math
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
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

if graph == 1:

    # Leitura de entrada dos dados

    epsilon_Rackham_ff_2 = np.genfromtxt("55Cnc_e_output_epsilon_Rackham(trans_lat=-24graus,f_spot=0.02,temp_spot=3781K).txt", delimiter=",")
    epsilon_Rackham_ff_4 = np.genfromtxt("55Cnc_e_output_epsilon_Rackham(trans_lat=-24graus,f_spot=0.04,temp_spot=3781K).txt", delimiter=",")
    epsilon_Rackham_ff_6 = np.genfromtxt("55Cnc_e_output_epsilon_Rackham(trans_lat=-24graus,f_spot=0.06,temp_spot=3781K).txt", delimiter=",")
    epsilon_Rackham_ff_8 = np.genfromtxt("55Cnc_e_output_epsilon_Rackham(trans_lat=-24graus,f_spot=0.08,temp_spot=3781K).txt", delimiter=",")
    epsilon_Rackham_ff_10 = np.genfromtxt("55Cnc_e_output_epsilon_Rackham(trans_lat=-24graus,f_spot=0.10,temp_spot=3781K).txt", delimiter=",")

    epsilon_Rackham_ff_facula_2 = np.genfromtxt("55Cnc_e_output_epsilon_Rackham(trans_lat=-24graus,f_facula=0.02,temp_facula=5272K).txt", delimiter=",")
    epsilon_Rackham_ff_facula_4 = np.genfromtxt("55Cnc_e_output_epsilon_Rackham(trans_lat=-24graus,f_facula=0.04,temp_facula=5272K).txt", delimiter=",")
    epsilon_Rackham_ff_facula_6 = np.genfromtxt("55Cnc_e_output_epsilon_Rackham(trans_lat=-24graus,f_facula=0.06,temp_facula=5272K).txt", delimiter=",")
    epsilon_Rackham_ff_facula_8 = np.genfromtxt("55Cnc_e_output_epsilon_Rackham(trans_lat=-24graus,f_facula=0.08,temp_facula=5272K).txt", delimiter=",")
    epsilon_Rackham_ff_facula_10 = np.genfromtxt("55Cnc_e_output_epsilon_Rackham(trans_lat=-24graus,f_facula=0.10,temp_facula=5272K).txt", delimiter=",")

    wavelengths = np.genfromtxt("55Cnc_e_output_wavelengths.txt", delimiter=",")

    table_epsilon_ourWork = np.genfromtxt('epsilon (our work).txt', delimiter=",", usecols=(0, 1, 2, 3, 4, 5),
                                          skip_header=1)
    table_epsilon_ourWork = np.transpose(table_epsilon_ourWork)

    table_epsilon_facula_ourWork = np.genfromtxt('epsilon (our work) - facula.txt', delimiter=",", usecols=(0, 1, 2, 3, 4, 5),
                                          skip_header=1)
    table_epsilon_facula_ourWork = np.transpose(table_epsilon_facula_ourWork)

    palette = sns.color_palette("Paired", 16)
    graph1 = fig.add_subplot(2, 1, 1)
    #graph1.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    graph1.set_ylabel('Contamination Factor ($\epsilon$)', fontsize=23, fontweight="bold",
                      labelpad=10) # labelpad é a distância entre o título e o eixo
    #graph1.set_xlabel('Wavelength (nm)', fontsize=23, fontweight="bold", labelpad=10)
    #graph1.plot(wavelengths, epsilon,  'o', linestyle='none', markersize=7, color='red', label='$\epsilon$')

    graph1.plot(wavelengths, table_epsilon_ourWork[5]/table_epsilon_ourWork[0], '-', linewidth=3, color=palette[9],
                label='$\mathbf{\epsilon - ff=10\%}$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_10,  'o', linestyle='none', markersize=5, color=palette[4])
    graph1.plot(wavelengths, epsilon_Rackham_ff_10,  '--', linewidth=3, color=palette[8],
                label='$\mathbf{\epsilon_{\mathbf{R}} - ff=10\%}$')
    graph1.plot(wavelengths, table_epsilon_ourWork[4]/table_epsilon_ourWork[0], '-', linewidth=3, color=palette[7],
                label='$\mathbf{\epsilon - ff=8\%}$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_8,  'o', linestyle='none', markersize=5, color=palette[3])
    graph1.plot(wavelengths, epsilon_Rackham_ff_8,  '--', linewidth=3, color=palette[6],
                label='$\mathbf{\epsilon_{\mathbf{R}} - ff=8\%}$')
    graph1.plot(wavelengths, table_epsilon_ourWork[3]/table_epsilon_ourWork[0], '-', linewidth=3, color=palette[5],
                label='$\mathbf{\epsilon - ff=6\%}$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_6,  'o', linestyle='none', markersize=5, color=palette[2])
    graph1.plot(wavelengths, epsilon_Rackham_ff_6,  '--', linewidth=3, color=palette[4],
                label='$\mathbf{\epsilon_{\mathbf{R}} - ff=6\%}$')
    graph1.plot(wavelengths, table_epsilon_ourWork[2]/table_epsilon_ourWork[0], '-', linewidth=3, color=palette[3],
                label='$\mathbf{\epsilon - ff=4\%}$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_4,  'o', linestyle='none', markersize=5, color=palette[1])
    graph1.plot(wavelengths, epsilon_Rackham_ff_4,  '--', linewidth=3, color=palette[2],
                label='$\mathbf{\epsilon_{\mathbf{R}} - ff=4\%}$')
    graph1.plot(wavelengths, table_epsilon_ourWork[1]/table_epsilon_ourWork[0], '-', linewidth=3, color=palette[1],
                label='$\mathbf{\epsilon- ff=2\%}$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_2,  'o', linestyle='none', markersize=5, color=palette[0])
    graph1.plot(wavelengths, epsilon_Rackham_ff_2,  '--', linewidth=3, color=palette[0],
                label='$\mathbf{\epsilon_{\mathbf{R}} - ff=2\%}$')


    plt.xlim(500, 1700)
    plt.ylim(1.000, 1.145)
    graph1.tick_params(axis="x", direction="in", labelsize=17)
    graph1.tick_params(axis="y", direction="in", labelsize=17)
    plt.subplots_adjust(top=0.9)
    #plt.plot(wavelengths, y1(wavelengths), "-", color='red') # ajuste polinomial
    legend = plt.legend(prop={'size': 19}, title_fontsize=19)
    plt.subplots_adjust(left=0.11, bottom=0.14, right=0.95, top=0.96)
    plt.tick_params(axis="x", direction="in", labelsize=17)
    plt.tick_params(axis="y", direction="in", labelsize=17)
    at1 = AnchoredText('$\mathbf{55\,Cnc\,E}:\;\mathbf{Trans. Lat. = 24.0^{\circ}}$', prop=dict(size=21),
                      frameon=True, loc='upper center') # frameon é o retângulo em volta do texto
    at2 = AnchoredText('$\mathbf{Starspots}$', prop=dict(size=21),
                      frameon=True, loc='upper left') # frameon é o retângulo em volta do texto
    graph1.add_artist(at1)
    graph1.add_artist(at2)

    palette = sns.color_palette("Paired", 16)
    graph1_2 = fig.add_subplot(2, 1, 2)
    #graph1.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    graph1_2.set_ylabel('Contamination Factor ($\epsilon$)', fontsize=23, fontweight="bold",
                      labelpad=10) # labelpad é a distância entre o título e o eixo
    graph1_2.set_xlabel('Wavelength (nm)', fontsize=23, fontweight="bold", labelpad=10)
    #graph1.plot(wavelengths, epsilon,  'o', linestyle='none', markersize=7, color='red', label='$\epsilon$')

    graph1_2.plot(wavelengths, table_epsilon_facula_ourWork[5]/table_epsilon_facula_ourWork[0], '-', linewidth=3, color=palette[9],
                label='$\mathbf{\epsilon - ff=10\%}$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_10,  'o', linestyle='none', markersize=5, color=palette[4])
    graph1_2.plot(wavelengths, epsilon_Rackham_ff_facula_10,  '--', linewidth=3, color=palette[8],
                label='$\mathbf{\epsilon_{\mathbf{R}} - ff=10\%}$')
    graph1_2.plot(wavelengths, table_epsilon_facula_ourWork[4]/table_epsilon_facula_ourWork[0], '-', linewidth=3, color=palette[7],
                label='$\mathbf{\epsilon - ff=8\%}$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_8,  'o', linestyle='none', markersize=5, color=palette[3])
    graph1_2.plot(wavelengths, epsilon_Rackham_ff_facula_8,  '--', linewidth=3, color=palette[6],
                label='$\mathbf{\epsilon_{\mathbf{R}} - ff=8\%}$')
    graph1_2.plot(wavelengths, table_epsilon_facula_ourWork[3]/table_epsilon_facula_ourWork[0], '-', linewidth=3, color=palette[5],
                label='$\mathbf{\epsilon - ff=6\%}$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_6,  'o', linestyle='none', markersize=5, color=palette[2])
    graph1_2.plot(wavelengths, epsilon_Rackham_ff_facula_6,  '--', linewidth=3, color=palette[4],
                label='$\mathbf{\epsilon_{\mathbf{R}} - ff=6\%}$')
    graph1_2.plot(wavelengths, table_epsilon_facula_ourWork[2]/table_epsilon_facula_ourWork[0], '-', linewidth=3, color=palette[3],
                label='$\mathbf{\epsilon - ff=4\%}$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_4,  'o', linestyle='none', markersize=5, color=palette[1])
    graph1_2.plot(wavelengths, epsilon_Rackham_ff_facula_4,  '--', linewidth=3, color=palette[2],
                label='$\mathbf{\epsilon_{\mathbf{R}} - ff=4\%}$')
    graph1_2.plot(wavelengths, table_epsilon_facula_ourWork[1]/table_epsilon_facula_ourWork[0], '-', linewidth=3, color=palette[1],
                label='$\mathbf{\epsilon- ff=2\%}$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_2,  'o', linestyle='none', markersize=5, color=palette[0])
    graph1_2.plot(wavelengths, epsilon_Rackham_ff_facula_2,  '--', linewidth=3, color=palette[0],
                label='$\mathbf{\epsilon_{\mathbf{R}} - ff=2\%}$')


    graph1_2.set_xlim(500, 1700)
    #graph1_2.set_ylim(0.960, 1.)
    graph1_2.tick_params(axis="x", direction="in", labelsize=17)
    graph1_2.tick_params(axis="y", direction="in", labelsize=17)
    plt.subplots_adjust(top=0.9)
    #plt.plot(wavelengths, y1(wavelengths), "-", color='red') # ajuste polinomial
    plt.legend(prop={'size': 19}, title_fontsize=19)
    plt.subplots_adjust(left=0.11, bottom=0.14, right=0.95, top=0.96)
    plt.tick_params(axis="x", direction="in", labelsize=17)
    plt.tick_params(axis="y", direction="in", labelsize=17)

    at3 = AnchoredText('$\mathbf{Faculae}$', prop=dict(size=21),
                      frameon=True, loc='upper left') # frameon é o retângulo em volta do texto
    graph1_2.add_artist(at3)
    graph1_2.yaxis.set_major_locator(MultipleLocator(0.005))

elif graph == 2:

    # Leitura de entrada dos dados

    ######### Mancha ##########
    transit_depth_tl_24_ff_0 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=-24graus,f_spot=0.00,temp_spot=3781K).txt", delimiter=",")
    transit_depth_tl_24_ff_2 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=-24graus,f_spot=0.02,temp_spot=3781K).txt", delimiter=",")
    transit_depth_tl_24_ff_4 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=-24graus,f_spot=0.04,temp_spot=3781K).txt", delimiter=",")
    transit_depth_tl_24_ff_6 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=-24graus,f_spot=0.06,temp_spot=3781K).txt", delimiter=",")
    transit_depth_tl_24_ff_8 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=-24graus,f_spot=0.08,temp_spot=3781K).txt", delimiter=",")
    transit_depth_tl_24_ff_10 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=-24graus,f_spot=0.10,temp_spot=3781K).txt", delimiter=",")
    ########## Temperatura da Mancha
    transit_depth_4281K = np.genfromtxt(
        "55Cnc_e_output_transit_depth(trans_lat=-24graus,f_spot=0.08,temp_spot=4281K).txt", delimiter=",")
    transit_depth_3781K = np.genfromtxt(
        "55Cnc_e_output_transit_depth(trans_lat=-24graus,f_spot=0.08,temp_spot=3781K).txt", delimiter=",")
    transit_depth_3281K = np.genfromtxt(
        "55Cnc_e_output_transit_depth(trans_lat=-24graus,f_spot=0.08,temp_spot=3281K).txt", delimiter=",")
    transit_depth_2781K = np.genfromtxt(
        "55Cnc_e_output_transit_depth(trans_lat=-24graus,f_spot=0.08,temp_spot=2781K).txt", delimiter=",")
    transit_depth_2281K = np.genfromtxt(
        "55Cnc_e_output_transit_depth(trans_lat=-24graus,f_spot=0.08,temp_spot=2281K).txt", delimiter=",")

    ######### Fácula ###########
    transit_depth_tl_24_ff_fac_10_T_5472K = np.genfromtxt(
        "55Cnc_e_output_transit_depth(trans_lat=-24graus,f_facula=0.10,temp_facula=5472K).txt", delimiter=",")
    transit_depth_tl_24_ff_fac_20_T_5472K = np.genfromtxt(
        "55Cnc_e_output_transit_depth(trans_lat=-24graus,f_facula=0.20,temp_facula=5472K).txt", delimiter=",")
    transit_depth_tl_24_ff_fac_30_T_5472K = np.genfromtxt(
        "55Cnc_e_output_transit_depth(trans_lat=-24graus,f_facula=0.30,temp_facula=5472K).txt", delimiter=",")
    transit_depth_tl_24_ff_fac_40_T_5472K = np.genfromtxt(
        "55Cnc_e_output_transit_depth(trans_lat=-24graus,f_facula=0.40,temp_facula=5472K).txt", delimiter=',')
    ########## Temperatura da Fácula
    transit_depth_5272K = np.genfromtxt(
        "55Cnc_e_output_transit_depth(trans_lat=-24graus,f_facula=0.08,temp_facula=5272K).txt", delimiter=",")
    transit_depth_5472K = np.genfromtxt(
        "55Cnc_e_output_transit_depth(trans_lat=-24graus,f_facula=0.08,temp_facula=5472K).txt", delimiter=",")
    transit_depth_5672K = np.genfromtxt(
        "55Cnc_e_output_transit_depth(trans_lat=-24graus,f_facula=0.08,temp_facula=5672K).txt", delimiter=",")
    transit_depth_6172K = np.genfromtxt(
        "55Cnc_e_output_transit_depth(trans_lat=-24graus,f_facula=0.08,temp_facula=6172K).txt", delimiter=",")
    transit_depth_7172K = np.genfromtxt(
        "55Cnc_e_output_transit_depth(trans_lat=-24graus,f_facula=0.08,temp_facula=7172K).txt", delimiter=",")

    wavelengths = np.genfromtxt("55Cnc_e_output_wavelengths.txt", delimiter=",")

    palette = sns.color_palette("mako", 12)
    palette1 = sns.color_palette("flare_r", 12)

    ######### Dados do Hubble ###############

    Cond_table5 = np.genfromtxt('55Cnc_e_WFC3.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table5 = np.transpose(Cond_table5)

    ########################

    graph2 = fig.add_subplot(2, 1, 1)
    plt.errorbar(Cond_table5[0] * 1000, Cond_table5[1], fmt='.', yerr=Cond_table5[2], color='gray', ms=10, alpha=0.7,
                 label='$\mathrm{\mathbf{Hubble\;WFC3}}$')
    graph2.plot(wavelengths, transit_depth_tl_24_ff_10, '-', color=palette[5], linewidth=3,
                label='$\mathbf{ff_{spot}=10\%}$')
    graph2.plot(wavelengths, transit_depth_tl_24_ff_8, '-', color=palette[6], linewidth=3,
                label='$\mathbf{ff_{spot}=8\%}$')
    graph2.plot(wavelengths, transit_depth_tl_24_ff_6, '-', color=palette[7], linewidth=3,
                label='$\mathbf{ff_{spot}=6\%}$')
    graph2.plot(wavelengths, transit_depth_tl_24_ff_4, '-', color=palette[8], linewidth=3,
                label='$\mathbf{ff_{spot}=4\%}$')
    graph2.plot(wavelengths, transit_depth_tl_24_ff_2, '-', color=palette[9], linewidth=3,
                label='$\mathbf{ff_{spot}=2\%}$')
    graph2.plot(wavelengths, transit_depth_tl_24_ff_0, '-', color=palette[10], linewidth=3,
                label='$\mathrm{\mathbf{ff=0\%}}$')

    graph2.plot(wavelengths, transit_depth_tl_24_ff_fac_10_T_5472K, '-', color=palette1[10], linewidth=3,
                label='$\mathbf{ff_{fac}=10\%}$')
    graph2.plot(wavelengths, transit_depth_tl_24_ff_fac_20_T_5472K, '-', color=palette1[8], linewidth=3,
                label='$\mathbf{ff_{fac}=20\%}$')
    graph2.plot(wavelengths, transit_depth_tl_24_ff_fac_30_T_5472K, '-', color=palette1[6], linewidth=3,
                label='$\mathbf{ff_{fac}=30\%}$')
    graph2.plot(wavelengths, transit_depth_tl_24_ff_fac_40_T_5472K, '-', color=palette1[4], linewidth=3,
                label='$\mathbf{ff_{fac}=40\%}$')

    graph2.tick_params(axis="x", direction="in", labelsize=15, which='both')
    graph2.tick_params(axis="y", direction="in", labelsize=15, which='both')
    plt.subplots_adjust(top=0.9)
    plt.xlim(470, 1700)
    plt.ylim(285, 440)
    plt.legend(prop={'size': 12}, title_fontsize=15, loc='upper right')
    #graph2.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    #graph2.set_ylabel('D$_{\mathrm{unnoc}}$ -- D$_{\mathrm{phot}}$ [ppm]', fontsize=25, fontweight="bold") # labelpad é a distância entre o título e o eixo
    graph2.set_ylabel('Transit Depth [ppm]', fontsize=19, fontweight="bold", labelpad=10) # labelpad é a distância entre o título e o eixo
    #graph2.set_xlabel('Wavelength (nm)', fontsize=22, fontweight="bold", labelpad=10)
    graph2.yaxis.set_major_locator(MultipleLocator(30))
    #graph2.xaxis.set_major_locator(MultipleLocator(30))
    #graph2.yaxis.set_minor_locator(MultipleLocator(10))
    #graph2.xaxis.set_minor_locator(MultipleLocator(10))
    at1 = AnchoredText("$\mathbf{55\,\mathrm{\mathbf{Cnc}}\,\mathrm{\mathbf{e}}}$", prop=dict(size=15),
                      frameon=True, loc='upper center')
    at2 = AnchoredText("$\mathbf{T_{star}=5172\,K;\,T_{spot}=3782\,K;\;T_{fac}=5472\,K}$", prop=dict(size=15),
                       frameon=True, loc='lower left')
    graph2.add_artist(at1)
    graph2.add_artist(at2)


    graph2_1 = fig.add_subplot(2, 1, 2)
    plt.errorbar(Cond_table5[0] * 1000, Cond_table5[1], fmt='.', yerr=Cond_table5[2], color='gray', ms=10, alpha=0.7,
                 label='$\mathrm{\mathbf{Hubble\;WFC3}}$')

    graph2_1.plot(wavelengths, transit_depth_2281K, '-', color=palette[9], linewidth=3,
                label='$\mathbf{T_\mathrm{\mathbf{spot}}=2281\,\mathrm{\mathbf{K}}}$')
    graph2_1.plot(wavelengths, transit_depth_2781K, '-', color=palette[8], linewidth=3,
                label='$\mathbf{T_\mathrm{\mathbf{spot}}=2781\,\mathrm{\mathbf{K}}}$')
    graph2_1.plot(wavelengths, transit_depth_3281K, '-', color=palette[7], linewidth=3,
                label='$\mathbf{T_\mathrm{\mathbf{spot}}=3281\,\mathrm{\mathbf{K}}}$')
    graph2_1.plot(wavelengths, transit_depth_3781K, '-', color=palette[6], linewidth=3,
                label='$\mathbf{T_\mathrm{\mathbf{spot}}=3781\,\mathrm{\mathbf{K}}}$')
    graph2_1.plot(wavelengths, transit_depth_4281K, '-', color=palette[5], linewidth=3,
                label='$\mathbf{T_\mathrm{\mathbf{spot}}=4281\,\mathrm{\mathbf{K}}}$')

    graph2_1.plot(wavelengths, transit_depth_5272K, '-', color=palette1[11], linewidth=3,
                label='$\mathbf{T_\mathrm{\mathbf{fac}}=5272\,\mathrm{\mathbf{K}}}$')
    graph2_1.plot(wavelengths, transit_depth_5472K, '-', color=palette1[9], linewidth=3,
                label='$\mathbf{T_\mathrm{\mathbf{fac}}=5472\,\mathrm{\mathbf{K}}}$')
    graph2_1.plot(wavelengths, transit_depth_5672K, '-', color=palette1[7], linewidth=3,
                label='$\mathbf{T_\mathrm{\mathbf{fac}}=5672\,\mathrm{\mathbf{K}}}$')
    graph2_1.plot(wavelengths, transit_depth_6172K, '-', color=palette1[5], linewidth=3,
                label='$\mathbf{T_\mathrm{\mathbf{fac}}=6172\,\mathrm{\mathbf{K}}}$')
    graph2_1.plot(wavelengths, transit_depth_7172K, '-', color=palette1[3], linewidth=3,
                label='$\mathbf{T_\mathrm{\mathbf{fac}}=7172\,\mathrm{\mathbf{K}}}$')

    graph2_1.tick_params(axis="x", direction="in", labelsize=12, which='both')
    graph2_1.tick_params(axis="y", direction="in", labelsize=12, which='both')
    plt.subplots_adjust(top=0.9)
    plt.xlim(470, 1700)
    plt.ylim(270, 435)
    graph2_1.legend( prop={'size': 12}, title_fontsize=15, loc='upper right')
    #graph3.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    #graph3.set_ylabel('D$_{\mathrm{unnoc}}$ -- D$_{\mathrm{phot}}$ [ppm]', fontsize=25, fontweight="bold") # labelpad é a distância entre o título e o eixo
    graph2_1.set_ylabel('Transit Depth [ppm]', fontsize=19, fontweight="bold", labelpad=10) # labelpad é a distância entre o título e o eixo
    graph2_1.set_xlabel('Wavelength (nm)', fontsize=19, fontweight="bold", labelpad=10)
    graph2_1.yaxis.set_major_locator(MultipleLocator(30))
    #graph2_1.xaxis.set_major_locator(MultipleLocator(30))
    #graph2_1.yaxis.set_minor_locator(MultipleLocator(10))
    #graph2_1.xaxis.set_minor_locator(MultipleLocator(10))
    at = AnchoredText("$\mathbf{T_{star}=5172\,K;\,ff=8\%}$", prop=dict(size=15),
                      frameon=True, loc='lower left') # frameon é o retângulo em volta do texto
    graph2_1.add_artist(at)



elif graph == 3:

    # Leitura de entrada dos dados

    transit_depth_phot_trans_lat_0 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=0graus,f_spot=0.00,temp_spot=3434K).txt", delimiter=",")
    transit_depth_phot_trans_lat_10 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-20graus,f_spot=0.00,temp_spot=3434K).txt", delimiter=",")
    transit_depth_phot_trans_lat_20 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-20graus,f_spot=0.00,temp_spot=3434K).txt", delimiter=",")
    transit_depth_phot_trans_lat_30 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-30graus,f_spot=0.00,temp_spot=3434K).txt", delimiter=",")
    transit_depth_phot_trans_lat_40 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-40graus,f_spot=0.00,temp_spot=3434K).txt", delimiter=",")
    transit_depth_phot_trans_lat_50 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-49graus,f_spot=0.00,temp_spot=3434K).txt", delimiter=",")
    transit_depth_phot_trans_lat_60 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-60graus,f_spot=0.00,temp_spot=3434K).txt", delimiter=",")
    transit_depth_phot_trans_lat_70 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-69graus,f_spot=0.00,temp_spot=3434K).txt", delimiter=",")
    transit_depth_phot_trans_lat_80 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-69graus,f_spot=0.00,temp_spot=3434K).txt", delimiter=",")
    wavelengths = np.genfromtxt("GJ9827d_output_wavelengths.txt", delimiter=",")

    palette = sns.color_palette("mako", 8)

    graph3 = fig.add_subplot(1, 1, 1)

    #graph3.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    #graph3.set_ylabel('D$_{\mathrm{unnoc}}$ -- D$_{\mathrm{phot}}$ [ppm]', fontsize=25, fontweight="bold") # labelpad é a distância entre o título e o eixo
    graph3.set_ylabel('Transit Depth [ppm]', fontsize=22, fontweight="bold", labelpad=10) # labelpad é a distância entre o título e o eixo
    graph3.set_xlabel('Wavelength (nm)', fontsize=22, fontweight="bold", labelpad=10)

    ######### Dados do Hubble ###############

    Cond_table5 = np.genfromtxt('GJ9827d_WFC3.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table5 = np.transpose(Cond_table5)

    plt.errorbar(Cond_table5[0] * 1000, Cond_table5[1] * 1.0e+4, fmt='.', yerr=Cond_table5[2] * 1.0e+4, color='gray', ms=10, alpha=0.7)
    ########################

    #graph3.plot(wavelengths, transit_depth_phot_trans_lat_0,  'o', linestyle='none', markersize=5, color=palette[0])
    graph3.plot(wavelengths, transit_depth_phot_trans_lat_0, '-', color=palette[0], linewidth=3,
                label='Trans. Lat. = 0$^{\circ}$')
    #graph3.plot(wavelengths, transit_depth_phot_trans_lat_10,  'o', linestyle='none', markersize=5, color=palette[0])
    graph3.plot(wavelengths, transit_depth_phot_trans_lat_10, '-', color=palette[1], linewidth=3,
                label='Trans. Lat. = 10$^{\circ}$')
    #graph3.plot(wavelengths, transit_depth_phot_trans_lat_20,  'o', linestyle='none', markersize=5, color=palette[1])
    graph3.plot(wavelengths, transit_depth_phot_trans_lat_20, '-', color=palette[2], linewidth=3,
                label='Trans. Lat. = 20$^{\circ}$')
    #graph3.plot(wavelengths, transit_depth_phot_trans_lat_30,  'o', linestyle='none', markersize=5, color=palette[2])
    graph3.plot(wavelengths, transit_depth_phot_trans_lat_30, '-', color=palette[3], linewidth=3,
                label='Trans. Lat. = 30$^{\circ}$')
    #graph3.plot(wavelengths, transit_depth_phot_trans_lat_40,  'o', linestyle='none', markersize=5, color=palette[2])
    graph3.plot(wavelengths, transit_depth_phot_trans_lat_40, '-', color=palette[4], linewidth=3,
                label='Trans. Lat. = 40$^{\circ}$')
    #graph3.plot(wavelengths, transit_depth_phot_trans_lat_50,  'o', linestyle='none', markersize=5, color=palette[3])
    graph3.plot(wavelengths, transit_depth_phot_trans_lat_50, '-', color=palette[5], linewidth=3,
                label='Trans. Lat. = 50$^{\circ}$')
    #graph3.plot(wavelengths, transit_depth_phot_trans_lat_60,  'o', linestyle='none', markersize=5, color=palette[3])
    graph3.plot(wavelengths, transit_depth_phot_trans_lat_60, '-', color=palette[6], linewidth=3,
                label='Trans. Lat. = 60$^{\circ}$')
    #graph3.plot(wavelengths, transit_depth_phot_trans_lat_70,  'o', linestyle='none', markersize=5, color=palette[3])
    graph3.plot(wavelengths, transit_depth_phot_trans_lat_70, '-', color=palette[7], linewidth=3,
                label='Trans. Lat. = 70$^{\circ}$')
    #graph3.plot(wavelengths, transit_depth_phot_trans_lat_80,  'o', linestyle='none', markersize=5, color=palette[4])
    #graph3.plot(wavelengths, transit_depth_phot_trans_lat_80, '-', color=palette[5], linewidth=3, label='Trans. Lat. = 80$^{\circ}$')

    plt.text(950, 16000, '$Photosphere$', fontsize=17, bbox=dict(facecolor='white', alpha=0.5))
    graph3.tick_params(axis="x", direction="in", labelsize=12)
    graph3.tick_params(axis="y", direction="in", labelsize=12)
    plt.subplots_adjust(top=0.9)
    plt.xlim(430, 1700)
    #plt.xlabel('Wavelength (nm)', labelpad=15)
    #plt.plot(wavelengths, y2(wavelengths), "-", color='red') # ajuste polinomial



elif (graph == 4):

    # Leitura de entrada dos dados

    transit_depth_trans_lat_0_fspot_0 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=0graus,f_spot=0.00,temp_spot=3781K).txt", delimiter=",")
    transit_depth_trans_lat_0_fspot_8 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=0graus,f_spot=0.08,temp_spot=3781K).txt", delimiter=",")
    transit_depth_trans_lat_20_fspot_0 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=-19graus,f_spot=0.00,temp_spot=3781K).txt", delimiter=",")
    transit_depth_trans_lat_20_fspot_8 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=-19graus,f_spot=0.08,temp_spot=3781K).txt", delimiter=",")
    transit_depth_trans_lat_30_fspot_0 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=-30graus,f_spot=0.00,temp_spot=3781K).txt", delimiter=",")
    transit_depth_trans_lat_30_fspot_8 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=-30graus,f_spot=0.08,temp_spot=3781K).txt", delimiter=",")
    transit_depth_trans_lat_40_fspot_0 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=-39graus,f_spot=0.00,temp_spot=3781K).txt", delimiter=",")
    transit_depth_trans_lat_40_fspot_8 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=-39graus,f_spot=0.08,temp_spot=3781K).txt", delimiter=",")
    transit_depth_trans_lat_50_fspot_0 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=-49graus,f_spot=0.00,temp_spot=3781K).txt", delimiter=",")
    transit_depth_trans_lat_50_fspot_8 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=-49graus,f_spot=0.08,temp_spot=3781K).txt", delimiter=",")
    transit_depth_trans_lat_60_fspot_0 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=-59graus,f_spot=0.00,temp_spot=3781K).txt", delimiter=",")
    transit_depth_trans_lat_60_fspot_8 = np.genfromtxt("55Cnc_e_output_transit_depth(trans_lat=-59graus,f_spot=0.08,temp_spot=3781K).txt", delimiter=",")
    wavelengths = np.genfromtxt("55Cnc_e_output_wavelengths.txt", delimiter=",")

    palette = sns.color_palette("Paired", 12)

    graph4 = fig.add_subplot(1, 1, 1)

    #graph4.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    #graph4.set_ylabel('D$_{\mathrm{unnoc}}$ -- D$_{\mathrm{phot}}$ [ppm]', fontsize=25, fontweight="bold") # labelpad é a distância entre o título e o eixo
    graph4.set_ylabel('Transit Depth [ppm]', fontsize=22, fontweight="bold", labelpad=10) # labelpad é a distância entre o título e o eixo
    graph4.set_xlabel('Wavelength (nm)', fontsize=22, fontweight="bold", labelpad=10)

    ######### Dados do Hubble ###############

    Cond_table5 = np.genfromtxt('55Cnc_e_WFC3.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table5 = np.transpose(Cond_table5)

    plt.errorbar(Cond_table5[0] * 1000, Cond_table5[1], fmt='.', yerr=Cond_table5[2], color='gray', ms=10, alpha=0.7,
                 label='$\mathbf{Hubble\;WFC3}$')
    ########################

    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_20,  'o', linestyle='none', markersize=5, color=palette[1])
    graph4.plot(wavelengths, transit_depth_trans_lat_0_fspot_8, '-', color=palette[0], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 0^{\circ}; ff=8\%}}$')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_0,  'o', linestyle='none', markersize=5, color=palette[0])
    graph4.plot(wavelengths, transit_depth_trans_lat_0_fspot_0, '-', color=palette[1], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 0^{\circ}; ff=0\%}}$')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_20,  'o', linestyle='none', markersize=5, color=palette[1])
    graph4.plot(wavelengths, transit_depth_trans_lat_20_fspot_8, '-', color=palette[2], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 20^{\circ}; ff=8\%}}$')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_0,  'o', linestyle='none', markersize=5, color=palette[0])
    graph4.plot(wavelengths, transit_depth_trans_lat_20_fspot_0, '-', color=palette[3], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 20^{\circ}; ff=0\%}}$')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_30,  'o', linestyle='none', markersize=5, color=palette[3])
    graph4.plot(wavelengths, transit_depth_trans_lat_30_fspot_8, '-', color=palette[4], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 30^{\circ}; ff=8\%}}$')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_0,  'o', linestyle='none', markersize=5, color=palette[0])
    graph4.plot(wavelengths, transit_depth_trans_lat_30_fspot_0, '-', color=palette[5], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 30^{\circ}; ff=0\%}}$')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_40,  'o', linestyle='none', markersize=5, color=palette[2])
    graph4.plot(wavelengths, transit_depth_trans_lat_40_fspot_8, '-', color=palette[6], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 40^{\circ}; ff=8\%}}$')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_40,  'o', linestyle='none', markersize=5, color=palette[2])
    graph4.plot(wavelengths, transit_depth_trans_lat_40_fspot_0, '-', color=palette[7], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 40^{\circ}; ff=0\%}}$')
    # graph4.plot(wavelengths, transit_depth_phot_trans_lat_50,  'o', linestyle='none', markersize=5, color=palette[3])
    graph4.plot(wavelengths, transit_depth_trans_lat_50_fspot_8, '-', color=palette[8], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 50^{\circ}, ff=8\%}}$')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_50,  'o', linestyle='none', markersize=5, color=palette[3])
    graph4.plot(wavelengths, transit_depth_trans_lat_50_fspot_0, '-', color=palette[9], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 50^{\circ}, ff=0\%}}$')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_60,  'o', linestyle='none', markersize=5, color=palette[4])
    graph4.plot(wavelengths, transit_depth_trans_lat_60_fspot_8, '-', color='wheat', linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 60^{\circ}; ff=8\%}}$')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_60,  'o', linestyle='none', markersize=5, color=palette[4])
    graph4.plot(wavelengths, transit_depth_trans_lat_60_fspot_0, '-', color=palette[11], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 60^{\circ}; ff=0\%}}$')


    #plt.text(550, 425, '55 Cnc e', fontsize=17, bbox=dict(facecolor='white', alpha=0.5))
    graph4.tick_params(axis="x", direction="in", labelsize=12)
    graph4.tick_params(axis="y", direction="in", labelsize=12)
    plt.subplots_adjust(top=0.9)
    plt.xlim(470, 1700)
    #plt.ylim(220, 450)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', prop={'size': 12})
    plt.tight_layout()
    #plt.xlabel('Wavelength (nm)', labelpad=15)
    #plt.plot(wavelengths, y2(wavelengths), "-", color='red') # ajuste polinomial
    at = AnchoredText("$\mathbf{55\,\mathrm{\mathbf{Cnc}}\,\mathrm{\mathbf{e}}}$", prop=dict(size=15),
                      frameon=True, loc='upper center') # frameon é o retângulo em volta do texto
    graph4.add_artist(at)


elif (graph == 5):

    # Leitura de entrada dos dados

    epsilon_Rackham_trans_lat_20_fspot_8 = np.genfromtxt("55Cnc_e_output_epsilon_Rackham(trans_lat=-24graus,f_spot=0.08,temp_spot=3781K).txt", delimiter=",")

    wavelengths = np.genfromtxt("55Cnc_e_output_wavelengths.txt", delimiter=",")

    table_epsilon_ourWork = np.genfromtxt('epsilon_fixed_fspot_8% (our work).txt', delimiter=",",
                                          usecols=(0, 1, 2, 3, 4, 5, 6, 7), skip_header=1)
    table_epsilon_ourWork = np.transpose(table_epsilon_ourWork)

    palette = sns.color_palette("tab10", 10)
    graph5 = fig.add_subplot(1, 1, 1)
    #graph5.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    graph5.set_ylabel('Contamination Factor ($\epsilon$)', fontsize=23, fontweight="bold",
                      labelpad=10) # labelpad é a distância entre o título e o eixo
    graph5.set_xlabel('Wavelength (nm)', fontsize=22, fontweight="bold", labelpad=10)

    #graph5.plot(wavelengths, table_epsilon_ourWork[0], '.', linestyle='none', markersize=5, color=palette[0])
    graph5.plot(wavelengths, table_epsilon_ourWork[7], '-', linewidth=3, color=palette[7],
                label='$\epsilon$ - Trans. Lat. = 70$^{\circ}$')
    graph5.plot(wavelengths, table_epsilon_ourWork[6], '-', linewidth=3, color=palette[4],
                label='$\epsilon$ - Trans. Lat. = 60$^{\circ}$')
    graph5.plot(wavelengths, table_epsilon_ourWork[5], '-', linewidth=3, color=palette[5],
                label='$\epsilon$ - Trans. Lat. = 50$^{\circ}$')
    graph5.plot(wavelengths, table_epsilon_ourWork[4], '-', linewidth=3, color=palette[6],
                label='$\epsilon$ - Trans. Lat. = 40$^{\circ}$')
    graph5.plot(wavelengths, table_epsilon_ourWork[3], '-', linewidth=3, color=palette[9],
                label='$\epsilon$ - Trans. Lat. = 30$^{\circ}$')
    graph5.plot(wavelengths, table_epsilon_ourWork[2], '-', linewidth=3, color=palette[1],
                label='$\epsilon$ - Trans. Lat. = 20$^{\circ}$')
    graph5.plot(wavelengths, table_epsilon_ourWork[1], '-', linewidth=3, color=palette[0],
                label='$\epsilon$ - Trans. Lat. = 10$^{\circ}$')
    graph5.plot(wavelengths, table_epsilon_ourWork[0], '-', linewidth=3, color=palette[2],
                label='$\epsilon$ - Trans. Lat. = 0$^{\circ}$')


    graph5.plot(wavelengths, epsilon_Rackham_trans_lat_20_fspot_8,  '-', linewidth=3, color='red',
                label='$\epsilon_{R}$')

    plt.text(950, 1.108, '$ff=8\%$', fontsize=21, bbox=dict(facecolor='white', alpha=0.5))
    #plt.xlim(430, 1700)
    graph5.tick_params(axis="x", direction="in", labelsize=17)
    graph5.tick_params(axis="y", direction="in", labelsize=17)
    plt.subplots_adjust(top=0.9)
    #plt.plot(wavelengths, y1(wavelengths), "-", color='red') # ajuste polinomial
    legend = plt.legend(prop={'size': 19})
    plt.subplots_adjust(left=0.11, bottom=0.14, right=0.95, top=0.96)
    plt.tick_params(axis="x", direction="in", labelsize=17)
    plt.tick_params(axis="y", direction="in", labelsize=17)

elif graph == 6:

    graph6 = fig.add_subplot(1, 3, 1)

    graph6.set_ylabel('Transit depth [ppm]', fontsize=22, fontweight="bold",
                      labelpad=10)  # labelpad é a distância entre o título e o eixo
    graph6.set_xlabel('Transit latitude [degree]', fontsize=22, fontweight="bold", labelpad=10)

    TL = [0, 20, 30, 40, 50, 60, 70]
    wave_452nm = [421.702154, 399.0390673, 368.9662141, 329.6637183, 281.9390945, 229.1255732, 172.9223357]
    wave_563nm = [382.8163897, 368.9050625, 349.3471909, 323.1423468, 289.417508, 249.643848, 202.7466675]
    wave_790nm = [355.7951136, 346.717192, 333.3344167, 315.4724578, 291.8799056, 263.6638526, 228.8377327]
    wave_1018nm = [343.1011744, 336.2449924, 325.7423313, 311.8092789, 293.015145, 270.2748884, 241.1392687]
    wave_1743nm = [321.7567214, 318.8963701, 313.3175355, 306.438898, 295.4859, 281.9035769, 261.6057927]

    palette = sns.color_palette("Spectral", 10)
    graph6.plot(TL, wave_452nm, 'o', linewidth=3, color='blue', label='$\mathrm{\mathbf{452\,nm}}$')
    graph6.plot(TL, wave_452nm, '-', linewidth=3, color='blue')
    graph6.plot(TL, wave_563nm, 'o', linewidth=3, color='darkturquoise', label='$\mathrm{\mathbf{563\,nm}}$')
    graph6.plot(TL, wave_563nm, '-', linewidth=3, color='darkturquoise')
    graph6.plot(TL, wave_790nm, 'o', linewidth=3, color='mediumseagreen', label='$\mathrm{\mathbf{790\,nm}}$')
    graph6.plot(TL, wave_790nm, '-', linewidth=3, color='mediumseagreen')
    graph6.plot(TL, wave_1018nm, 'o', linewidth=3, color='darkorange', label='$\mathrm{\mathbf{1018\,nm}}$')
    graph6.plot(TL, wave_1018nm, '-', linewidth=3, color='darkorange')
    graph6.plot(TL, wave_1743nm, 'o', linewidth=3, color='red', label='$\mathrm{\mathbf{1743\,nm}}$')
    graph6.plot(TL, wave_1743nm, '-', linewidth=3, color='red')

    legend = plt.legend(prop={'size': 12})
    plt.tick_params(axis="x", direction="in", labelsize=15)
    plt.tick_params(axis="y", direction="in", labelsize=15)
    at = AnchoredText("$\mathbf{ff = 0\%}$", prop=dict(size=15),
                      frameon=True, loc='lower left')
    graph6.add_artist(at)


    graph6 = fig.add_subplot(1, 3, 2)


    graph6.set_xlabel('Transit latitude [degree]', fontsize=22, fontweight="bold", labelpad=10)

    TL = [0, 20, 30, 40, 50, 60, 70]
    ff_4_wave_452nm = [429.6705122, 410.5803228, 382.8281552, 346.0353962, 300.1602303, 247.7219145, 189.2501477]
    ff_4_wave_563nm = [399.1618113, 385.6271847, 365.182773, 337.7900878, 302.5365335, 260.9599707, 211.9369847]
    ff_4_wave_790nm = [368.1274286, 359.3848375, 345.5131096, 326.9985469, 302.5440181, 273.2970646, 237.198539]
    ff_4_wave_1018nm = [353.2701816, 346.7116094, 335.8820221, 321.5152623, 302.1361055, 278.6879914, 248.6454396]
    ff_4_wave_1743nm = [328.649586, 326.0229794, 320.3194705, 313.2871111, 302.0893385, 288.2034814, 267.4520877]

    palette = sns.color_palette("Spectral", 10)
    graph6.plot(TL, ff_4_wave_452nm, 'o', linewidth=3, color='blue', label='$\mathrm{\mathbf{452\,nm}}$')
    graph6.plot(TL, ff_4_wave_452nm, '-', linewidth=3, color='blue')
    graph6.plot(TL, ff_4_wave_563nm, 'o', linewidth=3, color='darkturquoise', label='$\mathrm{\mathbf{563\,nm}}$')
    graph6.plot(TL, ff_4_wave_563nm, '-', linewidth=3, color='darkturquoise')
    graph6.plot(TL, ff_4_wave_790nm, 'o', linewidth=3, color='mediumseagreen', label='$\mathrm{\mathbf{790\,nm}}$')
    graph6.plot(TL, ff_4_wave_790nm, '-', linewidth=3, color='mediumseagreen')
    graph6.plot(TL, ff_4_wave_1018nm, 'o', linewidth=3, color='darkorange', label='$\mathrm{\mathbf{1018\,nm}}$')
    graph6.plot(TL, ff_4_wave_1018nm, '-', linewidth=3, color='darkorange')
    graph6.plot(TL, ff_4_wave_1743nm, 'o', linewidth=3, color='red', label='$\mathrm{\mathbf{1743\,nm}}$')
    graph6.plot(TL, ff_4_wave_1743nm, '-', linewidth=3, color='red')

    plt.legend(prop={'size': 12})
    plt.tick_params(axis="x", direction="in", labelsize=15)
    plt.tick_params(axis="y", direction="in", labelsize=15)
    at = AnchoredText("$\mathbf{ff = 4\%}$", prop=dict(size=15),
                      frameon=True, loc='lower left')
    graph6.add_artist(at)


    graph6 = fig.add_subplot(1, 3, 3)

    graph6.set_xlabel('Transit latitude [degree]', fontsize=22, fontweight="bold", labelpad=10)

    TL = [0, 20, 30, 40, 50, 60, 70]
    ff_8_wave_452nm = [448.9765803, 432.7477269, 403.4972081, 364.7179926, 316.3660072, 261.0965247, 199.4678427]
    ff_8_wave_563nm = [415.0518441, 403.68792, 382.2860003, 353.6103868, 316.7057427, 273.1819538, 221.8629907]
    ff_8_wave_790nm = [380.1246091, 372.8821713, 358.489466, 339.2795562, 313.9065943, 283.5612196, 246.10695]
    ff_8_wave_1018nm = [363.1474702, 357.7662348, 346.591354, 331.7665221, 311.7694762, 287.5737376, 256.5733028]
    ff_8_wave_1743nm = [335.3382359, 333.4477031, 327.6143046, 320.4217929, 308.9690065, 294.7669181, 273.5429401]

    palette = sns.color_palette("Spectral", 10)
    graph6.plot(TL, ff_8_wave_452nm, 'o', linewidth=3, color='blue', label='$\mathrm{\mathbf{452\,nm}}$')
    graph6.plot(TL, ff_8_wave_452nm, '-', linewidth=3, color='blue')
    graph6.plot(TL, ff_8_wave_563nm, 'o', linewidth=3, color='darkturquoise', label='$\mathrm{\mathbf{563\,nm}}$')
    graph6.plot(TL, ff_8_wave_563nm, '-', linewidth=3, color='darkturquoise')
    graph6.plot(TL, ff_8_wave_790nm, 'o', linewidth=3, color='mediumseagreen', label='$\mathrm{\mathbf{790\,nm}}$')
    graph6.plot(TL, ff_8_wave_790nm, '-', linewidth=3, color='mediumseagreen')
    graph6.plot(TL, ff_8_wave_1018nm, 'o', linewidth=3, color='darkorange', label='$\mathrm{\mathbf{1018\,nm}}$')
    graph6.plot(TL, ff_8_wave_1018nm, '-', linewidth=3, color='darkorange')
    graph6.plot(TL, ff_8_wave_1743nm, 'o', linewidth=3, color='red', label='$\mathrm{\mathbf{1743\,nm}}$')
    graph6.plot(TL, ff_8_wave_1743nm, '-', linewidth=3, color='red')

    legend = plt.legend(prop={'size': 12})
    plt.tick_params(axis="x", direction="in", labelsize=15)
    plt.tick_params(axis="y", direction="in", labelsize=15)
    at = AnchoredText("$\mathbf{ff = 8\%}$", prop=dict(size=15),
                      frameon=True, loc='lower left')
    graph6.add_artist(at)

elif graph == 7:

    graph7 = fig.add_subplot(1, 1, 1)

    graph7.set_ylabel('Transit depth [ppm]', fontsize=22, fontweight="bold",
                      labelpad=10)  # labelpad é a distância entre o título e o eixo
    graph7.set_xlabel('$\mathbf{Faculae\;Temperature\;[\%\,T_{star}]}$', fontsize=22, fontweight="bold", labelpad=10)

    temp_porcentagem = [101.933488, 105.800464, 109.6674401, 119.3348801, 138.6697602]
    wave_452nm = [375.6901754, 368.9072451, 361.395753, 339.855514, 289.6882987]
    wave_522nm = [363.3513206, 357.8450057, 351.8707572, 335.191029, 297.2376388]
    wave_563nm = [357.7526576, 352.7971998, 347.4737768, 332.8084986, 299.9013473]
    wave_614nm = [352.454755, 348.0317054, 343.3273515, 330.5420356, 302.284814]
    wave_665nm = [347.1948856, 343.2100116, 339.0066891, 327.7122908, 303.0832039]

    palette = sns.color_palette("Spectral", 10)
    graph7.plot(temp_porcentagem, wave_665nm, 'o', linewidth=3, color='red', label='$\mathrm{\mathbf{665\,nm}}$')
    graph7.plot(temp_porcentagem, wave_665nm, '-', linewidth=3, color='red')
    graph7.plot(temp_porcentagem, wave_614nm, 'o', linewidth=3, color='darkorange', label='$\mathrm{\mathbf{614\,nm}}$')
    graph7.plot(temp_porcentagem, wave_614nm, '-', linewidth=3, color='darkorange')
    graph7.plot(temp_porcentagem, wave_563nm, 'o', linewidth=3, color='mediumseagreen', label='$\mathrm{\mathbf{563\,nm}}$')
    graph7.plot(temp_porcentagem, wave_563nm, '-', linewidth=3, color='mediumseagreen')
    graph7.plot(temp_porcentagem, wave_522nm, 'o', linewidth=3, color='darkturquoise', label='$\mathrm{\mathbf{522\,nm}}$')
    graph7.plot(temp_porcentagem, wave_522nm, '-', linewidth=3, color='darkturquoise')
    graph7.plot(temp_porcentagem, wave_452nm, 'o', linewidth=3, color='blue', label='$\mathrm{\mathbf{452\,nm}}$')
    graph7.plot(temp_porcentagem, wave_452nm, '-', linewidth=3, color='blue')

    legend = plt.legend(prop={'size': 12})
    plt.tick_params(axis="x", direction="in", labelsize=15)
    plt.tick_params(axis="y", direction="in", labelsize=15)
    handles, labels = graph7.get_legend_handles_labels()  # get the handles
    graph7.legend(reversed(handles), reversed(labels))
    at = AnchoredText("$\mathbf{T_{star} = 5172\,K}$", prop=dict(size=15),
                      frameon=True, loc='lower left')
    graph7.add_artist(at)

#legend = plt.legend(prop={'size': 12})

plt.tick_params(axis="x", direction="in", labelsize=15)
plt.tick_params(axis="y", direction="in", labelsize=15)


plt.show()