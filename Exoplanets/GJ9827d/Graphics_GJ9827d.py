import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
import math
from matplotlib.offsetbox import AnchoredText

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

graph = 2

if graph == 1:

    # Leitura de entrada dos dados

    epsilon_Rackham_ff_2 = np.genfromtxt("GJ9827d_output_epsilon_Rackham(trans_lat=-62graus,f_spot=0.02,temp_spot=3434K).txt", delimiter=",")
    epsilon_Rackham_ff_4 = np.genfromtxt("GJ9827d_output_epsilon_Rackham(trans_lat=-62graus,f_spot=0.04,temp_spot=3434K).txt", delimiter=",")
    epsilon_Rackham_ff_6 = np.genfromtxt("GJ9827d_output_epsilon_Rackham(trans_lat=-62graus,f_spot=0.06,temp_spot=3434K).txt", delimiter=",")
    epsilon_Rackham_ff_8 = np.genfromtxt("GJ9827d_output_epsilon_Rackham(trans_lat=-62graus,f_spot=0.08,temp_spot=3434K).txt", delimiter=",")
    epsilon_Rackham_ff_10 = np.genfromtxt("GJ9827d_output_epsilon_Rackham(trans_lat=-62graus,f_spot=0.10,temp_spot=3434K).txt", delimiter=",")
    wavelengths = np.genfromtxt("GJ9827d_output_wavelengths.txt", delimiter=",")

    table_epsilon_ourWork = np.genfromtxt('epsilon (our work).txt', delimiter=",", usecols=(0, 1, 2, 3, 4, 5),
                                          skip_header=1)
    table_epsilon_ourWork = np.transpose(table_epsilon_ourWork)

    palette = sns.color_palette("Paired", 16)
    graph1 = fig.add_subplot(1, 1, 1)
    #graph1.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    graph1.set_ylabel('Contamination Factor ($\epsilon$)', fontsize=23, fontweight="bold",
                      labelpad=10) # labelpad é a distância entre o título e o eixo
    graph1.set_xlabel('Wavelength (nm)', fontsize=23, fontweight="bold", labelpad=10)
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
    plt.ylim(1.000, 1.135)
    graph1.tick_params(axis="x", direction="in", labelsize=17)
    graph1.tick_params(axis="y", direction="in", labelsize=17)
    plt.subplots_adjust(top=0.9)
    #plt.plot(wavelengths, y1(wavelengths), "-", color='red') # ajuste polinomial
    legend = plt.legend(prop={'size': 19}, title_fontsize=19)
    plt.subplots_adjust(left=0.11, bottom=0.14, right=0.95, top=0.96)
    plt.tick_params(axis="x", direction="in", labelsize=17)
    plt.tick_params(axis="y", direction="in", labelsize=17)
    at = AnchoredText('$\mathbf{GJ\,9827\,d}:\;\mathbf{Trans. Lat. = 62.8^{\circ}}$', prop=dict(size=21),
                      frameon=True, loc='upper center') # frameon é o retângulo em volta do texto
    graph1.add_artist(at)

elif graph == 2:

    # Leitura de entrada dos dados

    transit_depth_tl_63_ff_0 = np.genfromtxt(
        "GJ9827d_output_transit_depth(trans_lat=-63graus,f_spot=0.00,temp_spot=3434K).txt", delimiter=",")
    transit_depth_tl_63_ff_2 = np.genfromtxt(
        "GJ9827d_output_transit_depth(trans_lat=-63graus,f_spot=0.02,temp_spot=3434K).txt", delimiter=",")
    transit_depth_tl_63_ff_4 = np.genfromtxt(
        "GJ9827d_output_transit_depth(trans_lat=-63graus,f_spot=0.04,temp_spot=3434K).txt", delimiter=",")
    transit_depth_tl_63_ff_6 = np.genfromtxt(
        "GJ9827d_output_transit_depth(trans_lat=-63graus,f_spot=0.06,temp_spot=3434K).txt", delimiter=",")
    transit_depth_tl_63_ff_8 = np.genfromtxt(
        "GJ9827d_output_transit_depth(trans_lat=-63graus,f_spot=0.08,temp_spot=3434K).txt", delimiter=",")
    transit_depth_tl_63_ff_10 = np.genfromtxt(
        "GJ9827d_output_transit_depth(trans_lat=-63graus,f_spot=0.10,temp_spot=3434K).txt", delimiter=",")

    transit_depth_3934K = np.genfromtxt(
        "GJ9827d_output_transit_depth(trans_lat=-63graus,f_spot=0.08,temp_spot=3934K).txt", delimiter=",")
    transit_depth_3434K = np.genfromtxt(
        "GJ9827d_output_transit_depth(trans_lat=-63graus,f_spot=0.08,temp_spot=3434K).txt", delimiter=",")
    transit_depth_2934K = np.genfromtxt(
        "GJ9827d_output_transit_depth(trans_lat=-63graus,f_spot=0.08,temp_spot=2934K).txt", delimiter=",")
    transit_depth_2434K = np.genfromtxt(
        "GJ9827d_output_transit_depth(trans_lat=-63graus,f_spot=0.08,temp_spot=2434K).txt", delimiter=",")
    transit_depth_1934K = np.genfromtxt(
        "GJ9827d_output_transit_depth(trans_lat=-63graus,f_spot=0.08,temp_spot=1934K).txt", delimiter=",")

    wavelengths = np.genfromtxt("GJ9827d_output_wavelengths.txt", delimiter=",")

    palette = sns.color_palette("mako", 6)

    ######### Dados do Hubble ###############

    Cond_table5 = np.genfromtxt('GJ9827d_WFC3.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table5 = np.transpose(Cond_table5)

    ########################

    graph2 = fig.add_subplot(2, 1, 1)
    plt.errorbar(Cond_table5[0] * 1000, Cond_table5[1], fmt='.', yerr=Cond_table5[2], color='gray', ms=10, alpha=0.7,
                 label='$\mathrm{\mathbf{Hubble\;WFC3}}$')
    # graph2.plot(wavelengths, transit_depth_tl_46_ff_10,  'o', linestyle='none', markersize=5, color=palette[5])
    graph2.plot(wavelengths, transit_depth_tl_63_ff_10, '-', color=palette[5], linewidth=3,
                label='$\mathbf{ff=10\%}$')
    # graph2.plot(wavelengths, transit_depth_tl_46_ff_8,  'o', linestyle='none', markersize=5, color=palette[4])
    graph2.plot(wavelengths, transit_depth_tl_63_ff_8, '-', color=palette[4], linewidth=3,
                label='$\mathbf{ff=8\%}$')
    # graph2.plot(wavelengths, transit_depth_tl_46_ff_6,  'o', linestyle='none', markersize=5, color=palette[3])
    graph2.plot(wavelengths, transit_depth_tl_63_ff_6, '-', color=palette[3], linewidth=3,
                label='$\mathbf{ff=6\%}$')
    # graph2.plot(wavelengths, transit_depth_tl_46_ff_4,  'o', linestyle='none', markersize=5, color=palette[2])
    graph2.plot(wavelengths, transit_depth_tl_63_ff_4, '-', color=palette[2], linewidth=3,
                label='$\mathbf{ff=4\%}$')
    # graph2.plot(wavelengths, transit_depth_tl_46_ff_2,  'o', linestyle='none', markersize=5, color=palette[1])
    graph2.plot(wavelengths, transit_depth_tl_63_ff_2, '-', color=palette[1], linewidth=3,
                label='$\mathbf{ff=2\%}$')
    # graph2.plot(wavelengths, transit_depth_tl_46_ff_0,  'o', linestyle='none', markersize=5, color=palette[0])
    graph2.plot(wavelengths, transit_depth_tl_63_ff_0, '-', color=palette[0], linewidth=3,
                label='$\mathrm{\mathbf{ff=0\%}}$')

    graph2.tick_params(axis="x", direction="in", labelsize=15)
    graph2.tick_params(axis="y", direction="in", labelsize=15)
    plt.subplots_adjust(top=0.9)
    plt.xlim(470, 1700)
    plt.ylim(600, 1050)
    legend = plt.legend(prop={'size': 12}, title_fontsize=15, loc='lower right')
    # graph2.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    # graph2.set_ylabel('D$_{\mathrm{unnoc}}$ -- D$_{\mathrm{phot}}$ [ppm]', fontsize=25, fontweight="bold") # labelpad é a distância entre o título e o eixo
    graph2.set_ylabel('Transit Depth [ppm]', fontsize=19, fontweight="bold",
                      labelpad=10)  # labelpad é a distância entre o título e o eixo
    # graph2.set_xlabel('Wavelength (nm)', fontsize=22, fontweight="bold", labelpad=10)
    at = AnchoredText("$\mathrm{\mathbf{GJ}}\,\mathbf{9827}\,\mathrm{\mathbf{d}}$", prop=dict(size=15),
                      frameon=True, loc='lower left')
    graph2.add_artist(at)

    graph2_1 = fig.add_subplot(2, 1, 2)
    plt.errorbar(Cond_table5[0] * 1000, Cond_table5[1], fmt='.', yerr=Cond_table5[2], color='gray', ms=10, alpha=0.7,
                 label='$\mathrm{\mathbf{Hubble\;WFC3}}$')
    palette1 = sns.color_palette("flare_r", 5)
    graph2_1.plot(wavelengths, transit_depth_3934K, '-', color=palette1[0], linewidth=3,
                  label='$\mathbf{T_\mathrm{\mathbf{spot}}=3934\,\mathrm{\mathbf{K}}}$')
    graph2_1.plot(wavelengths, transit_depth_3434K, '-', color=palette1[1], linewidth=3,
                  label='$\mathbf{T_\mathrm{\mathbf{spot}}=3434\,\mathrm{\mathbf{K}}}$')
    graph2_1.plot(wavelengths, transit_depth_2934K, '-', color=palette1[2], linewidth=3,
                  label='$\mathbf{T_\mathrm{\mathbf{\mathbf{spot}}}=2934\,\mathrm{\mathbf{K}}}$')
    graph2_1.plot(wavelengths, transit_depth_2434K, '-', color=palette1[3], linewidth=3,
                  label='$\mathbf{T_\mathrm{\mathbf{spot}}=2434\,\mathrm{\mathbf{K}}}$')
    graph2_1.plot(wavelengths, transit_depth_1934K, '-', color=palette1[4], linewidth=3,
                  label='$\mathbf{T_\mathrm{\mathbf{spot}}=1934\,\mathrm{\mathbf{K}}}$')

    graph2_1.tick_params(axis="x", direction="in", labelsize=12)
    graph2_1.tick_params(axis="y", direction="in", labelsize=12)
    plt.subplots_adjust(top=0.9)
    plt.xlim(470, 1700)
    plt.ylim(600, 1050)
    legend = plt.legend(prop={'size': 12}, title_fontsize=15, loc='lower right')
    # graph3.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    # graph3.set_ylabel('D$_{\mathrm{unnoc}}$ -- D$_{\mathrm{phot}}$ [ppm]', fontsize=25, fontweight="bold") # labelpad é a distância entre o título e o eixo
    graph2_1.set_ylabel('Transit Depth [ppm]', fontsize=19, fontweight="bold",
                        labelpad=10)  # labelpad é a distância entre o título e o eixo
    graph2_1.set_xlabel('Wavelength (nm)', fontsize=19, fontweight="bold", labelpad=10)
    at = AnchoredText("$\mathrm{\mathbf{GJ}}\,\mathbf{9827}\,\mathrm{\mathbf{d}}\mathbf{: ff=8\%}}$", prop=dict(size=15),
                      frameon=True, loc='lower left')  # frameon é o retângulo em volta do texto
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



elif graph == 4:

    # Leitura de entrada dos dados

    transit_depth_trans_lat_0_fspot_0 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=0graus,f_spot=0.00,temp_spot=3434K).txt", delimiter=",")
    transit_depth_trans_lat_0_fspot_4 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=0graus,f_spot=0.08,temp_spot=3434K).txt", delimiter=",")
    transit_depth_trans_lat_20_fspot_0 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-20graus,f_spot=0.00,temp_spot=3434K).txt", delimiter=",")
    transit_depth_trans_lat_20_fspot_4 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-20graus,f_spot=0.08,temp_spot=3434K).txt", delimiter=",")
    transit_depth_trans_lat_30_fspot_0 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-30graus,f_spot=0.00,temp_spot=3434K).txt", delimiter=",")
    transit_depth_trans_lat_30_fspot_4 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-30graus,f_spot=0.08,temp_spot=3434K).txt", delimiter=",")
    transit_depth_trans_lat_40_fspot_0 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-40graus,f_spot=0.00,temp_spot=3434K).txt", delimiter=",")
    transit_depth_trans_lat_40_fspot_4 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-40graus,f_spot=0.08,temp_spot=3434K).txt", delimiter=",")
    transit_depth_trans_lat_50_fspot_0 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-49graus,f_spot=0.00,temp_spot=3434K).txt", delimiter=",")
    transit_depth_trans_lat_50_fspot_4 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-49graus,f_spot=0.08,temp_spot=3434K).txt", delimiter=",")
    transit_depth_trans_lat_60_fspot_0 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-60graus,f_spot=0.00,temp_spot=3434K).txt", delimiter=",")
    transit_depth_trans_lat_60_fspot_4 = np.genfromtxt("GJ9827d_output_transit_depth(trans_lat=-60graus,f_spot=0.08,temp_spot=3434K).txt", delimiter=",")
    wavelengths = np.genfromtxt("GJ9827d_output_wavelengths.txt", delimiter=",")

    palette = sns.color_palette("Paired", 12)

    graph4 = fig.add_subplot(1, 1, 1)

    #graph4.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    #graph4.set_ylabel('D$_{\mathrm{unnoc}}$ -- D$_{\mathrm{phot}}$ [ppm]', fontsize=25, fontweight="bold") # labelpad é a distância entre o título e o eixo
    graph4.set_ylabel('Transit Depth [ppm]', fontsize=22, fontweight="bold", labelpad=10) # labelpad é a distância entre o título e o eixo
    graph4.set_xlabel('Wavelength (nm)', fontsize=22, fontweight="bold", labelpad=10)

    ######### Dados do Hubble ###############

    Cond_table5 = np.genfromtxt('GJ9827d_WFC3.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table5 = np.transpose(Cond_table5)

    plt.errorbar(Cond_table5[0] * 1000, Cond_table5[1], fmt='.', yerr=Cond_table5[2],
                 color='gray', ms=10, alpha=0.7, label='$\mathbf{Hubble\;WFC3}$')
    ########################

    # graph4.plot(wavelengths, transit_depth_phot_trans_lat_20,  'o', linestyle='none', markersize=5, color=palette[1])
    graph4.plot(wavelengths, transit_depth_trans_lat_0_fspot_4, '-', color=palette[0], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 0^{\circ}; ff=8\%}}$')
    # graph4.plot(wavelengths, transit_depth_phot_trans_lat_0,  'o', linestyle='none', markersize=5, color=palette[0])
    graph4.plot(wavelengths, transit_depth_trans_lat_0_fspot_0, '-', color=palette[1], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 0^{\circ}; ff=0\%}}$')
    # graph4.plot(wavelengths, transit_depth_phot_trans_lat_20,  'o', linestyle='none', markersize=5, color=palette[1])
    graph4.plot(wavelengths, transit_depth_trans_lat_20_fspot_4, '-', color=palette[2], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 20^{\circ}; ff=8\%}}$')
    # graph4.plot(wavelengths, transit_depth_phot_trans_lat_0,  'o', linestyle='none', markersize=5, color=palette[0])
    graph4.plot(wavelengths, transit_depth_trans_lat_20_fspot_0, '-', color=palette[3], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 20^{\circ}; ff=0\%}}$')
    # graph4.plot(wavelengths, transit_depth_phot_trans_lat_30,  'o', linestyle='none', markersize=5, color=palette[3])
    graph4.plot(wavelengths, transit_depth_trans_lat_30_fspot_4, '-', color=palette[4], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 30^{\circ}; ff=8\%}}$')
    # graph4.plot(wavelengths, transit_depth_phot_trans_lat_0,  'o', linestyle='none', markersize=5, color=palette[0])
    graph4.plot(wavelengths, transit_depth_trans_lat_30_fspot_0, '-', color=palette[5], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 30^{\circ}; ff=0\%}}$')
    # graph4.plot(wavelengths, transit_depth_phot_trans_lat_40,  'o', linestyle='none', markersize=5, color=palette[2])
    graph4.plot(wavelengths, transit_depth_trans_lat_40_fspot_4, '-', color=palette[6], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 40^{\circ}; ff=8\%}}$')
    # graph4.plot(wavelengths, transit_depth_phot_trans_lat_40,  'o', linestyle='none', markersize=5, color=palette[2])
    graph4.plot(wavelengths, transit_depth_trans_lat_40_fspot_0, '-', color=palette[7], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 40^{\circ}; ff=0\%}}$')
    # graph4.plot(wavelengths, transit_depth_phot_trans_lat_50,  'o', linestyle='none', markersize=5, color=palette[3])
    graph4.plot(wavelengths, transit_depth_trans_lat_50_fspot_4, '-', color=palette[8], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 50^{\circ}, ff=8\%}}$')
    # graph4.plot(wavelengths, transit_depth_phot_trans_lat_50,  'o', linestyle='none', markersize=5, color=palette[3])
    graph4.plot(wavelengths, transit_depth_trans_lat_50_fspot_0, '-', color=palette[9], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 50^{\circ}, ff=0\%}}$')
    # graph4.plot(wavelengths, transit_depth_phot_trans_lat_60,  'o', linestyle='none', markersize=5, color=palette[4])
    graph4.plot(wavelengths, transit_depth_trans_lat_60_fspot_4, '-', color='wheat', linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 60^{\circ}; ff=8\%}}$')
    # graph4.plot(wavelengths, transit_depth_phot_trans_lat_60,  'o', linestyle='none', markersize=5, color=palette[4])
    graph4.plot(wavelengths, transit_depth_trans_lat_60_fspot_0, '-', color=palette[11], linewidth=3,
                label='$\mathrm{\mathbf{Trans. Lat. = 60^{\circ}; ff=0\%}}$')


    #plt.text(550, 425, '55 Cnc e', fontsize=17, bbox=dict(facecolor='white', alpha=0.5))
    graph4.tick_params(axis="x", direction="in", labelsize=12)
    graph4.tick_params(axis="y", direction="in", labelsize=12)
    plt.subplots_adjust(top=0.9)
    plt.xlim(470, 1700)
    plt.ylim(600, 1700)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', prop={'size': 12})
    plt.tight_layout()
    #plt.xlabel('Wavelength (nm)', labelpad=15)
    #plt.plot(wavelengths, y2(wavelengths), "-", color='red') # ajuste polinomial
    at = AnchoredText("$\mathrm{\mathbf{GJ}}\,\mathbf{9827}\,\mathrm{\mathbf{d}}}$", prop=dict(size=15),
                      frameon=True, loc='upper center') # frameon é o retângulo em volta do texto
    graph4.add_artist(at)


elif graph == 5:

    # Leitura de entrada dos dados

    epsilon_Rackham_trans_lat_62_fspot_8 = np.genfromtxt("GJ9827d_output_epsilon_Rackham(trans_lat=-62graus,f_spot=0.08,temp_spot=3434K).txt", delimiter=",")

    wavelengths = np.genfromtxt("GJ9827d_output_wavelengths.txt", delimiter=",")

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


    graph5.plot(wavelengths, epsilon_Rackham_trans_lat_62_fspot_8,  '-', linewidth=3, color='red',
                label='$\epsilon_{R}$')


    plt.text(870, 1.119, '$ff=8\%$', fontsize=21, bbox=dict(facecolor='white', alpha=0.5))
    plt.xlim(500, 1700)
    plt.ylim(1.000, 1.135)
    graph5.tick_params(axis="x", direction="in", labelsize=17)
    graph5.tick_params(axis="y", direction="in", labelsize=17)
    plt.subplots_adjust(top=0.9)
    #plt.plot(wavelengths, y1(wavelengths), "-", color='red') # ajuste polinomial
    legend = plt.legend(prop={'size': 19}, title='GJ$\,$9827$\,$d', title_fontsize=19)
    plt.subplots_adjust(left=0.11, bottom=0.14, right=0.95, top=0.96)
    plt.tick_params(axis="x", direction="in", labelsize=17)
    plt.tick_params(axis="y", direction="in", labelsize=17)



#legend = plt.legend(prop={'size': 12})

plt.tick_params(axis="x", direction="in", labelsize=15)
plt.tick_params(axis="y", direction="in", labelsize=15)


plt.show()