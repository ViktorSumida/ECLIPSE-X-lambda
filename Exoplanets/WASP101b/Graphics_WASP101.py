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

graph = 5

if (graph == 1):

    # Leitura de entrada dos dados

    epsilon_Rackham_ff_2 = np.genfromtxt("Exoplanets/WASP101b/output_epsilon_Rackham(trans_lat=-46graus,f_spot=0.02).txt", delimiter=",")
    epsilon_Rackham_ff_4 = np.genfromtxt("Exoplanets/WASP101b/output_epsilon_Rackham(trans_lat=-46graus,f_spot=0.04).txt", delimiter=",")
    epsilon_Rackham_ff_6 = np.genfromtxt("Exoplanets/WASP101b/output_epsilon_Rackham(trans_lat=-46graus,f_spot=0.06).txt", delimiter=",")
    epsilon_Rackham_ff_8 = np.genfromtxt("Exoplanets/WASP101b/output_epsilon_Rackham(trans_lat=-46graus,f_spot=0.08).txt", delimiter=",")
    epsilon_Rackham_ff_10 = np.genfromtxt("Exoplanets/WASP101b/output_epsilon_Rackham(trans_lat=-46graus,f_spot=0.10).txt", delimiter=",")
    wavelengths = np.genfromtxt("Exoplanets/WASP101b/output_wavelengths.txt", delimiter=",")

    table_epsilon_ourWork = np.genfromtxt('Exoplanets/WASP101b/output_epsilon (our work).txt', delimiter=",", usecols=(6, 7, 8, 9, 10),
                                          skip_header=1)
    table_epsilon_ourWork = np.transpose(table_epsilon_ourWork)

    palette = sns.color_palette("Paired", 16)
    graph1 = fig.add_subplot(1, 1, 1)
    #graph1.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    graph1.set_ylabel('Contamination Factor ($\epsilon$)', fontsize=22, fontweight="bold",
                      labelpad=10) # labelpad é a distância entre o título e o eixo
    graph1.set_xlabel('Wavelength (nm)', fontsize=22, fontweight="bold", labelpad=10)
    #graph1.plot(wavelengths, epsilon,  'o', linestyle='none', markersize=7, color='red', label='$\epsilon$')

    graph1.plot(wavelengths, table_epsilon_ourWork[4], '-', linewidth=3, color=palette[9],
                label='$\epsilon$ - f$_{spot}=10\%$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_10,  'o', linestyle='none', markersize=5, color=palette[4])
    graph1.plot(wavelengths, epsilon_Rackham_ff_10,  '--', linewidth=3, color=palette[8],
                label='$\epsilon_{R}$ - f$_{spot}=10\%$')
    graph1.plot(wavelengths, table_epsilon_ourWork[3], '-', linewidth=3, color=palette[7],
                label='$\epsilon$ - f$_{spot}=8\%$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_8,  'o', linestyle='none', markersize=5, color=palette[3])
    graph1.plot(wavelengths, epsilon_Rackham_ff_8,  '--', linewidth=3, color=palette[6],
                label='$\epsilon_{R}$ - f$_{spot}=8\%$')
    graph1.plot(wavelengths, table_epsilon_ourWork[2], '-', linewidth=3, color=palette[5],
                label='$\epsilon$ - f$_{spot}=6\%$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_6,  'o', linestyle='none', markersize=5, color=palette[2])
    graph1.plot(wavelengths, epsilon_Rackham_ff_6,  '--', linewidth=3, color=palette[4],
                label='$\epsilon_{R}$ - f$_{spot}=6\%$')
    graph1.plot(wavelengths, table_epsilon_ourWork[1], '-', linewidth=3, color=palette[3],
                label='$\epsilon$ - f$_{spot}=4\%$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_4,  'o', linestyle='none', markersize=5, color=palette[1])
    graph1.plot(wavelengths, epsilon_Rackham_ff_4,  '--', linewidth=3, color=palette[2],
                label='$\epsilon_{R}$ - f$_{spot}=4\%$')
    graph1.plot(wavelengths, table_epsilon_ourWork[0], '-', linewidth=3, color=palette[1],
                label='$\epsilon$ - f$_{spot}=2\%$')
    #graph1.plot(wavelengths, epsilon_Rackham_ff_2,  'o', linestyle='none', markersize=5, color=palette[0])
    graph1.plot(wavelengths, epsilon_Rackham_ff_2,  '--', linewidth=3, color=palette[0],
                label='$\epsilon_{R}$ - f$_{spot}=2\%$')


    plt.text(950, 1.14, 'Trans. Lat. = 46.2$^{\circ}$', fontsize=17, bbox=dict(facecolor='white', alpha=0.5))
    plt.xlim(430, 1700)
    graph1.tick_params(axis="x", direction="in", labelsize=12)
    graph1.tick_params(axis="y", direction="in", labelsize=12)
    plt.subplots_adjust(top=0.9)
    #plt.plot(wavelengths, y1(wavelengths), "-", color='red') # ajuste polinomial

elif (graph == 2):

    # Leitura de entrada dos dados

    transit_depth_tl_46_ff_0 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-46graus,f_spot=0.00).txt", delimiter=",")
    transit_depth_tl_46_ff_2 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-46graus,f_spot=0.02).txt", delimiter=",")
    transit_depth_tl_46_ff_4 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-46graus,f_spot=0.04).txt", delimiter=",")
    transit_depth_tl_46_ff_6 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-46graus,f_spot=0.06).txt", delimiter=",")
    transit_depth_tl_46_ff_8 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-46graus,f_spot=0.08).txt", delimiter=",")
    transit_depth_tl_46_ff_10 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-46graus,f_spot=0.10).txt", delimiter=",")
    wavelengths = np.genfromtxt("Exoplanets/WASP101b/output_wavelengths.txt", delimiter=",")

    palette = sns.color_palette("mako", 6)
    graph2 = fig.add_subplot(1, 1, 1)
    #graph2.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    #graph2.set_ylabel('D$_{\mathrm{unnoc}}$ -- D$_{\mathrm{phot}}$ [ppm]', fontsize=25, fontweight="bold") # labelpad é a distância entre o título e o eixo
    graph2.set_ylabel('Transit Depth [ppm]', fontsize=22, fontweight="bold", labelpad=10) # labelpad é a distância entre o título e o eixo
    graph2.set_xlabel('Wavelength (nm)', fontsize=22, fontweight="bold", labelpad=10)

    ######### Dados do Hubble ###############
    Cond_table3 = np.genfromtxt('Exoplanets/WASP101b/WASP101_STISG430.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table3 = np.transpose(Cond_table3)

    Cond_table4 = np.genfromtxt('Exoplanets/WASP101b//WASP101_STISG750.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table4 = np.transpose(Cond_table4)

    Cond_table5 = np.genfromtxt('Exoplanets/WASP101b/WASP101_WFC3.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table5 = np.transpose(Cond_table5)

    plt.errorbar(Cond_table3[0] * 1000, Cond_table3[1] * 1.0e+4, fmt='.', yerr=Cond_table3[2] * 1.0e+4, color='gray', ms=10, alpha=0.7,
                 label='Hubble data (STIS+WFC3)')
    plt.errorbar(Cond_table4[0] * 1000, Cond_table4[1] * 1.0e+4, fmt='.', yerr=Cond_table4[2] * 1.0e+4, color='gray', ms=10, alpha=0.7)
    plt.errorbar(Cond_table5[0] * 1000, Cond_table5[1] * 1.0e+4, fmt='.', yerr=Cond_table5[2] * 1.0e+4, color='gray', ms=10, alpha=0.7)
    ########################

    #graph2.plot(wavelengths, transit_depth_tl_46_ff_10,  'o', linestyle='none', markersize=5, color=palette[5])
    graph2.plot(wavelengths, transit_depth_tl_46_ff_10, '-', color=palette[5], linewidth=3,
                label='f$_{spot}=10\%$')
    #graph2.plot(wavelengths, transit_depth_tl_46_ff_8,  'o', linestyle='none', markersize=5, color=palette[4])
    graph2.plot(wavelengths, transit_depth_tl_46_ff_8, '-', color=palette[4], linewidth=3,
                label='f$_{spot}=8\%$')
    #graph2.plot(wavelengths, transit_depth_tl_46_ff_6,  'o', linestyle='none', markersize=5, color=palette[3])
    graph2.plot(wavelengths, transit_depth_tl_46_ff_6, '-', color=palette[3], linewidth=3,
                label='f$_{spot}=6\%$')
    #graph2.plot(wavelengths, transit_depth_tl_46_ff_4,  'o', linestyle='none', markersize=5, color=palette[2])
    graph2.plot(wavelengths, transit_depth_tl_46_ff_4, '-', color=palette[2], linewidth=3,
                label='f$_{spot}=4\%$')
    #graph2.plot(wavelengths, transit_depth_tl_46_ff_2,  'o', linestyle='none', markersize=5, color=palette[1])
    graph2.plot(wavelengths, transit_depth_tl_46_ff_2, '-', color=palette[1], linewidth=3,
                label='f$_{spot}=2\%$')
    #graph2.plot(wavelengths, transit_depth_tl_46_ff_0,  'o', linestyle='none', markersize=5, color=palette[0])
    graph2.plot(wavelengths, transit_depth_tl_46_ff_0, '-', color=palette[0], linewidth=3,
                label='Photosphere')


    plt.text(950, 14000, 'Trans. Lat. = 46.2$^{\circ}$', fontsize=17, bbox=dict(facecolor='white', alpha=0.5))
    graph2.tick_params(axis="x", direction="in", labelsize=12)
    graph2.tick_params(axis="y", direction="in", labelsize=12)
    plt.subplots_adjust(top=0.9)
    plt.xlim(430, 1700)
    #plt.xlabel('Wavelength (nm)', labelpad=15)
    #plt.plot(wavelengths, y2(wavelengths), "-", color='red') # ajuste polinomial


elif (graph == 3):

    # Leitura de entrada dos dados

    transit_depth_phot_trans_lat_0 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=0graus,f_spot=0.00).txt", delimiter=",")
    transit_depth_phot_trans_lat_10 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-9graus,f_spot=0.00).txt", delimiter=",")
    transit_depth_phot_trans_lat_20 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-19graus,f_spot=0.00).txt", delimiter=",")
    transit_depth_phot_trans_lat_30 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-30graus,f_spot=0.00).txt", delimiter=",")
    transit_depth_phot_trans_lat_40 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-40graus,f_spot=0.00).txt", delimiter=",")
    transit_depth_phot_trans_lat_50 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-50graus,f_spot=0.00).txt", delimiter=",")
    transit_depth_phot_trans_lat_60 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-59graus,f_spot=0.00).txt", delimiter=",")
    transit_depth_phot_trans_lat_70 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-69graus,f_spot=0.00).txt", delimiter=",")
    transit_depth_phot_trans_lat_80 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-79graus,f_spot=0.00).txt", delimiter=",")
    wavelengths = np.genfromtxt("Exoplanets/WASP101b/output_wavelengths.txt", delimiter=",")

    palette = sns.color_palette("mako", 8)

    graph3 = fig.add_subplot(1, 1, 1)

    #graph3.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    #graph3.set_ylabel('D$_{\mathrm{unnoc}}$ -- D$_{\mathrm{phot}}$ [ppm]', fontsize=25, fontweight="bold") # labelpad é a distância entre o título e o eixo
    graph3.set_ylabel('Transit Depth [ppm]', fontsize=22, fontweight="bold", labelpad=10) # labelpad é a distância entre o título e o eixo
    graph3.set_xlabel('Wavelength (nm)', fontsize=22, fontweight="bold", labelpad=10)

    ######### Dados do Hubble ###############
    Cond_table3 = np.genfromtxt('Exoplanets/WASP101b/WASP101_STISG430.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table3 = np.transpose(Cond_table3)

    Cond_table4 = np.genfromtxt('Exoplanets/WASP101b/WASP101_STISG750.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table4 = np.transpose(Cond_table4)

    Cond_table5 = np.genfromtxt('Exoplanets/WASP101b/WASP101_WFC3.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table5 = np.transpose(Cond_table5)

    plt.errorbar(Cond_table3[0] * 1000, Cond_table3[1] * 1.0e+4, fmt='.', yerr=Cond_table3[2] * 1.0e+4, color='gray', ms=10, alpha=0.7,
                 label='Hubble data (STIS+WFC3)')
    plt.errorbar(Cond_table4[0] * 1000, Cond_table4[1] * 1.0e+4, fmt='.', yerr=Cond_table4[2] * 1.0e+4, color='gray', ms=10, alpha=0.7)
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

    transit_depth_trans_lat_20_fspot_0 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-19graus,f_spot=0.00).txt", delimiter=",")
    transit_depth_trans_lat_20_fspot_4 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-19graus,f_spot=0.04).txt", delimiter=",")
    transit_depth_trans_lat_30_fspot_0 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-30graus,f_spot=0.00).txt", delimiter=",")
    transit_depth_trans_lat_30_fspot_4 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-30graus,f_spot=0.04).txt", delimiter=",")
    transit_depth_trans_lat_40_fspot_0 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-40graus,f_spot=0.00).txt", delimiter=",")
    transit_depth_trans_lat_40_fspot_4 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-40graus,f_spot=0.04).txt", delimiter=",")
    transit_depth_trans_lat_50_fspot_0 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-50graus,f_spot=0.00).txt", delimiter=",")
    transit_depth_trans_lat_50_fspot_4 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-50graus,f_spot=0.04).txt", delimiter=",")
    transit_depth_trans_lat_60_fspot_0 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-59graus,f_spot=0.00).txt", delimiter=",")
    transit_depth_trans_lat_60_fspot_4 = np.genfromtxt("Exoplanets/WASP101b/output_transit_depth(trans_lat=-59graus,f_spot=0.04).txt", delimiter=",")
    wavelengths = np.genfromtxt("Exoplanets/WASP101b/output_wavelengths.txt", delimiter=",")

    palette = sns.color_palette("Paired", 10)

    graph4 = fig.add_subplot(1, 1, 1)

    #graph4.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    #graph4.set_ylabel('D$_{\mathrm{unnoc}}$ -- D$_{\mathrm{phot}}$ [ppm]', fontsize=25, fontweight="bold") # labelpad é a distância entre o título e o eixo
    graph4.set_ylabel('Transit Depth [ppm]', fontsize=22, fontweight="bold", labelpad=10) # labelpad é a distância entre o título e o eixo
    graph4.set_xlabel('Wavelength (nm)', fontsize=22, fontweight="bold", labelpad=10)

    ######### Dados do Hubble ###############
    Cond_table3 = np.genfromtxt('Exoplanets/WASP101b/WASP101_STISG430.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table3 = np.transpose(Cond_table3)

    Cond_table4 = np.genfromtxt('Exoplanets/WASP101b/WASP101_STISG750.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table4 = np.transpose(Cond_table4)

    Cond_table5 = np.genfromtxt('Exoplanets/WASP101b/WASP101_WFC3.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table5 = np.transpose(Cond_table5)

    plt.errorbar(Cond_table3[0] * 1000, Cond_table3[1] * 1.0e+4, fmt='.', yerr=Cond_table3[2] * 1.0e+4, color='gray', ms=10, alpha=0.7,
                 label='Hubble data (STIS+WFC3)')
    plt.errorbar(Cond_table4[0] * 1000, Cond_table4[1] * 1.0e+4, fmt='.', yerr=Cond_table4[2] * 1.0e+4, color='gray', ms=10, alpha=0.7)
    plt.errorbar(Cond_table5[0] * 1000, Cond_table5[1] * 1.0e+4, fmt='.', yerr=Cond_table5[2] * 1.0e+4, color='gray', ms=10, alpha=0.7)
    ########################


    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_20,  'o', linestyle='none', markersize=5, color=palette[1])
    graph4.plot(wavelengths, transit_depth_trans_lat_20_fspot_4, '-', color=palette[0], linewidth=3,
                label='Trans. Lat. = 20$^{\circ}$; f$_{spot}=4\%$')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_0,  'o', linestyle='none', markersize=5, color=palette[0])
    graph4.plot(wavelengths, transit_depth_trans_lat_20_fspot_0, '-', color=palette[1], linewidth=3,
                label='Trans. Lat. = 20$^{\circ}$; photosphere')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_30,  'o', linestyle='none', markersize=5, color=palette[3])
    graph4.plot(wavelengths, transit_depth_trans_lat_30_fspot_4, '-', color=palette[2], linewidth=3,
                label='Trans. Lat. = 30$^{\circ}$; f$_{spot}=4\%$')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_0,  'o', linestyle='none', markersize=5, color=palette[0])
    graph4.plot(wavelengths, transit_depth_trans_lat_30_fspot_0, '-', color=palette[3], linewidth=3,
                label='Trans. Lat. = 30$^{\circ}$; photosphere')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_40,  'o', linestyle='none', markersize=5, color=palette[2])
    graph4.plot(wavelengths, transit_depth_trans_lat_40_fspot_4, '-', color=palette[4], linewidth=3,
                label='Trans. Lat. = 40$^{\circ}$; f$_{spot}=4\%$')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_40,  'o', linestyle='none', markersize=5, color=palette[2])
    graph4.plot(wavelengths, transit_depth_trans_lat_40_fspot_0, '-', color=palette[5], linewidth=3,
                label='Trans. Lat. = 40$^{\circ}$; photosphere')
    # graph4.plot(wavelengths, transit_depth_phot_trans_lat_50,  'o', linestyle='none', markersize=5, color=palette[3])
    graph4.plot(wavelengths, transit_depth_trans_lat_50_fspot_4, '-', color=palette[8], linewidth=3,
                label='Trans. Lat. = 50$^{\circ}$, f$_{spot}=4\%$')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_50,  'o', linestyle='none', markersize=5, color=palette[3])
    graph4.plot(wavelengths, transit_depth_trans_lat_50_fspot_0, '-', color=palette[9], linewidth=3,
                label='Trans. Lat. = 50$^{\circ}$, photosphere')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_60,  'o', linestyle='none', markersize=5, color=palette[4])
    graph4.plot(wavelengths, transit_depth_trans_lat_60_fspot_4, '-', color=palette[6], linewidth=3,
                label='Trans. Lat. = 60$^{\circ}$; f$_{spot}=4\%$')
    #graph4.plot(wavelengths, transit_depth_phot_trans_lat_60,  'o', linestyle='none', markersize=5, color=palette[4])
    graph4.plot(wavelengths, transit_depth_trans_lat_60_fspot_0, '-', color=palette[7], linewidth=3,
                label='Trans. Lat. = 60$^{\circ}$; photosphere')


    #plt.text(950, 16000, '$Photosphere$', fontsize=17, bbox=dict(facecolor='white', alpha=0.5))
    graph4.tick_params(axis="x", direction="in", labelsize=12)
    graph4.tick_params(axis="y", direction="in", labelsize=12)
    plt.subplots_adjust(top=0.9)
    plt.xlim(430, 1700)
    #plt.xlabel('Wavelength (nm)', labelpad=15)
    #plt.plot(wavelengths, y2(wavelengths), "-", color='red') # ajuste polinomial


elif (graph == 5):

    # Leitura de entrada dos dados

    epsilon_Rackham_trans_lat_20_fspot_8 = np.genfromtxt("Exoplanets/WASP101b/output_epsilon_Rackham(trans_lat=-46graus,f_spot=0.08).txt", delimiter=",")

    wavelengths = np.genfromtxt("Exoplanets/WASP101b/output_wavelengths.txt", delimiter=",")

    table_epsilon_ourWork = np.genfromtxt('Exoplanets/WASP101b/output_epsilon_fixed_fspot_8% (our work).txt', delimiter=",",
                                          usecols=(0, 1, 2, 3, 4, 5, 6, 7), skip_header=1)
    table_epsilon_ourWork = np.transpose(table_epsilon_ourWork)

    palette = sns.color_palette("tab10", 10)
    graph5 = fig.add_subplot(1, 1, 1)
    #graph5.set_title('WASP-101$\,$b', fontsize=29, fontweight='bold')
    graph5.set_ylabel('Contamination Factor ($\epsilon$)', fontsize=22, fontweight="bold",
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

    plt.text(950, 1.10, 'f$_{spot}=8\%$', fontsize=17, bbox=dict(facecolor='white', alpha=0.5))
    plt.xlim(430, 1700)
    plt.ylim(1.04, 1.11)
    graph5.tick_params(axis="x", direction="in", labelsize=12)
    graph5.tick_params(axis="y", direction="in", labelsize=12)
    plt.subplots_adjust(top=0.9)
    #plt.plot(wavelengths, y1(wavelengths), "-", color='red') # ajuste polinomial



legend = plt.legend(prop={'size': 12})

plt.tick_params(axis="x", direction="in", labelsize=15)
plt.tick_params(axis="y", direction="in", labelsize=15)


plt.show()