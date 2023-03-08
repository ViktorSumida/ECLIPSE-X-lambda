import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
import math

########### Fonte igual ao LaTeX ###### <https://matplotlib.org/stable/tutorials/text/usetex.html> ######
#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "serif",
#    "font.serif": ["Computer Modern Roman"]})
#########################################################################################################

# Leitura de entrada dos dados

transit_depth = np.genfromtxt("output_transit_depth.txt", delimiter=",")
epsilon_Rackham = np.genfromtxt("output_epsilon_Rackham.txt", delimiter=",")
wavelengths = np.genfromtxt("output_wavelengths.txt", delimiter=",")


########## Ajuste polinomial ###############################
deg = 2
z1 = np.polyfit(wavelengths, epsilon_Rackham, deg)
y1 = np.poly1d(z1)
z2 = np.polyfit(wavelengths, transit_depth, deg)
y2 = np.poly1d(z2)
############################################################

fig = plt.figure()
graph = 1
if (graph == 1):

    graph1 = fig.add_subplot(1, 1, 1)
    graph1.set_title('HD 69830 b', fontsize=20, fontweight='bold')
    graph1.set_ylabel('$\epsilon$', fontsize=30, fontweight="bold") # labelpad é a distância entre o título e o eixo
    graph1.set_xlabel('R$_{spot}$/R$_{star}$', fontsize=20, fontweight="bold")
    #graph1.plot(wavelengths, epsilon,  'o', linestyle='none', markersize=7, color='red', label='$\epsilon$')
    graph1.plot(wavelengths, epsilon_Rackham,  'o', linestyle='none', markersize=7, color='blue', label='$\epsilon_{R}$')
    graph1.tick_params(axis="x", direction="in", labelsize=12)
    graph1.tick_params(axis="y", direction="in", labelsize=12)
    plt.plot(wavelengths, y1(wavelengths), "-", color='red') # ajuste polinomial


elif (graph == 2):
    graph2 = fig.add_subplot(1, 1, 1)
    graph2.set_title('WASP 101$\,$b', fontsize=20, fontweight='bold')
    #graph2.set_ylabel('D$_{\mathrm{unnoc}}$ -- D$_{\mathrm{phot}}$ [ppm]', fontsize=25, fontweight="bold") # labelpad é a distância entre o título e o eixo
    graph2.set_ylabel('Transit Depth [ppm]', fontsize=25, fontweight="bold") # labelpad é a distância entre o título e o eixo
    graph2.set_xlabel('Wavelength (nm)', fontsize=20, fontweight="bold")
    graph2.plot(wavelengths, transit_depth,  'o', linestyle='none', markersize=7, color='red')
    graph2.plot(wavelengths, transit_depth, '-', color='black')
    graph2.tick_params(axis="x", direction="in", labelsize=12)
    graph2.tick_params(axis="y", direction="in", labelsize=12)
    #plt.plot(wavelengths, y2(wavelengths), "-", color='red') # ajuste polinomial



legend = plt.legend(prop={'size': 20})

plt.tick_params(axis="x", direction="in", labelsize=15)
plt.tick_params(axis="y", direction="in", labelsize=15)

plt.show()