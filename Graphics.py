import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
import math

########### Fonte igual ao LaTeX ###### <https://matplotlib.org/stable/tutorials/text/usetex.html> ######
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})
#########################################################################################################


epsilon = [1.0, 1.000395388, 1.001538534, 1.003494507, 1.006209571, 1.009750761, 1.014083514, 1.019235361,
           1.025265541, 1.032163413, 1.039957286]
epsilon_Rackham = [1.00000, 1.00032, 1.00128, 1.00288, 1.00513, 1.00803, 1.01161, 1.01586, 1.02082, 1.02650, 1.03292]

epsilon_Rackham_projeção = [1.00000, 1.00032, 1.00124, 1.00282, 1.00501, 1.00786, 1.01136, 1.01551, 1.02036, 1.02650,
                            1.03210]

frac_r_spot_r_star = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]

D_phot_unnoc = [16.86167443, 15.61290814, 14.359741, 13.47637323, 12.58029288, 11.92529956, 11.34251398, 10.79087153,
                10.33929163, 9.926751603]
wavelength = [445.6, 493.4, 541.5, 589.4, 637.7, 685.5, 733.4, 781.7, 829.6, 877.4]


########## Ajuste polinomial ###############################
deg = 2
z1 = np.polyfit(frac_r_spot_r_star, epsilon, deg)
y1 = np.poly1d(z1)

z2 = np.polyfit(wavelength, D_phot_unnoc, deg)
y2 = np.poly1d(z2)
############################################################



fig = plt.figure()

graph1 = fig.add_subplot(1, 1, 1)
graph1.set_title('HD 69830 b', fontsize=20, fontweight='bold')
graph1.set_ylabel('$\epsilon$', fontsize=30, fontweight="bold") # labelpad é a distância entre o título e o eixo
graph1.set_xlabel('R$_{spot}$/R$_{star}$', fontsize=20, fontweight="bold")
graph1.plot(frac_r_spot_r_star, epsilon,  'o', linestyle='none', markersize=7, color='red', label='$\epsilon$')
graph1.plot(frac_r_spot_r_star, epsilon_Rackham,  'o', linestyle='none', markersize=7, color='blue', label='$\epsilon_{R}$')
graph1.plot(frac_r_spot_r_star, epsilon_Rackham_projeção,  'o', linestyle='none', markersize=7, color='green', label='$\epsilon_{R_{\mathrm{proj}}}$')
graph1.tick_params(axis="x", direction="in", labelsize=12)
graph1.tick_params(axis="y", direction="in", labelsize=12)
plt.plot(frac_r_spot_r_star, y1(frac_r_spot_r_star), "-", color='red') # ajuste polinomial



#graph2 = fig.add_subplot(1, 1, 1)
##graph2.set_title('HD 69830 b', fontsize=20, fontweight='bold')
#graph2.set_ylabel('D$_{\mathrm{unnoc}}$ -- D$_{\mathrm{phot}}$ [ppm]', fontsize=25, fontweight="bold") # labelpad é a distância entre o título e o eixo
#graph2.set_xlabel('wavelength (nm)', fontsize=20, fontweight="bold")
#graph2.plot(wavelength, D_phot_unnoc,  'o', linestyle='none', markersize=7, color='red')
##graph2.plot(wavelength, D_phot_unnoc, '-', color='black')
#graph2.tick_params(axis="x", direction="in", labelsize=12)
#graph2.tick_params(axis="y", direction="in", labelsize=12)
#plt.plot(wavelength, y2(wavelength), "-", color='red') # ajuste polinomial

legend = plt.legend(prop={'size': 20})

plt.tick_params(axis="x", direction="in", labelsize=15)
plt.tick_params(axis="y", direction="in", labelsize=15)

plt.show()