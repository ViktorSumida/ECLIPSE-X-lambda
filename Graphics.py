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

frac_r_spot_r_star = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]




fig = plt.figure()

graph1 = fig.add_subplot(1, 1, 1)
graph1.set_title('HD 69830 b', fontsize=20, fontweight='bold')
graph1.set_ylabel('$\epsilon$', fontsize=25, labelpad=20, fontweight="bold") # labelpad é a distância entre o título e o eixo
graph1.set_xlabel('R$_{spot}$/R$_{star}$', fontsize=20, labelpad=20, fontweight="bold")
graph1.plot(frac_r_spot_r_star, epsilon,  'o', linestyle='none', markersize=7, color='red')
graph1.plot(frac_r_spot_r_star, epsilon_Rackham,  'o', linestyle='none', markersize=3, color='red')
#graph1.errorbar(data_ejeção, data_ejeção_Schutzer, yerr=erro_data_ejeção, linestyle='none', markersize=9, color='red')
#graph1.plot(data_ejeção_Schutzer, data_ejeção,  'o', linestyle='none', markersize=7, color='blue', label='Schutzer (2019)')
#graph1.plot(data_ejeção_Britzen, data_ejeção_meu_Britzen,  'D', linestyle='none', markersize=7, color='black', label='Britzen et al. (2019)')

graph1.plot(frac_r_spot_r_star, epsilon,  linestyle='-', color='red')

graph1.tick_params(axis="x", direction="in", labelsize=12)
graph1.tick_params(axis="y", direction="in", labelsize=12)

plt.show()