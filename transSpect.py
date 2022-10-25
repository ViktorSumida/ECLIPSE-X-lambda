import numpy as np

import matplotlib.pyplot as plt

graphic = 2

if graphic == 1:

    Cond_table3 = np.genfromtxt('WASP101_STISG430.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table3 = np.transpose(Cond_table3)

    Cond_table4 = np.genfromtxt('WASP101_STISG750.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table4 = np.transpose(Cond_table4)

    Cond_table5 = np.genfromtxt('WASP101_WFC3.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table5 = np.transpose(Cond_table5)

    plt.errorbar(Cond_table3[0], Cond_table3[1], fmt='.', yerr=Cond_table3[2], color='black', ms=10)
    plt.errorbar(Cond_table4[0], Cond_table4[1], fmt='.', yerr=Cond_table4[2], color='black', ms=10)
    plt.errorbar(Cond_table5[0], Cond_table5[1], fmt='.', yerr=Cond_table5[2], color='black', ms=10)

    plt.ylabel('Transit Depth [%]')
    plt.xlabel('Wavelength [microns]')

    plt.show()

if graphic == 2:

    Cond_table10 = np.genfromtxt('GJ1132_WFC3.txt', delimiter=",", usecols=(0, 1, 2), skip_header=1)
    Cond_table10 = np.transpose(Cond_table10)

    plt.errorbar(Cond_table10[0], Cond_table10[1], fmt='.', yerr=Cond_table10[2], color='black', ms=10)

    plt.ylabel('Transit Depth [%]')
    plt.xlabel('Wavelength [microns]')

    plt.show()