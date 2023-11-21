__author__ = "Adriana Valio, Beatriz Duque, Felipe Pereira Pinho"
__copyright__ = "..."
__credits__ = ["Universidade Presbiteriana Mackenzie, CRAAM"]
__license__ = ""
__version__ = ""
__maintainer__ = ""
__email__ = "biaduque7@hotmail.com"
__status__ = "Production"

'''
Este programa simula a plotagem de uma estrela com manchas, através de parâmetros como raio, intensidade, escurecimento 
de limbo, etc.
As bibliotecas importadas são: 
math
matplotlib
numpy
verify:função criada para validar entradas, por exemplo numeros nao float/int ou negativos
'''

import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from verify import Validar
import random
from ctypes import WinDLL, c_float, c_int, POINTER
from numpy.ctypeslib import ndpointer
import time
import sys
import platform
import ctypes


class Estrela:
    '''
    A classe estrela recebe como objeto o raio, intensidade maxima, coeficientes de escurecimento de limbo.
    A estrela é formata em uma matriz de tamanho defeault 856.
    São objetos pertencentes a classe os parâmetros passados à mancha, como: raio, intensidade, longitude e latitude
    em relação à estrela.
    ************ PARÂMETROS DA ESTRELA***************
    :parâmetro raio: O raio da estrela em pixel
    :parâmetro raioSun: O raio da estrela em unidades de Rsun
    :parâmetro intensidadeMaxima: Intensidade do centro da estrela
    :parâmetro coeficienteHum: Coeficiente de escurecimento de limbo
    :parâmetro coeficienteDois: Coeficiete de escurecimento de limbo
    :parâmetro tamanhoMatriz: Tamanho da matriz em que será construída a estrela
    :parâmetro estrela: Estrela construida com os coeficientes de escurecimento de limbo
    '''

    def __init__(self, raio, raioSun, intensidadeMaxima, coeficienteHum, coeficienteDois,
                 coeficienteTres, coeficienteQuatro, tamanhoMatriz, profile):

        self.raio = raio  # em pixel
        self.raioSun = raioSun
        self.intensidadeMaxima = intensidadeMaxima
        self.coeficienteHum = coeficienteHum
        self.coeficienteDois = coeficienteDois
        self.coeficienteTres = coeficienteTres
        self.coeficienteQuatro = coeficienteQuatro
        self.tamanhoMatriz = tamanhoMatriz
        # self.colors = ["gray","pink","hot"]
        self.profile = profile
        error = 0

        #print('razão entre intensidades sem mancha e dentro da fun: ', self.intensidadeMaxima)

        start = time.time()
         ## Verifica o SO e se o Python é 32 ou 64 bit
        if (platform.system() == "Windows"):
            if (platform.architecture()[0] == "32bit"):
                my_func = WinDLL('scripts/func32.dll', winmode=0x8)
            elif (platform.architecture()[0] == "64bit"):
                my_func = ctypes.CDLL('./func.dll', winmode=0x8)
        elif (platform.system() == "Darwin"):
            my_func = CDLL.LoadLibrary('scripts/func64.dylib')
        else:
            my_func = CDLL('scripts/func64.so')

        my_func.criaEstrela.restype = ndpointer(dtype=c_int, ndim=2, shape=(self.tamanhoMatriz,self.tamanhoMatriz))
        self.estrela = my_func.criaEstrela(self.tamanhoMatriz, self.tamanhoMatriz, self.tamanhoMatriz,
                                           c_float(self.raio), c_float(self.intensidadeMaxima),
                                           c_float(self.coeficienteHum), c_float(self.coeficienteDois),
                                           c_float(self.coeficienteTres), c_float(self.coeficienteQuatro))

        del my_func

        self.error = error
        self.Nx = self.tamanhoMatriz
        self.Ny = self.tamanhoMatriz
        self.color = "hot"

        # self.estrela = [[0.0 for i in range(self.tamanhoMatriz)] for j in range(self.tamanhoMatriz)]
        # for j in range(len(self.estrela)):
        #     for i in range(len(self.estrela[j])):
        #         distanciaCentro = math.sqrt(
        #             pow(i - self.tamanhoMatriz / 2, 2) + pow(j - self.tamanhoMatriz / 2, 2))
        #         if distanciaCentro <= self.raio:
        #             cosTheta = math.sqrt(1 - pow(distanciaCentro / self.raio, 2))
        #             #cosTheta = np.cos(np.arcsin(distanciaCentro/raio))
        #             self.estrela[i][j] = int(self.intensidadeMaxima * (1 - self.coeficienteHum * (1 - pow(cosTheta, 1/2)) -
        #                                                                self.coeficienteDois * (1 - cosTheta) -
        #                                                                self.coeficienteTres * (1 - pow(cosTheta, 3/2)) -
        #                                                                self.coeficienteQuatro * (1 - pow(cosTheta, 2))))
        #
        # self.error = error
        # self.Nx = self.tamanhoMatriz
        # self.Ny = self.tamanhoMatriz
        # # self.color = random.choice(self.colors)
        # self.color = "pink"


        #######  Inserção de manchas

    def manchas(self, r, intensidadeMancha, lat, longt):
        '''
        Função onde é criada a(s) mancha(s) da estrela. Todos os parâmetros
        são relacionados ao tamanho da estrela, podendo o usuário escolher valores
        ou selecionar a opção default.
        *********INICIO DOS PARÂMETROS DA MANCHA*******
        :parâmetro raioMancha: Raio da mancha em relação ao raio da estrela
        :parâmetro intensidadeMancha: Intensidade da mancha em funcao da intensidade maxima da estrela
        :parâmetro latitudeMancha: Coordenada de latitude da mancha em relação à estrela
        :parâmetro longitudeMancha: Coordenada de longitude da mancha em relação à estrela
        '''

        self.raioMancha = self.raio * r  # raio em funcao do raio da estrela em pixels
        self.intensidadeMancha = intensidadeMancha  # intensidade da mancha em funcao da intensidade maxima da estrela
        #print('intensidade da mancha razão dentro da func: ', self.intensidadeMancha)

        # coordenadas de posicionamento da mancha em graus

        degreeToRadian = np.pi / 180.  # A read-only variable containing the floating-point value used to convert degrees to radians.
        self.latitudeMancha = lat * degreeToRadian
        self.longitudeMancha = longt * degreeToRadian

        # posicao da mancha em pixels em relacao ao centro da estrela
        ys = self.raio * np.sin(self.latitudeMancha)
        xs = self.raio * np.cos(self.latitudeMancha) * np.sin(self.longitudeMancha)
        anguloHelio = np.arccos(np.cos(self.latitudeMancha) * np.cos(self.longitudeMancha))

        # efeito de projecao pela mancha estar a um angulo Heliocentrico do centro da estrela - elipcidade
        yy = ys + self.Ny / 2  # posicao em pixel com relacao à origem da matriz
        xx = xs + self.Nx / 2  # posicao em pixel com relacao à origem da matriz

        kk = np.arange(self.Ny * self.Nx)
        vx = kk - self.Nx * np.int64(1. * kk / self.Nx) - xx
        vy = kk / self.Ny - yy

        # angulo de rotacao da mancha
        anguloRot = np.abs(np.arctan(ys / xs))  # em radianos
        if self.latitudeMancha * self.longitudeMancha > 0:
            anguloRot = -anguloRot
        elif self.latitudeMancha * self.longitudeMancha == 0:
            anguloRot = 0

        ii, = np.where((((vx * np.cos(anguloRot) - vy * np.sin(anguloRot)) / np.cos(anguloHelio)) ** 2 + (
                    vx * np.sin(anguloRot) + vy * np.cos(anguloRot)) ** 2) <= self.raioMancha ** 2)

        spot = np.zeros(self.Ny * self.Nx) + 1

        spot[ii] = self.intensidadeMancha
        spot = spot.reshape([self.Ny, self.Nx])

        self.estrela = self.estrela * spot

        error = 0
        self.error = error
        return self.estrela  # retorna a decisão: se há manchas ou não

        #### Inserção de flares

    def faculas(self, estrela, count):

        # recebe como parâmetro a estrela atualizada
        '''
        Função onde são criadas as fáculas da estrela. Todos os parâmetros
        são relacionados ao tamanhdo da estrela, podendo o usuário escolher valores
        ou selecionar a opção default.
        ---Parametros ainda nao definidos
        *********INICIO DOS PARÂMETROS FÁCULA*******
        :parâmetro
        :parâmetro
        :parâmetro
        :parâmetro

        '''
        error = 0
        self.error = error
        # vai sobrescrever a estrela que ele está criando, sendo ela a estrela ou a estrelaManchada.
        self.estrela = estrela
        return self.estrela  # retorna a decisão: se há fácula ou não

    def flares(self, estrela, count):  # recebe como parâmetro a estrela atualizada
        '''
        Função onde são criadas os flares da estrela. Todos os parâmetros
        são relacionados ao tamanhdo da estrela, podendo o usuário escolher valores
        ou selecionar a opção default.
        ---Parametros ainda nao definidos
        *********INICIO DOS PARÂMETROS FLARES*******
        :parâmetro
        :parâmetro
        :parâmetro
        :parâmetro

        '''

        error = 0
        self.error = error
        # vai sobrescrever a estrela que ele está criando, sendo ela a estrela ou a estrelaManchada.
        self.estrela = estrela
        return self.estrela  # retorna a decisão: se há flare ou não

        #### Getters

    def getNx(self):
        '''
        Retorna parâmetro Nx, necessário para o Eclipse.
        '''
        return self.Nx

    def getNy(self):
        '''
        Retorna parâmetro Ny, necessário para o Eclipse.
        '''
        return self.Ny

    def getRaioStar(self):
        '''
        Retorna o raio da estrela em pixel, necessário para o programa Eclipse, visto que o raio do planeta se dá em
        relação ao raio da estrela.
        '''
        return self.raio

    def getEstrela(self):
        '''
        Retorna a estrela, plotada sem as manchas, necessário caso o usuário escolha a plotagem sem manchas.
        '''
        return self.estrela

    def getu1(self):
        return self.c1

    def getu2(self):
        return self.c2

    def getTamanhoMatriz(self):
        return self.tamanhoMatriz

    def getRaioSun(self):
        return self.raioSun

    def getIntensidadeMaxima(self):
        return self.intensidadeMaxima

    def getError(self):
        '''
        Retorna valor de erro. Se não houverem erros, a variável assumirá 0. Se houverem erros, o programa manterá
        o valor origem da variável (que é -1).
        '''
        return self.error

    def setStarName(self, starName):
        self.starName = starName

    def getStarName(self):
        return self.starName

    def setCadence(self, cadence):
        self.cadence = cadence

    def getCadence(self):
        return self.cadence

    def Plotar(self, tamanhoMatriz, estrela):
        Nx = tamanhoMatriz
        Ny = tamanhoMatriz
        plt.axis([0, Nx, 0, Ny])
        plt.imshow(estrela, self.color)
        plt.show()