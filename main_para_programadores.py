#######################################################################
# Viktor Y. D. Sumida e-mail: viktor.sumida@alumni.usp.br #############
# PhD student in Astrophysics at Mackenzie University #################
# Contact via e-mail: viktor.sumida@alumni.usp.br        ##############
#######################################################################

import numpy as np
import pandas as pd
from matplotlib import pyplot
from estrela_nv1 import estrela
from eclipse_nv1 import Eclipse
from verify import Validar, ValidarEscolha, calSemiEixo, calculaLat


########### Fonte igual ao LaTeX ###### <https://matplotlib.org/stable/tutorials/text/usetex.html> ######
#pyplot.rcParams.update({
#    "text.usetex": True,
#    "font.family": "serif",
#    "font.serif": ["Computer Modern Roman"]})
#########################################################################################################

'''
main programado para profissionais e estudantes familiarizados com a área 
--- estrela ---
parâmetro raio:: raio da estrela em pixel
parâmetro intensidadeMaxima:: intensidade da estrela que sera plotada 
parâmetro tamanhoMatriz:: tamanho em pixels da matriz estrela
parâmetro raioStar:: raio da estrela em relação ao raio do sol
parâmetro coeficienteHum:: coeficiente de escurecimento de limbo 1 (u1)
parâmetro coeficienteDois:: coeficiente de escurecimento de limbo 2 (u2)
objeto estrela_ :: é o objeto estrela onde é guardada a matriz estrela de acordo com os parâmetros. Chamadas das funções 
da classe estrela são feitas através dele.
parâmetro estrela :: variavel que recebe o objeto estrela 
--- planeta ---
parâmetro periodo:: periodo de órbita do planeta em dias 
parâmetro anguloInclinacao:: ângulo de inclinação do planeta em graus
parâmetro semieixoorbital:: semi-eixo orbital do planeta
parâmetro semiEixoRaioStar:: conversão do semi-eixo orbital em relação ao raio da estrela 
parâmetro raioPlanetaRstar:: conversão do raio do planeta em relação ao raio de Júpiter para em relação ao raio da 
estrela
---  mancha --- 
parâmetro latsugerida:: latitude sugerida para a mancha
parâmetro fa:: vetor com a área de cada mancha
parâmetro fi:: vetor com a intensidade de cada mancha
parâmetro li:: vetor com a longitude de cada mancha 
parâmetro quantidade:: variavel que armazena a quantidade de manchas
parâmetro r:: raio da mancha em relação ao raio da estrela
parâmetro intensidadeMancha:: intensidade da mancha em relação a intensidade da estrela
parâmetro lat:: latitude da mancha 
parâmetro longt:: longitude da mancha 
parâmetro raioMancha:: raio real da mancha
parâmetro area::  area da mancha 
--- eclipse ---
parâmetro eclipse:: variavel que guarda o objeto da classe eclipse que gera a curva de luz. Chamadas das funções da 
classe Eclipse () são feitas através dele. 
parâmetro tempoTransito:: tempo do transito do planeta 
parâmetro curvaLuz:: matriz curva de luz que sera plotada em forma de grafico 
parâmetro tempoHoras:: tempo do transito em horas
'''

##################################################################################################
############# Função para intensidade de um corpo negro a uma determinada temperatura ############
##################################################################################################
h_Planck = 6.626e-34
c_Light = 2.998e+8
k_Boltzmann = 1.38e-23
def planck(wavelength, Temp) :
    aux = h_Planck * c_Light / (wavelength * k_Boltzmann * Temp)
    intensity = (2 * h_Planck * (c_Light**2)) / ((wavelength**5) * (np.exp(aux) - 1.0))
    return intensity

##################################################################################################
############# Os coeficientes de limbo serão dados pelo arquivo .xml #############################
####### Leitura e escrita dos parâmetros da tabela obtida pelo ExoCTK: ###########################
################# https://exoctk.stsci.edu/limb_darkening ########################################
##################################################################################################

table_ExoCTK = pd.read_excel('ExoCTK_results.xlsx', engine='openpyxl')

profile_aux = table_ExoCTK['profile'].to_numpy()
profile_aux = np.delete(profile_aux, 0)    # removing the column reader "-----"
profile = profile_aux[0]
c1 = table_ExoCTK['c1'].to_numpy()
c1 = np.delete(c1, 0)    # removing the column reader "-----"
c2 = table_ExoCTK['c2'].to_numpy()
c2 = np.delete(c2, 0)    # removing the column reader "-----"
lambdaEff = table_ExoCTK['wave_eff'].to_numpy()
lambdaEff = np.delete(lambdaEff, 0) # removing the column reader "-----"
lambdaMin = table_ExoCTK['wave_min'].to_numpy()
lambdaMin = np.delete(lambdaEff, 0) # removing the column reader "-----"
lambdaMax = table_ExoCTK['wave_max'].to_numpy()
lambdaMax = np.delete(lambdaEff, 0) # removing the column reader "-----"
num_elements = len(c1)  # number of different wavelengths
temp = table_ExoCTK['Teff'].to_numpy()
temp = np.delete(temp, 0)    # removing the column reader "-----"
tempStar = temp[0]


if profile == 'linear': # Linear has one limb darkening coefficient
    c2 = [0 for j in range(num_elements)]
    c3 = [0 for i in range(num_elements)]
    c4 = [0 for j in range(num_elements)]
elif profile == '3-parameter':
    c3 = table_ExoCTK['c3'].to_numpy()
    c3 = np.delete(c3, 0)  # removing the column reader
    c4 = [0 for j in range(num_elements)]
elif profile == '4-parameter':
    c3 = table_ExoCTK['c3'].to_numpy()
    c3 = np.delete(c3, 0)  # removing the column reader
    c4 = table_ExoCTK['c4'].to_numpy()
    c4 = np.delete(c4, 0)  # removing the column reader
elif profile == 'quadratic' or 'square-root' or 'logarithmic' or 'exponential':
    c3 = [0 for i in range(num_elements)]
    c4 = [0 for j in range(num_elements)]

########################################################################################
######################### Parâmetros ###################################################

parameters = pd.read_excel('Parâmetros.xlsx', engine='openpyxl',
                           keep_default_na=False) # To read empty cell as empty string, use keep_default_na=False

raio = parameters['raio'].to_numpy()
nullAux = np.where(raio == '')
raio = np.delete(raio, nullAux)   # removendo os valores ''
raio = int(raio[0])         # necessário converter vetor para variável

intensidadeMaxima = parameters['intensidadeMaxima'].to_numpy()
nullAux = np.where(intensidadeMaxima == '')
intensidadeMaxima = np.delete(intensidadeMaxima, nullAux)  # removendo os valores ''
intensidadeMaxima = int(intensidadeMaxima[0]) # necessário converter vetor para variável

tamanhoMatriz = parameters['tamanhoMatriz'].to_numpy()
nullAux = np.where(tamanhoMatriz == '')
tamanhoMatriz = np.delete(tamanhoMatriz, nullAux)  # removendo os valores ''
tamanhoMatriz = int(tamanhoMatriz[0]) # necessário converter vetor para variável

raioStar = parameters['raioStar'].to_numpy()
nullAux = np.where(raioStar == '')
raioStar = np.delete(raioStar, nullAux)  # removendo os valores ''
raioStar = float(raioStar[0]) # necessário converter vetor em float

####### manchas ########

manchas = parameters['manchas'].to_numpy()
nullAux = np.where(manchas == '')
quantidade = np.delete(manchas, nullAux) # removendo os valores ''
quantidade = int(quantidade[0]) # necessário converter vetor para variável

lat = parameters['lat'].to_numpy()
nullAux = np.where(lat == '')
lat = np.delete(lat, nullAux) # removendo os valores ''

longt = parameters['longt'].to_numpy()
nullAux = np.where(longt == '')
longt = np.delete(longt, nullAux) # removendo os valores ''

r = parameters['r'].to_numpy()
nullAux = np.where(r == '')
r = np.delete(r, nullAux) # removendo os valores ''

ecc = parameters['ecc'].to_numpy()
nullAux = np.where(ecc == '')
ecc = np.delete(ecc, nullAux) # removendo os valores ''
ecc = float(ecc[0]) # necessário converter vetor para variável

anom = parameters['anom'].to_numpy()
nullAux = np.where(anom == '')
anom = np.delete(anom, nullAux) # removendo os valores ''
anom = float(anom[0]) # necessário converter vetor para variável

#raio = 2000  # default (pixel) 373.
#intensidadeMaxima = 5000  # default 240
#tamanhoMatriz = 4050  # default 856
#raioStar = 0.89  # parâmetro_mudar raio da estrela em relacao ao raio do sol
raioStar = raioStar * 696340  # multiplicando pelo raio solar em Km

#ecc = 0
#anom = 0

##########################################################################################


# cria estrela

############ Structure "while" to generate a star for each ######################
############     wavelength (stack each star in 3D matrix) ######################

stack_estrela_ = []  # this creates a 3D empty matrix

for i in range(tamanhoMatriz):  # create lines
    stack_estrela_.append([])

    for j in range(tamanhoMatriz):  # create columns
        stack_estrela_[i].append([])

        for k in range(num_elements):  # create depth
            stack_estrela_[i][j].append(0)


########################################################################################
stack_curvaLuz = [0 for i in range(2000)]  # array used later, inside the while(count3)
stack_tempoHoras = [0 for i in range(2000)]  # array used later, inside the while(count3)
########################################################################################

########### Matriz 3-D para armazenar valores das intensidades das estrelas sem machas ################
########### com diferentes compr. de ondaspara a normalização em Eclipse ##############################

estrelaSemManchas = []  # this creates a 3D empty matrix

for i in range(tamanhoMatriz):  # create lines
    estrelaSemManchas.append([])

    for j in range(tamanhoMatriz):  # create columns
        estrelaSemManchas[i].append([])

        for k in range(num_elements):  # create depth
            estrelaSemManchas[i][j].append(0)

#######################################################################################################

################################################################################
################ Lei de Wien ###################################################
################################################################################

#print('Temp da estrela: ', tempStar)
lambdaEfetivo = (0.0028976 / tempStar)
intensidadeEstrelaLambdaMaximo = planck(lambdaEfetivo, tempStar)
#print('Comprimento de onda para intensidade máxima por Wien:', lambdaEfetivo)
#print('Intensidade máxima da estrela lambda (Wien):', intensidadeEstrelaLambdaMaximo)

intensidadeEstrelaLambdaNormalizada = np.zeros(num_elements)
intensidadeEstrelaLambda = np.zeros(num_elements)

count7 = 0
while (count7 < num_elements):
    intensidadeEstrelaLambda[count7] = planck(lambdaEff[count7] * 1.0e-6, tempStar)
    intensidadeEstrelaLambdaNormalizada[count7] = intensidadeMaxima * intensidadeEstrelaLambda[count7] / intensidadeEstrelaLambdaMaximo

    count7 += 1
################################################################################
#print('Intensidade máxima (default):', intensidadeMaxima)
#print('Intensidade: ', intensidadeEstrelaLambda)
#print('Intensidade da estrela normalizada: ', intensidadeEstrelaLambdaNormalizada)

count1 = 0
while (count1 < num_elements):
    coeficienteHum = c1[count1]
    coeficienteDois = c2[count1]
    coeficienteTres = c3[count1]
    coeficienteQuatro = c4[count1]


    estrela_ = estrela(raio, intensidadeEstrelaLambdaNormalizada[count1], coeficienteHum, coeficienteDois,
                       coeficienteTres, coeficienteQuatro, tamanhoMatriz, profile)


    Nx = estrela_.getNx()  # Nx e Ny necessarios para a plotagem do eclipse
    Ny = estrela_.getNy()

    dtor = np.pi / 180.0
    #periodo = 8.667  # [em dias] parâmetro_mudar
    periodo = parameters['periodo'].to_numpy()
    nullAux = np.where(periodo == '')
    periodo = np.delete(periodo, nullAux)  # removendo os valores ''
    periodo = float(periodo[0])  # necessário converter vetor para variável
    #anguloInclinacao = 89.86  # [em graus] parâmetro_mudar
    anguloInclinacao = parameters['anguloInclinacao'].to_numpy()
    nullAux = np.where(anguloInclinacao == '')
    anguloInclinacao = np.delete(anguloInclinacao, nullAux)  # removendo os valores ''
    anguloInclinacao = float(anguloInclinacao[0])  # necessário converter vetor para variável (se for = 0, tirar float)

    # dec = ValidarEscolha("Deseja calular o semieixo Orbital do planeta através da 3a LEI DE KEPLER? 1. Sim 2.Não |") descomentar para voltar ao original
    kepler = parameters['Kepler'].to_numpy()
    nullAux = np.where(kepler == '')
    dec = np.delete(kepler, nullAux)  # removendo os valores ''
    dec = int(dec[0])  # necessário converter vetor para variável
    if dec == 1:
        #massStar = 0.86  # parâmetro_mudar [colocar massa da estrela em relação à massa do sol]
        massStar = parameters['massStar'].to_numpy()
        nullAux = np.where(massStar == '')
        massStar = np.delete(massStar, nullAux)  # removendo os valores ''
        massStar = float(massStar[0])  # necessário converter vetor para variável
        semieixoorbital = calSemiEixo(periodo, massStar)
        semiEixoRaioStar = ((semieixoorbital / 1000) / raioStar)
        # transforma em km para fazer em relação ao raio da estrela
    else:
        #semiEixoRaioStar = Validar('Semi eixo (em UA:)')
        # em unidades de Rstar
        semiEixoRaioStar = parameters['semiEixoRaioStar'].to_numpy()
        nullAux = np.where(semiEixoRaioStar == '')
        semiEixoRaioStar = np.delete(semiEixoRaioStar, nullAux)  # removendo os valores ''
        semiEixoRaioStar = float(semiEixoRaioStar[0])  # necessário converter vetor para variável
        semiEixoRaioStar = ((1.469 * (10 ** 8)) * semiEixoRaioStar) / raioStar
        # multiplicando pelas UA (transformando em Km) e convertendo em relacao ao raio da estrela

    #raioPlanetaRstar = 0.283  # parâmetro_mudar em relação ao raio de jupiter
    raioPlanetaRstar = parameters['raioPlaneta'].to_numpy()
    nullAux = np.where(raioPlanetaRstar == '')
    raioPlanetaRstar = np.delete(raioPlanetaRstar, nullAux)  # removendo os valores ''
    raioPlanetaRstar = float(raioPlanetaRstar[0])  # necessário converter vetor para variável

    raioPlanetaRstar = (raioPlanetaRstar * 69911) / raioStar  # multiplicando pelo raio de jupiter em km

    latsugerida = calculaLat(semiEixoRaioStar, anguloInclinacao)
    print("A latitude sugerida para que a mancha influencie na curva de luz da estrela é:", latsugerida)

    estrelaSemManchas[count1] = estrela_

    stack_estrela_[count1] = estrela_

    count1 += 1

####################################################################################
################## manchas #########################################################
####################################################################################

#quantidade = 1 # parâmetro_mudar quantidade de manchas desejadas, se quiser acrescentar, mude essa variavel
#lat = [-2.654914842784862] # parâmetro_mudar informação dada quando rodar o programa
#longt = [0] # parâmetro_mudar
#r = [0.05] # parâmetro_mudar Digite o raio da mancha em função do raio da estrela em pixels

#####################################################################################

# cria vetores do tamanho de quantidade para colocar os parametros das manchas
fa = [0.] * quantidade  # vetor area manchas
fi = [0.] * quantidade  # vetor intensidade manchas
li = [0.] * quantidade  # vetor longitude manchas


tempSpot = 0.418 * tempStar + 1620 # Temp. calculada em Rackham et al. 2018 p/ estrelas do tipo F-G-K
#tempSpot = 3500
#print("Tempurature: ", tempSpot)

intensidadeMancha = np.zeros(num_elements)
intensidadeManchaNormalizada = np.zeros(num_elements)
intensidadeManchaRazao = np.zeros(num_elements)

count3 = 0
while (count3 < num_elements):

    # intensidadeMancha = float(input('Digite a intensidade da mancha em função da intensidade máxima da estrela:')) ## esta intensidade
    # deverá ser mudada com o modelo de corpo negro

    intensidadeMancha[count3] = planck(lambdaEff[count3] * 1.0e-6, tempSpot)  # em [W m^-2 nm^-1]

    intensidadeManchaNormalizada[count3] = intensidadeMancha[count3] * intensidadeEstrelaLambdaNormalizada[count3] / intensidadeEstrelaLambda[count3]
    intensidadeManchaRazao[count3] = intensidadeManchaNormalizada[count3] / intensidadeEstrelaLambdaNormalizada[count3]
    print('valores da razão entre intensidade da mancha e intensidade da estrela em determinado '
          'comprimento de onda: ', intensidadeManchaRazao[count3])


    #print('Intensidade da Estrela normalizada = ', intensidadeEstrelaLambdaNormalizada)
    #print('Intensidade da Mancha normalizada = ', intensidadeManchaNormalizada)
    #print('Intensidade da Estrela =', intensidadeEstrelaLambda)
    #print('Intensidade da Mancha =', intensidadeMancha)
    #print('Razão entre a intensidade da mancha e a intensidade da estrela =', intensidadeManchaRazao[count3])

    count = 0
    while count != quantidade:  # o laço ira rodar a quantidade de manchas selecionadas pelo usuario

        #print('\033[1;35m\n\n══════════════════ Parâmetros da mancha ', count + 1, '═══════════════════\n\n\033[m')
        #r = Validar('Digite o raio da mancha em função do raio da estrela em pixels:')

        fi[count] = intensidadeManchaRazao[count3]
        #lat = float(input('Latitude da mancha:'))
        #longt = float(input('Longitude da mancha:'))
        li[count] = longt[count]

        raioMancha = r[count] * raioStar
        area = np.pi * (raioMancha ** 2) # área da mancha sem projeção
        fa[count] = area

        estrela = stack_estrela_[count3].manchas(r[count], intensidadeManchaRazao[count3], lat[count],
                                                 longt[count])  # recebe a escolha se irá receber manchas ou não

        print("Razão entre intensidades: ", intensidadeManchaRazao[count3])

        count += 1


    area_spot = np.sum(fa)
    area_star = np.pi * (raioStar ** 2)
    f_spot = area_spot / area_star
    print('razao A_spot/A_star: ', f_spot)
    fatorEpsilon = 1 / (1 - (f_spot * (1 - intensidadeManchaRazao[count3])))
    print('epsilon de Rackham = ', fatorEpsilon)


    estrela = stack_estrela_[count3].getEstrela()  #getEstrela(self)  Retorna a estrela, plotada sem as manchas,
                                                    # necessário caso o usuário escolha a plotagem sem manchas.


    # para plotar a estrela
    # caso nao queira plotar a estrela, comentar linhas abaixo
    #if (quantidade > 0):  # se manchas foram adicionadas. plotar
    #    estrela_.Plotar(tamanhoMatriz, estrela)

    # criando lua
    lua = False  # TRUE para simular com luas e FALSE para não
    eclipse = Eclipse(Nx, Ny, raio, estrela)


    ######## Plotar ou não a a estrela junto com a animação #######################
   
    plotGrafico = parameters['plotGrafico'].to_numpy()
    nullAux = np.where(plotGrafico == '')
    plotGrafico = np.delete(plotGrafico, nullAux)  # removendo os valores ''
    plotGrafico = int(plotGrafico[0])  # necessário converter vetor para variável
    
    plotEstrela = parameters['plotAnimacao'].to_numpy()
    nullAux = np.where(plotEstrela == '')
    plotEstrela = np.delete(plotEstrela, nullAux)  # removendo os valores ''
    plotEstrela = int(plotEstrela[0])  # necessário converter vetor para variável

    if (plotEstrela == 1 or plotGrafico == 1):
        stack_estrela_[count3].Plotar(tamanhoMatriz, estrela) # descomentar para imprimir a estrela

    #########################

    eclipse.geraTempoHoras()
    tempoHoras = eclipse.getTempoHoras()
    # instanciando LUA
    #rmoon = 0.5  # parâmetro_mudar em relacao ao raio da Terra
    rmoon = parameters['rMoon'].to_numpy()
    nullAux = np.where(rmoon == '')
    rmoon = np.delete(rmoon, nullAux)  # removendo os valores ''
    rmoon = float(rmoon[0])  # necessário converter vetor para variável
    rmoon = rmoon * 6371  # parâmetro_mudar multiplicando pelo R da terra em Km

    #massMoon = 0.001  # parâmetro_mudar em relacao a massa da Terra
    massMoon = parameters['massMoon'].to_numpy()
    nullAux = np.where(massMoon == '')
    massMoon = np.delete(massMoon, nullAux)  # removendo os valores ''
    massMoon = float(massMoon[0])  # necessário converter vetor para variável
    massMoon = massMoon * (5.972 * (10 ** 24))

    #massPlaneta = 0.0321  # parâmetro_mudar em relacao ao R de jupiter
    massPlaneta = parameters['massPlaneta'].to_numpy()
    nullAux = np.where(massPlaneta == '')
    massPlaneta = np.delete(massPlaneta, nullAux)  # removendo os valores ''
    massPlaneta = float(massPlaneta[0])  # necessário converter vetor para variável
    massPlaneta = massPlaneta * (1.898 * (10 ** 27))  # passar para gramas por conta da constante G
    G = (6.674184 * (10 ** (-11)))

    #perLua = 0.1  # parâmetro_mudar [em dias]
    perLua = parameters['perMoon'].to_numpy()
    nullAux = np.where(perLua == '')
    perLua = np.delete(perLua, nullAux)  # removendo os valores ''
    perLua = float(perLua[0])  # necessário converter vetor para variável

    distancia = ((((perLua * 24. * 3600. / 2. / np.pi) ** 2) * G * (massPlaneta + massMoon)) ** (1. / 3)) / raioStar
    distancia = distancia / 100
    moon = eclipse.criarLua(rmoon, massMoon, raio, raioStar, tempoHoras, anguloInclinacao, periodo, distancia)


    estrela = stack_estrela_[count3].getEstrela()

    # eclipse
    eclipse.criarEclipse(semiEixoRaioStar, raioPlanetaRstar, periodo, anguloInclinacao, lua, ecc, anom)

    print("Tempo Total (Trânsito):", eclipse.getTempoTransito())
    tempoTransito = eclipse.getTempoTransito()
    curvaLuz = eclipse.getCurvaLuz()
    tempoHoras = eclipse.getTempoHoras()

    ############ Plotagem da curva de luz individual #############
    #pyplot.plot(tempoHoras, curvaLuz)
    #pyplot.axis([-tempoTransito / 2, tempoTransito / 2, min(curvaLuz) - 0.001, 1.001])
    #pyplot.show()

    tempoHoras = np.asarray(tempoHoras)   # turning list into vector
    curvaLuz = np.asarray(curvaLuz)       # turning list into vector

    sizeCurvaLuz = np.size(curvaLuz)
    sizeTempoHoras = np.size(tempoHoras)


    stack_curvaLuz[count3] = curvaLuz     # previously declared
    stack_tempoHoras[count3] = tempoHoras # previously declared

    count3 += 1


############################################################################################
########## Plotting the light curves at different wavelengths in the same graph ############
############################################################################################
lambdaEff_nm = [0.] * num_elements
D_lambda = [0.] * num_elements
count4 = 0

while(count4 < num_elements):
    lambdaEff_nm[count4] = lambdaEff[count4] * 1000
    pyplot.plot(stack_tempoHoras[count4], stack_curvaLuz[count4], label=int(lambdaEff_nm[count4]))
    D_lambda[count4] = min(stack_curvaLuz[count4])
    D_lambda[count4] = (1 - D_lambda[count4]) * 1000000 # profundidade de trânsito está em ppm, CUIDADO!!!
    print("Máxima profundidade de trânsito: ", min(stack_curvaLuz[count4]))
    count4 += 1

pyplot.axis([-tempoTransito / 2, tempoTransito / 2, min(curvaLuz) - 0.001, 1.001])
legend = pyplot.legend()
legend.set_title("Wavelength [nm]")

#print("Máxima profundidade de trânsito: ", min(curvaLuz))

pyplot.xlabel("time from transit center (hr)", fontsize=25)
pyplot.ylabel("relative flux", fontsize=25)

pyplot.tick_params(axis="x", direction="in", labelsize=15)
pyplot.tick_params(axis="y", direction="in", labelsize=15)

d = {'Wavelength [nm]': lambdaEff_nm, 'D_lambda [ppm]': D_lambda}
df1 = pd.DataFrame(data=d)
#df1 = pd.DataFrame([D_lambda, lambdaEff_nm],
#                   index=['D_lamb', 'wave'])
df1.to_excel("output.xlsx")

print(D_lambda)

pyplot.show()