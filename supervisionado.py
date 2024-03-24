import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas

print ('Algoritmo supervisionado de regressão linear')
# Coleta e Integração
arquivo = pandas.read_csv('dados_dengue.csv') # lendo informações de arquivo com o pandas

# separando dados por tabela
anos = arquivo[['ano']]
casos = arquivo[['casos']]

############## Mineração #################
regr = LinearRegression() # definindo algoritmo de regressao linear
regr.fit(X=anos, y=casos) # treinando dadados

ano_futuro = [[2018]]
casos_2018 = regr.predict(ano_futuro) # prevendo dados futuros

print('Casos previstos para 2018 ->', int(casos_2018))

############ Pós-processamento ################
plt.scatter(anos, casos, color='black') # definindo cor preta para os casos na tabela
plt.scatter(ano_futuro, casos_2018, color='red') # cor vermelha para o numero de casos futuros
plt.plot(anos, regr.predict(anos), color='blue') # cor azul para uma linha que passa até os resultados previstos

plt.xlabel('Anos')
plt.ylabel('Casos de dengue')
plt.xticks([2018])
plt.yticks([int(casos_2018)])

plt.show()
