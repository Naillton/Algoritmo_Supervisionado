from sklearn.datasets import load_iris, fetch_kddcup99
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.svm import SVC

print ('Algoritmo de classificação de dados')
# Coleta e Integração de dados
iris = load_iris() # lendo dados

caracteristicas = iris.data # recuperando os dados
rotulos = iris.target # setando os alvos

print("Caracteristicas:\n", caracteristicas[:2])
print("Rótulos:\n", rotulos[:2])
print('########################################################')

# Partição dos dados
carac_treino, carac_teste, rot_treino, rot_teste = train_test_split(caracteristicas, rotulos)

# Mineração + Arvore de Decisão
arvore = DecisionTreeClassifier(max_depth=2)
arvore.fit(X=carac_treino, y=rot_treino)

rot_preditos = arvore.predict(carac_teste) # prevendos elementos
acuracia_arvore = accuracy_score(rot_teste, rot_preditos)

# Máquina de Vetor Suporte
clf = SVC()
clf.fit(X=carac_treino, y=rot_treino)

rot_preditos_svm = clf.predict(carac_teste)
acuracia_svm = accuracy_score(rot_teste, rot_preditos_svm)

# Pós-processamento
print("Acurácia Árvore de Decisão:", round(acuracia_arvore, 5))
print("Acurácia SVM:", round(acuracia_svm, 5))
print('########################################################')

r = export_text(arvore, feature_names=iris['feature_names'])
print("Estrutura da árvore")
print(r)
