import numpy as np
from scipy import stats as st

def flower_to_int(s):
    if s == "Iris-setosa\n":
        return 0
    elif s == "Iris-versicolor\n":
        return 1
    elif s == "Iris-virginica\n":
        return 2
    else:
        return -1

f = open("iris.data", "r")
lines = f.readlines()

raw_d = []

for line in lines:
    l_data = line.split(",")
    if len(l_data) == 5:
        raw_d.append([float(l_data[0]), float(l_data[1]), float(l_data[2]), float(l_data[3]), int(flower_to_int(l_data[4]))])

data = []
for item in raw_d:
    data.append(item[:-1]+[item[0]/item[1], item[2]/item[3]])

media_geral = np.mean(data, axis=0)
desvio_geral = np.std(data, axis=0)
moda_geral = st.mode(data, axis=0)


media_por_classe = []
desvio_por_classe = []
moda_por_classe = []

for i in range(3):
    aux_v = []
    for item in raw_d:
        if item[4] == i:
            aux_v.append(item[:-1]+[item[0]/item[1], item[2]/item[3]])
    media_por_classe.append(np.mean(aux_v, axis=0))
    desvio_por_classe.append(np.std(aux_v, axis=0))
    moda_por_classe.append(st.mode(aux_v, axis=0))

print("                      Comp. Sépala  Larg. Sépala  Comp. Pétala  Larg. Pétala  Relação Sépala  Relação Pétala")
print("Médias geral:        ", media_geral)
print("Desvio Padrão geral: ", desvio_geral)
print("Moda geral:          ", moda_geral)
for i in range(3):
    print("Médias (", i, "):        ", media_por_classe[i])
    print("Desvio Padrão (", i, "): ", desvio_por_classe[i])
    print("Moda (", i, "):          ", moda_por_classe[i])
