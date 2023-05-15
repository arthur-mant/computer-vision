import numpy as np
import random
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

training_percentage = 0.8
random.shuffle(raw_d)

training = raw_d[:int(training_percentage*len(raw_d))]
validation = raw_d[int((training_percentage)*len(raw_d)):]

print("training:\n", training)
print("validation:\n", validation)

media_geral = np.mean(training, axis=0)[:-1]
desvio_geral = np.std(training, axis=0)[:-1]
moda_geral = st.mode(training, axis=0)[:-1]

media_por_classe = []
desvio_por_classe = []
moda_por_classe = []

for i in range(3):
    aux_v = []
    for item in training:
        if item[4] == i:
            aux_v.append(item[:-1]+[item[0]/item[1], item[2]/item[3]])
    media_por_classe.append(np.mean(aux_v, axis=0))
    desvio_por_classe.append(np.std(aux_v, axis=0))
    moda_por_classe.append(st.mode(aux_v, axis=0))
    
print(media_por_classe)
print(desvio_por_classe)
print(moda_por_classe)
