# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy as sc
import tensorflow.python.framework.dtypes
from sklearn.datasets import make_circles


# ************************** Creacion de datos **************************************

# Genero un conjunto de datos que caen en circulos concentricos
# Total de puntos generados: 500
# Ruido gaussiano en datos de 0.05
# Factor de escala entre ambos circulos de 0.5 (va entre 0 y 1)
X, Y = make_circles(n_samples=500, factor=0.5, noise=0.05)

# Genero dos vectores con 100 puntos equidistantes entre -1,5 y 1,5
res = 100
x0 = np.linspace(-1.5, 1.5, res)
x1 = np.linspace(-1.5, 1.5, res)

# Armo la matriz
mX = np.array(np.meshgrid(x0, x1)).T.reshape(-1, 2)

# Matriz de ceros (0.5 para no anular datos)
mY = np.zeros((res, res)) + 0.5


# ************************** Construccion de la arquitectura *********************************
print("Construimos arquitectura")

#Defino entrada de la red para la matriz X e Y: capa de entrada y capa de salida
eX = tf.placeholder('float', shape=[None, X.shape[1]])
eY = tf.placeholder('float', shape=[None])

ta = 0.00 #taza de aprendizaje
nn = [2,16,8,1] #numero de neuronas por capa

#Capa 1
c1 = tf.Variable(tf.random_normal([nn[0], nn[1]]), name= 'Capa1')
b1 = tf.Variable(tf.random_normal([nn[1]]), name ='bias1')
#Funcion de activacion ReLU
l1 = tf.nn.relu(tf.add(tf.matmul(eX, c1), b1))

#Capa 2
c2 = tf.Variable(tf.random_normal([nn[1], nn[2]]), name= 'Capa2')
b2 = tf.Variable(tf.random_normal([nn[2]]), name ='bias2')
#Funcion de activacion ReLU
l2 = tf.nn.relu(tf.add(tf.matmul(l1, c2), b2))

#Capa 3
c3 = tf.Variable(tf.random_normal([nn[2], nn[3]]), name= 'Capa3')
b3 = tf.Variable(tf.random_normal([nn[3]]), name ='bias3')
#Funcion sigmoide para acotar los valores en 0 y 1
sY = tf.nn.sigmoid(tf.add(tf.matmul(l2,c3), b3))[:, 0]

#Calculo del error
error = tf.losses.mean_squared_error(sY, eY)

#Optimizador para la red, para que minimice el error
opt = tf.train.GradientDescentOptimizer(learning_rate =0.05).minimize(error)


# ************************** Entrenamiento de la red neuronal *********************************
print("Entrenamos la red")

ciclos = 1000 #Numero de ciclos de entrenamiento

evY = [] #Para guardar la evolucion

with tf.Session() as sess:
	#Inicializamos los parametros de la red
	sess.run(tf.global_variables_initializer())

	#Iteramos n veces de entrenamiento
	for ciclo in range(ciclos):
		#Evaluamos optimizador, costo y tensor de salida pY
		#La evaluacion del optimizador producira el entrenamiento de la red, correción mY

		_, _error, mY = sess.run([opt, error, sY], feed_dict={ eX : X, eY: Y})

		#Cada 25 iteraciones, imprimimos metricas
		if ciclo % 25 == 0:
			media = np.mean(np.round(mY) == Y)
			print('Ciclo', ciclo, '/', ciclos, '- Error = ', _error, ' - Media = ', media)
			mY = sess.run(sY, feed_dict = { eX: mX }).reshape((res, res))
			evY.append(mY)


# ************************************* Resultados **********************************************
print("Imprimimos resultados")

for r in range(len(evY)):
	print(evY[r])
 
		