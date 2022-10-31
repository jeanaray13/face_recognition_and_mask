#Programa que entrena el reconocimiento de rostros

#Importación de las librerias cv2, os y numpy
import cv2
import os
import numpy as np

dataPath = 'Datapath+/Data' #Ruta de los datos
peopleList = os.listdir(dataPath) #Lista las personas capturadas
print("Lista de personas: ", peopleList)

labels = []
facesData = []
label = 0

#Especifica la ruta del directorio
for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir
	print("Leyendo las imágenes...")

	#Lectura de cada rostro
	for fileName in os.listdir(personPath):
		print('Rostros: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread('Data/'+nameDir+'/'+fileName,0)) #Cada imagen obtenida se lo pasa a escala de grises
		image = cv2.imread('Data/'+nameDir+'/'+fileName,0)

	label = label + 1

#Entrenamiento LBPH (Local Binary Pathern Histogram)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

#Almacenando el modelo obtenido
face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado...")
