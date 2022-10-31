#Programa que entrena el reconocimiento de rostros y de mascarilla

#Importación de las librerias cv2, os y numpy
import cv2
import os
import imutils
import mediapipe as mp

#Especificamos la ruta en de las personas que se va reconocer y obtener el nombre de los mismos
dataPath = 'Datapath+/Data' #Ruta de los datos
imagePaths = os.listdir(dataPath) #Lista las personas capturadas
print("imagePaths=", imagePaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Leyendo el modelo de reconocimiento facial
face_recognizer.read('modeloLBPHFace.xml')

mp_face_detection = mp.solutions.face_detection

LABELS = ["Sin_Mascarilla","Con_Mascarilla"]

#Lectura del modelo de mascarillas
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Imágenes y Videos de Prueba/Jeff.mp4')

faceClassific = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:

	while True:
		ret, frame = cap.read()
		if ret == False: break
		frame = imutils.resize(frame, width=640)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		auxFrame = gray.copy()

		faces = faceClassific.detectMultiScale(gray,1.3,5)

		height, width, _ = frame.shape
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = face_detection.process(frame_rgb)

		#Ubicación del rostro
		for (x,y,w,h) in faces:
		
			rostro = auxFrame[y:y+h,x:x+w]
			rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
			result = face_recognizer.predict(rostro) #Predice la etiqueta y la confianza asociada

			#LBPHFace
			if result[1] < 70:
				cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,0.8,(0,255,0),1,cv2.LINE_AA)
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			else:
				cv2.putText(frame,'Desconocido',(x-100,y-25),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

		if results.detections is not None:
			for detection in results.detections:
				xmin = int(detection.location_data.relative_bounding_box.xmin * width)
				ymin = int(detection.location_data.relative_bounding_box.ymin * height)
				w = int(detection.location_data.relative_bounding_box.width * width)
				h = int(detection.location_data.relative_bounding_box.height * height)

			#Si hay valores negativos
			if xmin<0 and ymin<0: continue

			#Capturando el rostro de la imagen en pequeño en escala de grises 72x72
			face_image = frame[ymin:ymin+h,xmin:xmin+h]
			face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
			face_image = cv2.resize(face_image,(72,72),interpolation=cv2.INTER_CUBIC) #Se redimensiona al tamaño de las imágenes del dataset

			#Aplicando el modelo entrenado
			result1 = face_mask.predict(face_image)

			if result1[1]<150:
				if LABELS[result1[0]] == "Con_Mascarilla": 
					#cv2.putText(frame,'{}'.format(LABELS[result1[0]]),(x+100,y-25),2,0.8,(0,255,0),1,cv2.LINE_AA) #Ejecutar cuando sea video
					cv2.putText(frame,'{} {}'.format(imagePaths[result[0]],LABELS[result1[0]]),(x,y-25),2,0.8,(0,255,0),1,cv2.LINE_AA) #Se obtiene el valor de predicción y valor de confianza
					cv2.rectangle(frame,(xmin,ymin),(xmin+w,ymin+h),(0,255,0),2)
				else:
					cv2.putText(frame,'{}'.format(LABELS[result1[0]]),(x+100,y-25),2,0.8,(0,0,255),1,cv2.LINE_AA)

		cv2.imshow('frame',frame)
		k = cv2.waitKey(1)
		if k == 27: break

cap.release()
cv2.destroyAllWindows()
