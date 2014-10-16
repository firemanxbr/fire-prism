import cv
import time
from PIL import Image

cv.NamedWindow("camera", 1)
capture = cv.CreateCameraCapture(0)

width = None
height = None
width = 640
height = 320

if width is None:
    width = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH))
else:
	cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FRAME_WIDTH,width)    

if height is None:
	height = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT))
else:
	cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FRAME_HEIGHT,height) 

result = cv.CreateImage((width,height),cv.IPL_DEPTH_8U,3) 

def Load():

	return (faceCascade, eyeCascade)

def Display(image):
	cv.NamedWindow("Olho Vermelho Test")
	cv.ShowImage("Olho Vermelho Test", image)
	cv.WaitKey(0)
	cv.DestroyWindow("Olho Vermelho Test")

def DetectRedEyes(image, faceCascade, eyeCascade):
	min_size = (20,20)
	image_scale = 2
	haar_scale = 1.2
	min_neighbors = 2
	haar_flags = 0

	# Alocacao Temporaria das imagens
	gray = cv.CreateImage((image.width, image.height), 8, 1)
	smallImage = cv.CreateImage((cv.Round(image.width / image_scale),cv.Round (image.height / image_scale)), 8 ,1)

	# Converter a imagem em tons de cinza e entrada de cor
	cv.CvtColor(image, gray, cv.CV_BGR2GRAY)

	"""
	"""
	# Imagem de entrada escala para processamento mais rapido
	cv.Resize(gray, smallImage, cv.CV_INTER_LINEAR)

	# Equalizar o histograma
	cv.EqualizeHist(smallImage, smallImage)

	# Detectar os rostos
	faces = cv.HaarDetectObjects(smallImage, faceCascade, cv.CreateMemStorage(0),
	haar_scale, min_neighbors, haar_flags, min_size)

	# Se exixtirem Rostos
	if faces:
		for ((x, y, w, h), n) in faces:
		#a entrada para cv.HaarDetectObjects foi redimensionada, para dimensionar a
		#caixa delimitadora de cada rosto e converte-lo em dois CvPoints
			pt1 = (int(x * image_scale), int(y * image_scale))
			pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
			cv.Rectangle(image, pt1, pt2, cv.RGB(0, 0, 255), 3, 8, 0)
			face_region = cv.GetSubRect(image,(x,int(y + (h/4)),w,int(h/2)))

		cv.SetImageROI(image, (pt1[0],
			pt1[1],
			pt2[0] - pt1[0],
			int((pt2[1] - pt1[1]) * 0.7)))
		eyes = cv.HaarDetectObjects(image, eyeCascade,
		cv.CreateMemStorage(0),
		haar_scale, min_neighbors,
		haar_flags, (15,15))	

		if eyes:
		#Para cada olho encontrado
			for eye in eyes:
				# Desenhe um retangulo em volta do olho
				cv.Rectangle(image,
				(eye[0][0],
				eye[0][1]),
				(eye[0][0] + eye[0][2],
				eye[0][1] + eye[0][3]),
				cv.RGB(0, 0, 255), 1, 8, 0)

	cv.ResetImageROI(image)
	return image

faceCascade = cv.Load("haarcascade_frontalface_alt.xml")
eyeCascade = cv.Load("haarcascade_eye.xml")

while True:
	img = cv.QueryFrame(capture)

	image = DetectRedEyes(img, faceCascade, eyeCascade)
	cv.ShowImage("camera", image)
	k = cv.WaitKey(10);
	if k == 'f':
		break
