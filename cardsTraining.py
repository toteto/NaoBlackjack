import sys
import numpy as np
from optparse import OptionParser
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/") 
import cv2
import time
import Image

def rectify(h):
	try:
		h = h.reshape((4,2))
	except:
		#print "Failed to rectify card"
		return False
	hnew = np.zeros((4,2),dtype = np.float32)

	add = h.sum(1)
	hnew[0] = h[np.argmin(add)]
	hnew[2] = h[np.argmax(add)]
	 
	diff = np.diff(h,axis = 1)
	hnew[1] = h[np.argmin(diff)]
	hnew[3] = h[np.argmax(diff)]

	return hnew

def preprocess(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),2 )
	thresh, o = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)#.adaptiveThreshold(blur,255,1,1,5,1)
	return o

def getCards(im, numcards=4):
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(1,1),1000)
	flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)   
	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	contours = sorted(contours, key=cv2.contourArea,reverse=True)[:numcards]  

	# Debug: Uncomment to see how the countours of the cards detected
	# cv2.drawContours(im, contours, -1, (0,255,0), 3)
	# newWidth = 1024
	# newHeigh = im.shape[0] / (im.shape[1]/newWidth)
	# imx = cv2.resize(im,(newWidth,newHeigh))
	# cv2.imshow("getCards", imx)
	# cv2.waitKey(0)

	for card in contours:
		peri = cv2.arcLength(card,True)
		poly = cv2.approxPolyDP(card,0.02*peri,True)
		approx = rectify(poly)      
		
		if type(approx)!=bool:
			h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)
			transform = cv2.getPerspectiveTransform(approx,h)
			warp = cv2.warpPerspective(im,transform,(450,450))

			yield warp

def createTrainingFile(training_labels_filename="training.tst", training_image_filename="training.png", num_training_cards=52):
	# Creating file in write mode
	train_labels_file = open(training_labels_filename, "w+")  
	print "Started training"
	print ""
	print "Suit=Symbol\nHarts=H\nDiamond=D\nSpades=S\nClubs=C"

	im = cv2.imread(training_image_filename)
	for i, c in enumerate(getCards(im,num_training_cards)):
		pcard=preprocess(c)
		cv2.imshow("Preprocessed Card", pcard)
		cv2.waitKey(1)
		human_input = raw_input("Enter value of the card in format 'VALUE SUIT':\n")
		value, suit = human_input.strip().upper().split()
		train_labels_file.write(str(i) + "\t" + value + "\t" + suit + "\n")
	train_labels_file.close()
	cv2.destroyAllWindows() 
	print "Done training"

if __name__ == '__main__':
	training_image_filename = raw_input("Enter the name of the training image (ex. training.png):\n")
	training_labels_filename = raw_input("Enter the name of the training file that will be created (ex. training.tst):\n")
	num_training_cards = int(raw_input("Enter number of cards on training image (ex. 52):\n"))

	createTrainingFile(training_labels_filename, training_image_filename, num_training_cards)
	raw_input("Press enter to exit")
