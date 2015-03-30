"""
Card Recognition using OpenCV
Code from the blog post 
http://arnab.org/blog/so-i-suck-24-automating-card-games-using-opencv-and-python

Usage: 

  ./card_img.py filename num_cards training_image_filename training_labels_filename num_training_cards

Example:
  ./card_img.py test.JPG 4 train.png train.tsv 56
  card_img.py test3.JPG 4 train2.jpg train2.tsv 52
  card_img.py test3.JPG 4 train3.png train3.tsv 52
  card_img.py test3.JPG 4 train4.jpg train4.tsv 52
  black_nao.py naotest2.png 4 train4.jpg train4.tsv 52
  black_nao.py naotest3.png 9 train4.jpg train4.tsv 52
  black_nao.py naotest4.png 5 train4.jpg train4.tsv 52
  black_nao.py test4.jpg 4 train4.jpg train4.tsv 52
  black_nao.py train4.jpg 52 train4.jpg train4.tsv 52
  black_nao.py one_card.png 1 train4.jpg train4.tsv 52
  black_nao.py black_cloth1.png 4 train4.jpg train4.tsv 52 

Note: The recognition method is not very robust; please see SIFT / SURF for a good algorithm.  

"""

from naoqi import ALProxy
import sys
import numpy as np
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/") 
import cv2


###############################################################################
# Utility code from 
# http://git.io/vGi60A
# Thanks to author of the sudoku example for the wonderful blog posts!
###############################################################################

IP = "169.254.28.162"
PORT = 9559
STAND, HIT, OVER = 0, 1, 2

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

###############################################################################
# Image Matching
###############################################################################
def preprocess(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(5,5),2 )
  thresh, o = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)#.adaptiveThreshold(blur,255,1,1,5,1)
  return o
  
def imgdiff(img1,img2):
  img1 = cv2.GaussianBlur(img1,(5,5),5)
  img2 = cv2.GaussianBlur(img2,(5,5),5)    
  diff = cv2.absdiff(img1,img2)  
  diff = cv2.GaussianBlur(diff,(5,5),5)    
  flag, diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY)
  return np.sum(diff)  

def find_closest_card(training,img):
  features = preprocess(img)
  return sorted(training.values(), key=lambda x:imgdiff(x[1],features))[0][0]
  
   
###############################################################################
# Card Extraction
###############################################################################  
def getCards(im, numcards=4):
  gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(1,1),1000)
  flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)   
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  contours = sorted(contours, key=cv2.contourArea,reverse=True)[:numcards]  

  # print len(contours)
  cv2.drawContours(im, contours, -1, (0,255,0), 3)
  width = im.shape[1]
  newWidth = 1024
  newHeigh = im.shape[0] / (im.shape[1]/newWidth)
  imx = cv2.resize(im,(newWidth,newHeigh))
  cv2.imshow("naoimg", imx)
  cv2.waitKey(0)
  # cv2.imwrite('img_contours.jpg', imx)
  for card in contours:
    peri = cv2.arcLength(card,True)
    poly = cv2.approxPolyDP(card,0.02*peri,True)
    approx = rectify(poly)
    ### debbugging
    # cv2.drawContours(im, [poly], 0, (0,255,0),10,cv2.CV_AA)
    # imx = cv2.rearraySize(im,(2000,600))
    # cv2.imshow('a', imx)
    # cv2.waitKey(0)
    ### exit debugging

    # box = np.int0(approx)
    # cv2.drawContours(im,[box],0,(255,255,0),6)
    # imx = cv2.rearraySize(im,(1500,500))
    # cv2.imshow('a',imx)      
    
    if type(approx)!=bool:
      h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)

      transform = cv2.getPerspectiveTransform(approx,h)
      warp = cv2.warpPerspective(im,transform,(450,450))
      # if numcards==4:
      #   cv2.imshow(str(cv2.contourArea(card))+ " " + str(type(approx)), warp)
      #   cv2.waitKey(0)
      yield warp

def count_cards(im): 
  gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(1,1),1000)
  flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)   
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  contours = sorted(contours, key=cv2.contourArea,reverse=True)
  prev=0
  count=0
  curr=0.0
  for i, card in enumerate(contours):
    peri = cv2.arcLength(card,True)
    poly = cv2.approxPolyDP(card,0.02*peri,True)
    approx = rectify(poly)

    if type(approx)==bool: #check if the countours are rectifiable
      continue
    curr = cv2.contourArea(card)
    if (prev*0.8)<=curr: 
      # print prev, curr
      # cv2.drawContours(im, card, -1, (255, 0 ,0), 3)
      # imx = cv2.rearraySize(im,(1000,400))
      # cv2.imshow(str(cv2.contourArea(card)), imx)
      # cv2.waitKey(0)
      count+=1
      prev=curr
    else:
      break
    # print str(i), cv2.contourArea(card)
  return count

def get_training(training_labels_filename,training_image_filename,num_training_cards,avoid_cards=None):
  training = {}
  
  labels = {}
  for line in file(training_labels_filename): 
    key, num, suit = line.strip().split()
    labels[int(key)] = num
  
  # creating file in write mode
  # train_labels_file = open(training_labels_filename, "w+")  
  print "Training"

  im = cv2.imread(training_image_filename)
  for i,c in enumerate(getCards(im,num_training_cards)):
    if avoid_cards is None or (labels[i][0] not in avoid_cards[0] and labels[i][1] not in avoid_cards[1]):
      ### human assisted training
      #pcard=preprocess(c)
      #cv2.imshow("Preprocessed Card", pcard)
      #cv2.waitKey(0)
      # human_input = raw_input("Enter the value of the card: ")
      # train_labels_file.write(str(i) + "\t" + human_input.upper() + "\n")
      ### end human assited training

      training[i] = (labels[i], preprocess(c))
      #training[i] = (labels[i], preprocess(c))
  
  print "Done training"
  return training
  
def card_value(card, result):
	if card=='K' or card=='Q' or card=='J' or card=='10':
		return 10
	elif card=='A':
		return 11
	else:
		return int(card)

def sum_cards(nao_cards):
	result=0
	aces = 0
	for card in nao_cards:
		if card=='A':
			aces+=1
		result+=card_value(card, result)
		while result>21 and aces>0:
			result-=10
			aces-=1
	return result

def decide(nao_cards, dealer_card):
  summed = sum_cards(nao_cards)
  dealer = card_value(dealer_card, 0)
  if summed<=11:
    return HIT
  elif summed==12 and 4 > dealer > 6:
    return HIT
  elif 13 <= summed <= 16 and 2 > dealer > 6:
    return HIT
  elif summed>21:
    return OVER
  else:
    return STAND


if __name__ == '__main__':
  if len(sys.argv) == 6:
    filename = sys.argv[1]
    num_cards = int(sys.argv[2])
    training_image_filename = sys.argv[3]
    training_labels_filename = sys.argv[4]    
    num_training_cards = int(sys.argv[5])
    
    training = get_training(training_labels_filename,training_image_filename,num_training_cards)

    if True:
      im = cv2.imread(filename)
      
      width = im.shape[0]
      height = im.shape[1]
      if width < height and False:
        im = cv2.transpose(im)
        im = cv2.flip(im,1)

      # Debug: uncomment to see registered images
      for i,c in enumerate(getCards(im,num_cards)):
        card = find_closest_card(training,c,)
        cv2.imshow(str(card) + " " + str(i),c)
      cv2.waitKey(0)

      print "Cards on image: " + str(count_cards(im))
      cards = [find_closest_card(training,c) for c in getCards(im,num_cards)]
      print cards
      
      #### IF CONNECTED TO NAO SET TO TRUE
      if False:
        tts = ALProxy("ALTextToSpeech", IP, PORT)
        decision = decide(cards, '7')
        if decision==HIT:
        	tts.say("Hit")
        elif decision==STAND:
        	tts.say("Stand")
        elif decision==OVER:
          tts.say("Damn I lost!")

      raw_input("Press enter to finish")
      
    else:
      print __doc__
	