# Robot Nao playing Blackjack
This is a project I made for my course “Introduction in robotics”. It uses simple image processing and simple decision tree so the Nao robot is able to play simple Blackjack (with only Hit and Stand). To use this you are going to need the Nao robot, [Python 2.7.9](https://www.python.org/downloads/), [NaoSDK for Python](http://doc.aldebaran.com/1-14/dev/python/install_guide.html), [OpenCV](http://opencv.org/) and [Nyphy](https://pypi.python.org/pypi/numpy).

Please refer to the images located in **Training** folder and **Images** folder to see how your images should look.
#### Video demonstration: https://www.youtube.com/watch?v=yPWiHUWcCJ8

# Instructions:
1. First you need to prepare a training file using the **cardsTraining.py**. To use this, just start it and follow the instructions. The training image should contain all the cards in the deck you will be playing with. This is going to generate a TSV file for later use. You only need to do this once.
2. Position Nao in that way that it can see as much cards as possible, while keeping his legs and arms out of his field of view, also trying to avoid other rectangle objects in frame that may interfere with the image recognition. You need to use Choregraphe software to switch to his down facing camera on his chin.
3. Start **nao_blackjack.py arg1 arg2**, where
  * **arg1** is the location of the image used for training
  * **arg2** is the TSV file we generated in step 1. 

   Example: **nao_blackjack.py training.jpg training.tsv**
4. Deal two cards to each player. Also deal two cards to yourself (dealer) but **keep the card that is face-down under the face-up card**. To avoid this, you can also deal yourself just one card and later when it is your turn just deal another instead turn the one you have.
5. When it is Nao’s turn, touch his front tactile sensor on his head.
6. Follow the decisions that Nao makes.
7. When the round is over, Nao will give you instructions how to start new round (touch front tactile sensor on his head). If you would like stop playing, just touch the rear tactile sensor on his head.

For more informations on how the card image recognition works, please read on: http://arnab.org/blog/so-i-suck-24-automating-card-games-using-opencv-and-pytho. I have made some modifications to is for better card recognition and automatically counting how many cards are on the frame.



