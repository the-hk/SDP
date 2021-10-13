#/usr/bin/python3

from datetime import datetime
import numpy as np


#detect grasshopper and water cup and food


# start grasshopper voice detector

# start and get real-tim grasshopper behavior classification result

# if one of the movement classified call callback function which is named as callbackSaveClassifcation
    # check the voice that the time n before and after

        # question is that how can we understand this two stuff enough close to each other
        # train YO-LO by photos that taken while it was eating or drinking or pairing stuff

    #if there is voice before and after save them with class name

        # class names are 
        # * grasshopperFeeding
        # * drinkingWater
        # * grasshopperWants2Pair

class grasshopperIssueSaveSounds:

    def __init__(self,className,soundBefore,soundAfter):

        self.className = className
        self.soundBefore = soundBefore
        self.soundAfter = soundAfter

    def callbackSaveClassifcation(self):
        now = datetime.now()
        dt_string = now.strftime("date_%d.%m.%Y_time_%H:%M:%S")
        filename = dt_string 
        np.save("/home/hk/Desktop/sdp/"+self.className+"/"+"before_"+filename +".npy", self.soundBefore)
        np.save("/home/hk/Desktop/sdp/"+self.className+"/"+"after_"+filename +".npy", self.soundAfter)

  



