#bin/python

from datetime import datetime
import numpy as np

#detect grasshopper and water cup and food


# start grasshopper voice detector

# start and get real-tim grasshopper behavior classification result

# if one of the movement classified call callback function which is named as callbackSaveClassifcation
    # check the voice that the time n before and after

    #if there is voice before and after save them with class name
        # * class names are grasshopperFeeding
        # * grasshopper drinking water
        # * grasshopperWants2Pair

class grasshopperIssue:

    def __init__(self,className,soundBefore,soundAfter):

        self.className = className
        self.soundBefore = soundBefore
        self.soundAfter = soundAfter

    def callbackSaveClassifcation(self):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        filename = dt_string + str(className)
        np.save(filename +" before", soundBefore)
        np.save(filename +" after", soundAfter)


