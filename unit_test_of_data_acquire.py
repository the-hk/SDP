#/usr/bin/python3
from acquire_sound_data import grasshopperIssueSaveSounds
import numpy as np

className   = "drinkingWater"
soundBefore = np.array([1,2,3,4,5,6])
soundAfter  = np.array([1,2,3,4,5,6])

hk = grasshopperIssueSaveSounds(className,soundBefore,soundAfter)

hk.callbackSaveClassifcation()