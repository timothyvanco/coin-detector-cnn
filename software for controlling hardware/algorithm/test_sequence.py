import time
import numpy as np
import random
import cv2

from algorithm.algorithm_interface import AlgorithmInterface
from hw.hardware import Hardware

class TestSequence(AlgorithmInterface):

    def alg(self, iterations: int) -> dict:
        """
        Represents a demonstration and camera checking algorithm. 
        Results and internals of any other algorithm may mimic the ones shown in here.

        The best known setting for camera:
        Top and side leds, iso 200, shutter 1/250 (without paper)
        Top and side leds, iso ??, shutter 1/?? (with paper, doesn't even help with reflections)
        """

        # self.__seq('top', True, False, False, iso = 50)
        # self.__seq('side', False, True, False, iso = 400)
        # self.__seq('top_side', True, True, False, iso = 5)
        # self.__seq('top_bottom', True, False, True, iso = 20)
        # self.__seq('side_bottom', False, True, True, iso = 400)
        # self.__seq('top_side_bottom', True, True, True, iso = 5)    

        self.__seq('bottom_' + str(iterations), False, False, True, iso = 50, shutter = (1 / 400))
        im = self.__seq('master_' + str(iterations), True, True, False, iso = 200, shutter = (1 / 250))

        coins = {
            ' 1Kc': 0, 
            ' 2Kc': 0, 
            ' 5Kc': 0,
            '10Kc': 0,
            '20Kc': 0,
            '50Kc': 0,
            '  1c': 0,
            '  2c': 0,
            '  5c': 0,
            ' 10c': 0,
            ' 20c': 0,
            ' 50c': 0,
            '1EUR': 0,
            '2EUR': 0
        }

        found = False
        if random.randint(0,3) != 0:
            found = True
            for key in coins:
                coins[key] = random.randint(0,1)

        return {
            'found': found,
            'image_annotated': cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
            'image_original': cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
            'coins': coins
        }


    def __seq(self, name, top, side, bottom, iso = 300, shutter     = 0):
        Hardware.leds(top, side, bottom)
        time.sleep(0.1)
        im = Hardware.cam(iso = iso, shutter = shutter)
        cv2.imwrite('./img/{0}.jpg'.format(name), im)
        Hardware.leds(False, False, False)
        return im 

