import time
import cv2
import time
import sys

from hw.disp import Display
from hw.hardware import Hardware
from trigger.motion_detect import MotionDetect 
from dto.thread_sharing import (
    post_result, 
    post_start, 
    read_algorithm, 
    read_mode, 
    read_leds, 
    read_iso,
    read_shutter,
    read_currency,
    MODE_RECOGNITION, 
    MODE_CAPTURING
)

from algorithm.algorithm_interface import AlgorithmInterface 
from algorithm.test_sequence import TestSequence 
from algorithm.kd import KD 
from algorithm.kdvn import KDVN

def get_algorithm() -> AlgorithmInterface:

    algType = read_algorithm()
    print(algType)
    alg = None
    if algType == 'Randomized test sequence':
        alg = TestSequence()
    elif algType == 'KDVN-algorithm':
        alg = KDVN(False)
    elif algType == 'KDVN-algorithm-CNN (exp.)':
        alg = KDVN(True)

    return alg

def demonstrator_proc():
    print("Starting program MROZ 2020 - coins")

    Hardware.init(16,4)
    detector = MotionDetect()
    disp = Display(4, [
        "   MROZ  2020   ",
        "  Matous  Hybl  ",
        " Timotej  Vanco ", 
        " Lukas  Korinek "
    ])

    iterations = 0
    while True:
        iterations += 1
        sys.stdout.flush()

        time.sleep(0.1)
        if(iterations % 3 == 0 or iterations == 0):
                disp.do_show()

        mode = read_mode()
        if mode == MODE_RECOGNITION:

            if detector.acknowledge():  
                post_start(True)
                res = get_algorithm().alg(iterations, read_currency())
                post_start(False)
                post_result(res)

                if res['found']:
                    disp.set_values(res['coins'])    
                else:
                    disp.set_values({})
                disp.do_show()
            else:
                pass 

        elif mode == MODE_CAPTURING:
            disp.set_values({})
            time.sleep(1)
            if(iterations % 3 == 0):

                (top, side, bottom) = read_leds()
                iso = read_iso()
                shutter = read_shutter()
                
                post_start(True)
                Hardware.leds(top, side, bottom)
                im = Hardware.cam(iso = iso, shutter = shutter)
                cv2.imwrite('img/captured_{0}.png'.format(iterations), im)
                Hardware.leds(False, False, False)
                post_start(False)
                post_result(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))




