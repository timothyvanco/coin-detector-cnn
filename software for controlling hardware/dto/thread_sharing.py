import _thread
import copy

# This module handles mutexed resource sharing between UI and demonstrator threads
# Event producers are posting values, event consumers are reading them
# We have only 2 threads, thus a single lock should be sufficient without noticeable performance issues
resultLock = _thread.allocate_lock()

# Algorithm result sharing
result = None
def post_result(res):
    global result 
    resultLock.acquire()
    result = copy.deepcopy(res) 
    resultLock.release()

def read_result():
    global result
    resultLock.acquire()
    temp = copy.deepcopy(result) 
    result = None
    resultLock.release()
    return temp

# Algorithm started sharing
started = None
def post_start(start):
    global started 
    resultLock.acquire()
    started = start 
    resultLock.release()
    
def read_start():
    global started 
    temp = ''
    resultLock.acquire()
    temp = started 
    resultLock.release()
    
    return temp

# Algorithm selection
algorithm = None

def post_algorithm(alg):
    global algorithm 
    resultLock.acquire()
    algorithm = alg 
    resultLock.release()
    
def read_algorithm():
    global algorithm 
    temp = ''
    resultLock.acquire()
    temp = algorithm 
    resultLock.release()
    
    return temp

# Program mode selection
MODE_RECOGNITION = 0
MODE_CAPTURING = 1
mode = MODE_RECOGNITION

def post_mode(md):
    global mode 
    resultLock.acquire()
    mode = md 
    resultLock.release()
    
def read_mode():
    global mode 
    temp = 0
    resultLock.acquire()
    temp = mode 
    resultLock.release()
    
    return temp

# Leds selection 
leds = (False, False, False)

def post_leds(tuple):
    global leds 
    resultLock.acquire()
    leds = tuple
    resultLock.release()
    
def read_leds():
    global leds 
    temp = (False, False, False)
    resultLock.acquire()
    temp = leds 
    resultLock.release()
    
    return temp

# Shutter speed
shutter = 1 / 60

def post_shutter(sh):
    global shutter
    resultLock.acquire()
    shutter = sh
    resultLock.release()
    
def read_shutter():
    global shutter 
    temp = 0
    resultLock.acquire()
    temp = shutter 
    resultLock.release()
    
    return temp


# ISO
iso = 100

def post_iso(i):
    global iso
    resultLock.acquire()
    iso = i
    resultLock.release()
    
def read_iso():
    global iso 
    temp = 0
    resultLock.acquire()
    temp = iso 
    resultLock.release()
    
    return temp

# Currency
currency = 'CZK'

def post_currency(curr):
    global currency
    resultLock.acquire()
    currency = curr
    resultLock.release()
    
def read_currency():
    global currency 
    temp = ''
    resultLock.acquire()
    temp = currency 
    resultLock.release()
    
    return temp