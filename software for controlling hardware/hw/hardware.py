import time
import board
import pulseio
import digitalio
import numpy as np
import RPi.GPIO as GPIO
import adafruit_character_lcd.character_lcd as LCD

from picamera import PiCamera

class Hardware:
    """Provides low level functions for IO control of the demonstrator."""

    # Constants for led control
    LEDS_TOP      = 7
    LEDS_BOTTOM   = 11
    LEDS_SIDE     = 13

    # Camera constants
    CAM_RES_WIDTH = 1664 
    CAM_RES_BUFF_WIDTH=1664
    CAM_RES_WIDTH_SM = 192
    CAM_RES_HEIGHT = 1232
    CAM_RES_BUFF_HEIGHT = 1232
    CAM_RES_HEIGHT_SM = 128
    CAM_IMG_REGULAR = 0
    CAM_IMG_SMALL = 1

    # Pinmap
    leds_top = digitalio.DigitalInOut(board.D4)
    leds_bottom = digitalio.DigitalInOut(board.D17)
    leds_side = digitalio.DigitalInOut(board.D27)

    lcd_rs = digitalio.DigitalInOut(board.D11)
    lcd_en = digitalio.DigitalInOut(board.D5)
    lcd_d7 = digitalio.DigitalInOut(board.D26)
    lcd_d6 = digitalio.DigitalInOut(board.D19)
    lcd_d5 = digitalio.DigitalInOut(board.D13)
    lcd_d4 = digitalio.DigitalInOut(board.D6)
    lcd_backlight = digitalio.DigitalInOut(board.D21) # unconnected wire

    # Working  vars
    lcd = None
    lcd_height = None
    lcd_last_mess = None
    camera = None
    topPWM = None

    @staticmethod 
    def init(disp_x, disp_y):
        """Initializes the hardware."""

        # Leds
        #GPIO.setmode(GPIO.BCM)
        #GPIO.setup(4, GPIO.OUT)
        #Hardware.topPWM = GPIO.PWM(4, 500)
        #Hardware.topPWM.start(0)
        Hardware.leds_top.direction = digitalio.Direction.OUTPUT
        Hardware.leds_side.direction = digitalio.Direction.OUTPUT
        Hardware.leds_bottom.direction = digitalio.Direction.OUTPUT
        Hardware.leds(False, False, False)


        # Display
        Hardware.lcd_height = disp_y
        Hardware.lcd = LCD.Character_LCD_Mono(
            Hardware.lcd_rs, 
            Hardware.lcd_en, 
            Hardware.lcd_d4, 
            Hardware.lcd_d5, 
            Hardware.lcd_d6,
            Hardware.lcd_d7,
            disp_x, 
            disp_y, 
            Hardware.lcd_backlight)
        Hardware.lcd.clear()
        Hardware.lcd.text_direction = Hardware.lcd.LEFT_TO_RIGHT

        # Camera
        Hardware.camera = PiCamera()
        Hardware.camera.resolution = (Hardware.CAM_RES_WIDTH, Hardware.CAM_RES_HEIGHT)
        Hardware.camera.iso = 100
        time.sleep(0.1)
        Hardware.camera.shutter_speed = Hardware.camera.exposure_speed
        Hardware.camera.exposure_mode = 'off'
        Hardware.camera.awb_mode = 'off'
        Hardware.camera.awb_gains = 1.5

    @staticmethod 
    def leds(top: bool, side : bool, bottom: bool):
        """Turns state of all three LED strips of the demonstrator."""

        Hardware.led(Hardware.LEDS_TOP, top)
        Hardware.led(Hardware.LEDS_BOTTOM, bottom)
        Hardware.led(Hardware.LEDS_SIDE, side)

    @staticmethod
    def led(kind: int, on: bool):
        """Turns state of LED strips of a specified 'kind' to an 'on' state."""

        if kind == Hardware.LEDS_TOP:
            #if on: 
            #    Hardware.topPWM.ChangeDutyCycle(100)
            #else:
            #    Hardware.topPWM.ChangeDutyCycle(0)
            Hardware.leds_top.value = on
        elif kind == Hardware.LEDS_SIDE:
            Hardware.leds_side.value = on
        elif kind == Hardware.LEDS_BOTTOM:
            Hardware.leds_bottom.value = on
        else:
            raise Exception('LEDs must be one of a kind specified in Hardware class constants.')

    @staticmethod 
    def cam(iso = 100, shutter = 0, type = 0):
        """Gets a picture from camera with a passed settings and stores it in a numpy array."""

        w = Hardware.CAM_RES_WIDTH
        h = Hardware.CAM_RES_HEIGHT
        wbuff = Hardware.CAM_RES_BUFF_WIDTH
        hbuff = Hardware.CAM_RES_BUFF_HEIGHT
        if type == Hardware.CAM_IMG_SMALL:
            w = Hardware.CAM_RES_WIDTH_SM
            h = Hardware.CAM_RES_HEIGHT_SM
            wbuff = w
            hbuff = h

        Hardware.camera.resolution = (w, h)
        Hardware.camera.iso = iso
        time.sleep(0.1)
        if shutter == 0:
            Hardware.camera.shutter_speed = Hardware.camera.exposure_speed
        else:
            Hardware.camera.shutter_speed = int(shutter * 1E6)

        output = np.empty((hbuff, wbuff, 3), dtype=np.uint8)
        Hardware.camera.capture(output, 'bgr')

        return output

    @staticmethod 
    def disp_clear():
        """Clears the display and returns cursor to the start position."""
        Hardware.lcd.clear()

    @staticmethod
    def disp_out(lines):
        """Outputs a list of lines to the display."""
        
        # Has to account for memory shifts in 3rd and 4th line
        index = 0
        msg = ''
        for line in lines:
            if index < 2:
                msg = msg + '    ' + line + '\n'
            elif index < Hardware.lcd_height - 1:
                msg = msg + line + '\n'
            else:
                msg = msg + line
            index += 1

        # Do change only when there is one to avoid blinking
        if Hardware.lcd_last_mess is None or msg != Hardware.lcd_last_mess:
            Hardware.lcd.clear()
            Hardware.lcd.message = Hardware.lcd_last_mess = msg
            Hardware.lcd.move_left()
            Hardware.lcd.move_left()
            Hardware.lcd.move_left()
            Hardware.lcd.move_left()
