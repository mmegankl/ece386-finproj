#!/bin/bash

apt install gpiod
pip install Jetson.GPIO

# Find name for GPIO01
gpioinfo | grep 105
# line 105: "PQ.05" unused input active-high

# Should print 0 or 1 to the terminal
gpioget gpiochip0 105

# Run echo on rising edge
gpiomon -r -n 1 gpiochip0 105 | while read line; do echo "event $line"; done

#docker run line
--device=/dev/gpiochip0

# In terminal
export JETSON_MODEL_NAME=JETSON_ORIN_NANO

'''Prints 'UP' or 'DOWN' based on edges on Jetson pin #29'''
import Jetson.GPIO as GPIO

# Init as digital input
my_pin = 29
GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
GPIO.setup(my_pin, GPIO.IN)  # digital input

print('Starting Demo! Move pin 29 between 0V and 3.3V')

while True:
    GPIO.wait_for_edge(my_pin, GPIO.RISING)
    print('UP!')
    GPIO.wait_for_edge(my_pin, GPIO.FALLING)
    print('down')