from machine import Pin
import time

In1 = Pin(6, Pin.OUT)
In2 = Pin(7, Pin.OUT)
EN_A = Pin(8, Pin.OUT)

In3 = Pin(5, Pin.OUT)
In4 = Pin(4, Pin.OUT)
EN_B = Pin(3, Pin.OUT)

EN_A.high()
EN_B.high()

In5 = Pin(13, Pin.OUT)
In6 = Pin(14, Pin.OUT)
EN_C = Pin(15, Pin.OUT)

In7 = Pin(12, Pin.OUT)
In8 = Pin(11, Pin.OUT)
EN_D = Pin(10, Pin.OUT)

EN_C.high()
EN_D.high()

#In1 = right back
#In2 = right forward
#In3 = left forward
#In4 = left back

#In5 = left back
#In6 = left forward
#In7 = right forward
#In8 = right back

# All four forward
def move_forward():
    In1.low()
    In2.high()
    In3.high()
    In4.low()
    In5.low()
    In6.high()
    In7.high()
    In8.low()

# All four backward
def move_backward():
    In1.high()
    In2.low()
    In3.low()
    In4.high()
    In5.high()
    In6.low()
    In7.low()
    In8.high()
    
# Left pair backward, right pair forward
def turn_left():
    In1.low()
    In2.high()
    In3.low()
    In4.high()
    In5.high()
    In6.low()
    In7.high()
    In8.low()

# Right pair backward, left pair forward
def turn_right():
    In1.high()
    In2.low()
    In3.high()
    In4.low()
    In5.low()
    In6.high()
    In7.low()
    In8.high()

# All motors off
def stop():
    In1.low()
    In2.low()
    In3.low()
    In4.low()
    In5.low()
    In6.low()
    In7.low()
    In8.low()
    