from connections import connect_mqtt, connect_internet
from time import sleep
from motors import move_forward, move_backward, turn_left, turn_right, stop

# Function to handle an incoming message

def cb(topic, msg):
    
    # Car movement calls
    if topic == b"direction":
        if msg == b"forward":
            print("Forward")
            move_forward()

        elif msg == b"backward":
            print("Backward")
            move_backward()

        elif msg == b"left":
            print("left")
            turn_left()

        elif msg == b"right":
            print("right")
            turn_right()

        elif msg == b"stop":
            print("Stop")
            stop()

            
            


def main():
    try:
        connect_internet("network name",password="network password")
        client = connect_mqtt("mqtt broker ip", "client name", "client password")

        client.set_callback(cb)
        client.subscribe("direction")
        while True:
            client.check_msg()
            sleep(0.1)
    except KeyboardInterrupt:
        print('keyboard interrupt')
        
        
if __name__ == "__main__":
    main()

