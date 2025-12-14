import sys
from camera import download_image
from send_image import query
import paho.mqtt.client as paho
from paho import mqtt

def main():
    client = paho.Client(client_id="client name", userdata=None, protocol=paho.MQTTv5)
    client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
    client.username_pw_set("client name", "client pass")
    client.connect("mqtt broker ip", 8883)

    # Main loop with input options
    try:
        while True:
            user_input = input("Enter 'w' to start the car, '1' to use base, '2' to use crf, '3' to use grounded, '4' to use grounded + crf, '5' to use SAM, '6' to use SAM2, 's' to manually stop the car, or 'q'to exit: ").strip().lower()
            
            if user_input == '1' or user_input == '2' or user_input == '3' or user_input == '4' or user_input == '5' or user_input == '6':
                client.publish("direction", "stop")
                download_image()
                if user_input == '1':
                    action = query("./downloaded_image.jpg", model="cnn", variant="base")["action"]
                if user_input == '2':
                    action = query("./downloaded_image.jpg", model="cnn_crf", variant="base_crf")["action"]
                elif user_input == '3':
                    action = query("./downloaded_image.jpg", model="cnn", variant="grounded")["action"]
                elif user_input == '4':
                    action = query("./downloaded_image.jpg", model="cnn_crf", variant="grounded_crf")["action"]
                elif user_input == '5':
                    action = query("./downloaded_image.jpg", model="cnn_sam", variant="base")["action"]
                elif user_input == '6':
                    action = query("./downloaded_image.jpg", model="cnn_sam2", variant="base")["action"]
                client.publish("direction", action)
            elif user_input == 'q':
                client.publish("direction", "stop")
                client.disconnect()
                print("Exiting the program.")
                break
            elif user_input == 'w':
                client.publish("direction", "forward")
            elif user_input == 's':
                client.publish("direction", "stop")
            else:
                print("Invalid input. Please enter 'update' or 'quit'.")
    except Exception:
        print("Exiting the program.")
        client.disconnect()
                
if __name__ == '__main__':
    main()