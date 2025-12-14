import requests
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, "downloaded_image.jpg")

url = "ip address/size(_x_).jpg"

# Function to download the image
def download_image():
    response = requests.get(url)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Image saved to: {filename}")
    else:
        print("Failed to download image. Status code:", response.status_code)

download_image()
