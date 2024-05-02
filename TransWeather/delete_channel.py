import os
import random
from PIL import Image

# specify your path
path = "/home/ec2-user/TransWeather/data/train/input"

# get list of all files
files = os.listdir(path)
# filter out all non-image files
images = [file for file in files if file.endswith(('jpg', 'png', 'jpeg'))]

# pick a random image
random_image_name = random.choice(images)
random_image_path = os.path.join(path, random_image_name)

# open the image
img = Image.open(random_image_path)

# check if it has 4 channels
if len(img.split()) == 4:
    print("Start working")
    # loop through all images
    for image_name in images:
        image_path = os.path.join(path, image_name)
        img = Image.open(image_path)
        # if image has 4 channels
        if len(img.split()) == 4:
            # delete the fourth channel
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            # save back to the original image
            img.save(image_path)
    print("Finish working")