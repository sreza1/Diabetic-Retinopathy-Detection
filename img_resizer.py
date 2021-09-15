import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

directory = "data/test"
counter = 0
for filename in os.listdir(directory):
    counter+=1
    print(counter)
    image_file = os.path.join(directory, filename)
    if image_file != os.path.join(directory, '.DS_Store'):
        image = Image.open(image_file)
        new_image = image.resize((650, 650))
        new_image.save('data/images_resized_650/test/'+filename)

