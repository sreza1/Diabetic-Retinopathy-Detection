import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

directory = "data/downloaded/train"
counter = 0
for filename in os.listdir(directory):
    counter+=1
    print(counter)
    image_file = os.path.join(directory, filename)
    if image_file != os.path.join(directory, '.DS_Store'):
        image = Image.open(image_file)
        new_image = image.resize((650, 650))
        new_image.save('data/resized_data_650/'+filename)

