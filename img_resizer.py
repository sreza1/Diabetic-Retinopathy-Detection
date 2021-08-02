import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

directory = "data/data_copy/test_001"
for filename in os.listdir(directory):

    image_file = os.path.join(directory, filename)
    if image_file != os.path.join(directory, '.DS_Store'):
        image = Image.open(image_file)
        new_image = image.resize((400, 400))
        new_image.save('data/resized_data/test_001/'+filename)

