from PIL import Image
import os

dataset_path = r'c:\Users\student\Desktop\GAN model\mc_skin_faces'
first_image = os.listdir(dataset_path)[0]
img = Image.open(os.path.join(dataset_path, first_image))
print(f"Image: {first_image}, Size: {img.size}")
