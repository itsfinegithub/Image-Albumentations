import cv2
import glob
from zipfile import ZipFile
import os
import shutil
import albumentations as A

def unzip():
    with ZipFile(r"C:\Users\Lakshmi M\Desktop\task_3\tiny_shoppee_train.zip") as zip:
        zip.extractall()
        
    folders = []
    for name in glob.glob(r'tiny_shoppee_train/*'):
        F = name.split('\\')[-1]
        folders.append(F)

    for i in folders:
        os.makedirs(f'task3_folders/{i}')

    transform = A.Compose([
        A.Blur(blur_limit=3),
        A.GridDistortion(),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2)
        ])
    
    for img in glob.glob(r"C:/Users/Lakshmi M/Desktop/task_3/tiny_shoppee_train/*/*"):
        image= cv2.imread(img)
        transformed = transform(image=image)
        transformed_image = transformed["image"] 
        transform = A.Compose([A.RandomBrightnessContrast(brightness_limit=1)])
        transformed_image_1 = transform(image=image)['image']
        folder_names = img.split('\\')[-2]
        images_name = img.split('\\')[-1]
        

        if folder_names in img:
            cv2.imwrite(f'task3_folders/{folder_names}/{images_name}',transformed_image_1)

    return "Images are augumented"       
shutil.make_archive('task3_folders','zip','task3_folders') 
if __name__=="__main__":
    unzip()