from create_sub_database import split_image
import cv2
import numpy as np

# folder = "D:\\OneDrive\\Documents\\FruitSpec\\datasets\\South Africa\\L3_2"
ysteps = 128
xsteps = 128
folder = "D:\\OneDrive\\Documents\\FruitSpec\\datasets\\South Africa\\L3"
filename = "NDRIrgb.jpg"
for fileind in range(42):
    img = cv2.imread(folder + "\\"+str(fileind+1)+"\\"+"NDRI_L"+str(fileind+1)+".jpg")
    # out_folder = folder + "\\"+str(fileind+1)+"\\sub_images"
    out_folder = "D:\OneDrive\Documents\FruitSpec\datasets\South Africa\L3\sub_images_all"
    sub_imgs, xTopLeft, yTopLeft, xBottomRight, yBottomRight = split_image(img, xsteps, ysteps)
    for rind in range(np.shape(sub_imgs)[0]):
        for cind in range(np.shape(sub_imgs)[1]):
            # creating and saving XML file of small image annotation
            try:
                img = sub_imgs[rind][cind]

            except Exception:
                continue

            # saving image and XML
            img = cv2.resize(img,(0,0),fx=4,fy=4)
            cv2.imwrite((out_folder + "\\" + filename[:-4] +"L"+str(fileind+1)+ "_{}_{}.jpg".format(rind, cind)), img)
