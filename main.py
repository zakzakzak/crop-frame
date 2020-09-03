"""
Mengambil gambar di salah satu film ghibli
"""

import cv2
import numpy as np
# pengambilan video dari folder
cap = cv2.VideoCapture('D:/anime/Coquelicot-zaka kara BD/Coquelicot-zaka kara BD.mkv')


arr_dataset = np.ones((0,0))
counter = 0
# loop forever gambar dari video
while cap.isOpened():
    # read frame setiap video
    ret, frame = cap.read()

    # counter untuk perhitungan frame
    counter = counter+1
    # perhitungan dilakukan setiap 5 frame untuk alasan kecepatan komputasi
    if counter%5==0:
        # print(counter)

        # mengubah gambar video (frame) jadi gray
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize gambar--------------------------------------------
        scale_percent = 50 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        # Resize gambar--------------------------------------------
        
        # [ y (atas bawah) , x (kiri->kanan) ]
        # [ mulai 10, sampai 350], 170 x 170
        # 170 x 3 = 510, total 640
        # 65 [ 510 ] 65
        # crop_img1 = frame[10:10+170 ,         65 :         65+170]
        # crop_img2 = frame[10:10+170 ,     65+170 :     65+170+170]
        # crop_img3 = frame[10:10+170 , 65+170+170 : 65+170+170+170]

        # crop_img4 = frame[180:180+170 ,         65 :         65+170]
        # crop_img5 = frame[180:180+170 ,     65+170 :     65+170+170]
        # crop_img6 = frame[180:180+170 , 65+170+170 : 65+170+170+170]

        # case 2 : full
        crop_img_full1 = frame[10:10+340 , 0:340] 
        crop_img_full2 = frame[10:10+340 , 150:150+340] 
        crop_img_full3 = frame[10:10+340 , 150+150:150+150+340]

        gray1 = cv2.cvtColor(crop_img_full1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(crop_img_full2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(crop_img_full3, cv2.COLOR_BGR2GRAY)


        # Resize gambar--------------------------------------------
        scale_percent = 20 # percent of original size
        width = int(gray1.shape[1] * scale_percent / 100)
        height = int(gray1.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        gray1 = cv2.resize(gray1, dim, interpolation = cv2.INTER_AREA)
        # Resize gambar--------------------------------------------

        # Resize gambar--------------------------------------------
        scale_percent = 20 # percent of original size
        width = int(gray2.shape[1] * scale_percent / 100)
        height = int(gray2.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        gray2 = cv2.resize(gray2, dim, interpolation = cv2.INTER_AREA)
        # Resize gambar--------------------------------------------

        # Resize gambar--------------------------------------------
        scale_percent = 20 # percent of original size
        width = int(gray3.shape[1] * scale_percent / 100)
        height = int(gray3.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        gray3 = cv2.resize(gray3, dim, interpolation = cv2.INTER_AREA)
        # Resize gambar--------------------------------------------

        # --view------------------------
        flat1 = gray1.flatten()
        flat2 = gray2.flatten()
        flat3 = gray3.flatten()

        if arr_dataset.shape[0] == 0:
            arr_dataset = np.array([flat1])
            arr_dataset = np.append(arr_dataset, [flat2], axis=0)
            arr_dataset = np.append(arr_dataset, [flat3], axis=0)
        else:
            arr_dataset = np.append(arr_dataset, [flat1], axis=0)
            arr_dataset = np.append(arr_dataset, [flat2], axis=0)
            arr_dataset = np.append(arr_dataset, [flat3], axis=0)
        

        print(arr_dataset[-1])
        print(arr_dataset.shape[0])
        print("-----------")
        resh = flat3.reshape((68,68))
        cv2.imshow("feed", resh)

        if cv2.waitKey(40) == 27:
            break
        if(arr_dataset.shape[0] > 40000):
            break

        # --view------------------------

import pandas as pd 
pd.DataFrame(arr_dataset).to_csv("D:/boku no projecto/python/ghibli_dataset/file.csv")

cv2.destroyAllWindows()
cap.release()
