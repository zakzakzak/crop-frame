import cv2
import numpy as np
from numpy import genfromtxt

my_data = genfromtxt('D:/boku no projecto/python/ghibli_dataset/file.csv', delimiter=',')

print(my_data[1:,1:].shape)
arr_dataset = my_data[1:,1:]

for i in arr_dataset:
	print(i)

	i = np.array(i, dtype = np.uint8)

	resh = i.reshape((68,68))
	cv2.imshow("feed", resh)

	if cv2.waitKey(40) == 27:
		break

cv2.destroyAllWindows()
cap.release()
