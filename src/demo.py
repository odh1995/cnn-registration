from __future__ import print_function
import src.Registration
import matplotlib.pyplot as plt
from src.utils.utils import *
import cv2

PATH_1A = '../img/1a.jpg'
PATH_1B = '../img/1b.jpg'
PATH_MISSING ='../img/missingCoffee.jpg'
PATH_COFFEE = '../img/Coffee.png'
PATH_PLANOGRAM = '../img/planogram.png'
PATH_ACTUAL = '../img/actualPic.jpg'
PATH_CURRY_ACTUAL = '../img/curryShelves.jpg'
PATH_CURRY_PLANOGRAM = '../img/Planogram_curry.png'

# designate image path here
IX_path = PATH_CURRY_PLANOGRAM
IY_path = PATH_CURRY_ACTUAL

IX = cv2.imread(IX_path)
IY = cv2.imread(IY_path)

#initialize
reg = src.Registration.CNN()

#register
X, Y, Z = reg.register(IX, IY)
#generate regsitered image using TPS
registered = tps_warp(Y, Z, IY, IX.shape)

# #Show checkboard
# cb = checkboard(IX, registered, 11)
#
# plt.subplot(131)
# plt.title('reference')
# plt.imshow(cv2.cvtColor(IX, cv2.COLOR_BGR2RGB))
# plt.subplot(132)
# plt.title('registered')
# plt.imshow(cv2.cvtColor(registered, cv2.COLOR_BGR2RGB))
# plt.subplot(133)
# plt.title('checkboard')
# plt.imshow(cv2.cvtColor(cb, cv2.COLOR_BGR2RGB))
#
#plt.show()

#Show matches
res = np.zeros(shape=(IX.shape[0] * 2, IX.shape[1], 3), dtype=np.uint8)
res[:IX.shape[0], :, :] = IX
res[IX.shape[0]:, :, :] = registered

for i, pnt in enumerate(X):
	src_x = int(pnt[1])
	src_y = int(pnt[0])
	dst_x = int(Z[i][1])
	dst_y = int(Z[i][0] + IX.shape[0])

	cv2.line(res, (src_x, src_y), (dst_x, dst_y), (255, 0, 0), 2)
	cv2.circle(res, (src_x, src_y), 5, (0, 255, 0), -1)
	cv2.circle(res, (dst_x, dst_y), 5, (0, 255, 0), -1)

cv2.imshow('matching', res)
cv2.imwrite('../output/curry.jpg', res)
cv2.waitKey(0)
