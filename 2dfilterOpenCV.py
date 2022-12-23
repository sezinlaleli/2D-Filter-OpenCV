# sezin laleli 191101040

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# noise distribution
Ps = 0.1
Pp = 0.05
Pv = (1 - Ps - Pp)


# utility func - resize
def resizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)


# read image from file
img = cv.imread('original.jpg')
# resize
img = resizeWithAspectRatio(img, width=480)
noisyImg = np.copy(img)

# grab image dimensions. I prefer to call them rows and columns.
(rows, cols) = img.shape[:2]

# produce the noise "map"
# 1-salt 2-pepper 3-clean
pixelTypes = [1, 2, 3]
# salt and pepper probabilities
noiseDist = [Ps, Pp, Pv]
noiseMap = np.random.choice(pixelTypes, (rows, cols), True, noiseDist)

noisyImg[np.where(noiseMap == 1)] = 255  # salt
noisyImg[np.where(noiseMap == 2)] = 0  # pepper

# BORDER_REPLICATE padding
imageborder = cv.copyMakeBorder(noisyImg, 10, 10, 10, 10, cv.BORDER_REPLICATE, None, value=0)


def med_filter(image):
    final_image = []
    final_image = np.zeros((len(image), len(image[0]), 3))
    temp = []

    for i in range(rows):
        for j in range(cols):
            for k in range((i - 2), (i + 3)):
                for l in range((j - 2), (j + 3)):
                    temp.append(imageborder[k + 10][l + 10][0])
            temp.sort()
            for m in range(3):
                final_image[i][j][m] = temp[12]
            temp = []

    final_image = np.round_(final_image)
    final_image = final_image.astype(np.uint8)
    return final_image


def weighted_med_filter(image):
    final_image = []
    final_image = np.zeros((len(image), len(image[0]), 3))
    temp = []

    for i in range(rows):
        for j in range(cols):
            for k in range((i - 2), (i + 3)):
                for l in range((j - 2), (j + 3)):
                    temp.append(imageborder[k + 10][l + 10][0])
            temp.append(imageborder[i + 10][j + 10][0])
            temp.append(imageborder[i + 10][j + 10][0])
            temp.sort()
            for m in range(3):
                final_image[i][j][m] = temp[13]
            temp = []

    final_image = np.round_(final_image)
    final_image = final_image.astype(np.uint8)
    return final_image


# my median filter
myMedianOutput = med_filter(noisyImg)
window_name = 'My Median Filter Output'
cv.imshow(window_name, myMedianOutput)

# my weighted median filter
myWeightedMedianOutput = weighted_med_filter(noisyImg)
window_name = 'My Weighted Median Filter Output'
cv.imshow(window_name, myWeightedMedianOutput)

# Opencv .blur() API - NORMALIZED box filter
boxOutput = cv.blur(noisyImg, (5, 5))

# Opencv .GaussianBlur() API - Gaussian filter
gaussOutput = cv.GaussianBlur(noisyImg, (7, 7), 0)

# Opencv .medianBlur() API - median filter
medianOutput = cv.medianBlur(noisyImg, 5)

window_name = 'Open cv Median Filter Output'
cv.imshow(window_name, medianOutput)

# uncomment the following for the homework
cv.imshow('Diff Image', resizeWithAspectRatio(myMedianOutput - medianOutput))

psnr = cv.PSNR(img, noisyImg)
psnrBOX = cv.PSNR(img, boxOutput)
psnrGAUSS = cv.PSNR(img, gaussOutput)
psnrMEDIAN = cv.PSNR(img, medianOutput)
psnrMyMedian = cv.PSNR(img, myMedianOutput)
psnrMyWeightedMedian = cv.PSNR(img, myWeightedMedianOutput)

print('1. Your own median filter output: ' + str(psnrMyMedian))
print('2. OpenCV’s box filter output: ' + str(psnrBOX))
print('3. OpenCV’s Gaussian filter output: ' + str(psnrGAUSS))
print('4. OpenCV’s median filter output: ' + str(psnrMEDIAN))
print('5. Your own weighted median filter output: ' + str(psnrMyWeightedMedian))

plt.subplot(231), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(noisyImg), plt.title('Noisy PSNR: ' + str(psnr))
plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(boxOutput), plt.title('Box PSNR: ' + str(psnrBOX))
plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(gaussOutput), plt.title('Gaussian PSNR: ' + str(psnrGAUSS))
plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(medianOutput), plt.title('Median PSNR: ' + str(psnrMEDIAN))
plt.xticks([]), plt.yticks([])

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
