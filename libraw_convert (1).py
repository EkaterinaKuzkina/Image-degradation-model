import libraw              # for loading
import rawpy
import numpy as np         # for processing
import cv2
import math




proc = libraw.LibRaw()     # create RAW processor
proc.open_file("IMG_9542.cr2") # open file
proc.unpack()              # extract mosaic from file
proc_rawpy = rawpy.imread("IMG_9542.cr2")
mosaic = cv2.imread('C:/Users/kWX597368/Desktop/low.tiff',-1)

#mosaic = np.resize(mos,(3464, 5202))

## Listing 1: Mapping to Linear Values
# black = proc.imgdata.color.black
# saturation = proc.imgdata.color.maximum
# black = 2048
# saturation = 12279
#
mosaic= np.float64(mosaic) #avoid the noise in case of overflow
# mosaic -= black                     # black subtraction
# uint14_max = 2**16 - 1
# mosaic *= int(uint14_max/(saturation - black))
# mosaic = np.clip(mosaic,0,uint14_max)  # clip to range
print('>> Mapping to Linear Values')

## Listing 2: White Balancing
assert(proc.imgdata.idata.cdesc == b"RGBG")


print('>> White Balancing')

# Adding Poison and Gauss noise to image
# a = 2
# b = 0.9
# chi = 1/a
# norm = np.random.randn(mosaic.shape[0], mosaic.shape[1])
# max_r = np.amax(mosaic[0::2, 0::2])
# max_b = np.amax(mosaic[1::2, 1::2])
# max_g1 = np.amax(mosaic[0::2, 1::2])
# max_g2 = np.amax(mosaic[1::2, 0::2])
#
# mosaic[0::2, 0::2] /= max_r
# mosaic[1::2, 1::2] /= max_b
# mosaic[0::2, 1::2] /= max_g1
# mosaic[1::2, 0::2] /= max_g2
#
# mosaic[0::2, 0::2] += np.sqrt(a*mosaic[0::2, 0::2] +b)*norm[0::2, 0::2]
# mosaic[1::2, 1::2] += np.sqrt(a*mosaic[1::2, 1::2] +b)*norm[1::2, 1::2]
# mosaic[0::2, 1::2] += np.sqrt(a*mosaic[0::2, 1::2] +b)*norm[0::2, 1::2]
# mosaic[1::2, 0::2] += np.sqrt(a*mosaic[1::2, 0::2] +b)*norm[1::2, 0::2]
#
# mosaic[0::2, 0::2] *= max_r
# mosaic[1::2, 1::2] *= max_b
# mosaic[0::2, 1::2] *= max_g1
# mosaic[1::2, 0::2] *= max_g2
# mosaic = np.clip(mosaic,0,255)# clip to range

# cam_mul = proc.imgdata.color.cam_mul # RGB multipliers
# cam_mul /= cam_mul[1]                # scale green to 1
# # mosaic[0::2, 0::2] *= cam_mul[0]     # scale reds
# # mosaic[1::2, 1::2] *= cam_mul[2]     # scale blues
# mosaic[0::2, 1::2] *= 2.077148   # scale reds
# mosaic[1::2, 0::2] *= 1.563477

## Listing 3: Demosaicing
mosaic = mosaic.astype(np.uint16)
img = cv2.cvtColor(mosaic, cv2.COLOR_BAYER_GB2BGR)
print('>> Demosaicing')

# z = clahe_contrast(img)
z = img
z1 = cv2.normalize(z, None, 0, 255, cv2.NORM_MINMAX)
z1 = z1.astype(np.uint8)
cv2.imwrite('low_nothing.tiff', z1)

## Listing 4: Color Space Conversion
cam2srgb = proc.imgdata.color.rgb_cam[:, 0:3]
cam2srgb = np.round(cam2srgb*255).astype(np.int16)
img = img // 2**8        # reduce dynamic range to 8bpp
shape = img.shape
pixels = img.reshape(-1, 3).T     # 3xN array of pixels
pixels = cam2srgb.dot(pixels)//255
img = pixels.T.reshape(shape)
img = np.clip(img, 0, 255).astype(np.uint16)
print('>> Color Space Conversion')

## Listing 5: Gamma Correction
gcurve = [(i / 255) ** (1 / 2.2) * 255 for i in range(256)]
gcurve = np.array(gcurve * 255, dtype=np.uint8)
img = gcurve[img]  # apply gamma LUT
print('>> Gamma Correction')

## show info and save output
import matplotlib.image
matplotlib.image.imsave("low_1.png", img)


