import os

import cv2
import numpy as np

# #####################  IMX385 RAW -> sRGB ############################################
# def demosicing(raw):
#     raw = np.expand_dims(raw, axis=2)
#     h, w = raw.shape[:2]
#     bgrg = np.concatenate((raw[0:h:2, 0:w:2, :],
#                            raw[0:h:2, 1:w:2, :],
#                            raw[1:h:2, 1:w:2, :],
#                            raw[1:h:2, 0:w:2, :]), axis=2)
#     srgb = cv2.cvtColor(bgrg[:, :, :3], cv2.COLOR_BGR2RGB)
#     return srgb
#
#
# if __name__ == '__main__':
#
#     input = '/home/dabingreat666/data/20210708test/imx385/2021-07-08_14_44_06Z_raw'
#     output = '/home/dabingreat666/data/20210708test/imx385/2021-07-08_14_44_06Z_rgb'
#     assert os.path.isdir(input), 'input path must be a dir!'
#     for i, img in enumerate(os.listdir(input)):
#         if img.endswith('.PNG'):
#             raw_path = os.path.join(input, img)
#             srgb_path = os.path.join(output, img.split('.')[0] + '.jpg')
#             raw = cv2.imread(raw_path, 0)
#             srgb = demosicing(raw)
#             print(f'has processed {i} raw to jpg...')
#             cv2.imwrite(srgb_path, srgb * 8)
#         else:
#             continue
#     print('total finish!')


if __name__ == '__main__':
    l = 15
    input = '/home/dabingreat666/darkfarm/s2/sRGB/51200/l3/DSC02908.JPG'
    output = f'/home/dabingreat666/data/20210719_sony_a7s3_1/pro/51200_3_x{l}.JPG'
    img = cv2.imread(input)
    img = img * l
    cv2.imwrite(output, img)


########################## ############################################
# import rawpy
#
#
# def pack_gbrg_raw(raw):
#     # pack rggb Bayer raw to 4 channels
#     black_level = 240
#     white_level = 2**12-1
#     im = raw.astype(np.float32)
#     im = np.maximum(im - black_level, 0) / (white_level-black_level)
#
#     im = np.expand_dims(im, axis=2)
#     img_shape = im.shape
#     H = img_shape[0]
#     W = img_shape[1]
#
#     out = np.concatenate((im[0:H:2, 0:W:2, :],  # R
#                           im[0:H:2, 1:W:2, :],  # RG
#                           im[1:H:2, 1:W:2, :],  # B
#                           im[1:H:2, 0:W:2, :]),  # GB
#                          axis=2)
#     # srgb = cv2.cvtColor(out[:, :, :3], cv2.COLOR_BGR2RGB)
#     srgb = out[:, :, :3]
#     return srgb, out
#
#
# if __name__ == '__main__':
#
#     path = '/home/dabingreat666/data/20210708test/a7s3/raw/DSC02618.ARW'
#     raw = rawpy.imread(path)
#     rgbg, rgb = pack_gbrg_raw(raw.raw_image)
#     rgbg, rgb = rgb.clip(0, 1.0) * 255.0, rgbg.clip(0, 1.0) * 255.0
#
#     rgb = rgb.astype(np.uint8)
#     rgbg = rgbg.astype(np.uint8)
#     cv2.imwrite('/home/dabingreat666/data/20210708test/a7s3/rgb/DSC02618.jpg', rgb)
#     c = ['R', 'RG', 'B', 'GB']
#     for i in range(rgbg.shape[-1]):
#         img_s = rgbg[:, :, i]
#         cv2.imwrite(f'/home/dabingreat666/data/20210708test/a7s3/rgb/DSC02618_{c[i]}.jpg', img_s)



