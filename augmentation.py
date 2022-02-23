import cv2
import numpy as np
import os
import random
from PIL import ImageEnhance, Image
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

"""
数据增强，所有函数输入为0-255，输出为0-255，浮点数或者int
"""

"""
旋转，rotate mask
平移：move mask
随机裁剪：crop mask
翻转：flip mask
弹性形变：xingbian mask
中心缩放：zoom mask
边缘补0：padding mask

亮度：brihgt
锐度：sharp
对比度：contrast
gamma变换：gamma_transform

滤波：filter
分辨率：resolution

椒盐噪声：sp_noise
高斯噪声：gususs_noise
噪声：add_noise
"""

def rotate(mats, masks=None): # (n, h, w) (0-1) ndarry
    """
    旋转
    :param mat:
    :param mask:
    :return:
    """
    new_mats = []
    rote = random.choice(range(-90, 90))
    rows, cols = mats.shape[1], mats.shape[2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rote, 1)
    for i in range(mats.shape[0]):
        mat = mats[i, :, :]
        mat = cv2.warpAffine(mat, M, (cols, rows))
        new_mats.append(mat)
    new_mats = np.array(new_mats)
    if masks is None:
        return new_mats
    else:
        new_masks = []
        for i in range(masks.shape[0]):
            mask = masks[i, :, :]
            mask = cv2.warpAffine(mask, M, (cols, rows), flags=cv2.INTER_NEAREST)
            new_masks.append(mask)
        new_masks = np.array(new_masks)
        return new_mats, new_masks

def bright(images):
    """
    亮度
    :param image:
    :return:
    """
    images = images * 255
    images = images.astype(np.uint8)
    brightness = random.choice(range(8, 20)) / 10
    new_images = []
    for i in range(images.shape[0]):
        image = images[i, :, :]
        image = Image.fromarray(image)
        enh_bri = ImageEnhance.Brightness(image)
        image_brightened = enh_bri.enhance(brightness)
        img = np.array(image_brightened)
        new_images.append(img)
    new_images = np.array(new_images)
    return new_images / 255

def sharp(images):
    """
    锐度
    :param image:
    :return:
    """
    images = images * 255
    images = images.astype(np.uint8)
    new_images = []
    sharpness = random.choice(range(1, 5))
    for i in range(images.shape[0]):
        image = images[i, :, :]
        image = Image.fromarray(image)
        enh_sha = ImageEnhance.Sharpness(image)
        image_sharped = enh_sha.enhance(sharpness)
        img = np.array(image_sharped)
        new_images.append(img)
    new_images = np.array(new_images)
    return new_images / 255

def contrast(images):
    """
    对比度
    :param image:
    :return:
    """
    images = images * 255
    images = images.astype(np.uint8)
    new_images = []
    contrast = random.choice(range(7, 20)) / 10
    for i in range(images.shape[0]):
        image = images[i, :, :]
        image = Image.fromarray(image)
        enh_con = ImageEnhance.Contrast(image)
        image_contrasted = enh_con.enhance(contrast)
        img = np.array(image_contrasted)
        new_images.append(img)
    new_images = np.array(new_images)
    return new_images / 255

def filt(images):
    """
    滤波
    :param image:
    :return:
    """
    new_images = []
    type = random.choice(["gasuss", "mid", "avg"])
    for i in range(images.shape[0]):
        image = images[i, :, :]
        image = image.astype(np.uint8)
        ksize = random.choice([1, 3, 5])
        if type == "gasuss":
            image = cv2.GaussianBlur(image, (ksize, ksize), 3)
        elif type == "mid":
            image = cv2.medianBlur(image, ksize)
        else:
            image = cv2.blur(image, (ksize, ksize))
        new_images.append(image)
    new_images = np.array(new_images)
    return new_images

def move(mats, masks=None):
    """
    平移
    :param mat:
    :param mask:
    :return:
    """
    x = random.choice(range(-50, 50))
    y = random.choice(range(-50, 50))
    new_mats = []
    for i in range(mats.shape[0]):
        mat = mats[i, :, :]
        mat = cv2.warpAffine(mat, np.float32([[1, 0, x], [0, 1, -y]]), (mat.shape[1], mat.shape[0]))
        new_mats.append(mat)
    new_mats = np.array(new_mats)
    if masks is None:
        return new_mats
    else:
        new_masks = []
        for i in range(masks.shape[0]):
            mask = masks[i, :, :]
            mask = cv2.warpAffine(mask, np.float32([[1, 0, x], [0, 1, -y]]), (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
            new_masks.append(mask)
        new_masks = np.array(new_masks)
        return new_mats, new_masks

def gamma_transform(images):
    """
    gamma变换
    :param image:
    :param gamma:
    :return:
    """
    images = images * 255
    images = images.astype(np.uint8)
    gamma = random.choice(range(5, 15))
    gamma = gamma / 10
    new_images = []
    for i in range(images.shape[0]):
        image = images[i, :, :]
        max_value = np.max(image)
        min_value = np.min(image)
        value_l = max_value - min_value
        if value_l != 0:
            image = (image - min_value) / value_l
        else:
            image = (image - min_value) / (value_l + 1e-6)
        image = np.power(image, gamma)
        image = image * 255
        new_images.append(image)
    new_images = np.array(new_images)
    return new_images / 255

def crop(mats, masks=None):
    """
    随机裁剪
    :param mat:
    :param mask:
    :return:
    """
    new_mats = []
    min_ratio = 0.8
    ratio = random.random()
    h, w = mats[0, :, :].shape
    scale = min_ratio + ratio * (1.0 - min_ratio)
    new_h = int(h * scale)
    new_w = int(w * scale)
    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)
    for i in range(mats.shape[0]):
        mat = mats[i, :, :]
        mat = mat[y:y + new_h, x:x + new_w]
        mat = cv2.resize(mat, (w, h))
        new_mats.append(mat)
    new_mats = np.array(new_mats)
    if masks is None:
        return new_mats
    else:
        new_masks = []
        for i in range(masks.shape[0]):
            mask = masks[i, :, :]
            mask = mask[y:y+new_h, x:x+new_w]
            mask = cv2.resize(mask, (w, h),interpolation=cv2.INTER_NEAREST)
            new_masks.append(mask)
        new_masks = np.array(new_masks)
        return new_mats, new_masks

def resolution(mats):
    tall, wide = mats[0, :, :].shape[0], mats[0, :, :].shape[1]
    scale = random.randint(4, 20)
    scale = scale / 10
    w, h = int(wide*scale), int(tall*scale)
    new_mats = []
    for i in range(mats.shape[0]):
        mat = mats[i, :, :]
        mat = cv2.resize(mat, (w, h))
        mat = cv2.resize(mat, (wide, tall))
        new_mats.append(mat)
    new_mats = np.array(new_mats)
    return new_mats

def flip(imgs, masks=None):
    """
    翻转
    :param img:
    :return:
    """
    f3 = random.choice([-1, 0, 1, 2])
    new_imgs = []
    for i in range(imgs.shape[0]):
        img = imgs[i, :, :]
        mat = cv2.flip(img, f3)
        new_imgs.append(mat)
    new_imgs = np.array(new_imgs)
    if masks is None:
        return new_imgs
    else:
        new_masks = []
        for i in range(masks.shape[0]):
            mask = masks[i, :, :]
            mask = cv2.flip(mask, f3)
            new_masks.append(mask)
        new_masks = np.array(new_masks)
        return new_imgs, new_masks

def sp_noise(images):
    """
    椒盐噪声
    :param image:
    :return:
    """
    images = images * 255
    images = images.astype(np.uint8)
    n = random.choice(range(20))
    prob = n / 1000
    thres = 1 - prob

    new_images = []
    for k in range(images.shape[0]):
        image = images[k, :, :]
        output = np.zeros(image.shape,np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        new_images.append(output)
    new_images = np.array(new_images)
    return new_images / 255

def gasuss_noise(images):
    """
    高斯噪声
    :param image:
    :return:
    """
    images = images * 255
    images = images.astype(np.uint8)
    mean = 0
    n = random.choice(range(5))
    var = n / 50000
    new_images = []
    for k in range(images.shape[0]):
        image = images[k, :, :]
        image = np.array(image, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        new_images.append(out * 255)
    new_images = np.array(new_images)
    return new_images / 255

def add_noise(images):
    """
    添加噪声
    :param image:
    :return:
    """
    images = images * 255
    images = images.astype(np.uint8)
    new_images = []
    for k in range(images.shape[0]):
        image = images[k, :, :]
        threshold = random.choice(range(32))
        noise = np.random.uniform(low=-1, high=1, size=image.shape)
        image = image + noise * threshold
        image = np.clip(image, 0, 255)
        new_images.append(image)
    new_images = np.array(new_images)
    return new_images / 255



"""
数据增强，所有函数输入为0-255，输出为0-255，浮点数或者int
"""

"""
旋转，rotate mask
平移：move mask
随机裁剪：crop mask
翻转：flip mask

亮度：brihgt
锐度：sharp
对比度：contrast
gamma变换：gamma_transform

滤波：filter
分辨率：resolution
"""
def augment(imgs, masks=None, choices=[True]): # imgs(0, 1) mask[0, 1]
    n = random.choice(choices)
    if n:
        funcs1 = [contrast, gamma_transform, bright, sharp]
        func1 = random.choice(funcs1)
        funcs2 = [rotate, flip]
        func2 = random.choice(funcs2)
        funcs3 = [move, crop]
        func3 = random.choice(funcs3)
        funcs4 = [filt, resolution]
        func4 = random.choice(funcs4)
        funcs5 = [sp_noise, gasuss_noise, add_noise]
        func5 = random.choice(funcs5)
        if masks is None:
            for func in [func1, func2, func3, func4]:
                boo = random.choice([True, False])
                if boo:
                    imgs = func(imgs)
            return imgs
        else:
            for func in [func1, func4]:
                boo = random.choice([True, False])
                if boo:
                    imgs = func(imgs)
            for func in [func2, func3]:
                boo = random.choice([True, False])
                if boo:
                    imgs, masks = func(imgs, masks)
            return imgs, masks
    else:
        if masks is None:
            return imgs
        else:
            return imgs, masks
