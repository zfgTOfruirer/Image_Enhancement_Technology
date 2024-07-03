#zfg 限制对比度的自适应直方图均衡
import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from scipy.signal import savgol_filter

def gamma_correction(image, gamma):
    """
    伽马变换函数，将输入图像进行伽马变换
    :param image: 输入图像
    :param gamma: 伽马值
    :return: 变换后的图像
    """
    gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(image, gamma_table)

def contrast_stretching_adaptive_gamma_correction(image, clipLimit=7.2,grid_size=(8, 8)):
    """
    对比度拉伸自适应伽马变换
    :param image: 输入图像
    :param clip_limit: 对比度限制因子，控制对比度拉伸的程度
    :param grid_size: 网格大小，用于划分图像
    :return: 变换后的图像
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=grid_size)
    hist_eq = clahe.apply(gray)
    # 计算对比度拉伸参数
    p_min = np.percentile(hist_eq, 1)
    p_max = np.percentile(hist_eq, 99)
    epsilon = 1e-8  # 添加一个很小的常数，避免零除错误
    gamma = np.log10(0.5) / np.log10((p_max - p_min) / 255.0 + epsilon)
    # 应用伽马变换
    gamma_corrected = gamma_correction(gray, gamma)
    # 融合增强后的图像和直方图均衡化图像
    enhanced = cv2.addWeighted(gamma_corrected, 1.8, hist_eq, -0.3, 0)
    return enhanced


if __name__ == '__main__':
    image = cv2.imread('path_to_your_image.jpg')
    result_img = contrast_stretching_adaptive_gamma_correction(image)
    cv2.imshow('Result', result_img)
    cv2.destroyAllWindows()
    cv2.imwrite('path_to_save_result.jpg', result_img)








