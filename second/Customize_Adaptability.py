#zfg 适应度函数

# 自定义适应度函数的结合
# 图像质量评价方法即为适应度函数
# “灰度峰值”        [1,14]  小
# “灰度分布区间”     [0,50]  小
# "灰度标准方差”     [1,10]   小
# "信息熵E"         [1000,6000]    小
# "均方误差"        [110,120]   大
# "信噪改变量"       [0,50]  小（重叠）
# "紧致度"          [7*e8,3.5*e9]  大
# "特定对比度"       [2*e5,1.5*e6]   小
# "原始对比度"     [0.0005,3*e-5]   大
# ”边缘强度之和、边缘点数和图像熵值“   [2*e4,1.5*e5]  小
# "F3"  [0,4]
'''

论文中的结合方式：
细菌觅食优化算法研究及其在图像增强中的应用
f_{Fimess}=E\times I_{nc}\times\left[F_{ac}+2.5\times C\right]+F_{br}

基于改进量子粒子群的红外图像增强算法
f_{\mathrm{itoess}}=\gamma_{1}H_{1}+\gamma_{2}H_{2}+\gamma_{3}H_{3}+\gamma_{4}H_{4}+\gamma_{5}H_{5}

'''

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math

epsilon = 1e-20


# 计算图像对比度
def contrast_0(image, contrast_threshold):
    mean_intensity = np.mean(image)
    contrast_map = np.abs(image - mean_intensity)
    defect_area = np.sum(contrast_map > contrast_threshold)
    total_contrast = np.sum(contrast_map)
    fitness = defect_area + total_contrast
    return abs(fitness)


def calculate_H1(image):
    # 使用Sobel边缘检测算法
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # 计算边缘像素数量
    h1 = np.sum(sobel_magnitude) / (image.shape[0] * image.shape[1])
    return h1


def calculate_H3(image, enhanced_image):
    signal_mean = np.mean(enhanced_image)
    noise_stddev = np.std(image - enhanced_image)
    h3 = 10 * math.log10((signal_mean ** 2) / (epsilon + (noise_stddev ** 2)))
    return h3


def calculate_H5(image, enhanced_image):
    h5 = np.mean((image - enhanced_image) ** 2)
    return h5


def calculate_F1(image):
    M, N = image.shape
    mean_intensity = np.mean(image)
    f_squared_mean = np.mean(image ** 2)
    F1 = f_squared_mean - mean_intensity ** 2
    return F1 + 10e10


def calculate_F2(image):
    M, N = image.shape
    # 使用Sobel算子计算图像的边缘强度
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # 计算边缘像素数量
    edge_pixel_count = np.sum(edge_strength > 0)
    # 计算图像的熵
    image_normalized = (image / 255)  # 归一化图像像素值到 [0, 1] 范围
    entropy = -np.sum(image_normalized * np.log2(image_normalized + 1e-10))
    epsilon = 1e-10  # 小的正数用于避免除零错误
    F2 = np.log(np.log(np.sum(edge_strength) + epsilon)) * ((edge_pixel_count / (epsilon + (M * N))) * entropy)
    return F2


def calculate_edge_pixel_count(image):
    # 使用Sobel算子计算图像的边缘强度
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # 计算边缘像素数量
    edge_pixel_count = np.sum(edge_strength > 0)
    return edge_pixel_count


def calculate_edge_strength(image):
    # 使用Sobel算子计算图像的边缘强度
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return np.sum(edge_strength)


def Compactness(image):
    perimeter = np.sum(np.abs(image[:-1, :] - image[1:, :])) + np.sum(np.abs(image[:, :-1] - image[:, 1:]))
    area = np.sum(image)
    compactness = (perimeter ** 2) / area  # 紧致度
    return compactness


def Gray_peak(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    peak_value = np.argmax(hist)
    return peak_value


def Gray_scal(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    indices = [i for i, value in enumerate(hist) if 4000 < value < 5000]
    if indices:
        b = min(indices)
    else:
        # 处理空序列的情况，例如返回一个默认值或抛出一个更具体的错误
        b = 0  # 例如 0 或其他适当的值
    return b


def compute_D_uv_mean(image):
    rows, cols = image.shape
    D_uv_sum = 0
    for u in range(rows):
        for v in range(cols):
            D_uv = np.sqrt((u - rows // 2) ** 2 + (v - cols // 2) ** 2)
            D_uv_sum += D_uv
    D_uv_mean = D_uv_sum / (rows * cols)
    return D_uv_mean


# zfg 搜索1
def homomorphic_filter_auto(image, k):
    # 1. 图像预处理：将原始图像分解为照射分量和反射分量
    D_uv_mean = compute_D_uv_mean(image)
    image_log = np.log1p(np.float32(image))
    image_fft = np.fft.fftshift(np.fft.fft2(image_log))
    # 2. 同态滤波，对原始图像的对数进行滤波操作
    H = np.zeros_like(image)
    rows, cols = image.shape
    for u in range(rows):
        for v in range(cols):
            D_uv = np.sqrt((u - rows//2)**2 + (v - cols//2)**2)
            #单参数传递函数
            H[u, v] = 0.1*k/ (1 + np.exp(-k * (D_uv_mean)/((D_uv + 1e-6))))  # 添加一个小的修正项
    filtered_fft = H * image_fft
    # 3. 反变换到空域
    filtered_log = np.fft.ifftshift(filtered_fft)
    filtered = np.real(np.fft.ifft2(filtered_log))
    # 4. 取指数得到最终结果
    filtered = np.exp(filtered) - 1
    # 调整灰度范围，如果极差为零则处理为全白
    diff = np.max(filtered) - np.min(filtered)

    filtered = (filtered - np.min(filtered)) / diff * 255
    return filtered.astype(np.uint8)


# zfg 搜索1适应度函数

epsilon = 1e-20



# 裁剪512*512

# zfg 搜索2适应度函数

def fitness_function(paras, image):
    M, N = image.shape
    costs = np.zeros(paras.shape[0])
    for i in range(paras.shape[0]):
        k = paras[i][0]  # 从数组中提取单个数值k
        enhanced_image = homomorphic_filter_auto(image, k)

        # MSE = calculate_H5(image, enhanced_image)
        Fi = Compactness(enhanced_image)
        H = calculate_F1(enhanced_image)
        S = calculate_edge_strength(enhanced_image)
        E = calculate_edge_pixel_count(enhanced_image)
        Sarea = (M * N) ** 2
        Gp = Gray_peak(enhanced_image)
        Gs = Gray_scal(enhanced_image)
        Gstd = np.std(enhanced_image)
        # E = np.sum((enhanced_image / 255) * np.log2((enhanced_image / 255) + 1e-10))
        # Sc = np.sum(enhanced_image > 128)
        C1 = contrast_0(enhanced_image, 110)
        # F2 = calculate_F2(enhanced_image)
        # F3 = calculate_F3(image)

        # 计算适应度分数
        # result=(MSE*F1*C2)/(Sarea*Gp*Gs*Gstd*E*Sc*C1*F2*F3)

        # result = (MSE * C2) / (Sarea * Gp * Gs * Gstd * Sc * C1 * F3 * Fi + 1e-10)  # F3,F2,F1三者之间存在一定的包含关系

        result = (Gp*Gstd*H * S * abs(E) * Gs)/ Sarea *(C1 *Fi )  # zfg 1.3改进版本---论文

        # result = F3  # 或其他适应度计算方法
        costs[i] = result  #

    return costs

#           #高斯传递函数
# H[u, v] = (gamma_h - gamma_l) * (1 - np.exp(-c * (D_uv**2 / cutoff_freq**2))) + gamma_l
# 巴特沃斯传递含糊
# H[u, v] = (gamma_h - gamma_l) * (1 / (c * (cutoff_freq ** 2 / (D_uv ** 2 + + 1e-6)))) + gamma_l
# 指数型传递函数
# H[u, v] = (gamma_h - gamma_l) * np.exp(-c * (cutoff_freq ** 2 / D_uv ** 2)) + gamma_l

# 同态频分论文里的高通滤波器
# H[u, v] = 1 / (1 + (D_uv ** (-k))) #效果还行

# 传递函数叠加   效果不行
# H[u, v] = (1 / (1 + np.exp(-k * (D_uv_mean) / ((D_uv + 1e-6))))   +    1 / (1 + (D_uv ** (-k))))*(1/2)
