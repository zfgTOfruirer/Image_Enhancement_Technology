#zfg 技术2：基于粒子群优化的图像增强

import cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyswarms as ps
import os
from pyswarm import pso
from Auto_improve.end_unfast.faster_detect.Customize_adaptability import fitness_function
import time
start_time = time.time()
def pltshow(img):
    plt.imshow(img,cmap="gray")
    plt.show()

def homomorphic_filter(image, cutoff_freq, gamma_l, gamma_h, c):
    # 1. 图像预处理：将原始图像分解为照射分量和反射分量
    image_log = np.log1p(np.float32(image))
    image_fft = np.fft.fftshift(np.fft.fft2(image_log))
    # 2. 同态滤波，对原始图像的对数进行滤波操作
    H = np.zeros_like(image)
    rows, cols = image.shape
    for u in range(rows):
        for v in range(cols):
            D_uv = np.sqrt((u - rows//2)**2 + (v - cols//2)**2)
            H[u, v] = (gamma_h - gamma_l) * (1 - np.exp(-c * (D_uv**2 / cutoff_freq**2))) + gamma_l

    filtered_fft = H * image_fft
    # 3. 反变换到空域
    filtered_log = np.fft.ifftshift(filtered_fft)
    filtered = np.real(np.fft.ifft2(filtered_log))
    # 4. 取指数得到最终结果
    filtered = np.exp(filtered) - 1
    # 调整灰度范围
    filtered = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered)) * 255
    return filtered.astype(np.uint8)

#白图参数调整到最佳状态，作为整个自适应同态滤波的初始阶段
cutoff_freq = 70
gamma_l = 0.5
gamma_h = 2.5    #2.5   7
c = 70
# cutoff_freq = 70
# gamma_l = 1.62
# gamma_h = 3.27  #2.5   7
# c = 70

img_path =r"E:\Grege_silk\Auto_improve\temp_small\46.png"  #46

img = cv2.imread(img_path,0)
filtered_image = homomorphic_filter(img, cutoff_freq, gamma_l, gamma_h, c)

def homomorphic_filter_auto(image, D_uv_mean, k):
    # 1. 图像预处理：将原始图像分解为照射分量和反射分量
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
    # if diff == 0:
    #     filtered = np.ones_like(filtered) * 255
    # else:
    #     filtered = (filtered - np.min(filtered)) / diff * 255
    filtered = (filtered - np.min(filtered)) / diff * 255
    return filtered.astype(np.uint8)
def compute_D_uv_mean(image):
    rows, cols = image.shape
    D_uv_sum = 0
    for u in range(rows):
        for v in range(cols):
            D_uv = np.sqrt((u - rows//2)**2 + (v - cols//2)**2)
            D_uv_sum += D_uv
    D_uv_mean = D_uv_sum / (rows * cols)
    return D_uv_mean

image = filtered_image

k = 15 #11 #zfg 后续通过粒子群找到最佳的K

D_uv_mean = compute_D_uv_mean(image)
# # 应用同态滤波
# filtered_image = homomorphic_filter_auto(image, D_uv_mean, k)

options = {
    'c1': 0.95,    # 个体学习因子
    'c2': 0.95,    # 社会学习因子
    'w': 0.1     # 惯性权重
}

# 设置搜索范围
constraints = (np.array([13]), np.array([17]))  # 定义搜索空间的下界和上界

# 实例化GlobalBestPSO
optimizer = ps.single.GlobalBestPSO(
    n_particles=4,          # 群体大小
    dimensions=1,           # 问题维数（这里是1维，因为只优化一个参数k）
    options=options,        # 参数设置
    bounds=constraints      # 约束条件
)

# 执行优化
best_cost, best_pos = optimizer.optimize(fitness_function, iters=5, image=image)

# 使用最佳参数
k = best_pos[0]

print("best K:", k)
img2 = homomorphic_filter_auto(image, D_uv_mean, k)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time}秒")
plt.imshow(img2, cmap="gray")
plt.show()



