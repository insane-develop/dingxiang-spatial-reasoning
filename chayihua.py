# -*- coding: utf-8 -*-
"""
Created on 2025-06-05 15:44:01
---------
@summary: 
---------
@author: Kevin
@email:  bug_hunter@foxmail.com
"""
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from loguru import logger

# 初始化设备和模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(pretrained=True).to(device)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # 移除分类层
model.eval()  # 评估模式

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_feature_vector(image):
    """从裁剪图像中提取特征向量"""
    tensor = preprocess(image).unsqueeze(0).to(device)  # 添加batch维度
    with torch.no_grad():
        features = model(tensor)  # 输出 [1, 2048, 1, 1]
    return features.flatten().cpu().numpy()  # 展平成2048维向量


def cosine_similarity(vec_a, vec_b):
    """计算余弦相似度"""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)


def find_different_target(bounding_boxes, original_image):
    """
    主函数：找出差异化目标
    参数：
        bounding_boxes: 目标检测返回的边界框列表 [ (x1, y1, x2, y2), ... ]
        original_image: 原始PIL格式验证码图像
    返回：
        different_index: 差异目标的索引（0-3）
    """
    # 1. 裁剪图像区域
    crop_images = []
    for box in bounding_boxes:
        crop = original_image.crop(box)  # (left, upper, right, lower)
        crop_images.append(crop)

    # 2. 提取特征向量
    features = [get_feature_vector(img) for img in crop_images]

    # 3. 计算每个目标的平均相似度
    avg_similarities = []
    for i, vec_i in enumerate(features):
        similarities = []
        for j, vec_j in enumerate(features):
            if i != j:
                sim = cosine_similarity(vec_i, vec_j)
                similarities.append(sim)
        avg_sim = np.mean(similarities)
        avg_similarities.append(avg_sim)
        logger.info(f"目标 {i} 的平均相似度: {avg_sim:.4f}")

    # 4. 找到相似度最低的目标
    different_index = np.argmin(avg_similarities)
    logger.info(f"差异化目标: {different_index}")
    return different_index


# 示例用法
if __name__ == "__main__":
    import cv2

    img_path = '20250605154442.jpg'
    # 假设有四个边界框坐标 (格式 x1,y1,x2,y2)
    # data = [(0.238786 * 379, 0.309091 * 165, 0.124011 * 379, 0.242424 * 165),
    #         (0.424802 * 379, 0.478788 * 165, 0.158311 * 379, 0.303030 * 165),
    #         (0.583113 * 379, 0.745455 * 165, 0.126649 * 379, 0.242424 * 165),
    #         (0.862797 * 379, 0.575758 * 165, 0.126649 * 379, 0.242424 * 165)]
    # print(data)
    image = cv2.imread(img_path)
    print(image.shape[:2])
    # 边界框列表，格式为(x1, y1, x2, y2)
    bounding_boxes = [(67, 31, 114, 71),
                      (137, 54, 191, 103),
                      (197, 103, 245, 143),
                      (302, 75, 351, 115)]
    # 设置框的颜色（BGR格式）和线宽
    color = (0, 255, 0)  # 绿色
    thickness = 2
    # 遍历每个边界框并绘制
    # for box in bounding_boxes:
    #     x1, y1, x2, y2 = box
    #     cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    #
    # # 显示结果
    # cv2.imshow('Bounding Boxes', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    original_image = Image.open("20250605154442.jpg").convert('RGB')  # 加载验证码
    # #
    # # # 找出不同目标
    result = find_different_target(bounding_boxes, original_image)
    x1,y1,x2,y2 = bounding_boxes[result]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.imshow('Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()