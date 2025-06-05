# -*- coding: utf-8 -*-
"""
Created on 2025-06-04 17:19:35
---------
@summary:
---------
@author: Kevin
@email:  bug_hunter@foxmail.com
"""
import os
import math
import cv2
import numpy as np
import logging

os.environ['LOGLEVEL'] = 'ERROR'

from ultralytics import YOLO
from paddleocr import PaddleOCR
from loguru import logger
from PIL import Image

logging.getLogger('ppocr').setLevel(logging.WARNING)


def get_dominant_color(img):
    """判断图像中红、绿、蓝哪个颜色最多"""
    # img = cv2.imread(image_path)  # 读取图片（BGR格式）
    if img is None:
        return None

    # 计算每个通道的像素总和
    blue_sum = np.sum(img[:, :, 0])  # B通道
    green_sum = np.sum(img[:, :, 1])  # G通道
    red_sum = np.sum(img[:, :, 2])  # R通道

    # 找到最大值对应的颜色
    color_dict = {"红色": red_sum, "绿色": green_sum, "蓝色": blue_sum}
    dominant_color = max(color_dict, key=color_dict.get)

    return dominant_color


def extract_text_from_image(image) -> str:
    result = ocr.predict(
        input='target.jpg')
    # result = ocr.predict(
    #     input=image)
    # 可视化结果并保存 json 结果
    valid_texts = []
    for res in result:
        valid_texts.extend(res['rec_texts'])
        # res.save_to_img("output")
        # res.save_to_json("output")
    if not valid_texts:
        return ''
    return ''.join(valid_texts).replace('-', '').replace('学', '字').replace('育', '请').replace('肩', '请').replace(
        '−', '')


def use_yolo_onnx(image):
    result = model.predict(
        source=image,
        device='cpu',
        show=False, save=False, imgsz=320
        , project=os.path.join(os.getcwd(), 'runs'), verbose=False)

    # 提取第一个结果（假设单张图片输入）
    detections = result[0]
    ret = []
    # 打印所有检测框的坐标和类别
    for i, box in enumerate(detections.boxes):
        # 获取坐标（xyxy格式：左上x,左上y,右下x,右下y）
        coordinates = [int(i) for i in box.xyxy[0].tolist()]
        # 获取置信度
        confidence = box.conf[0].item()
        if confidence <= 0.85:
            continue
        # 获取类别ID
        class_id = box.cls[0].item()
        # 获取类别名称
        class_name = detections.names[int(class_id)]
        if class_name == '正方体':
            class_name = '立方体'
        x1, y1, x2, y2 = coordinates
        cropped_img = image[y1:y2, x1:x2]
        color = get_dominant_color(cropped_img)
        centre_x = (x1 + x2) / 2
        centre_y = (y1 + y2) / 2
        ret.append([centre_x, centre_y, class_name, color])
    return ret


# info=(centre_x,centre_y,model.names[class_id],color)
def choose_model(text, infos):
    res_point = (0, 0)
    if "大写" in text:
        for info in infos:
            if info[2] == "大写字母":
                res_point = (info[0], info[1])
                break
    elif "小写" in text:
        for info in infos:
            if info[2] == "小写字母":
                res_point = (info[0], info[1])
                break
    elif "请点击平面图形" in text:
        for info in infos:
            if info[2].endswith("形"):
                res_point = (info[0], info[1])
                break
    elif "请点击立体图形" in text:
        for info in infos:
            if info[2].endswith("体"):
                res_point = (info[0], info[1])
                break
    elif "左侧的" in text:
        index = text.index("左侧的")
        origin = text[3:index]
        target = text[index + 3:]
        if origin == "球":
            origin = "球体"
        logger.info(f"origin: {origin}, target: {target}")
        target_points = []
        for info in infos:
            if origin == info[2] or origin in info[2]:
                origin_point = (info[0], info[1])
            elif "字母" in info[2]:
                target_points.append((info[0], info[1]))
        for target_point in target_points:
            if target_point[0] < origin_point[0]:
                res_point = target_point
                break
    elif "右侧的" in text:
        index = text.index("左侧的")
        origin = text[3:index]
        target = text[index + 3:]
        if origin == "球":
            origin = "球体"
        logger.info(f"origin: {origin}, target: {target}")
        target_points = []
        for info in infos:
            if origin == info[2] or origin in info[2]:
                origin_point = (info[0], info[1])
            elif "字母" in info[2]:
                target_points.append((info[0], info[1]))
        for target_point in target_points:
            if target_point[0] > origin_point[0]:
                res_point = target_point
                break
    elif "与" in text:
        index = text.index("与")
        origin = text[3:index]
        target = text[index + 1:-5]
        logger.info(f"origin: {origin}, target: {target}")
        target_color = ""
        for info in infos:
            if target == info[2] or target in info[2]:
                target_color = info[3]
                break
        for info in infos:
            if info[3] == target_color and target not in info[2]:
                res_point = (info[0], info[1])
                break
    elif '离' in text:
        index = text.index("最近的")
        origin = text[4:index]
        target = text[index + 3:]
        for info in infos:
            if origin == info[2] or origin in info[2]:
                target_point = (info[0], info[1])
                break
        min_dis = 0
        for info in infos:
            if '字母' in info[2]:
                res_dist = find_nearest_letter(info[0], info[1], target_point[0], target_point[1])
                if min_dis == 0 or res_dist < min_dis:
                    min_dis = res_dist
                    res_point = (info[0], info[1])
    elif '色的' in text:
        index = text.index("的")
        target_color = text[3:index]
        target = text[index + 1:]
        for info in infos:
            if info[3] == target_color and target == '立体图形' and info[2].endswith("体"):
                res_point = (info[0], info[1])
                break
            elif info[3] == target_color and target == '平面图形' and info[2].endswith("形"):
                res_point = (info[0], info[1])
                break

    logger.info(f'目标中心点坐标 -> {res_point}')
    return res_point


def find_nearest_letter(ax1, ay1, bx1, by1):
    """
    计算圆柱体(ax1,ay1)与字母(bx1,by1)之间的距离
    参数:
    ax1, ay1 - 圆柱体中心坐标
    bx1, by1 - 字母中心坐标

    返回:
    两点之间的欧几里得距离
    """
    return math.sqrt((bx1 - ax1) ** 2 + (by1 - ay1) ** 2)


def main(img_path):
    image = cv2.imread(img_path)
    # 获取图片尺寸
    height, width = image.shape[:2]
    # 计算要截取的高度（总高度的20%）
    crop_height = int(height * 0.15)
    # 截取顶部部分（从第0行到crop_height行，所有列）
    top_part = image[0:crop_height, :]
    cv2.imwrite('target.jpg', top_part)
    target_text = extract_text_from_image(top_part)
    logger.info(f'文本语义识别 -> {target_text}')
    ret = use_yolo_onnx(image)
    logger.info(f'识别结果 -> {ret}')
    model = choose_model(target_text, ret)
    logger.info(f'目标中心点坐标 -> {model}')
    x, y = model  # 示例坐标
    color = (255, 255, 255)  # 白色(BGR格式)
    radius = 3
    thickness = -1  # -1表示填充圆
    cv2.circle(image, (int(x), int(y)), radius, color, thickness)
    cv2.imshow('', image)
    cv2.waitKey(0)  # 等待任意按键


if __name__ == '__main__':
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        ocr_version='PP-OCRv5',
    )
    model = YOLO(r"best.onnx", task='detect',
                 verbose=False)
    # ret = reg_words('./out.jpg')
    #
    # print({'res':ret})
    base_dir = r'G:\workspace\captcha\dingxiang-spatial-reasoning\captcha_screenshots'
    files = os.listdir(base_dir)
    for file in files:
        img_path = os.path.join(base_dir, file)
        # img_path = os.path.join(base_dir, '60ee0baa5a222ba54fa910061418e933.png')
        # print(img_path)
        main(
            img_path=img_path)
        # break
