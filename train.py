# -*- coding: utf-8 -*-
"""
Created on 2025-06-04 10:04:43
---------
@summary: 
---------
@author: Kevin
@email:  bug_hunter@foxmail.com
"""
import os
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11m.pt", task='detect')
    # 训练
    train_results = model.train(
        data=r"G:\workspace\captcha\dingxiang-spatial-reasoning\datasets\cooc.yaml",
        # Path to dataset configuration file
        epochs=200,
        cache=True,
        imgsz=320,  # 增加图像大小
        batch=4,  # 增加批次大小
        workers=4,  # 增加工作线程数
        device='0',  # 或者 '0,1' 根据GPU情况
        save=True,
        lr0=0.01,  # 初始学习率
        hsv_h=0.015,  # 数据增强
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.5,
        translate=0.1,
        scale=0.5,
        shear=0.5,
        project=os.path.join(os.getcwd(), 'runs')
    )
    model.export(format='onnx', imgsz=320, simplify=True)
