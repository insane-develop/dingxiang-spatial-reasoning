import cv2
import numpy as np
from loguru import logger

def rotate_and_paste(background_path, circle_path, angle):
    """
    旋转圆形图片并贴回背景图的中心洞中

    Args:
        background_path: 带洞背景图路径
        circle_path: 独立圆形图路径
        angle: 圆形旋转的角度 (度)

    Returns:
        复原后的图片
    """
    # 读取带透明通道的图片
    background = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)
    circle_img = cv2.imread(circle_path, cv2.IMREAD_UNCHANGED)

    # 确保图片正确读取
    if background is None or circle_img is None:
        raise ValueError("图片读取失败，请检查路径")

    # 为背景图添加透明通道（如果不存在）
    if background.shape[2] == 3:
        background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

    # 获取背景图尺寸和中心点
    bg_height, bg_width = background.shape[:2]
    center_x, center_y = bg_width // 2, bg_height // 2

    # 从圆形图创建掩码（非透明区域）
    _, alpha_mask = cv2.threshold(circle_img[:, :, 3], 1, 255, cv2.THRESH_BINARY)

    # 计算圆形半径（取宽高中较小值的50%）
    circle_radius = min(circle_img.shape[0], circle_img.shape[1]) // 2

    # 创建临时背景用于旋转（包含透明通道）
    circle_on_bg = np.zeros_like(background)
    circle_on_bg = cv2.cvtColor(circle_on_bg, cv2.COLOR_BGRA2RGBA)  # 确保透明通道兼容

    # 获取圆形图中心点
    circle_center_x = circle_img.shape[1] // 2
    circle_center_y = circle_img.shape[0] // 2

    # 计算在背景中心放置圆形图的坐标
    start_x = center_x - circle_center_x
    start_y = center_y - circle_center_y
    end_x = start_x + circle_img.shape[1]
    end_y = start_y + circle_img.shape[0]

    # 在临时背景上放置圆形图
    circle_on_bg[start_y:end_y, start_x:end_x] = circle_img

    # 旋转组合图片（角度为负实现逆时针旋转）
    rotation_matrix = cv2.getRotationMatrix2D(
        (center_x, center_y),
        -angle,
        1.0
    )
    rotated = cv2.warpAffine(
        circle_on_bg,
        rotation_matrix,
        (bg_width, bg_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)  # 透明边框
    )

    # 创建背景图洞区域掩码（中心为透明）
    hole_mask = np.zeros((bg_height, bg_width), dtype=np.uint8)
    cv2.circle(hole_mask, (center_x, center_y), circle_radius, 255, -1)

    # 创建洞区域的倒置掩码（洞区域为透明，其他区域不透明）
    inverted_hole_mask = cv2.bitwise_not(hole_mask)

    # 将背景洞区域设置为透明
    background = cv2.cvtColor(background, cv2.COLOR_BGRA2RGBA)
    background[:, :, 3] = cv2.bitwise_and(
        background[:, :, 3],
        inverted_hole_mask
    )

    # 合并旋转后的圆形图和背景
    result = cv2.add(background, rotated)

    # 转换回BGR格式（去掉透明通道）
    result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)

    return result


def calculate_gradient_magnitude(image):
    """计算图像的梯度强度"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    return gradient_magnitude


if __name__ == '__main__':
    background_path = 'bg.jpg'  # 带洞背景图
    circle_path = 'tg.jpg'  # 独立圆形图
    min_gradient = float('inf')
    best_result = None
    best_angle = 0

    # 测试0-360度，步长为5度
    for angle in range(0, 360, 2):
        try:
            logger.info(f"测试角度: {angle}度")
            result = rotate_and_paste(background_path, circle_path, angle)

            # 计算梯度强度
            gradient = calculate_gradient_magnitude(result)
            total_gradient = np.sum(gradient)
            logger.info(f"总梯度强度: {total_gradient}")

            # 更新最佳结果
            if total_gradient < min_gradient:
                min_gradient = total_gradient
                best_angle = angle
                best_result = result
                logger.info(f"发现新最佳角度: {angle}度, 梯度值: {min_gradient}")

        except Exception as e:
            logger.info(f"角度 {angle} 处理出错: {str(e)}")

    logger.info(f"最佳角度: {best_angle}度, 最小梯度值: {min_gradient}")

    # 显示并保存结果
    if best_result is not None:
        cv2.imshow('best_angle', best_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('best_reconstruction.jpg', best_result)
    else:
        logger.info("未找到有效解决方案")