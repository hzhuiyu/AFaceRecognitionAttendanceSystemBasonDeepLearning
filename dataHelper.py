import cv2
import os
import time

def process_images_and_save_faces(input_folder, output_folder):
    """
    处理输入文件夹中的所有图像，检测人脸并保存到指定的输出文件夹中。

    Args:
    - input_folder (str): 包含输入图像的文件夹路径。
    - output_folder (str): 输出文件夹的路径。
    """
    # 初始化人脸检测器（使用OpenCV自带的Haar级联分类器）
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    count = 0

    # 遍历输入文件夹中的所有图片
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 读取图片
            image_path = os.path.join(input_folder, filename)
            img = cv2.imread(image_path)

            # 将图像转换为灰度
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 使用Haar特征检测人脸
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # 遍历每张检测到的人脸
            for (x, y, w, h) in faces:
                # 构造文件名（基于时间戳和计数）
                timestamp = int(time.time())
                fileName = f'{timestamp}_{count}.jpg'

                # 裁剪出人脸并调整大小为 (200, 200)
                face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

                # 保存人脸图像
                cv2.imwrite(os.path.join(output_folder, fileName), face_img)

                # 增加计数
                count += 1

    print('处理完成！')


input_folder = 'data/zhz'    # 输入文件夹路径，包含要处理的图像
output_folder = 'dataset/zhz'  # 输出文件夹路径，保存处理后的人脸图像

# 调用函数处理图像并保存人脸
process_images_and_save_faces(input_folder, output_folder)
