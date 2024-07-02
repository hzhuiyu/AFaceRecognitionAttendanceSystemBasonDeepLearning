import os
import cv2
from faceRegnigtionModel import Model

threshold = 0.7  # 如果模型认为概率高于70%则显示为模型中已有的人物


def read_name_list(path):
    '''
    读取训练数据集
    '''
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list


class Camera_reader(object):
    def __init__(self):
        self.model = Model()
        self.model.load()
        self.img_size = 128

    def build_camera(self):
        '''
        调用摄像头来实时人脸识别
        '''
        face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_alt.xml')
        name_list = read_name_list('dataset/')
        cameraCapture = cv2.VideoCapture(0)

        detected_name = None

        while True:
            success, frame = cameraCapture.read()
            if not success:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                ROI = gray[y:y + h, x:x + w]
                ROI = cv2.resize(ROI, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                label, prob = self.model.predict(ROI)

                if prob > threshold:
                    show_name = name_list[label]
                    detected_name = show_name
                else:
                    show_name = "Stranger"

                cv2.putText(frame, show_name, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("Camera", frame)

            # 如果检测到人脸，就销毁窗口并返回detected_name
            if detected_name:
                cameraCapture.release()
                cv2.destroyAllWindows()
                return detected_name

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cameraCapture.release()
        cv2.destroyAllWindows()
        return detected_name


if __name__ == '__main__':
    camera = Camera_reader()
    detected_name = camera.build_camera()
    print(f"识别到的姓名为: {detected_name}")
