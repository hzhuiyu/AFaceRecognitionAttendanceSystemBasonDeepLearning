import os
import cv2

def cameraAutoForPictures(saveDir='data/'):
    '''
    调用电脑摄像头来自动获取图片
    '''
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    count=1
    cap=cv2.VideoCapture(0)
    width,height,w=640,480,360
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
    crop_w_start=(width-w)//2
    crop_h_start=(height-w)//2
    print('width: ',width)
    print('height: ',height)
    while True:
        ret,frame=cap.read()
        frame=frame[crop_h_start:crop_h_start+w,crop_w_start:crop_w_start+w]
        frame=cv2.flip(frame,1,dst=None)
        cv2.imshow("capture", frame)
        action=cv2.waitKey(1) & 0xFF
        if action==ord('c'):
            saveDir=input(u"请输入新的存储目录：").strip()
            saveDir = os.path.join('data', saveDir)
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
        elif action==ord('p'):
            filePath = os.path.join(saveDir, f"{count}.jpg")
            cv2.imwrite(filePath, cv2.resize(frame, (224, 224),interpolation=cv2.INTER_AREA))
            print(f"{saveDir}: {count} 张图片")
            count+=1
        if action==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    #xxx替换为自己的名字
    cameraAutoForPictures(saveDir=r'data/zhz/')
