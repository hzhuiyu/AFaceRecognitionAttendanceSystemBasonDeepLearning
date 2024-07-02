from flask_uploads import UploadSet, IMAGES, configure_uploads, patch_request_class
from flask import Flask, request, redirect, url_for, render_template
import os
import cv2
import time
import numpy as np
import mysql.connector
from faceRegnigtionModel import Model
from cameraDemo import Camera_reader
from werkzeug import FileStorage
from getCameraPics import cameraAutoForPictures
from faceRegnigtionModel import DataSet
from dataHelper import process_images_and_save_faces

app = Flask(__name__)
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'bmp'}

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app, size=None)

if not os.path.exists(app.config['UPLOADED_PHOTOS_DEST']):
    os.makedirs(app.config['UPLOADED_PHOTOS_DEST'])

db_config = {
    'user': 'root',
    'password': '976419zhz',
    'host': 'localhost',
    'database': 'face_recognition'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def dest(name):
    return os.path.join(app.config['UPLOADED_PHOTOS_DEST'], name)

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        if 'photo' in request.files:
            file = request.files['photo']
            filename = photos.save(file)
            return redirect(url_for('show', name=filename))

        else:
            return 'No file part in the request!'
    return render_template('upload.html')


@app.route('/photo/<name>')
def show(name):
    if not name:
        return "出错了！"

    url = photos.url(name)
    start_time = time.time()
    res = detectOnePicture(dest(name))
    end_time = time.time()
    execute_time = str(round(end_time - start_time, 2))
    tsg = f' 总耗时为： {execute_time} 秒'
    return render_template('show.html', url=url, xinxi=res, shijian=tsg)

def endwith(s, *endstring):
    return any(map(s.endswith, endstring))

def read_file(path):
    img_list = []
    label_list = []
    dir_counter = 0
    IMG_SIZE = 128

    for child_dir in os.listdir(path):
        child_path = os.path.join(path, child_dir)
        for dir_image in os.listdir(child_path):
            if endwith(dir_image, 'jpg'):
                img = cv2.imread(os.path.join(child_path, dir_image))
                resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                recolored_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                img_list.append(recolored_img)
                label_list.append(dir_counter)
        dir_counter += 1

    img_list = np.array(img_list)
    return img_list, label_list, dir_counter

def read_name_list(path):
    return [child_dir for child_dir in os.listdir(path)]

def detectOnePicture(path):
    model = Model()
    model.load()
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    picType, prob = model.predict(img)

    if picType != -1:
        name_list = read_name_list('dataset/')
        res = f"识别为： {name_list[picType]} 的概率为： {prob}"
    else:
        res = "抱歉，未识别出该人！请尝试增加数据量来训练模型！"
    return res

@app.route("/collect", methods=['GET', 'POST'])
def collect():
    if request.method == 'POST':
        name = request.form['name']
        saveDir = os.path.join('data', name)
        cameraAutoForPictures(saveDir)
        return redirect(url_for('collect'))
    return render_template('collect.html')

@app.route("/process", methods=['POST'])
def process():
    name = request.form['name']
    srcDir = os.path.join('data', name)
    dstDir = os.path.join('dataset', name)
    process_images_and_save_faces(srcDir, dstDir)
    return redirect(url_for('collect'))

@app.route("/train", methods=['POST'])
def train():
    dataset = DataSet('dataset/')
    model = Model()
    model.read_trainData(dataset)
    model.build_model()
    model.train_model()
    model.save()
    return render_template('trainresult.html')

@app.route("/")
def init():
    return render_template("index.html", title='Home')

@app.route("/camera/")
def camera():
    return render_template("camera.html", title='签到签退')

@app.route("/checkin", methods=['POST'])
def checkin():
    return handle_check("checkin")

@app.route("/checkout", methods=['POST'])
def checkout():
    return handle_check("checkout")


def handle_check(action):
    camera = Camera_reader()
    detected_name = camera.build_camera()

    if detected_name:
        log_check(detected_name, action)
        result_message = f"{detected_name} {action} 成功！"
        return render_template('result.html', result_message=result_message)
    else:
        return "未检测到姓名，请重新尝试。"


def log_check(name, action):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    time_now = time.strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO check_log (name, action, time) VALUES (%s, %s, %s)", (name, action, time_now))
    connection.commit()
    cursor.close()
    connection.close()

@app.route("/records")
def records():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    cursor.execute("SELECT name, action, time FROM check_log ORDER BY time DESC")
    records = cursor.fetchall()
    cursor.close()
    connection.close()
    return render_template("records.html", records=records)

@app.route("/clear_records")
def clear_records():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    cursor.execute("DELETE FROM check_log")
    connection.commit()
    cursor.close()
    connection.close()
    return redirect(url_for('records'))


if __name__ == "__main__":
    print('faceRegnitionDemo')
    app.run(debug=True)
