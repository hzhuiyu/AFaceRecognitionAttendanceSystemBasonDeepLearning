# -
重庆大学大数据与软件学院大二暑期实训基于深度识别的AI人脸识别系统
# 项目简介
本项目为重庆大学大数据与软件学院大二暑假实训的一个项目，即基于深度学习的AI人脸识别系统，本项目在《python项目开发实战》源代码的基础上将所有功能都在htlm界面上实现出来，并与MySQL数据库连接增加了签到签退记录功能，添加背景图进行了一定程度的美化。本人实训开始前系统出了问题含泪清空磁盘重装系统，为防止悲剧重演，特此将此次实训的代码上传GitHub备份。

运行文件应该安装的包：opencv，flask，numpy，tensorflow，sklearn，如果有报错的话自己检查一下哪里报错网上寻找解决办法吧，或者往下看每个文件的介绍，这里漏了的包我会在下面介绍的。编译环境（应该是叫这个吧）是pycharm，python版本是3.11，我不确定其他版本会不会出现不兼容的问题。

以下是对各个py文件功能的介绍

getCameraPics.py：调用摄像头来拍照，在摄像头界面输入c可以在pycharm里输入字符在data文件夹里创建新的文件夹储存照片，一个人的照片都储存在一个文件夹里，注意这里如果输入中文的话会出现无法保存图片的情况，我在网上找了很久也没找到有用的解决办法就摆烂了，所以文件夹名只能是英文。在摄像头界面输入p会拍摄一张照片并保存到你刚刚创建的文件夹里，如果没有创建新的文件夹的话就是保存到预设的文件夹里，预设的文件夹为“zhz”，修改这个就行了。

dataHelper.py：这个文件是用来处理data里的文件的图片的，将data里的图像灰化然后识别并裁剪出其中的人脸，保存到dataset/xxx文件夹里，有的图可能识别不出其中的人脸，所以建议前面拍照的时候多拍几张。如果要直接运行main3的话记得把这里最下面的

1. input\_folder = 'data/zhz'    *# 输入文件夹路径，包含要处理的图像*
1. output\_folder = 'dataset/zhz'  *# 输出文件夹路径，保存处理后的人脸图像*

1. *# 调用函数处理图像并保存人脸*
1. process\_images\_and\_save\_faces(input\_folder, output\_folder)


删掉，这是单独运行的时候要加的代码，运行main的时候因为会调用如果不删掉的话会报错。

faceRecognitionModel.py：这个文件是读取dataset里的文件，用深度学习的方法训练出模型，为人脸识别提供模型的，可能对电脑配置有一定的要求，我这里的配置是处理器：i7-1260P 内存：16G 显卡是：NVIDIA geforce mx570，不过我这是轻薄本如果是游戏本应该就没问题了。这里头文件导包可能会报错：

1. from tensorflow.keras.utils import to\_categorical
1. from tensorflow.keras.models import Sequential, load\_model

不用管，只要你安装了tensorflow，可以运行的

训练结束之后在文件夹里会多出个face.h5文件，就是训练好的模型。

cameraDemo.py：这个文件是调用摄像头识别人脸的，如果识别出来与模型里训练过的一个人脸相同就会关闭摄像头，并返回识别出的人的名字。

mian3：…nm我现在才发现这个文件名打错了，罢了就这样吧懒得改了。这里东西有点多，我按顺序讲下来吧。

前面几行跟上传图片的格式有关，这个应该不用改。

1. db\_config = {
1. `    `'user': 'root',
1. `    `'password': '976419zhz',
1. `    `'host': 'localhost',
1. `    `'database': 'face\_recognition'
1. }

这里是对MySQL数据库的设定，因为是用MySQL保存签到签退数据的你得先下个MySQL然后设置登陆密码，用户名一般是默认root的，自己没改的话就不用改，password改成你自己设定的密码，database是我保存签到信息的数据类型，你得先运行

1. CREATE DATABASE face\_recognition;

1. USE face\_recognition;

1. CREATE TABLE check\_log (
1. `    `id INT AUTO\_INCREMENT PRIMARY KEY,
1. `    `name VARCHAR(100) NOT NULL,
1. `    `action VARCHAR(10) NOT NULL,
1. `    `time DATETIME NOT NULL
1. );

来创建face\_recognition这个数据类型。

接下来到

1. @app.route('/upload', methods=['POST', 'GET'])

这行代码之前都不用动，这行代码是上传图片识别图片里的人脸功能界面的路由，这里用到的函数导包可能有点问题，如果使用的时候运行报错“storage must be a werkzeug.FileStorage“的话，报错应该有一行末尾包括packages\flask\_uploads.py”，点进去应该会给你弹出来一个文件夹的内容，点开flask\_uploads.py文件，把里面引用的这一行头文件

1. from werkzeug import secure\_filename, FileStorage

改成

1. from werkzeug.utils import secure\_filename
1. from werkzeug.datastructures import FileStorage

我记得前面flask的某一行导入也有这种问题，如果报错的不止我提到的那个可以无视的报错的话，把报错的代码复制出来到网上搜一下，看看是什么问题，如果也是导包的问题的话可以参考这里的修改方式进行修改。mian3.py要注意的几个点应该就这些了，接下来我介绍一下htlm界面。

template/index.htlm：htlm文件都存放在template文件夹里。运行程序后，点进显示出来的链接，显示出的首页就是index界面，这里可以跳转到其他所有功能：签到签退，上传图片识别，采集图片，查询记录等，如果想更换背景图片的话进入template文件夹里，把”目录.jpg”文件删掉，添加你喜欢的背景图片，改名为目录即可。也可以修改index里的代码

1. `        `body {
1. `            `background-image: url({{ url\_for('static', filename='background/目录.jpg')}});
1. `            `background-size: cover;
1. `            `font-family: Arial, sans-serif;
1. `            `color: white;
1. `            `text-align: center;
1. `            `padding-top: 50px;
1. `        `}

把background/目录.jpg换成你要更改的背景图片的位置就可以了，也可以把第一个url（）里的所有内容” {{ url\_for('static', filename='background/目录.jpg')}}”更换成具体的图片网址，不过如果用网上的图片作为背景的话进入界面就会有一个加载时间，所以更推荐把图片下载到本地，再把本地图片的地址填进去。其他几个页面的背景图也是这样改，如果你对页面的布局不满意的话也可以自己学习flask设计htlm的方法，自己进行修改。

camera.htlm：签到签退界面，点签到会弹出摄像头识别人脸，识别成功后跳转到result.htlm，显示识别出的人脸名字，签退同理，不过显示的文字是签退。result界面可以返回首页。

upload.htlm：图片识别界面，在这里可以上传一张本地的图片，识别图片中的人脸，可能是深度学习过拟合了，即使上传的图片没有训练过也会显示训练过的第一个人的名字，并显示识别成功，目前没想到怎么解决。识别完成后会跳转到show界面，显示识别结果和识别时间，show界面可返回主页。

collect.htlm：采集图片界面，在这里可以进行采集图片，处理图片，训练模型等功能，并且提供了跳转到其他功能的按钮。输入姓名再点击开始采集就可调用摄像头，输入一次p即可采集一张图片，采集多少张图片手动控制。下面一行输入姓名即可处理采集到的图片，这里不做过多赘述。点击训练模型就可以读取处理好的所有照片进行模型训练，训练完成后会跳转到trainresult界面显示训练成功。点击查看签到记录记录可跳转到records界面。

records.htlm：显示签到签退记录，可清空记录。我的想法是一天考勤完了就清空记录，可以根据自己的想法结合MySQL进行修改。
