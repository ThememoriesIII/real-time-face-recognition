# Real-Time-Face-Recognition (OpenCV)

Create a fast real-time face recognition app with few lines of python code.
#โปรเจ็คนี้สร้างขึ้นมาเพื่อสร้างโปรแกรมจดจำใบหน้าแบบเรียลทามด้วยภาษาไพทอน
<img src = 'https://github.com/medsriha/Real-Time-Face-Recognition/blob/master/gif.gif?raw=true'><center>

## Steps:

`cmd: python face_taker.py`
1) Take pictures using the `face_taker.py` script. The script will save 30 images of your face in the `images` folder after you entered the ID number (MUST be integer and incremental (starts with 1 then 2, 3, ...)
Note: Make sure your face is centered. The window will collapse when all the 30 pictures are taken.
คอมมาน python face_taker.py
ใช้สำหรับนำรูปจากกล้องถ่ายรูปโดยสคริปจะทำการเซฟรูปภาพเป็นจำนวน 30 รูปโดยรูปภาพทั้งหมดจะเป็นรูปภาพใบหน้า และทำการเก็บไว้ในโฟลเดอร์ images ในการถ่วยภาพใบหน้าจะมีพรอมให่เนสใใส่หมายเลขใบหน้า ID number:

`cmd: python face_train.py`

2) The `face_tain.py` script will train a model to recognize all the faces from the 30 images taken using `face_taker.py` script, and save the training output in the `training.yml` file.
 คอมมาน face_train.py ใช้สำหรับเทรนโมเดล AI โดยใช้ภภาพในโฟลเดอร์ Images โดยจะมีเอาต์พุตเป็นไฟล์ชื่อ Training.yml


`cmd: python face_recognizer.py`

3) The `face_recognizer.py` is the main script. You need to append the name of each person with the picture taken in the `face_taker.py` script. The program will recognize the face according to the id given in the `face_taker.py` script. If Joe has an id 1, his name should appear in the list as index 1 like such `names = ['None', 'Joe'] # keep None and append a name into this list`
 คอมมาน face_recognizer.py ใช้สำหรับเริ่มต้นโปรแกรมตรวจจับใบหน้าโดยในบรรทัดที่ 40 จะมี name= ['None', 'Joe'] ตรงคำว่าnone เป็นคำที่เว้นว่างไว้สำหรับคนที่ไม่มีหน้าในระบบส่วน joe เป็นชื่อสำหรับใบหน้าที่ 1 ที่เราทำการถ่ายด้วยคอมมาน face_taker.py เช่นถ้ารูปแรกที่เราใช้คอมมาน face_taker.py มีชื่อว่าทิติวัฒ เราก็ต้องใส่ชื่อ ทิติวัฒแทน joe

Requirements:

- `pip install opencv-python`
- `pip install opencv-contrib-python --upgrade` or `pip install opencv-contrib-python --user.`
