import cv2 #อิมพอทโมดูลไลบรารี่ของ open cv เป็นไลบรารี่สำหรับการทำ computer vision หรือการสร้างดวงตาให้กับคอมพิวเตอร์เพื่อเข้าใจภาพของมนุษย์
import numpy as np #อิมพอทโมดูลไลบรารี่ numpy อ่านว่านัมพาย หรือ นัมปี้ เป็นโมดูลสำหรับคำนวณทางคณิตศาสตร์สูตรบางที่อยู่นอกเหรือจากไลบรารี่พื้นฐานของไพทอนจะใช้ไลบรารี่นี้แทนเนื่อจากบ่งสูตรนั้นมีความซัพซ้อนในการเขียนหากเขียนเองอาจมีความผิดพลาดได้หรืออาจช้ากว่าเดิม
import os #อิมพอทโมดูลไลบรารี่ OS เข้ามาในโปรแกรมเป็นโมดูลสำหรับเรียกใช้งานฟังชั่นของวินโดว

recognizer = cv2.face.LBPHFaceRecognizer_create() #สร้างไฟล์ตัวแปลชื่อเรคคอไนทเซอร์สำหรับเป็นที่เก็บออปเจ็ค lowbinarypatternhistrogram เป็น object สำหรับใช้ในการตรวจสอบคุณสมบัต่ของรูปโปรไฟล์
                                                #binary profile แตกต่างกับ low binary profile โดยไบนารี่โปรไฟล์จะแทนสีด้วยสีสองสีคือขาว ดำแทนด้วยตัวเลข 0 และ 1 ส่วนโลวโปรไฟล์จะแทนด้วยเฉดสีดำไปหาขาวแทนตัวเลขด้วย 0-255
recognizer.read('trainer.yml') #โหลดโมเดลหน้าสำหรับให้ AI ตรวจสอบจากไฟล์ชื่อ trainer.yml

face_cascade_Path = "haarcascade_frontalface_default.xml" #โหลดเซ็นข้อมูลที่ใช้สำหรับกำหนดพื้นที่ตรวจสอบใบหน้าจากไฟล์  haarcascade_frontalface_default.xml เก็บในตัวแปลชื่อ face_cascade_Path


faceCascade = cv2.CascadeClassifier(face_cascade_Path)#กำหนดไฟล์พื้นที่สำหรับตรวจสอบใบหน้าของแต่ละหน้าที่บันทึกจากไฟล์  haarcascade_frontalface_default.xml

font = cv2.FONT_HERSHEY_SIMPLEX #กำหนดฟร้อนที่ใช้สำหรับตรวจจับใบหน้า

id = 0 #ตัวแปล id ใช้สำหรับเก็บ index รูปภาพที่ใช้สำหรับตรวจสอบ
# names related to ids: The names associated to the ids: 1 for Mohamed, 2 for Jack, etc...
names = ['None'] # add a name into this list #ตัวแปล names ใช้สำหรับเก็บ ชื่อของรูปภาพหรือสิ่งของที่ต้องการตรวจจับโดยชื่อแรกจะเป็น  NONE ชื่อสำหรับหน้าที่ไม่ได้ทำการจดจำหรือทำการเทรนให้ AI รู้จัก
#Video Capture
cam = cv2.VideoCapture(0) #คำสั่งสำหรับให้เลือก video cature ที่ 0 หรือก็คือ กล้องเริ่มต้นของอุปกรณ์คอมพิวเตอร์
cam.set(3, 640)#เซ็ตขนาดึวามกว้างของหน้าจอหน้าจอ
cam.set(4, 480)#เซ็ตขนาดความสูงของหน้าจอหน้าจอ
# Min Height and Width for the  window size to be recognized as a face
minW = 0.1 * cam.get(3) #กำหนดขวานความกว้างหน้าจอต่ำสุด
minH = 0.1 * cam.get(4) #กำหนดขวานความสูงหน้าจอต่ำสุด
while True: #เงื่อนไขทำซ้ำไม่รู้จบ โดยคำสั่งใดที่อยู่ภายใดต้ tag while จะทำงานซ้ำเรื่อยๆไม่รู้จบ
    ret, img = cam.read() # อ่านค่าจากกล้องเพื่อเก็บภาพ แล้วนำข้อมูลภาพที่ได้เก็บลงในตัวแปลชื่อ ret img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # ทำภาพให้เป็นเกรสเกลหรือภาพขาวดำ

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    ) #บรรทัดที่ 30-35 คือการกำหนดพื้นที่ตรวจสอบใบหน้า

    for (x, y, w, h) in faces: #กำหนดเงื่อไขการทำซ้ำโดยดึงค่าตัวแปล x y w h จาก ตัวแปน face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) #สร้างภาพสี่เหลี่ยมครอบบนใบหน้า
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])#ดึงค่า id ของหน้าที่ถูกบันทึกไว้จากรูปภาพและค่าเป็นเซ็นความคล้ายครึงของของแพทเทิล
        if (confidence < 100):#ถ้าค่าความคล้ายครึงในตัวแปน confidence อยู่ในช่วงที่กำหนดให้ id เท่ากับชื่อจอวคนที่เก็บอยู่ในตัวแปร name และทำการแปรงค่าคอนไฟเดนเป็นตัวอักษรเพื่อใช้สำหรับพพิมพ์
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:#ถ้าเงื่อนไขไม่เ็นจริงให้id มีค่าเท่าดับ who are you คอนไฟเดนต่ำกว่า 0
            # Unknown Face
            id = "Who are you ?"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2) #นำข้อความในตัวแปร id มาแปะลงบนกล่องสี่เหลี่ยม
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1) #นำข้อความในตัวแปร confidence มาแปะลงบนกล่องสี่เหลี่ยม

    cv2.imshow('camera', img)
    # Escape to exit the webcam / program
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
print("\n [INFO] Exiting Program.")
cam.release()
cv2.destroyAllWindows()
