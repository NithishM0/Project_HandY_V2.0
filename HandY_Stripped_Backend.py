import cv2
import mediapipe as mp
import time
import mouse
import numpy as np
import math
from tkinter import *
import threading

cap = cv2.VideoCapture(0)


import numpy as np
mLocOld = np.array([0,0])
mouseLoc = np.array([0,0])
(camx,camy) = (480,320)
Df = 4.2
cap.set(3, camx)
cap.set(4, camy)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
                      static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1
    )
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
landmarks = []
lis = []


while True:

    success, img = cap.read()
    img.flags.writeable = False
    h1, w1, c1 = img.shape


    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    img.flags.writeable = False


    if results.multi_hand_landmarks:




        for handLms in results.multi_hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w1
            y_min = h1
            for _id, lm in enumerate(handLms.landmark):



                        h, w, c = img.shape

                        cx, cy, cz = int(lm.x *w), int(lm.y*h), int(lm.z*w)
                        cv2.circle(img, (cx,cy), 3, (255,255,255), cv2.FILLED)
                        landmarks.append([_id, cx, cy, cz])
                        if cx > x_max:
                            x_max = cx
                        if cx < x_min:
                            x_min = cx
                        if cy > y_max:
                            y_max = cy
                        if cy < y_min:
                             y_min = cy



                        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)







    if landmarks != []:





        xa, ya, za = landmarks[4][ 1], landmarks[4][ 2], landmarks[4][ 3]
        xb, yb, zb = landmarks[8][1], landmarks[8][2],landmarks[8][ 3]
        xb2, yb2, zb2 = landmarks[5][1], landmarks[5][2],landmarks[5][ 3]



        cv2.circle(img, (xa, ya), 7, (255, 255, 255), cv2.FILLED)
        cv2.circle(img, (xb, yb), 7, (255, 255, 255), cv2.FILLED)


        cv2.line(img, (xa, ya), (xb, yb), (255, 255, 255), 3)

        xa1, ya1, za1 = landmarks[8][ 1], landmarks[8][ 2], landmarks[8][ 3]
        xb1, yb1, zb1 = landmarks[12][1], landmarks[12][2],landmarks[12][ 3]



        cv2.circle(img, (xa1, ya1), 7, (255, 255, 255), cv2.FILLED)
        cv2.circle(img, (xb1, yb1), 7, (255, 255, 255), cv2.FILLED)


        cv2.line(img, (xa1, ya1), (xb1, yb1), (255, 255, 255), 3)

        dist = math.sqrt((xa - xb)**2 + (ya - yb)**2)
        dist2 = math.sqrt((xa1 - xb1)**2 + (ya1- yb1)**2)
        lis.append(dist2)

        midx, midy = int((xa + xb)/2), int((ya+yb)/2)

        cv2.line(img, (midx, midy), (xb2, yb2), (255, 255, 255), 3)


        cv2.circle(img, (midx, midy), 7, (255, 255, 255), cv2.FILLED)
        dist1 = math.sqrt((midx - xb2)**2 + (midy - yb2)**2)

       




        sx, sy = 1366, 768
        
        mouseLoc = mLocOld + ((int(xb),int(yb)) - mLocOld)/Df

        hx, hy = (sx-(mouseLoc[0]*sx/camx),mouseLoc[1]*sy/camy)
        mouse.move(hx, hy)
        mLocOld = mouseLoc



        inter = np.interp(dist2,(15,100), (1,-1))
        mouse.wheel(inter)


        if dist/2 < dist1-15:
            mouse.click(button='left')





        landmarks = []








    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
  

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break



