import time
print('Initializing Project HandY....')
time.sleep(0.5)


import cv2
import mediapipe as mp
import time
import mouse
import numpy as np
import math
from tkinter import *
import threading
import PySimpleGUI as sg
from pygrabber.dshow_graph import FilterGraph
import numpy as np
print('All Dependencies Imported....')
time.sleep(0.5)






graph = FilterGraph()
pos = None
def select_camera(last_index):
    number = 0
    hint = "Select a camera (0 to " + str(last_index) + "): "
    try:
        number = int(input(hint))
        # select = int(select)
    except Exception :
        print("It's not a number!")
        return select_camera(last_index)

    if number > last_index:
        print("Invalid number! Retry!")
        return select_camera(last_index)

    return number
device_list = graph.get_input_devices()
index = 0

for name in device_list:
    print(str(index) + ': ' + name)
    index += 1
last_index = index - 1

if last_index < 0:
    print("No device is connected")
    exit()

camera_number = select_camera(last_index)
cap = cv2.VideoCapture(camera_number)
cv2.setUseOptimized(True)



mLocOld = np.array([0,0])
mouseLoc = np.array([0,0])
(camx,camy) = (480,320)
print('Add a Mouse Dampening field greater than 1.0')
time.sleep(0.5)
Df = float(input("Enter Mouse Dampening field (Recommended Value: 2.4): "))
cap.set(3, camx)
cap.set(4, camy)
sg.theme('DarkBlack')
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
liframe = [
    [sg.Text('Frames Per Second', font='Any 20'), sg.Text('0', font='Any 20', key = 'fps')],
    
    [sg.Text('Mouse Position: ', font='Any 20'), sg.Text('-----', font='Any 20', key = 'f')],
    
    ]
rframe = [
    [sg.Image(r'', key = 'img')],
    [sg.Text('Webcam Input', font='Any 20')],
    [sg.Frame('Statistics', layout = liframe)]
    
    
    ]
R  = [
    [sg.Frame('Backend', layout = rframe)]
    
            ]
layout = [
    [sg.Column(R, element_justification = 'c'),sg.VSeparator()
    ]
    ]
window = sg.Window('Project HandY - Nithish Murugavenkatesh',
                       layout, location=(0, 0),no_titlebar=False,margins=(0,0),grab_anywhere=True, finalize=True)
print('All Variables, Classes, and Functions Defined.')

print('Running.... Project HandY 3.0')

while True:

    success, img = cap.read()
    
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



        


        cv2.line(img, (xa, ya), (xb, yb), (255, 255, 255), 3)

        
        xb1, yb1, zb1 = landmarks[12][1], landmarks[12][2],landmarks[12][ 3]



       

        

        dist = math.sqrt((xa - xb)**2 + (ya - yb)**2)
        
        

        midx, midy = int((xa + xb)/2), int((ya+yb)/2)

        cv2.line(img, (midx, midy), (xb2, yb2), (255, 255, 255), 3)


        cv2.circle(img, (midx, midy), 7, (255, 255, 255), cv2.FILLED)
        dist1 = math.sqrt((midx - xb2)**2 + (midy - yb2)**2)

       




        sx, sy = 1366, 768
        
        mouseLoc = mLocOld + ((int(xb),int(yb)) - mLocOld)/Df

        hx, hy = (sx-(mouseLoc[0]*sx/camx),mouseLoc[1]*sy/camy)
        mouse.move(hx, hy)
        mLocOld = mouseLoc



        


        if dist/2 < dist1-19:
            mouse.click(button='left')





        landmarks = []
        pos = str(cx)+','+str(cy)








    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    window['fps'].update(str(int(fps)))
    if pos != None:
        window['f'].update(str(pos))
    
   
    
    imgbytes = cv2.imencode('.png', img)[1].tobytes()
    window["img"].update(data=imgbytes)
    window.refresh()
    if cv2.waitKey(1) & 0xff == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()



