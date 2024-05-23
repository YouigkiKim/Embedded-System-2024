import cv2
import pygame
from jetracer.nvidia_racecar import NvidiaRacecar
import datetime
import time
import os

#define model
import torch
import torchvision
import PIL.Image
from cnn.center_dataset import TEST_TRANSFORMS
import numpy as np

# sign ?��?�� yolo model ?��?��
from ultralytics import YOLO



#classes = YOLO("yolov8n.pytorch.pt", task='detect').names
classes_sign = YOLO("best4.pt", task='detect').names
model_sign = YOLO("best4.pt", task='detect')
colors = np.random.randn(len(classes_sign), 3)
colors = (colors * 255.0).astype(np.uint8)
# visualize_pred_fn = lambda img, pred: draw_boxes(img, pred, classes_sign, colors)

#define and init camera
sensor_id = 0
downscale = 2
width, height = (1280, 720)
_width, _height = (width // downscale, height // downscale)
frame_rate = 30
flip_method = 0
contrast = 1.0
brightness = 0.1

gstreamer_pipeline = (
    "nvarguscamerasrc sensor-id=%d ! "
    "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
    "nvvidconv flip-method=%d, interpolation-method=1 ! "
    "videobalance contrast=%.1f brightness=%.1f ! "
    "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink"
    % (
        sensor_id,
        width,
        height,
        frame_rate,        
        flip_method,
        contrast,
        brightness,
        _width,
        _height,
    )
)

camera = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

#define joystick
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()



# define car
car = NvidiaRacecar()
running = True
throttle_range = [-0.3, 0.3]
throttle_offset = 0
steering_range = [-1, 1]

# lowbattery
# throttle_gain_range = [0.7,0.7]
#AutoThrottle
# auto_throttle = [0.575, 0.59]
# auto_throttle = [0.6, 0.61]
auto_throttle = [0.56, 0.57]
steering_throttle_offset = 0.02
# battery charged
# throttle_gain_range = [0.58 ,0.63]
car.throttle = 0
steering_offset = 0.296
car.steering = 0

def preprocess(image,device):  
    if image is not None:
        image = PIL.Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)) 
        image = TEST_TRANSFORMS(image).to(device)
        return image[None, ...]
    else:
        return None
def get_model():
    model = torchvision.models.alexnet(num_classes=2, dropout=0.0)
    return model

def get_steering(x):
    dot_range = [230, 460]
    # dot_range = [260, 430]
    if x> dot_range[1]:
        newsteering = steering_range[1]
    elif x< dot_range[0]:
        newsteering = steering_range[0]
    else:
        newsteering = min(steering_range[1],max(((x-320)/350),steering_range[0])) - steering_offset
    #throttle
    if x> dot_range[1]:
        newthrottle = auto_throttle[1] + steering_throttle_offset
    elif x< dot_range[0]:
        newthrottle = auto_throttle[1] + steering_throttle_offset
    else:
        newthrottle = auto_throttle[1]
    return newsteering,newthrottle 

device = torch.device('cuda')
model1 = get_model()
model1.load_state_dict(torch.load('Drive05202228.pth'))
model1 = model1.to(device)
model_left = get_model()
model_left.load_state_dict(torch.load('left3.pth'))
model_left = model_left.to(device)
model_straight = get_model()
model_straight.load_state_dict(torch.load('Straight05210021.pth'))
model_straight = model_straight.to(device)
model_right = get_model()
#model_right.load_state_dict(torch.load('right0521002106.pth'))
model_right.load_state_dict(torch.load('right05202106.pth'))
model_right = model_right.to(device)

# option True or Fals
t0 = 0
t1 = 0
t2 = 0 # yolomodel
intersection_count = 0 #intersection
intersection_time = 0
t4 = 0 # throttle_offset

time_bus = 0 #start slow drive
stop_time = 0 # check crosswalk sign starting time
monitor = False
auto = False    
contrast = False
stop_signal_detected = False
intersection_signal_detected = False # 교차로
class_name = 'str'
direction = ''
clss = []

    
x = 320
while running:
    pygame.event.pump()

    if joystick.get_button(7):
        if t4+0.5 < time.time():
            throttle_offset =throttle_offset+0.002
            print('throttle_offset : ',throttle_offset)
    elif joystick.get_button(6):
        if t4+0.5 < time.time():
            throttle_offset  = throttle_offset -0.002
            print('throttle_offset : ',throttle_offset)
    if joystick.get_button(0):
        running = False

    if joystick.get_button(1):
        if t1+1 < time.time():  
            if auto:
                print('Manual mode')
                car.steering = -steering_offset
                car.throttle = 0
                auto = False
                t1 = time.time()
            else:
                auto = True
                print('Auto mode')            
                t1 = time.time()
    if auto:
        if len(clss) != 0:
            if  'crosswalk' in class_name:
                if stop_time + 5 > time.time():
                    print('Crosswalk : Before 5sec')
                    steering,throttle= get_steering(x)
                    car.throttle = 0.05
                    car.steering = steering
                else:
                    print('Crosswalk : After 5ec')
                    steering,throttle= get_steering(x)
                    car.throttle = throttle+ throttle_offset + 0.005
                    car.steering = steering
            elif 'bus' in class_name:
                time_bus = time.time()  #버스 인식되는 동안은 time_bus계속 초기화, 속도 느려짐 적용
                # print('BUS : Drive slowly')
                steering,_= get_steering(x)
                car.throttle = auto_throttle[0]+ throttle_offset
                car.steering = steering
            else:
                steering,throttle = get_steering(x)
                car.throttle = throttle+ throttle_offset
                car.steering = steering                
        else:
            if time_bus +2 > time.time():  #버스 인식 안되면 2초간 스로틀 낮은 상태 유지
                steering,throttle= get_steering(x)
                car.throttle = auto_throttle[0]+ throttle_offset
                car.steering = steering
            else:
                steering,_= get_steering(x)
                if intersection_time+3 > time.time():
                    car.throttle = auto_throttle[1] + throttle_offset
                else:
                    car.throttle = auto_throttle[1]+ throttle_offset
                car.steering = steering
    else:

        steering = joystick.get_axis(2)
        car.steering = steering - steering_offset
        car.throttle_gain = 0.6
        throttle = -joystick.get_axis(1)
        throttle = max(throttle_range[0], min(throttle_range[1], throttle))

        car.throttle = throttle+0.19/car.throttle_gain
    _,frame = camera.read()

    image = preprocess(frame,device)
    with torch.no_grad():
        if 'left' in clss :
            output= model_left(image).detach().cpu().numpy()
            intersection_time = time.time()
            print('left_left_left')
            
        elif 'straight' in clss :
            output = model_straight(image).detach().cpu().numpy()
            intersection_time = time.time()
            print('straight_straight_straight')
            
        elif 'right' in clss :
            output = model_right(image).detach().cpu().numpy()
            intersection_time = time.time()
            print('right_right_right')
            
        else: #교차로가 아닌 상황에
            if intersection_time +3 > time.time(): #마지막 direction 모델 약 3초간 유지(count>2일때)
                if  direction == 'left':                                      
                    output = model_left(image).detach().cpu().numpy()
                    print('After signal | Left')
                elif  direction == 'right':
                    output = model_right(image).detach().cpu().numpy()
                    print('After signal | right')
                elif  direction == 'straight':
                    output = model_straight(image).detach().cpu().numpy()
                    print('After signal | straight')
            else: 
                intersection_time = 0
                intersection_count = 0
                intersection_signal_detected = False
                output = model1(image).detach().cpu().numpy()
    
    x,_ = output[0]
    x = (x/2+0.5)*(width //downscale)
    print(x)
    if monitor:
        cv2.circle(frame, (int(x),281),10,(0,0,255),-1)
        cv2.imshow('Live Streaming',frame)
        if cv2.waitKey(1)==ord('q'):
            break
    if joystick.get_button(10):
        car.throttle = 0.193
        break

    if model_sign is not None:
        # run model
        if t2 +0.05 < time.time():
            yolo_pred = model_sign(frame, stream=True)
            clss = []
            for r in yolo_pred:
                for box in r.boxes:
                    score = round(float(box.conf[0]), 2)

                    label = int(box.cls[0])
                    class_name = classes_sign[label]
                    _,y1,_,y2 = r.boxes.xyxy[0]
                    # print(score)
                    height = abs(y1-y2)
                    print(height, score)
                    if score > 0.8:
                        if height > 50:
                            clss.append(class_name)
                            if class_name == 'left':
                                print('',end='')
                            if 'crosswalk' in class_name:
                                if not stop_signal_detected:
                                    stop_signal_detected = True
                                    # 멈춤 시작 시간 기록, 문제: crosswalk가 안찍혔다가 다시 찍히면 stoptime 초기화
                                    stop_time = time.time()    
                            if 'left'  in  clss or 'straight' in clss or 'right'  in clss:
                                intersection_signal_detected = True
                                intersection_time = time.time()
                                intersection_count = intersection_count+1
                                if 'straight' in clss:
                                    direction = 'straight'
                                elif 'left' in clss:
                                    direction = 'left'
                                elif 'right' in clss:
                                    direction = 'right'
                                else:
                                    direction = ''
                print('Classes:',clss)
            if 'crosswalk' not in clss:
                stop_signal_detected = False
            if 'left'  not in  clss or 'straight'not in clss or 'right' not in clss:
                intersection_signal_detected = False
            t2 = time.time()
                
                    
# multi

camera.release()
cv2.destroyAllWindows()


#export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD