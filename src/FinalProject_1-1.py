import cv2
import pygame
from jetracer.nvidia_racecar import NvidiaRacecar
import time
import os
import torchvision.transforms as transforms
from simple_pid import PID
#define model
import torch
import torchvision
import PIL.Image
import numpy as np

# sign ?��?�� yolo model ?��?��
from ultralytics import YOLO

TEST_TRANSFORMS = transforms.Compose([
    # transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#classes = YOLO("yolov8n.pytorch.pt", task='detect').names
classes_sign = YOLO("best0608.pt", task='detect').names
model_sign = YOLO("best0608.pt", task='detect')
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
auto_throttle = [0.55, 0.58]
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
t1=0
t4 = 0 # throttle_offset
t2=0
time_bus = 0 #start slow drive
stop_time = 0 # check car sign starting time
monitor = False
auto = False    
class_name = 'str'
direction = ''
clss = []
accheight = 100
pid =PID(Kp = 0.0005,Ki = 0.00001, Kd = 0.001, setpoint = accheight)
dthrottle = 0
minthrottle = 0.1
maxthrottle = auto_throttle[1]+0.1

thorttle_offset = 0
x=320
while running:
    pygame.event.pump()

    if joystick.get_button(7):  #쓰로틀 값 실시간 보정
        if t4+0.5 < time.time():
            throttle_offset =throttle_offset+0.002
            print('throttle_offset : ',throttle_offset)
    elif joystick.get_button(6):  #쓰로틀 값 실시간 보정
        if t4+0.5 < time.time():
            throttle_offset  = throttle_offset -0.002
            print('throttle_offset : ',throttle_offset)
    if joystick.get_button(10): # select
        break
    if joystick.get_button(0):  # A버튼: running 종료 
        running = False

    if joystick.get_button(1):  # B버튼: auto <-> Manual 전환
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
                
    # get X BY alexnetFF                   
    _,frame = camera.read()
    image = preprocess(frame,device)
    with torch.no_grad():
        output = model1(image).detach().cpu().numpy()
    x,_ = output[0]
    x = (x/2+0.5)*(width //downscale)
    
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
                    height = int(abs(y1-y2))
                    print(height, score)
                    if score > 0.75:
                        if height > 50:  #점수랑 y값 기준 통과할 때만 클래스에 'car'추가
                            clss.append(class_name)
                            # print('height: ',height)
                            dthrottle = pid(height)     
                print('Classes:',clss)
            t2 = time.time()
                            
                
    if auto:
        print(x)
        if len(clss) != 0:  
            Throttle = max(min(auto_throttle[0]+ dthrottle + throttle_offset, auto_throttle[1]),minthrottle)
            print(f"height = {height}, throttle = {Throttle}, dthrottle = {dthrottle}")
            steering,_ = get_steering(x)
            car.throttle = Throttle
            car.steering = steering                
        else:
            steering,_= get_steering(x)
            dthrottle = 0
            car.throttle = auto_throttle[1] + throttle_offset
            car.steering = steering
    else: #조이스틱 주행
        steering = joystick.get_axis(2)
        car.steering = steering - steering_offset
        car.throttle_gain = 0.6
        throttle = -joystick.get_axis(1)
        throttle = max(throttle_range[0], min(throttle_range[1], throttle))
        car.throttle = throttle+0.19/car.throttle_gain
    

camera.release()
cv2.destroyAllWindows()
car.throttle = 0


#export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD 