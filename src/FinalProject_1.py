
from mpu9250_jmdev.registers import *
from mpu9250_jmdev.mpu_9250 import MPU9250
from collections import deque
import time
import cv2
import pygame
from jetracer.nvidia_racecar import NvidiaRacecar
import os
import torch
import torchvision
import PIL.Image
import numpy as np
import torchvision.transforms as transforms
from simple_pid import PID
from ultralytics import YOLO

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),    
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



def initIMU():
    mpu.calibrate() 
    mpu.configure()  

def initvelocity():
    x_data = []
    y_data = []
    inittime = time.time()
    while inittime + 3 > time.time():
        data = mpu.readAccelerometerMaster()
        x_data.append(data[0])
        y_data.append(data[1])
    x_offset = sum(x_data) / len(x_data)
    y_offset = sum(y_data) / len(y_data)
    print(f"x offset : {x_offset}, y offset : {y_offset}")
    return x_offset, y_offset


mpu = MPU9250(
    address_ak=AK8963_ADDRESS,  # AK8963 magnetometer 
    address_mpu_master=MPU9050_ADDRESS_68,  
    bus=1,  
    gfs=GFS_250,  
    afs=AFS_16G,  
    mfs=AK8963_BIT_16,  
    mode=AK8963_MODE_C8HZ  
)
initIMU()
print(mpu.abias)
x_offset, y_offset = initvelocity()


prev_time = time.time()
velocity = [0, 0, 0]  
t0 = 0
accelprint = False
auto = False
accelprinttime = 0.0
printtime = 0.0
autoprint = 0.0
window_size = 20
xwindow = deque(maxlen=window_size)
ywindow = deque(maxlen=window_size)
sumx = 0
sumy = 0

gyro_z_window = deque(maxlen=window_size)
gyro_z_sum = 0

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
steering_range = [-1, 1]

throttle_gain_range = [0.68,0.75]

car.throttle_gain = 0
car.throttle = 0
steering_offset = 0.290
before = 0
now_steering = 0

classes_sign = YOLO("best0608.pt", task='detect').names
model_sign = YOLO("best0608.pt", task='detect')
colors = np.random.randn(len(classes_sign), 3)
colors = (colors * 255.0).astype(np.uint8)

def preprocess(image, device):  
    if image is not None:
        image = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
        image = TEST_TRANSFORMS(image).to(device)
        return image[None, ...]
    else:
        return None

def get_model():
    model = torchvision.models.alexnet(num_classes=2, dropout=0.0)
    return model

def decrease_contrast(image, factor):
    if image.dtype == np.uint8:
        max_value = 255
    elif image.dtype == np.uint16:
        max_value = 65535
    elif image.dtype == np.float32:
        max_value = 1.0
    else:
        raise TypeError("대비 변환실패")
    new_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)

    if image.dtype == np.float32:
        new_image = new_image.astype(np.float32) / max_value
    return new_image

def get_steering(x):
    dot_range = [240, 450]
    if x > dot_range[1]:
        newsteering = steering_range[1]
    elif x < dot_range[0]:
        newsteering = steering_range[0]
    else:
        newsteering = min(steering_range[1], max(((x - 320) / 400), steering_range[0])) - steering_offset
    
    return newsteering

device = torch.device('cuda')
model = get_model()
model.load_state_dict(torch.load('Drive05202228.pth'))
model = model.to(device)

# option True or False
t0 = 0
t1 = 0
t2 = 0 #used for car detect 
t8 = time.time() #corner
t9 = time.time() #corner flag
stop_time=0
class_name = 'str'
direction = ''
clss = []
accheight = 100
pid =PID(Kp = 0.0005,Ki = 0.00001, Kd = 0.001, setpoint = accheight)
dthrottle = 0

auto = False
contrast = False
corner = False
x = 320
throttle_offset = 0
#auto_throttle으로 쓰로틀 범위 제한
auto_throttle = [0.55, 0.59] 
minthrottle = 0.1

# task 3 
yaw = 0
prev_yaw = 0
count = 0
prev_time = time.time()
Get_into_position = False
avoidance_height = 300
avoidance_task = False
corner_offset = 0
height1 = 0
while running:
    pygame.event.pump()
    if joystick.get_button(10):
        car.throttle = 0.193
        break

    if joystick.get_button(4): # print sensorvalue
        if time.time() > autoprint + 1:
            if auto:
                auto = False
                auto = time.time()
                throttle = 0.24
            else:
                auto = True
                autoprint = time.time()
                throttle = 0.1

    if joystick.get_button(12):
        initIMU()
    if joystick.get_button(3):
        velocity = [0.0, 0.0]
        x_offset, y_offset = initvelocity()
    if joystick.get_button(7):
        throttle_offset = throttle_offset + 0.002
        print('throttle_offset : ', throttle_offset)
    elif joystick.get_button(6):
        throttle_offset = throttle_offset - 0.002
        print('throttle_offset : ', throttle_offset)

    if joystick.get_button(0):
        running = False

    if joystick.get_button(1):
        if t1 + 1 < time.time():  
            if auto:
                print('Manual mode')
                auto = False
                t1 = time.time()
            else:
                auto = True
                print('Auto mode')            
                t1 = time.time()
                
    #joystick 조종하면서 yaw값 출력 
    # gyro_data = mpu.readGyroscopeMaster()
    # gyro_z = gyro_data[2]
    # gyro_z_window.append(gyro_z)
    # gyro_z_sum = sum(gyro_z_window)
    # filtered_gyro_z = gyro_z_sum / len(gyro_z_window)            
    # current_time = time.time()
    # dt = current_time - prev_time
    # prev_time = current_time
    # yaw += filtered_gyro_z*dt
    # print('@@@@@@@@@@@yaw: ',yaw)
    if auto:
        steering = get_steering(x)
        #task3
        print("height1: ",height1)
        if height1 > avoidance_height:
            avoidance_task = True
            print("@@@@@@@@@@@@@@@task start@@@@@@@@@@@@@@@@@@")
          
        if avoidance_task == True:
            print("abs yaw: ",abs(prev_yaw - yaw))
            count = count+1
            if count==1:
                print("prev_yaw initialize!!")
                prev_yaw = yaw
            steering = steering_range[0]
            
            if abs(prev_yaw - yaw) >20:
                
                Get_into_position=True
                avoidance_task = False
                print("recovery start!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
               
        if Get_into_position == True :
            steering = steering_range[1]
            if abs(prev_yaw - yaw) <5:
                
                steering = (steering_range[0]+steering_range[1]*1.2)/2
                Get_into_position=False
                count = 0

        car.throttle = max(min(auto_throttle[0]+ dthrottle + throttle_offset + corner_offset, auto_throttle[1]),minthrottle)
        # car.throttle = 0.4+throttle_offset+corner_offset+dthrottle
        car.steering = steering 

        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
    
        gyro_data = mpu.readGyroscopeMaster()
        gyro_z = gyro_data[2]
        
        gyro_z_window.append(gyro_z)   #gyro_z 값을 deque를 통해 moving average filter 적용
        gyro_z_sum = sum(gyro_z_window)
        filtered_gyro_z = gyro_z_sum / len(gyro_z_window)
        dot_range = [240, 450]

        yaw += filtered_gyro_z*dt
        print('yaw: ',yaw)
        
        if len(clss) != 0:
            print("dthrottle: ",dthrottle)
        else:
            # print("height1 and dthrottle initiallized!!!")
            height1 = 0
            dthrottle = 0
        
        if (x > dot_range[1] or x < dot_range[0]) and abs(filtered_gyro_z) < 3:  #코너링 시 정지 감지
            if t8 + 1.0 <time.time():  
                corner = True
            else:
                corner = False
                corner_offset = 0
        else: 
            t8 = time.time()
            corner = False    
            
        if corner:
            if t9 + 0.6 < time.time():
                print("@@@@@@@@@@throttle up@@@@@@@@@@")
                print("filterd zyro z: ",filtered_gyro_z)
                corner_offset += 0.01
                t9 = time.time()
        else:
            t9 = time.time()
            
        # if not corner and corner_offset != 0:
        #     print("Corner exit detected, resetting throttle.")
        #     corner_offset = 0
              

    else:
        count = 0
        Get_into_position = False
        steering = joystick.get_axis(2)
        car.steering = steering - steering_offset
        car.throttle_gain = 0.6
        throttle = -joystick.get_axis(1)
        throttle = max(throttle_range[0], min(throttle_range[1], throttle))
        car.throttle = throttle + 0.19 / car.throttle_gain

    _, frame = camera.read()
    if contrast:
        frame = decrease_contrast(frame, 0.8)
    image = preprocess(frame, device)
    with torch.no_grad():
        output = model(image).detach().cpu().numpy()
    x, _ = output[0]
    x = (x / 2 + 0.5) * (width // downscale)
    


    if model_sign is not None: #YOLO 모델 로드 성공하면
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
                    height1 = int(abs(y1-y2))
                    print(height1, score)
                    if score > 0.75:
                        if height1 > 80 and height1 < avoidance_height:  
                            clss.append(class_name)
                            # print('height: ',height)
                            dthrottle = pid(height1)
                        if height1 > avoidance_height:
                            dthrottle = 0.0
                print('Classes:',clss)
                                
            t2 = time.time()
            
    
camera.release()
cv2.destroyAllWindows()


#export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD 
