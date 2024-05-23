import cv2
import pygame
from jetracer.nvidia_racecar import NvidiaRacecar
import datetime
import time
import os

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
# lowbattery
# throttle_gain_range = [0.60,0.64]
throttle_gain_range = [0.68,0.75]
# battery charged
# throttle_gain_range = [0.58 ,0.63]
car.throttle_gain = 0
car.throttle = 0
steering_offset = 0.290
before = 0
now_steering = 0



#define model
import torch
import torchvision
import PIL.Image
from cnn.center_dataset import TEST_TRANSFORMS
import numpy as np

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
def decrease_contrast(image,factor):
    if image.dtype == np.uint8:
        max_value = 255
    elif image.dtype ==np.uint16:
        max_value = 65535
    elif image.dtype == np.float32:
        max_value = 1.0
    else:
        raise TypeError("대비 변환실패")
    new_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    # 결과 이미지가 원본 데이터 타입과 같도록 조정
    if image.dtype == np.float32:
        new_image = new_image.astype(np.float32) / max_value
    return new_image
def get_steering(x):
    dot_range = [240, 450]
    if x> dot_range[1]:
        newsteering = steering_range[1]
    elif x< dot_range[0]:
        newsteering = steering_range[0]
    else:
        newsteering = min(steering_range[1],max(((x-320)/400),steering_range[0])) - steering_offset
    car.throttle_gain = min(throttle_gain_range[1],
                            max(throttle_gain_range[0],
                                (throttle_gain_range[1]-throttle_gain_range[0])*abs(x-320)/60)+throttle_gain_range[0])
    return newsteering

device = torch.device('cuda')
model = get_model()
model.load_state_dict(torch.load('intersection_right.pth'))
model = model.to(device)

# option True or False
t0 = 0
t1 = 0
monitor = False
auto = False
contrast = False
    
x = 320
while running:
    pygame.event.pump()

    if joystick.get_button(7):
        throttle_offset =throttle_offset+0.002
        print('throttle_offset : ',throttle_offset)
    elif joystick.get_button(6):
        throttle_offset  = throttle_offset -0.002
        print('throttle_offset : ',throttle_offset)
        
    if joystick.get_button(0):
        running = False

    if joystick.get_button(1):
        if t1+1 < time.time():  
            if auto:
                print('Manual mode')
                auto = False
                t1 = time.time()
            else:
                auto = True
                print('Auto mode')            
                t1 = time.time()
    if auto:
        steering= get_steering(x)
        car.throttle = 0.5
        car.steering = steering
    else:
        steering = joystick.get_axis(2)
        car.steering = steering - steering_offset
        car.throttle_gain = 0.6
        throttle = -joystick.get_axis(1)
        throttle = max(throttle_range[0], min(throttle_range[1], throttle))
        car.throttle = throttle+0.19/car.throttle_gain
    _,frame = camera.read()
    if contrast:
        frame = decrease_contrast(frame,0.8)
    image = preprocess(frame,device)
    with torch.no_grad():
        output = model(image).detach().cpu().numpy()
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


camera.release()
cv2.destroyAllWindows()

