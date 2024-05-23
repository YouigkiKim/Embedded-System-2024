
import os
import pygame
from jetracer.nvidia_racecar import NvidiaRacecar
import cv2 
import datetime
import time




# from jetcam.csi_camera import CSICamera
# camera = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30)
 
car = NvidiaRacecar()

# For headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()
running = True
throttle_range = (-0.3, 0.3)
i = 0

# init camera
sensor_id = 0
downscale = 2
width, height = (1280, 720)
_width, _height = (width // downscale, height // downscale)
frame_rate = 30
flip_method = 0
contrast = 1.3
brightness = 0.2

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
t0 = 0
t1=0
record = False
i = 0
while running:
    pygame.event.pump()

    throttle = -joystick.get_axis(1)
    throttle = max(throttle_range[0], min(throttle_range[1], throttle))

    steering = joystick.get_axis(2)
    
    
    if joystick.get_button(10):
    # if (i%100000==0):
        if t0+0.5 <time.time():
            if record == False:    
                t0 = time.time()
                record = True
                # print(record, i)
                print('no.image: ',i)
            # timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            # image_path = f'pracdataset/{timestamp}.jpg'
            # _, frame = camera.read()
            # cv2.imwrite(image_path, frame)
            elif record == True:
                record = False
                t0 = time.time()
                print(record)

    if record:
        if t1+0.3 < time.time():
            print("chal kak")
            i=i+1
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            image_path = f'dataset/05202302_straight/{timestamp}.jpg'
            _, frame = camera.read()
            cv2.imwrite(image_path, frame)
            # cv2.imshow(frame)
            t1=time.time()
        
    
    car.steering = steering-0.296
    print(car.steering)
    car.throttle_gain = 0.6
    car.throttle = throttle+0.19/car.throttle_gain

    if joystick.get_button(11): # start button
        camera.release()
        running = False
