"""Use a pre-trained model to drive the car.

Based on code from Udacity.
"""
import argparse
import base64
from collections import deque
from datetime import datetime
import os
import time
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import tqdm

from nnUtils import loadModel

from gauges import RotaryScale, root

sio = socketio.Server()
app = Flask(__name__)

maxThrottle = 2
minThrottle = -1
minSpeed = 1.0
slowdownFactor = 2
baseSpeedTarget = 20  # Can do 30 on the main track; need more like 20 for the jungle.


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.
        self.verbose = False
        self.setPointGauge = RotaryScale(
            max_value=30, unit='MPH', name='Speed Set Point',
        )

    def set_desired(self, desired):
        self.set_point = desired
        self.setPointGauge.set_value(desired)

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement
        if self.verbose: print('PI error = %s - %s = %s' % (self.set_point, measurement, self.error))

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral

    def reset(self):
        '''Reset the integrator. Combats integral windup.'''
        self.integral = 0


class Smoother(object):

    def __init__(self, initial=0, callback=lambda mean: mean, **kwargs):
        self.q = deque(**kwargs)
        self.callback = callback
        self(initial)

    def __call__(self, value):
        self.q.append(value)
        return self.callback(self.value)

    def __len__(self):
        return len(self.q)

    @property
    def value(self):
        return np.mean(self.q)


# Make persistent objects.
pbar = tqdm.tqdm(unit='frames')

steeringAngleGauge = RotaryScale(
    max_value=180., 
    unit='°',
    name='Steering Angle',
    img_data='emptyGauge',
    angleDirect=True,
    needle_color='red',
)

controller = SimplePIController(0.05, 0.001)
controller.set_desired(baseSpeedTarget)

throttleGauge = RotaryScale(
    max_value=maxThrottle,
    min_value=maxThrottle/-2.,
    unit='',
    name='Throttle',
)

setPointSmoother = Smoother(maxlen=60, callback=controller.set_desired)

@sio.on('telemetry')
def telemetry(sid, data):
    pbar.update()
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        # PIL is RGBA, I think.
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)

        inp = image_array[None, :, :, :]
        pred = model.predict(inp, batch_size=1)
        _steering = pred.ravel()[0]
        #_steering, _throttle, _brake = pred.ravel()
        steering_angle = float(_steering)
        
        # If we've been turning sharply lately, decrease speed.
        speedTarget = float(baseSpeedTarget)
        speedTarget *= np.abs(1 - slowdownFactor * np.abs(steering_angle))
        setPointSmoother(max(speedTarget, minSpeed))

        # It looks like angles actually aren't reported in radians or degrees.
        degsPerUnit = 25.
        steeringAngleGauge.set_value(
            degsPerUnit * steering_angle + 90. + 180,
            degsPerUnit * steering_angle
        )
        throttle = min(controller.update(float(speed)), maxThrottle)
        if throttle == maxThrottle:
            print('Stuck on a bump?')
            # If we've reached maxThrottle, probably it's because we're stuck,
            # and there was integral windup. Reset the integral term,
            # but also give a sharp negative jostle to unstick the car.
            # Sometimes it takes more than one.
            controller.reset()
            send_control(steering_angle, -4*throttle)
            # Allow a short application period for the jolt to take effect.
            time.sleep(.2)
        elif throttle == minThrottle:
            # Try not to send us careening backwards while ascending hills.
            controller.reset()
        else:
            throttleGauge.set_value(throttle)
            send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        controller.reset()
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)
    root.update()


@sio.on('connect')
def connect(sid, environ):
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    model = loadModel(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
