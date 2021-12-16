import io
import numpy as np
import torch
torch.set_num_threads(1)
import torchaudio
import matplotlib
import matplotlib.pylab as plt
torchaudio.set_audio_backend("soundfile")
import pyaudio
global new_confidence1
global new_confidence2
new_confidence1=0
new_confidence2=0
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)
(get_speech_ts,
 get_speech_ts_adaptive,
 save_audio,
 read_audio,
 state_generator,
 single_audio_stream,
 collect_chunks) = utils

def validate(model,inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/abs_max
    sound = sound.squeeze() 
    return sound

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)
audio = pyaudio.PyAudio()

frames_to_record = 20 
frame_duration_ms = 250

from jupyterplot import ProgressPlot
import threading


continue_recording = True

def start_recording1(threadname):
    stream1 = audio.open(format=FORMAT,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=1)
    data1 = []
    voiced_confidences1 = []
    global new_confidence1
    continue_recording1 = True
    #pp1 = ProgressPlot(plot_names=["Primates Dev Detector"],line_names=["speech probabilities"], x_label="audio chunks")
    
    while continue_recording1:
        audio_chunk1 = stream1.read(int(SAMPLE_RATE * frame_duration_ms / 1000.0))
        data1.append(audio_chunk1)
        audio_int161 = np.frombuffer(audio_chunk1, np.int16)
        audio_float321 = int2float(audio_int161)
        vad_outs1 = validate(model, torch.from_numpy(audio_float321))
        new_confidence1 = vad_outs1[:,1].numpy()[0].item()
        voiced_confidences1.append(new_confidence1)
        #pp1.update(new_confidence1)
    #pp1.finalize()
    
    
from jupyterplot import ProgressPlot
import threading


def start_recording2(threadname):
    stream2 = audio.open(format=FORMAT,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=2)
    data2 = []
    voiced_confidences2 = []
    global new_confidence2
    continue_recording2 = True
    #pp2 = ProgressPlot(plot_names=["Primates Dev Detector"],line_names=["speech probabilities"], x_label="audio chunks")
    
    while continue_recording2:
        audio_chunk2 = stream2.read(int(SAMPLE_RATE * frame_duration_ms / 1000.0))
        data2.append(audio_chunk2)
        audio_int162 = np.frombuffer(audio_chunk2, np.int16)
        audio_float322 = int2float(audio_int162)
        vad_outs2 = validate(model, torch.from_numpy(audio_float322))
        new_confidence2 = vad_outs2[:,1].numpy()[0].item()
        voiced_confidences2.append(new_confidence2)
        #pp2.update(new_confidence2)
    #pp2.finalize()

import io
import socket
import struct
from PIL import Image
import cv2
import numpy
import sys


def start_webcam(threadname):

    i=0
    
    img1 = None
    img2 = None
    while True:
        #image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]

        '''
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        image_stream.seek(0)
        image = Image.open(image_stream)
        im1 = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
        ret, im2 = vid.read()


        image_stream2 = io.BytesIO()
        image_stream2.write(connection.read(image_len))
        image_stream2.seek(0)
        image2 = Image.open(image_stream2)
        im2 = cv2.cvtColor(numpy.array(image2), cv2.COLOR_RGB2BGR)
        '''


        if(new_confidence1>0.60 and new_confidence2>0.60):
            im1=cv2.imread('./speakerone.jpg')
            im2=cv2.imread('./speakertwo.jpg')
            images_1_2_h = np.hstack((im1, im2))
            cv2.imshow('Video',images_1_2_h)
            print("thisconditionworks")
            i=0
        if(new_confidence1>0.60 and new_confidence2<0.60):
            im1=cv2.imread('./speakerone.jpg')
            cv2.imshow('Video',im1)
            print("thisconditionworkstwo")
            i=1
        if(new_confidence2>0.60 and new_confidence1<0.60):
            im2=cv2.imread('./speakertwo.jpg')
            print("thisconditionworksthree")
            cv2.imshow('Video',im2)
            i=2
        if(new_confidence1<0.60 and new_confidence2<0.60):
            im3=cv2.imread('./noone.jpg')
            print("noworks")
            cv2.imshow('Video',im3)
            i=3

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




from threading import Thread
import time

thread4 = Thread(target=start_recording1, args=("Thread-1", ))
thread5 = Thread(target=start_recording2, args=("Thread-2", ))
thread7 = Thread(target=start_webcam, args=("webcam", ))
#thread9 = Thread(target=start_webcam2, args=("webcam2", ))
#thread8 = Thread(target=startw, args=("webcamm", ))


def startall():
    thread4.start()
    thread5.start()
    thread7.start()
    #thread9.start()
    #thread8.start()
    
startall()