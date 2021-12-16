import io
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn.functional as F
from typing import List
from itertools import repeat
from collections import deque
import time
import pyaudio
global new_confidence1
global new_confidence2
new_confidence1=0
new_confidence2=0
import config
import onnxruntime

import io
import socket
import struct
from PIL import Image
import cv2
import numpy
import sys
import time
import threading

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 44100
CHUNK = 1024
audio = pyaudio.PyAudio()

frames_to_record = 20 
frame_duration_ms = 250



continue_recording = True

def init_onnx_model(model_path: str):
    return onnxruntime.InferenceSession(model_path)
    

def validate_onnx(model, inputs):
    with torch.no_grad():
        ort_inputs = {'input': inputs.cpu().numpy()}
        outs = model.run(None, ort_inputs)
        outs = [torch.Tensor(x) for x in outs]
    return outs[0]

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/abs_max
    sound = sound.squeeze() 
    return sound



def get_speech_ts(wav: torch.Tensor,
                  model,
                  trig_sum: float = 0.25,
                  neg_trig_sum: float = 0.07,
                  num_steps: int = 8,
                  batch_size: int = 200,
                  num_samples_per_window: int = 4000,
                  min_speech_samples: int = 10000, #samples
                  min_silence_samples: int = 500,
                  run_function=validate_onnx,
                  visualize_probs=False,
                  smoothed_prob_func='mean',
                  device='cpu'):

    assert smoothed_prob_func in ['mean', 'max'],  'smoothed_prob_func not in ["max", "mean"]'
    num_samples = num_samples_per_window
    assert num_samples % num_steps == 0
    step = int(num_samples / num_steps)  # stride / hop
    outs = []
    to_concat = []

    for i in range(0, len(wav),step):
        chunk = wav[i: i+num_samples]
        if len(chunk) < num_samples:
            chunk = F.pad(chunk, (0, num_samples - len(chunk)))
        to_concat.append(chunk.unsqueeze(0))
        if len(to_concat) >= batch_size:
            chunks = torch.Tensor(torch.cat(to_concat, dim=0)).to(device)
            out = run_function(model, chunks)
            outs.append(out)
            to_concat = []

    if to_concat:
        chunks = torch.Tensor(torch.cat(to_concat, dim=0)).to(device)
        out = run_function(model, chunks)
        outs.append(out)

    outs = torch.cat(outs, dim=0)

    buffer = deque(maxlen=num_steps)  # maxlen reached => first element dropped
    triggered = False
    speeches = []
    current_speech = {}
    if visualize_probs:
      import pandas as pd
      smoothed_probs = []

    speech_probs = outs[:, 1]  # this is very misleading
    return speech_probs

def start_recording1(threadname):
    import config
    model = init_onnx_model(model_path='./model.onnx')
    stream1 = audio.open(format=FORMAT,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=11)
    data1 = []
    global new_confidence1
    continue_recording1 = True    
    while continue_recording1:
        audio_chunk1 = stream1.read(int(SAMPLE_RATE * frame_duration_ms / 1000.0),exception_on_overflow = False)
        data1.append(audio_chunk1)
        audio_int161 = np.frombuffer(audio_chunk1, np.int16)
        audio_float321 = int2float(audio_int161)
        vad_outs1=get_speech_ts(torch.from_numpy(audio_float321), model, num_steps=4, run_function=validate_onnx)
        na = vad_outs1.numpy()
        config.nc1=np.max(na)
        print("Mic1",config.nc1)

def start_recording2(threadname):
    import config
    model2 = init_onnx_model(model_path='./model.onnx')
    stream2 = audio.open(format=FORMAT,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=12)
    data2 = []
    global new_confidence2
    continue_recording2 = True    
    while continue_recording2:
        audio_chunk2 = stream2.read(int(SAMPLE_RATE * frame_duration_ms / 1000.0),exception_on_overflow = False)
        data2.append(audio_chunk2)
        audio_int162 = np.frombuffer(audio_chunk2, np.int16)
        audio_float322 = int2float(audio_int162)
        vad_outs2=get_speech_ts(torch.from_numpy(audio_float322), model2, num_steps=4, run_function=validate_onnx)
        na2 = vad_outs2.numpy()
        config.nc2=np.max(na2)
        print("Mic2",config.nc2)


def decisionblockforcameraone(threadname):
    while True:
        import config
        if(config.nc1>0.72):
            if(config.db1>-28):
                config.cameraoneon=True
                while(config.nc1>0.75):
                    config.cameraoneon=True
                    time.sleep(2)
                    if(config.nc1<0.75):
                        time.sleep(5)
                time.sleep(5)
        else:
            config.cameraoneon=False
def decisionblockforcameratwo(threadname):
    while True:
        import config
        if(config.nc2>0.72):
            if(config.db2>-28):
                config.cameratwoon=True
                while(config.nc2>0.75):
                    config.cameratwoon=True
                    time.sleep(2)
                    if(config.nc2<0.75):
                        time.sleep(5)
                time.sleep(5)
        else:
            config.cameratwoon=False    
        
def start_webcam(threadname):

    i=0
    
    img1 = None
    img2 = None
    global db1
    global db2
    '''
    server_socket = socket.socket()
    server_socket.bind(('0.0.0.0', 5454))  
    server_socket.listen(0)
    print("Listening")
    connection = server_socket.accept()[0].makefile('rb')
    vid = cv2.VideoCapture(0
    '''
    #vid1=cv2.VideoCapture(0)
    #vid2=cv2.VideoCapture(1)
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
        

        from combineoftwo import db1
        from combineoftwo import db2
        print("db1:",db1)
        print("db2:",db2)
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break

        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        image_stream.seek(0)
        image = Image.open(image_stream)
        im1 = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
        ret, im2 = vid.read()
        '''

        #rval1, im1 = vid1.read()
        #rval2, im2 = vid2.read()
        import config
        print("db1:",config.db1)
        print("db1:",config.db2)
        print("nc1:",config.nc1)
        print("nc2:",config.nc2)
        if(config.cameraoneon==True and config.cameratwoon==True):
            im1=cv2.imread('./speakerone.jpg')
            im2=cv2.imread('./speakertwo.jpg')
            images_1_2_h = np.hstack((im1, im2))
            cv2.imshow('Video',images_1_2_h)
            #print("thisconditionworks")
            i=0
        if(config.cameraoneon==True and config.cameratwoon==False):
            im1=cv2.imread('./speakerone.jpg')
            cv2.imshow('Video',im1)
            #print("thisconditionworkstwo")
            i=1
        if(config.cameratwoon==True and config.cameraoneon==False):
            im2=cv2.imread('./speakertwo.jpg')
            #print("thisconditionworksthree")
            cv2.imshow('Video',im2)
            i=2
        if(config.cameraoneon==False and config.cameratwoon==False):
            im3=cv2.imread('./noone.jpg')
            #print("noworks")
            cv2.imshow('Video',im3)
            i=3

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




