import io
import numpy as np
import torch
torch.set_num_threads(1)
#import torchaudio
import matplotlib
import matplotlib.pylab as plt
#torchaudio.set_audio_backend("soundfile")
import pyaudio
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
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)
audio = pyaudio.PyAudio()

frames_to_record = 20 
frame_duration_ms = 250



continue_recording = True

def init_onnx_model(model_path: str):
    return onnxruntime.InferenceSession(model_path)
    
model = init_onnx_model(model_path='./model.onnx')


def validate(model,inputs: torch.Tensor):
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
  
import config
stream1 = audio.open(format=FORMAT,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=2)
data1 = []
voiced_confidences1 = []
continue_recording1 = True
#pp1 = ProgressPlot(plot_names=["Primates Dev Detector"],line_names=["speech probabilities"], x_label="audio chunks")

while continue_recording1:
    audio_chunk1 = stream1.read(int(SAMPLE_RATE * frame_duration_ms / 1000.0))
    data1.append(audio_chunk1)
    print(data1)
