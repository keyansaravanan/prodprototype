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

def validate(model,
             inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs
 
import config
stream1 = audio.open(format=FORMAT,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=1)
data1 = []
voiced_confidences1 = []
continue_recording1 = True
#pp1 = ProgressPlot(plot_names=["Primates Dev Detector"],line_names=["speech probabilities"], x_label="audio chunks")
outs = []
to_concat = []
while continue_recording1:
    audio_chunk1 = stream1.read(int(SAMPLE_RATE * frame_duration_ms / 1000.0))
    to_concat.append(audio_chunk1.unsqueeze(0))
    chunks = torch.Tensor(torch.cat(to_concat, dim=0)).to(device)
    out = validate(model, chunks)
    print(out)
