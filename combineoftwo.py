import numpy as np
import pyaudio
from pathlib import Path
import time
import argparse
import os
from reprint import output
from helpers import Interpolator, ratio_to_db, dbFS, rangemap
global db1
PREDICTION_THRES = 0.8 # confidence
DBLEVEL_THRES = -40 # dB

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = RATE
MICROPHONES_DESCRIPTION = []
FPS = 60.0
OUTPUT_LINES = 33

db1=0
db2=0
import config


output_lines = []*OUTPUT_LINES
audio_rms = 0
candidate = ("-",0.0)

interpolators1 = []
for k in range(31):
    interpolators1.append(Interpolator())
    
interpolators2 = []
for k in range(31):
    interpolators2.append(Interpolator())

def audio_samples(in_data, frame_count, time_info, status_flags):
    global graph
    global output_lines
    global interpolators1
    global audio_rms
    global candidate
    np_wav = np.fromstring(in_data, dtype=np.int16)
    rms1 = np.sqrt(np.mean(np_wav**2))
    db = dbFS(rms1)
    interp = interpolators1[30]
    interp.animate(interp.end, db, 1.0)
    return (in_data, pyaudio.paContinue)


def audio_samplestwo(in_data, frame_count, time_info, status_flags):
    global graph2
    global output_lines2
    global interpolators2
    global audio_rms
    global candidate
    np_wav = np.fromstring(in_data, dtype=np.int16) / 32768.0 # Convert to [-1.0, +1.0]
    rms2 = np.sqrt(np.mean(np_wav**2))
    db2 = dbFS(rms2)
    interp2 = interpolators2[30]
    interp2.animate(interp2.end, db2, 1.0)
    return (in_data, pyaudio.paContinue)

def helloone(threadname):
    p1 = pyaudio.PyAudio()
    stream1 = p1.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=audio_samples, input_device_index=11)
    stream1.start_stream()
    while stream1.is_active():
        with output(initial_len=OUTPUT_LINES, interval=0) as output_lines:
            while True:
                time.sleep(1.0/FPS) # 60fps
                print(config.db1)
                interp = interpolators1[30]
                db1 = interp.update()
                val = rangemap(db1, -50, 0, 0, 100)
                config.db1=db1
                

def hello(threadname):                
    global db2
    p2 = pyaudio.PyAudio()
    stream2 = p2.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=audio_samplestwo, input_device_index=12)
    stream2.start_stream()
    while stream2.is_active():
        with output(initial_len=OUTPUT_LINES, interval=0) as output_lines:
            while True:
                time.sleep(1.0/FPS) # 60fps
                interp2 = interpolators2[30]
                db2 = interp2.update()
                val = rangemap(db2, -50, 0, 0, 100)
                config.db2=db2
