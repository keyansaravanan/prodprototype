import numpy as np
import pyaudio
from pathlib import Path
import time
import argparse
import os
from reprint import output
from helpers import Interpolator, ratio_to_db, dbFS, rangemap
global db1
# thresholds
PREDICTION_THRES = 0.8 # confidence
DBLEVEL_THRES = -40 # dB

# Variables
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = int(RATE / 10)
MICROPHONES_DESCRIPTION = []
FPS = 60.0
OUTPUT_LINES = 33

###########################
# Model download

db1=0
db2=0
import config
###########################
# Check Microphone
###########################
print("=====")
print("1 / 2: Checking Microphones... ")
print("=====")

import microphones
desc, mics, indices = microphones.list_microphones()
if (len(mics) == 0):
    print("Error: No microphone found.")
    exit()





##############################
# Setup Audio Callback
##############################
output_lines = []*OUTPUT_LINES
audio_rms = 0
candidate = ("-",0.0)

# Prediction Interpolators
interpolators1 = []
for k in range(31):
    interpolators1.append(Interpolator())
    
interpolators2 = []
for k in range(31):
    interpolators2.append(Interpolator())

# Audio Input Callback
def audio_samples(in_data, frame_count, time_info, status_flags):
    global graph
    global output_lines
    global interpolators1
    global audio_rms
    global candidate
    np_wav = np.fromstring(in_data, dtype=np.int16) / 32768.0 # Convert to [-1.0, +1.0]

    # Compute RMS and convert to dB
    rms1 = np.sqrt(np.mean(np_wav**2))
    db = dbFS(rms1)
    interp = interpolators1[30]
    interp.animate(interp.end, db, 1.0)
    '''
    # Make Predictions
    x = waveform_to_examples(np_wav, RATE)
    predictions = []
    with graph.as_default():
        if x.shape[0] != 0:
            x = x.reshape(len(x), 96, 64, 1)
            pred = model.predict(x)
            predictions.append(pred)

        for prediction in predictions:
            m = np.argmax(prediction[0])
            candidate = (ubicoustics.to_human_labels[label[m]],prediction[0,m])
            num_classes = len(prediction[0])
            for k in range(num_classes):
                interp = interpolators[k]
                prev = interp.end
                interp.animate(prev,prediction[0,k],1.0)
    '''
    return (in_data, pyaudio.paContinue)


# Audio Input Callback
def audio_samplestwo(in_data, frame_count, time_info, status_flags):
    global graph2
    global output_lines2
    global interpolators2
    global audio_rms
    global candidate
    np_wav = np.fromstring(in_data, dtype=np.int16) / 32768.0 # Convert to [-1.0, +1.0]

    # Compute RMS and convert to dB
    rms2 = np.sqrt(np.mean(np_wav**2))
    db2 = dbFS(rms2)
    interp2 = interpolators2[30]
    interp2.animate(interp2.end, db2, 1.0)
    '''
    # Make Predictions
    x = waveform_to_examples(np_wav, RATE)
    predictions = []
    with graph.as_default():
        if x.shape[0] != 0:
            x = x.reshape(len(x), 96, 64, 1)
            pred = model.predict(x)
            predictions.append(pred)

        for prediction in predictions:
            m = np.argmax(prediction[0])
            candidate = (ubicoustics.to_human_labels[label[m]],prediction[0,m])
            num_classes = len(prediction[0])
            for k in range(num_classes):
                interp = interpolators[k]
                prev = interp.end
                interp.animate(prev,prediction[0,k],1.0)
    '''
    return (in_data, pyaudio.paContinue)


p1 = pyaudio.PyAudio()

stream1 = p1.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=audio_samples, input_device_index=11)

##############################
# Start Non-Blocking Stream
##############################
#print("# Live Prediction Using Microphone: %s" % (mic_desc))
stream1.start_stream()
while stream1.is_active():
    with output(initial_len=OUTPUT_LINES, interval=0) as output_lines:
        while True:
            time.sleep(1.0/FPS) # 60fps
            #for k in range(30):
                #interp = interpolators[k]
                #val = interp.update()
                #bar = ["|"] * int((val*100.0))
                #output_lines[k] = "%20s: %.2f %s" % (ubicoustics.to_human_labels[label[k]], val, "".join(bar))

            # dB Levels
            interp = interpolators1[30]
            db1 = interp.update()
            val = rangemap(db1, -50, 0, 0, 100)
            config.db1=db1
            print(db1)
                #if(db1>=-28):
                #print(db1)
                #print("Person is Near")
            #bar = ["|"] * min(100,int((val)))
            #output_lines[30] = "%20s: %.1fdB [%s " % ("Audio Level", db, "".join(bar))

            # Display Thresholds
            #output_lines[31] = "%20s: confidence = %.2f, db_level = %.1f" % ("Thresholds", PREDICTION_THRES, DBLEVEL_THRES)

            # Final Prediction
            
            pred = "-"
            event,conf = candidate
            if (conf > PREDICTION_THRES and db > DBLEVEL_THRES):
                pred = event
            output_lines[32] = "%20s: %s" % ("Prediction", pred.upper())
            

def hello(threadname):                
    global db2
    ##############################
    # Setup Audio
    ##############################
    p2 = pyaudio.PyAudio()
    stream2 = p2.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=audio_samplestwo, input_device_index=2)

    ##############################
    # Start Non-Blocking Stream
    ##############################
    os.system('cls' if os.name == 'nt' else 'clear')
    #print("# Live Prediction Using Microphone: %s" % (mic_desc))
    stream2.start_stream()
    while stream2.is_active():
        with output(initial_len=OUTPUT_LINES, interval=0) as output_lines:
            while True:
                time.sleep(1.0/FPS) # 60fps
                #for k in range(30):
                    #interp = interpolators[k]
                    #val = interp.update()
                    #bar = ["|"] * int((val*100.0))
                    #output_lines[k] = "%20s: %.2f %s" % (ubicoustics.to_human_labels[label[k]], val, "".join(bar))

                # dB Levels
                interp2 = interpolators2[30]
                db2 = interp2.update()
                val = rangemap(db2, -50, 0, 0, 100)
                config.db2=db2
                #if(db2>=-28):
                    #print(db2)
                    #print("Person2 is Near")
                #bar = ["|"] * min(100,int((val)))
                #output_lines[30] = "%20s: %.1fdB [%s " % ("Audio Level", db, "".join(bar))

                # Display Thresholds
                #output_lines[31] = "%20s: confidence = %.2f, db_level = %.1f" % ("Thresholds", PREDICTION_THRES, DBLEVEL_THRES)

                # Final Prediction
                '''
                pred = "-"
                event,conf = candidate
                if (conf > PREDICTION_THRES and db > DBLEVEL_THRES):
                    pred = event
                output_lines[32] = "%20s: %s" % ("Prediction", pred.upper())
                '''
