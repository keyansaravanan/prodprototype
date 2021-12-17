from threading import Thread
import time
from combinedone import start_recording1
from combinedone import start_recording2
from combinedone import start_webcam
from combineoftwo import hello
from combineoftwo import helloone
from combinedone import decisionblockforcameraone
from combinedone import decisionblockforcameratwo

thread4 = Thread(target=start_recording1, args=("Thread-1", ))
thread5 = Thread(target=start_recording2, args=("Thread-2", ))
thread7 = Thread(target=start_webcam, args=("webcam", ))
thread9 = Thread(target=hello, args=("webcam2", ))
thread8 = Thread(target=helloone, args=("webcamm", ))
thread10 = Thread(target=decisionblockforcameraone, args=("decsionone",))
thread11 = Thread(target=decisionblockforcameratwo, args=("decisiontwo",))


def startall():
    thread7.start()
    thread4.start()
    time.sleep(10)
    thread5.start()
    time.sleep(5)
    thread9.start()
    time.sleep(5)
    thread8.start()
    print("Mic's all started")
    thread10.start()
    thread11.start()
    
    
startall()
