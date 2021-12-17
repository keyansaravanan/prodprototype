from threading import Thread
import time
from combinedone import start_recording1
from combinedone import start_recording2
from combinedone import start_webcam
from combineoftwo import hello
from combineoftwo import helloone
from combinedone import decisionblockforcameraone
from combinedone import decisionblockforcameratwo

thread1 = Thread(target=start_recording1, args=("Thread-1", ))
thread2 = Thread(target=start_recording2, args=("Thread-2", ))
thread3 = Thread(target=hello, args=("Thread-3", ))
thread4 = Thread(target=helloone, args=("Thread-4", ))
thread5 = Thread(target=decisionblockforcameraone, args=("Thread-5", ))
thread6 = Thread(target=decisionblockforcameratwo, args=("Thread-6", ))
thread7 = Thread(target=start_webcam, args=("Thread-7", ))


thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()
thread6.start()
thread7.start()
