from threading import Thread
import time
from combinedone import start_recording1
from combinedone import start_recording2


thread5 = Thread(target=start_recording2, args=("Thread-2", ))
thread5.start()
