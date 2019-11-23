import sys
from auth import MiBand3
#from constants import ALERT_TYPES
import time
import os
import threading

threadLock = threading.Lock()
class HeartRate():
    MAC_ADDR = "C9:A1:81:64:BC:BB"
    band = None
    value = 0
    breakFlag = False
    def __init__(self):
        print('Attempting to connect to ', self.MAC_ADDR)
        self.band = MiBand3(self.MAC_ADDR, debug=True)
    
    #update the heart rate. This is just a callback function to update the value in this class
    #you don't need to call it
    def __update_heart_rate(self,x):
        #print('Realtime heart BPM:', x)
        if x != 0:
            threadLock.acquire()
            self.value = x
            threadLock.release()

    def run(self):
        self.band.start_raw_data_realtime(heart_measure_callback=self.__update_heart_rate, heartRate=self)

    #return a string with the current heart rate value
    def get_my_heart_rate(self):
        while self.value == 0:
            time.sleep(0.1)            
        return self.value
            
    def start(self):
    # Authenticate the MiBand
        self.band.authenticate()
    
    def change_flag(self, value):
        self.breakFlag = value

    def close(self):
        self.band.stop_realtime()
        self.band.disconnect()

#termination
#request
#request number
def update_value(heartRateObject):
    heartRateObject.run()
    heartRateObject.close()
    print("Terminating heartbeat controller part 2")
    return

def process(heartRateObject, heartbeatQueue):
    while True:
        if not heartbeatQueue.empty():
            #termination case
            print('im ready to get item from queue')
            item = heartbeatQueue.get()
            if item == "terminate":
                threadLock.acquire()
                heartRateObject.breakFlag=True
                threadLock.release()
                print("Terminating heartbeat controller part 1")
                return
            if item == "request":
                value = heartRateObject.get_my_heart_rate()                
                heartbeatQueue.put(heartRateObject.get_my_heart_rate())
                while not heartbeatQueue.empty:
                    pass  
                    

def run(heartbeatQueue):    
    heartRateObject = HeartRate()
    heartRateObject.start()
    t1 = threading.Thread(target=update_value, args=[heartRateObject], name="update") #This thread is for updating heart rate value
    t2 = threading.Thread(target=process, args=(heartRateObject, heartbeatQueue), name="process") #This thread is for listening to the request from queue and interact with the request
    t1.start()
    t2.start()
    t2.join()
    t1.join()
