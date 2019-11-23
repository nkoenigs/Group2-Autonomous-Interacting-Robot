import queue as qu
import threading as th
import multiprocessing as mp
import time

from flask import Flask, render_template
from flask_ask import Ask, statement, question

import openCVController
import heartbeatController
import chatController

# this file calls all others as threads then moves data from queue

# "terminate" - end the thread please
# "halt", "personFollow", "aprilFollow", "eyeballFollow"
# "request"

app = Flask(__name__)
ask = Ask(app, '/')

cvQueue = mp.JoinableQueue()
heartbeatQueue = qu.Queue()
chatQueue = qu.Queue()
checkChatQueueQueue = qu.Queue()

allQueues = []
allQueues.append(cvQueue)
allQueues.append(heartbeatQueue)
allQueues.append(chatQueue)
allQueues.append(checkChatQueueQueue)

@app.route('/')
def hello_world():
	return "Flask Server is Online!"

@ask.launch
def launched():
	return question("hello. what would you like me to do?").reprompt(
		"if you don't need me, please put me to sleep.")


@ask.intent('AMAZON.FallbackIntent')
def default():
	return question("Sorry, I don't understand that command. What would you like me to do?").reprompt(
		"What would you like me to do now?")

@ask.intent('SleepIntent')
def sleep():
	return statement('goodnight')

@ask.intent('terminate')
def terminate():
    for queue in allQueues:
        queue.put("terminate")
    return statement('processes are terminated')

@ask.intent('testComunication')
def testComunications():
	return statement('Jetson Backend is opperational. If it was not	I would not be alive')

@ask.intent('trackMe')
def trackMe():
    cvQueue.put("personFollow")
    cvQueue.join()
    return statement('Im following you now. Please do not run')

@ask.intent('lookAtMe')
def lookAtMe():
    cvQueue.put('eyeballFollow')
    cvQueue.join()
    return statement('I can see you')

@ask.intent('stopActing')
def stopActing():
    cvQueue.put("halt")
    cvQueue.join()
    return statement('Im done following you')

@ask.intent('callForHelp')
def callForHelp():
    # ask CV for my coords
    cvQueue.put("getCoordinates")
    cvQueue.join()
    x_cord = cvQueue.get()
    z_cord = cvQueue.get()
    cvQueue.task_done()
    cvQueue.task_done()
    # pass my coords to the chat server
    chatQueue.put("sendDistress")
    chatQueue.put(x_cord)
    chatQueue.put(z_cord)
    chatQueue.join()
    return statement('distress signal sent')

@ask.intent('getHeartbeat')
def getHeartbeat():
    heartbeatQueue.put("request")
    while not heartbeatQueue.empty():
        pass
    heartrate = heartbeatQueue.get()
    return statement("Your heartrate is " + str(heartrate))

def checkChatQueue():
    threadActive = True
    while threadActive:
        time.sleep(.1)
        if not checkChatQueueQueue.empty():
            if checkChatQueueQueue.get() == "terminate":
                threadActive = False
        if not chatQueue.empty():
            request = chatQueue.get()
            chatQueue.task_done()
            if request == "terminate":
                chatQueue.put(request)
            elif request == "recevedDistress":
                x_cord = chatQueue.get()
                z_cord = chatQueue.get()
                chatQueue.task_done()
                chatQueue.task_done()
                cvQueue.put("aprilFollow")
                cvQueue.put(x_cord)
                cvQueue.put(z_cord)
                cvQueue.join()
                # do the stuff to handle cv return after reaching dest

if __name__ == '__main__':
    alexaTh = th.Thread(target= app.run)
    cvTh = mp.Process(target= openCVController.run, args= (cvQueue, ))
    heartbeatTh = th.Thread(target= heartbeatController.run, args= (heartbeatQueue, ))
    chatTh = th.Thread(target= chatController.run, args= (chatQueue, ))
    chatQueueCheckTh = th.Thread(target= checkChatQueue)

    allThreads = []
    allThreads.append(alexaTh)
    allThreads.append(cvTh)
    allThreads.append(heartbeatTh)
    allThreads.append(chatTh)
    allThreads.append(chatQueueCheckTh)

    for thread in allThreads:
        try:
            thread.start()
        except:
            print("error starting thread")

    for thread in allThreads:
        thread.join()