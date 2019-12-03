import queue as qu
import threading as th
import multiprocessing as mp
import time

from flask import Flask, render_template
from flask_ask import Ask, statement, question

import openCVController
import heartbeatController
import chatController
import aprilTags

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
    return statement('I\'m following you now. Please do not run')

@ask.intent('lookAtMe')
def lookAtMe():
    cvQueue.put('eyeballFollow')
    cvQueue.join()
    return statement('I can see you')

@ask.intent('stopActing')
def stopActing():
    for queue in allQueues:
        queue.put("terminate")
    for thread in allThreads:
        thread.join()
    for thread in allThreads:
        thread.start()
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
    return statement('Distress signal sent at ' + str(x_cord) + ", " + str(z_cord))

@ask.intent('getHeartbeat')
def getHeartbeat():
    heartbeatQueue.put("request")
    while not heartbeatQueue.empty():
        pass
    heartrate = heartbeatQueue.get()
    return statement("Your heartrate is " + str(heartrate) + " beats per minute")

@ask.intent('takePhoto')
def takePhoto():
    cvQueue().put('photo')
    cvQueue.join()
    return("Your photo will be taken and emailed to you")

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
            elif request == "receivedDistress":
                x_cord = chatQueue.get()
                z_cord = chatQueue.get()
                chatQueue.task_done()
                chatQueue.task_done()
                final_target  = aprilTags.getClosestTag(x_cord, z_cord)
                cvQueue.put("aprilFollow")
                cvQueue.put(final_target[0])
                cvQueue.put(final_target[1])
                cvQueue.join()

if __name__ == '__main__':
    print("main")
    alexaTh = th.Thread(target= app.run)
    alexaTh.daemon = True
    cvTh = mp.Process(target= openCVController.run, args= (cvQueue, ))
    heartbeatTh = th.Thread(target= heartbeatController.run, args= (heartbeatQueue, ))
    chatTh = th.Thread(target= chatController.run, args= (chatQueue, ))
    chatQueueCheckTh = th.Thread(target= checkChatQueue)

    print("allth")
    allThreads = []
    allThreads.append(cvTh)
    allThreads.append(heartbeatTh)
    allThreads.append(chatTh)
    allThreads.append(chatQueueCheckTh)

    print("b4")
    alexaTh.start()

    print("test1")

    for thread in allThreads:
        print("ima thread")
        thread.start()
    
    print('ready for opperation')

    for thread in allThreads:
        thread.join()
    
    print("exiting")