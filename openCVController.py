# import the necessary packages
import sys  # command line lib
import glob
import math
import argparse
import threading
from collections import deque
from queue import Queue

from scipy import ndimage  # median filter
from threading import Thread
from imutils.video import WebcamVideoStream
import numpy as np
import cv2
import imutils
import serial
import serial.tools.list_ports  # for listing serial ports
import time
import os
from enum import Enum
import pygame
from pygame.locals import *
import json
import apriltag


# Class/Enums to keep track of directions sent or to send down to Arduino via Serial communication
class Direction(Enum):
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4
    STOP = 5


# Main class of the file
class OpenCVController:
    # PyGame constants
    RGB_WHITE = (255, 255, 255)
    RGB_BLACK = (0, 0, 0)
    EYEBALL_RADIUS = 30

    # Font for text on webcam display
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Define lower & upper boundaries in HSV color space (for object following/tracking)
    greenLower = np.array([48, 100, 50])
    greenUpper = np.array([85, 255, 255])

    def __init__(self):
        self.serialPort = self.com_connect() if True else None  # TODO True for running - False for testing
        # Variables to hold last command sent to Arduino and when it was sent (epoch seconds)
        self.lastCommandSentViaSerial = None
        self.lastCommandSentViaSerialTime = None
        # Connect to webcam video source and allow camera to warm up......
        self.vs = WebcamVideoStream(src=1).start()
        time.sleep(2.0)
        # TODO need some way to kill my loops nicely - how about these 2 lines below?
        # self.runningPersonFollowing = False
        # self.runningAprilFollowing = False

    # TODO remove the timeout in serial setting ... confirm it doesn't do anythin, then add write_timeout=0 (
    #  nonblocking)
    @staticmethod
    def com_connect():
        ser = None
        connection_made = False
        while not connection_made:
            if os.path.exists('/dev/ttyUSB0'):
                connection_made = True
                ser = serial.Serial('/dev/ttyUSB0', 9600, write_timeout=0)
                print("Connected to Serial")
            if os.path.exists('/dev/ttyACM1'):
                connection_made = True
                ser = serial.Serial('/dev/ttyACM1', 115200, timeout=1)
        return ser

    # TODO there's a section of this code that needs to be uncommented
    def send_serial_command(self, direction_enum, dataToSend):
        # If this command is different than the last command sent, then we should sent it
        # Or if it's the same command but it's been 1 second since we last sent a command, then we should send it
        if self.serialPort is not None:
            if self.lastCommandSentViaSerial != direction_enum:
                self.serialPort.write(dataToSend)
                self.lastCommandSentViaSerial = direction_enum
                self.lastCommandSentViaSerialTime = time.time()
            # elif (time.time() - self.lastCommandSentViaSerialTime > 1):
            #    self.serialPort.write(dataToSend)
            #    self.lastCommandSentViaSerialTime = time.time()
            else:
                pass  # Do nothing - same command sent recently

    def cleanup_resources(self):
        self.vs.stream.release()
        cv2.destroyAllWindows()
        self.vs.stop()

    def april_following(self, desiredTag, desiredDistance, cvQueue: Queue):

        # Frame is considered to be 600x600 (after resize)
        # Below are variables to set what we consider center and in-range
        radiusInRangeLowerBound, radiusInRangeUpperBound = desiredDistance-20, desiredDistance+20
        centerRightBound, centerLeftBound = 400, 200
        radiusTooCloseLowerLimit = 250

        # Creating a window for later use
        cv2.namedWindow('result')
        cv2.resizeWindow('result', 600, 600)

        # Variables to 'smarten' the following procedure
        objectSeenOnce = False  # Object has never been seen before
        leftOrRightLastSent = None  # Keep track of whether we sent left or right last

        # Initialize apriltag detector
        det = apriltag.Detector()
        firstTimeObjectNotSeen = None;
        # TODO delete this block when done
        start = time.time();
        num_frames = 0;
        inPosition = False

        while True:
            # Grab frame - break if we don't get it (some unknown error occurred)
            frame = self.vs.read()
            if frame is None:
                break
            # TODO delete this block when done
            end = time.time();
            seconds = end - start;
            num_frames += 1;
            fps = 0 if (seconds == 0) else num_frames / seconds;

            frame = imutils.resize(frame, width=600)
            # Use grayscale image for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = det.detect(gray)

            commandString = None

            tagObject = None
            for r in res:
                if r.tag_id == desiredTag:
                    tagObject = r

            if tagObject is None:  # We don't see the tag
                # Don't see the tag? Possibly just bad frame, lets wait 2 seconds and then start turning
                if firstTimeObjectNotSeen is None:
                    firstTimeObjectNotSeen = time.time()
                    self.send_serial_command(Direction.STOP, b'h')
                    commandString = "STOP";
                else:
                    secondsOfNoTag = time.time() - firstTimeObjectNotSeen
                    if secondsOfNoTag > 2: # Haven't seen our tag for more than 2 seconds
                        if leftOrRightLastSent is not None:
                            if leftOrRightLastSent == Direction.RIGHT:
                                self.send_serial_command(Direction.RIGHT, b'r');
                                commandString = "SEARCHING: GO RIGHT"
                            elif leftOrRightLastSent == Direction.LEFT:
                                self.send_serial_command(Direction.LEFT, b'l');
                                commandString = "SEARCHING: GO LEFT"
                        else:  # variable hasn't been set yet (seems unlikely), but default to left
                            self.send_serial_command(Direction.LEFT, b'l');
                            commandString = "DEFAULT SEARCHING: GO LEFT"

                        # We've sent the command now wait half a second and then send a halt
                        time.sleep(0.5)
                        self.send_serial_command(Direction.STOP, b'h');
                    else: # Keep waiting - 2 seconds haven't elapsed
                        self.send_serial_command(Direction.STOP, b'h');
                        commandString = "STOP";

            else:  # We see the tag!

                # Reset firstTimeObjectNotSeen to None for the next time we can't find the tag
                if firstTimeObjectNotSeen is not None:
                    firstTimeObjectNotSeen = None

                # Set objectSeenOnce to True if isn't already
                if not objectSeenOnce:
                    objectSeenOnce = True

                # Get the corners and draw a minimally enclosing circle of it
                # and get the x/y/radius information to use in navigation
                corners = np.array(tagObject.corners, dtype=np.float32).reshape((4, 2, 1))
                cornersList = []
                for c in corners:
                    cornersList.append([int(x) for x in c])
                cornersList = np.array(cornersList, dtype=np.int32)
                ((x, y), radius) = cv2.minEnclosingCircle(cornersList)
                M = cv2.moments(cornersList)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                filteredPtsRadius = [radius]
                filteredPtsX = [center[0]]
                filteredPtsY = [center[1]]

                # Draw circle and center
                cv2.circle(frame, (int(x), int(y)), int(filteredPtsRadius[0]), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                if filteredPtsRadius[0] > radiusTooCloseLowerLimit:
                    commandString = "MOVE BACKWARD - TOO CLOSE TO TURN"
                    self.send_serial_command(Direction.BACKWARD, b'b')
                elif filteredPtsX[0] > centerRightBound:
                    commandString = "GO RIGHT"
                    self.send_serial_command(Direction.RIGHT, b'r')
                    if leftOrRightLastSent != Direction.RIGHT:
                        leftOrRightLastSent = Direction.RIGHT
                elif filteredPtsX[0] < centerLeftBound:
                    commandString = "GO LEFT"
                    self.send_serial_command(Direction.LEFT, b'l')
                    if leftOrRightLastSent != Direction.LEFT:
                        leftOrRightLastSent = Direction.LEFT
                elif filteredPtsRadius[0] < radiusInRangeLowerBound:
                    commandString = "MOVE FORWARD"
                    self.send_serial_command(Direction.FORWARD, b'f')
                elif filteredPtsRadius[0] > radiusInRangeUpperBound:
                    commandString = "MOVE BACKWARD"
                    self.send_serial_command(Direction.BACKWARD, b'b')
                elif radiusInRangeLowerBound < filteredPtsRadius[0] < radiusInRangeUpperBound:
                    commandString = "STOP MOVING - IN RANGE"
                    self.send_serial_command(Direction.STOP, b'h')
                    inPosition = True

                # Put text on the camera image to display on the screen
                cv2.putText(frame, 'center coordinate: (' + str(filteredPtsX[0]) + ',' + str(filteredPtsY[0]) + ')',
                            (10, 60), self.font, 0.5, (200, 255, 155), 1, cv2.LINE_AA)
                cv2.putText(frame, 'filtered radius: (' + str(filteredPtsRadius[0]) + ')', (10, 90), self.font, 0.5,
                            (200, 255, 155), 1, cv2.LINE_AA)

            # Show FPS (TODO delete this later)
            cv2.putText(frame, commandString, (10, 30), self.font, 0.5, (200, 255, 155), 1, cv2.LINE_AA)
            cv2.putText(frame, 'FPS: (' + str(fps) + ')', (10, 120), self.font, 0.5,
                        (200, 255, 155), 1, cv2.LINE_AA)

            # Display frame
            cv2.imshow("result", frame)

            # Close application on 'q' key press
            # Infinite loop has been broken out of ... teardown now
            # Release the camera & close all windows
            key = cv2.waitKey(1) & 0xFF
            if (key == ord("q")) or (not cvQueue.empty()) or inPosition:
                # We've been requested to leave ...
                # Dont destroy everything - just destroy cv2 windows ... webcam still runs
                cv2.destroyAllWindows()
                break

    def get_coordinates(self):
        return (1, 2)

    def person_following(self, run_py_eyes, cvQueue: Queue):

        # Frame is considered to be 600x600 (after resize)
        # Below are variables to set what we consider center and in-range
        radiusInRangeLowerBound, radiusInRangeUpperBound = 80, 120
        centerRightBound, centerLeftBound = 400, 200
        radiusTooCloseLowerLimit = 250

        # Creating a window for later use
        cv2.namedWindow('result')
        cv2.resizeWindow('result', 600, 600)

        # Variables to 'smarten' the following procedure
        objectSeenOnce = False  # Object has never been seen before
        leftOrRightLastSent = None  # Keep track of whether we sent left or right last

        # TODO delete this block when done
        start = time.time();
        num_frames = 0;

        # PyEyes Setup
        if run_py_eyes:
            screen = pygame.display.set_mode((1024, 600), DOUBLEBUF)
            screen.set_alpha(None)
            surface = pygame.display.get_surface()
            screen.fill(self.RGB_WHITE)  # Fill PyGame screen (white background)
            pygame.draw.circle(surface, self.RGB_BLACK, (256, 300), 255, 15)
            pygame.draw.circle(surface, self.RGB_BLACK, (768, 300), 255, 15)
            pygame.display.flip()
            rects = []

        while True:
            # Reset to default pupil coordinates and width (in case no object is found on this iteration)
            leftx, lefty, width = 256, 350, 0

            # Grab frame - break if we don't get it (some unknown error occurred)
            frame = self.vs.read()
            if frame is None:
                break

            # TODO delete this block when done
            end = time.time();
            seconds = end - start;
            num_frames += 1;
            fps = 0 if (seconds == 0) else num_frames / seconds;

            # Resize the frame, blur it, and convert it to the HSV color space
            frame = imutils.resize(frame, width=600)
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # construct a mask for the desired color, then perform a series of dilations and erosions to
            # remove any small blobs left in the mask
            mask = cv2.inRange(hsv, self.greenLower, self.greenUpper)
            mask = cv2.erode(mask, None, iterations=2)  # TODO: these were 3 or 5 before
            mask = cv2.dilate(mask, None, iterations=2)

            # find contours in the mask and initialize the current (x, y) center of the ball
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            commandString = None

            # Only proceed if at least one contour was found
            # If nothing is found => look around OR send the STOP command to halt movement (depends on situation)
            if len(cnts) == 0:
                # If we haven't seen the object before, then we'll stay halted until we see one
                # If we HAVE seen the object before, then we'll move in the direction (left or right) that we did most recently
                if not objectSeenOnce:
                    self.send_serial_command(Direction.STOP, b'h');
                    commandString = "STOP";
                else:  # Object has been seen before
                    if leftOrRightLastSent is not None:
                        if leftOrRightLastSent == Direction.RIGHT:
                            self.send_serial_command(Direction.RIGHT, b'r');
                            commandString = "SEARCHING: GO RIGHT"
                        elif leftOrRightLastSent == Direction.LEFT:
                            self.send_serial_command(Direction.LEFT, b'l');
                            commandString = "SEARCHING: GO LEFT"
                    else:  # variable hasn't been set yet (seems unlikely), but default to left
                        self.send_serial_command(Direction.LEFT, b'l');
                        commandString = "DEFAULT SEARCHING: GO LEFT"

            elif len(cnts) > 0:  # Else if we are seeing some object...

                # Find the largest contour in the mask and use it to compute the minimum enclosing circle and centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                filteredPtsRadius = [radius]

                # Only consider it to a valid object if it's big enough - else it could be some other random thing
                if filteredPtsRadius[0] <= 25:
                    # TODO this is the same code as the block above - I should extract these out to a function
                    # If we haven't seen the object before, then we'll stay halted until we see one
                    # If we HAVE seen the object before, then we'll move in the direction (left or right) that we did most recently
                    if not objectSeenOnce:
                        self.send_serial_command(Direction.STOP, b'h');
                        commandString = "STOP";
                    else:  # Object has been seen before
                        if leftOrRightLastSent is not None:
                            if leftOrRightLastSent == Direction.RIGHT:
                                self.send_serial_command(Direction.RIGHT, b'r');
                                commandString = "SEARCHING: GO RIGHT"
                            elif leftOrRightLastSent == Direction.LEFT:
                                self.send_serial_command(Direction.LEFT, b'l');
                                commandString = "SEARCHING: GO LEFT"
                        else:  # variable hasn't been set yet (seems unlikely), but default to left
                            self.send_serial_command(Direction.LEFT, b'l');
                            commandString = "DEFAULT SEARCHING: GO LEFT"

                else:  # This object isn't super small ... we should proceed with the tracking

                    # Set objectSeenOnce to True if isn't already
                    if not objectSeenOnce:
                        objectSeenOnce = True

                    # only draw the circle on the frame if the radius meets a certain size if the radius meets a minimum size
                    # TODO remove this eventually - could speed things up
                    cv2.circle(frame, (int(x), int(y)), int(filteredPtsRadius[0]), (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

                    filteredPtsX = [center[0]]
                    filteredPtsY = [center[1]]

                    # Update PyGame Values
                    if run_py_eyes:
                        lefty = int(filteredPtsY[0] + 100)
                        leftx = int(abs(filteredPtsX[0] - 600) / 2 + 106)
                        width = int(filteredPtsRadius[0])

                    # Check radius and center of the blob to determine robot action
                    # What actions should take priority?
                    # 1. Moving Backward (only if it's super close)
                    # 2. Moving Left/Right
                    # 3. Moving Forward/Backward
                    # Why? Because if we're too close any turn would be too extreme. We need to take care of that first

                    if filteredPtsRadius[0] > radiusTooCloseLowerLimit:
                        commandString = "MOVE BACKWARD - TOO CLOSE TO TURN"
                        self.send_serial_command(Direction.BACKWARD, b'b')
                    elif filteredPtsX[0] > centerRightBound:
                        commandString = "GO RIGHT"
                        self.send_serial_command(Direction.RIGHT, b'r')
                        if leftOrRightLastSent != Direction.RIGHT:
                            leftOrRightLastSent = Direction.RIGHT
                    elif filteredPtsX[0] < centerLeftBound:
                        commandString = "GO LEFT"
                        self.send_serial_command(Direction.LEFT, b'l')
                        if leftOrRightLastSent != Direction.LEFT:
                            leftOrRightLastSent = Direction.LEFT
                    elif filteredPtsRadius[0] < radiusInRangeLowerBound:
                        commandString = "MOVE FORWARD"
                        self.send_serial_command(Direction.FORWARD, b'f')
                    elif filteredPtsRadius[0] > radiusInRangeUpperBound:
                        commandString = "MOVE BACKWARD"
                        self.send_serial_command(Direction.BACKWARD, b'b')
                    elif radiusInRangeLowerBound < filteredPtsRadius[0] < radiusInRangeUpperBound:
                        commandString = "STOP MOVING - IN RANGE"
                        self.send_serial_command(Direction.STOP, b'h')

                    # Put text on the camera image to display on the screen
                    cv2.putText(frame, 'center coordinate: (' + str(filteredPtsX[0]) + ',' + str(filteredPtsY[0]) + ')',
                                (10, 60), self.font, 0.5, (200, 255, 155), 1, cv2.LINE_AA)
                    cv2.putText(frame, 'filtered radius: (' + str(filteredPtsRadius[0]) + ')', (10, 90), self.font, 0.5,
                                (200, 255, 155), 1, cv2.LINE_AA)

            # The below steps are run regardless of whether we see a valid object or not ...

            # Show FPS (TODO delete this later)
            cv2.putText(frame, commandString, (10, 30), self.font, 0.5, (200, 255, 155), 1, cv2.LINE_AA)
            cv2.putText(frame, 'FPS: (' + str(fps) + ')', (10, 120), self.font, 0.5,
                        (200, 255, 155), 1, cv2.LINE_AA)

            # show webcam video with object drawings on the screen
            cv2.imshow("result", frame)

            if run_py_eyes:
                if leftx < 106: leftx = 106
                if leftx > 406: leftx = 406
                if lefty < 150: lefty = 150
                if lefty > 450: lefty = 450
                # rightx = leftx + 400 + 112 - width
                # if rightx < 568:
                #     rightx = 568
                # if rightx > 968:
                #     rightx = 968
                rects.append(pygame.draw.circle(surface, self.RGB_BLACK, (leftx, lefty), self.EYEBALL_RADIUS, 0))
                rects.append(
                    pygame.draw.circle(surface, self.RGB_BLACK, (leftx + 500 + 12, lefty), self.EYEBALL_RADIUS, 0))
                pygame.display.update(rects)
                rects = [pygame.draw.circle(surface, self.RGB_WHITE, (leftx, lefty), self.EYEBALL_RADIUS, 0),
                         pygame.draw.circle(surface, self.RGB_WHITE, (leftx + 500 + 12, lefty), self.EYEBALL_RADIUS, 0)]

            # Close application on 'q' key press
            # Infinite loop has been broken out of ... teardown now
            # Release the camera & close all windows
            key = cv2.waitKey(1) & 0xFF
            if (key == ord("q")) or (not cvQueue.empty()):
                # We've been requested to leave ...
                # Dont destroy everything - just destroy cv2 windows ... webcam still runs
                cv2.destroyAllWindows()
                if run_py_eyes:
                    pygame.display.quit()
                    pygame.quit()
                break


def thread_closer(person_following_thread: Thread, april_following_thread: Thread, cvObject: OpenCVController):
    # If either of the following threads are running then we need to close them and then run our operation since it
    # uses the shared webcam resources
    # Only one of these below should be able to run on any call to thread_closer
    if cvObject.runningPersonFollowing:
        cvObject.runningPersonFollowing = False
        person_following_thread.join()
    if cvObject.runningAprilFollowing:
        cvObject.runningAprilFollowing = False;
        april_following_thread.join()

    return (None, None)


def run(cvQueue: Queue):
    # Initialize an object for the class - this will connect to the webcam and serial port and begin grabbing frames
    # (threaded)
    cvObject = OpenCVController()

    # person_following_thread: Thread = None  # Keep track of threads
    # april_following_thread: Thread = None

    while True:
        if not cvQueue.empty():  # If there's something in the queue
            commandFromQueue = cvQueue.get()
            cvQueue.task_done()
            if commandFromQueue == "terminate":
                cvObject.cleanup_resources()
                print("Terminate OpenCV")
                return
            elif commandFromQueue == "halt":
                cvObject.send_serial_command(Direction.STOP, b'h');
                print("Sent halt command")
            elif commandFromQueue == "getCoordinates":
                print("got command getCoordinates")
                x, z = cvObject.get_coordinates()
                print("sending coordinates")
                cvQueue.put(x)
                cvQueue.put(z)
                cvQueue.join()
                # while not cvQueue.empty():  # Wait here until the queue gets emptied
                #    pass
            elif commandFromQueue == "personFollow":
                cvObject.person_following(False, cvQueue)
            elif commandFromQueue == "eyeballFollow":
                cvObject.person_following(True, cvQueue)
            elif commandFromQueue == "aprilFollow":
                # We know the next 2 items in the queue in this case are the x and z coordinates - grab them
                #  Note: get() commands will block until it can get something
                print("Receive april Tag request")
                target_tag_number = cvQueue.get()
                target_tag_radius = cvQueue.get()
                cvObject.april_following(target_tag_number, target_tag_radius, cvQueue)
                cvQueue.task_done()
                cvQueue.task_done()


if __name__ == "__main__":
    # classObject = OpenCVController()
    # classObject.runningPersonFollowing = True
    # classObject.person_following(True)

    cvQueue = Queue()
    cvQueue.put("personFollow")
    run(cvQueue)
