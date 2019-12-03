# TODO add break condition for cv to avoid unnessesary calibration

# import the necessary packages
from queue import Queue
import subprocess
from imutils.video import WebcamVideoStream
import numpy as np
import cv2
import imutils
import serial
import serial.tools.list_ports  # for listing serial ports
import time
import os
from enum import Enum
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *
import json
import apriltag
import aprilTags
from scipy import ndimage  # median filter
from threading import Thread
import sys  # command line lib
import glob
import math
import argparse
import threading
from collections import deque
from math import atan2, asin
import smtplib
import os
import time
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

#from email.MIMEText import MIMEText
#from email.MIMEImage import MIMEImage


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
    imageNumber = 0

    # Define lower & upper boundaries in HSV color space (for object following/tracking)
    greenLower = np.array([48, 100, 50])  # [48, 100, 50] 31
    greenUpper = np.array([85, 255, 255])

    def __init__(self):

        self.serialPort = self.com_connect() if False else None  # True for running - False for testing

        # Variables to hold last command sent to Arduino and when it was sent (epoch seconds)
        self.lastCommandSentViaSerial = None
        self.lastCommandSentViaSerialTime = None

        # Connect to webcam video source
        self.WebcamVideoStreamObject = WebcamVideoStream(src=1)

        # Save the original exposure and gain of the webcam for restoring later...
        self.originalExposure = self.WebcamVideoStreamObject.stream.get(cv2.CAP_PROP_EXPOSURE)
        self.originalGain = self.WebcamVideoStreamObject.stream.get(cv2.CAP_PROP_GAIN)
        # Note: original exposure is probably like 0.015401540324091911 and adjusts automatically

        # <Insert any webcam modifications here>

        # Start the webcam streaming operation and set the object attribute to use...
        self.vs = self.WebcamVideoStreamObject.start()
        # Allow camera to warm up...
        time.sleep(2.0)

        # subprocess.check_call("v4l2-ctl -d /dev/video1 -c exposure_absolute=40",shell=True)
        # print("New Gain") print(str(self.WebcamVideoStreamObject.stream.get(cv2.CAP_PROP_GAIN)))
        # print("New exposure") print(str(self.WebcamVideoStreamObject.stream.get(cv2.CAP_PROP_EXPOSURE)))
        # print("Frame width") print(str(self.WebcamVideoStreamObject.stream.get(cv2.CAP_PROP_FRAME_WIDTH)))
        # print("Frame height") print(str(self.WebcamVideoStreamObject.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    @staticmethod
    def internet_on():
      try:
        urllib.urlopen('http://216.58.192.142', timeout = 1)
        return True
      except urllib2.URLError as err:
        return False

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
                ser = serial.Serial('/dev/ttyACM1', 115200, write_timeout=0)
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
            # elif (time.time() - self.lastCommandSentViaSerialTime > 1): # TODO also need null check here
            #    self.serialPort.write(dataToSend)
            #    self.lastCommandSentViaSerialTime = time.time()
            else:
                pass  # Do nothing - same command sent recently

    def cleanup_resources(self):
        # TODO change the order of stream release and stop() - see what happens
        self.vs.stop()
        #self.vs.stream.release() # TODO looks like this is already handled by stop() ??
        cv2.destroyAllWindows() # Not necessary since I do it at the end of every method
        if self.serialPort is not None:  # Close serialPort if it exists
            self.serialPort.close()  # TODO make sure this doesn't ruin anything

    def take_photo(self):
        
        # Creating a window for later use
        cv2.namedWindow('result')
        cv2.resizeWindow('result', 600, 600)
        startTime = time.time()
        waitTimeSec = 5
        strSec = "54321"

        while True:
            # Grab frame - break if we don't get it (some unknown error occurred)
            frame = self.vs.read()
            if frame is None:
                print("ERROR - frame read a NONE")
                break

            frame = imutils.resize(frame, width=600)
            
            currTime = time.time()
            timeDiff = int(currTime - startTime) # seconds
            
            if timeDiff < waitTimeSec: # TODO <= ?
                # Put time countdown on the frame (TODO resize)
                cv2.putText(frame, strSec[timeDiff], (10, 30), self.font, 0.5, (200, 255, 155), 1, cv2.LINE_AA)
                cv2.imshow("result", frame)
            else: # Time's up - take the photo and do post-processing
                cv2.imshow("result", frame)
                time.sleep(1) # So it's obvious photo was taken
                cv2.imwrite(filename='saved_img_' + str(self.imageNumber) + '.jpg', img=frame)
                self.imageNumber += 1 # Increment for next possible photo
                cv2.destroyAllWindows() # Close the CV windows
                # Send email
                msg = MIMEMultipart()
                msg.attach(MIMEImage(file('saved_img_' + str(self.imageNumber) + '.jpg').read()))
                conn = smtplib.SMTP('imap.gmail.com', 587)
                conn.ehlo()
                conn.starttls()
                conn.login('UmdJetson2@gmail.com', 'jetson1994')
                #conn.sendmail('UmdJetson2@gmail.com', 'huy1994@gmail.com', IPAddr)
                #time.sleep(1)
                #conn.sendmail('UmdJetson2@gmail.com', 'ndkoenigsmark@gmail.com', IPAddr)
                #time.sleep(1)
                conn.sendmail('UmdJetson2@gmail.com', 'nikhilu@terpmail.umd.edu', msg.as_string())
                conn.quit()
                break
                #  and send via email here (actually send it once out of the loop

            # Close application on 'q' key press or if new stuff added to queue
            key = cv2.waitKey(1) & 0xFF
            if (key == ord("q")) or (not cvQueue.empty()):
                # We've been requested to leave ...
                # Don't destroy everything - just destroy cv2 windows ... webcam still runs
                cv2.destroyAllWindows()
                break

    def april_following(self, desiredTag, desiredDistance, cvQueue: Queue):

        # Tune the webcam to better see april tags while robot is moving
        # (compensating for motion blur). Restore settings when done
        self.WebcamVideoStreamObject.stream.set(cv2.CAP_PROP_EXPOSURE, 0.2)
        self.WebcamVideoStreamObject.stream.set(cv2.CAP_PROP_GAIN, 1)

        # Frame is considered to be 600x600 (after resize)
        # Below are variables to set what we consider center and in-range
        radiusInRangeLowerBound, radiusInRangeUpperBound = desiredDistance - 20, desiredDistance + 20
        centerRightBound, centerLeftBound = 400, 200
        radiusTooCloseLowerLimit = 250

        # When turning to search for the desiredTag, we specify time to turn,
        # and time to wait after each semi-turn
        searchingTimeToTurn = 0.5  # seconds
        searchingTimeToHalt = 0.5  # seconds
        # TODO change the above for max turning and minimal halting that still works

        # Creating a window for later use
        cv2.namedWindow('result')
        cv2.resizeWindow('result', 600, 600)

        # Variables to 'smarten' the following procedure
        objectSeenOnce = False  # Object has never been seen before
        leftOrRightLastSent = None  # Keep track of whether we sent left or right last
        firstTimeObjectNotSeen = None;

        # Initialize apriltag detector
        options = apriltag.DetectorOptions(
            families='tag36h11',
            border=1,
            nthreads=1,
            quad_decimate=1.0,
            quad_blur=0.0,
            refine_edges=True,
            refine_decode=True,
            refine_pose=False,
            debug=False,
            quad_contours=True)
        det = apriltag.Detector(options)

        # TODO delete this block when done
        start = time.time();
        num_frames = 0;
        inPosition = False
        numHalts = 0

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
            # frame = cv2.filter2D(frame, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])) # Sharpen image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Use grayscale image for detection
            res = det.detect(gray)

            commandString = None

            # Check if the desiredTag is visible
            tagObject = None
            for r in res:
                if r.tag_id == desiredTag:
                    tagObject = r

            if tagObject is None:  # We don't see the tag

                # Don't see the tag? Possibly just bad frame, lets wait 2 seconds and then start turning
                # TODO change this -  we probably don't need to do these half-ass turns anymore
                numHalts += 1  # TODO delete
                if firstTimeObjectNotSeen is None:
                    firstTimeObjectNotSeen = time.time()
                    self.send_serial_command(Direction.STOP, b'h')
                    commandString = "STOP";
                else:
                    secondsOfNoTag = time.time() - firstTimeObjectNotSeen
                    if secondsOfNoTag > 2:  # Haven't seen our tag for more than 2 seconds
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
                        time.sleep(searchingTimeToTurn)
                        self.send_serial_command(Direction.STOP, b'h');
                        time.sleep(searchingTimeToHalt)
                    else:  # Keep waiting - 2 seconds haven't elapsed
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

                # Determine command to send to arudino/motors
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

            # Show FPS and number of halts (TODO delete this later)
            cv2.putText(frame, commandString, (10, 30), self.font, 0.5, (200, 255, 155), 1, cv2.LINE_AA)
            cv2.putText(frame, 'FPS: (' + str(fps) + ')', (10, 120), self.font, 0.5,
                        (200, 255, 155), 1, cv2.LINE_AA)
            cv2.putText(frame, 'numHalts: (' + str(numHalts) + ')', (10, 150), self.font, 0.5,
                        (200, 255, 155), 1, cv2.LINE_AA)

            # Display frame
            cv2.imshow("result", frame)

            # Close application on 'q' key press, new stuff on queue, or if we've reached our destination
            key = cv2.waitKey(1) & 0xFF
            if (key == ord("q")) or (not cvQueue.empty()) or inPosition:
                # Restore webcam settings
                self.WebcamVideoStreamObject.stream.set(cv2.CAP_PROP_EXPOSURE, self.originalExposure)
                self.WebcamVideoStreamObject.stream.set(cv2.CAP_PROP_GAIN, self.originalGain)
                subprocess.check_call("v4l2-ctl -d /dev/video1 -c exposure_auto=3", shell=True)
                # Dont destroy everything - just destroy cv2 windows ... webcam still runs
                cv2.destroyAllWindows()
                break

    @staticmethod
    def calc_weight(p1, p2):
        max_weight = 150
        dist = np.linalg.norm(p1 - p2)
        return max_weight / dist

    def get_coordinates(self):

        # When turning to search for the desiredTag, we specify time to turn,
        # and time to wait after each semi-turn
        searchingTimeToTurn = 2.5  # seconds
        searchingTimeToHalt = 2.5  # seconds
        # TODO change the above for max turning and minimal halting that still works

        options = apriltag.DetectorOptions(
            families='tag36h11',
            border=1,
            nthreads=1,
            quad_decimate=1.0,
            quad_blur=0.0,
            refine_edges=True,
            refine_decode=True,
            refine_pose=True,
            debug=False,
            quad_contours=True)
        det = apriltag.Detector(options)

        # Load camera data
        with open('cameraParams.json', 'r') as f:
            data = json.load(f)
        cameraMatrix = np.array(data['cameraMatrix'], dtype=np.float32)
        distCoeffs = np.array(data['distCoeffs'], dtype=np.float32)

        # Load world points
        world_points = {}
        with open('worldPoints.json', 'r') as f:
            data = json.load(f)
        for k, v in data.items():
            world_points[int(k)] = np.array(v, dtype=np.float32).reshape((4, 3, 1))

        # Variables for final decision
        coordinates_list = []
        iterationNumber = 1
        numIterations = 5

        while True:
            # Rotate camera by going left by some amount (TODO fine tune)
            # TODO can cut down on halt time? lengthen turn time? need to play around..
            self.send_serial_command(Direction.LEFT, b'l');
            time.sleep(searchingTimeToTurn)
            self.send_serial_command(Direction.STOP, b'h');
            time.sleep(searchingTimeToHalt)

            # Now lets read the frame (while the robot is halted so that image is clean)
            frame = self.vs.read()
            if frame is None:
                print("ERROR - frame read a NONE")
                break

            # Use grayscale image for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = det.detect(gray)

            # Check how many tags we see... if 0 then ignore this
            numTagsSeen = len(res)
            print("\nNumber of tags seen", numTagsSeen) # TODO remove

            if numTagsSeen > 0:

                poses = []  # Store poses from each tag to average them over
                tagRadiusList = []  # Store tag radius' to determine the largest

                for r in res:  # Iterate over each tag in the frame
                    corners = r.corners
                    tag_id = r.tag_id
                    corners = np.array(corners, dtype=np.float32).reshape((4, 2, 1))
                    cornersList = []
                    for c in corners:
                        cornersList.append([int(x) for x in c])
                    cornersList = np.array(cornersList, dtype=np.int32)

                    # Draw circle around tag using its corners & get radius of that tag
                    ((x, y), radius) = cv2.minEnclosingCircle(cornersList) # TODO make _ (dont care)
                    filteredPtsRadius = [radius]

                    # Solve pose ((x,z) coordinates)
                    r, rot, t = cv2.solvePnP(world_points[tag_id], corners, cameraMatrix,
                                             distCoeffs)  # get rotation and translation vector using solvePnP
                    rot_mat, _ = cv2.Rodrigues(rot)  # convert to rotation matrix
                    R = rot_mat.transpose()  # Use rotation matrix to get pose = -R * t (matrix mul w/ @)
                    pose = -R @ t
                    weight = self.calc_weight(pose, world_points[tag_id][0])
                    poses.append((pose, weight))
                    tagRadiusList.append(filteredPtsRadius)

                # Done iterating over the tags that're seen in the frame...
                # Now get the average pose across the tags and get the largest radius
                # We will store the (x,z) coordinate that we calculate, and we'll also
                # store the largest radius for a tag that we've seen in this frame.
                avgPose = sum([x * y for x, y in poses]) / sum([x[1] for x in poses])
                largestTagRadius = max(tagRadiusList)
                coordinates = (avgPose[0][0], avgPose[2][0], largestTagRadius)
                print(str(coordinates))  # TODO remove this
                coordinates_list.append(coordinates)

            # Display frame
            cv2.imshow('frame', frame)

            # If we've completed our numIterations, then choose the coordinate
            # and return (do closing operations too)
            if iterationNumber == numIterations:
                if len(coordinates_list) > 0:
                    # TODO 2 things we can try here ...
                    #   1) The coordinate to return is the one with the smallest z-coordinate
                    #      (which essentially means it's closest to those tags that it used)
                    #      BUT this value seems to vary a lot and I don't think it's reliable
                    #   2) I have saved the largest radius for a tag seen for each of these
                    #      coordinates, so I can use that (which I bet is more reliable)
                    # I will go with approach number 2

                    # coordinateToReturn = min(coordinates_list, key=lambda x: x[1]) # Approach (1)
                    coordinateToReturn = max(coordinates_list, key=lambda x: x[2])  # Approach (2)
                    coordinateToReturn = (int(coordinateToReturn[0]), int(coordinateToReturn[1]))  # This stays regardless
                else:
                    coordinateToReturn = (1, 2)  # TODO set to some default outside the door
                    # TODO if this happens then we'll have to get moving too
                    #   ask team -  do we already move in response to our own distress? Can we make it do so
                    #   somehow but not always?
                cv2.destroyAllWindows()
                print("Value to return:")  # TODO remove
                return coordinateToReturn
            else:  # Still have iterations to go, increment the value
                iterationNumber += 1

            # Q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

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
        start = time.time()
        num_frames = 0

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
                print("ERROR - frame read a NONE")
                break
            # TODO delete this block when done
            end = time.time()
            seconds = end - start
            num_frames += 1
            fps = 0 if (seconds == 0) else num_frames / seconds

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
                # If we haven't seen the object before, then we'll stay halted until we see one. If we HAVE seen the
                # object before, then we'll move in the direction (left or right) that we did most recently
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
                    # If we HAVE seen the object before, then we'll move in the direction (left or right) that we did
                    # most recently
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
                # Don't destroy everything - just destroy cv2 windows ... webcam still runs
                cv2.destroyAllWindows()
                if run_py_eyes:
                    pygame.display.quit()
                    pygame.quit()
                break


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
                final_target_tag_number = cvQueue.get()
                final_target_tag_radius = cvQueue.get()
                cvQueue.task_done()
                cvQueue.task_done()
                for target_pair in aprilTags.aprilTargets:
                    cvObject.april_following(target_pair[0], target_pair[1], cvQueue)
                cvObject.april_following(final_target_tag_number, final_target_tag_radius, cvQueue)


if __name__ == "__main__":
    classObject = OpenCVController()
    cvQueue = Queue()
    # classObject.runningPersonFollowing = True
    #classObject.april_following(22, 80,  cvQueue)
    #print("here?")
    #classObject.person_following(True,  cvQueue)
    # classObject.april_following(22, 80,  cvQueue)
    #print(str(classObject.get_coordinates()))
    #classObject.cleanup_resources()
    classObject.take_photo()
    exit()

    # cvQueue = Queue()
    # cvQueue.put("personFollow")
    # run(cvQueue)
