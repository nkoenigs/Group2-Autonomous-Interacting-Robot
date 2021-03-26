# Author: Nikhil Uplekar
# ENEE408I-0101 Fall 2019 Group 2

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
# os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *
import json
import apriltag
import aprilTags
from scipy import ndimage  # median filter
import sys  # command line lib
import glob
import math
import smtplib
import os
import urllib.request
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from os.path import basename
from urllib.request import urlopen


# Class/Enums to keep track of some of our last directions sent and to
# send down commands to Arduino via Serial communication
class Direction(Enum):
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4
    STOP = 5


# Main class of the file
class OpenCVController:
    # PyGame constants (RGB value for white & black and the radius of the eyeballs to draw)
    RGB_WHITE = (255, 255, 255)
    RGB_BLACK = (0, 0, 0)
    EYEBALL_RADIUS = 30

    # Font for text on webcam display (put here to simply our writing commands elsewhere and enable uniformity)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Incremented after each photo is taken - this allows each photo to have a unique name and not be overwritten
    imageNumber = 0

    # Define lower & upper boundaries in HSV color space (for object following/tracking)
    greenLower = np.array([48, 100, 50])
    greenUpper = np.array([85, 255, 255])

    # Constructor/initalizer for this class
    def __init__(self):

        # Connect to the Arduino via serial port connection. If we're testing without the Arduino then set it to
        # False in the line below so we don't get stuck waiting/hanging for the serial connection to complete
        self.serialPort = self.com_connect() if True else None  # True for running - False for testing

        # Variables to hold last command sent to Arduino and when it was sent (epoch seconds). Note:
        # lastCommandSentViaSerialTime ended up not being utilized - but its purpose was to send duplicate commands
        # after some certain amount of time in case the Arduino needed it for some reason (it did not with our
        # current design)
        self.lastCommandSentViaSerial = None
        self.lastCommandSentViaSerialTime = None

        # Connect to webcam video source - WebcamVideoStream is a threaded class. By including it here,
        # all our functions will be able to use it (since only one is run at a time) and we won't have to restart our
        # video streaming each time we want to run a different function/capability.
        self.WebcamVideoStreamObject = WebcamVideoStream(src=1)

        # Save the original exposure and gain of the webcam for restoring later... (we will modify these parameters
        # when we notice that the motion of the robot is impairing the OpenCV/image-processing capabilities (motion
        # blur makes it hard to detect april tags, so by changing these parameters we can capture/detect these tags
        # even while the robot is moving quite fast). We'd like to restore the camera to its original settings when
        # it's not needed to specialize/adjust them because that enables the camera to choose the best settings for
        # the environment and change its exposure automatically.
        self.originalExposure = self.WebcamVideoStreamObject.stream.get(cv2.CAP_PROP_EXPOSURE)
        self.originalGain = self.WebcamVideoStreamObject.stream.get(cv2.CAP_PROP_GAIN)
        # Note: original exposure is probably like 0.015 or something small (depending on the lightning) and adjusts
        # automatically

        # Start the webcam streaming operation and set the object attribute to use...
        self.vs = self.WebcamVideoStreamObject.start()
        # Allow camera to warm up...
        time.sleep(2.0)

        # subprocess.check_call("v4l2-ctl -d /dev/video1 -c exposure_absolute=40",shell=True)
        # print("New Gain") print(str(self.WebcamVideoStreamObject.stream.get(cv2.CAP_PROP_GAIN)))
        # print("New exposure") print(str(self.WebcamVideoStreamObject.stream.get(cv2.CAP_PROP_EXPOSURE)))
        # print("Frame width") print(str(self.WebcamVideoStreamObject.stream.get(cv2.CAP_PROP_FRAME_WIDTH)))
        # print("Frame height") print(str(self.WebcamVideoStreamObject.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Tries to open a specific URL to confirm that we are connected to the internet - this is never used in the
    # codebase because some of our startup steps for our code already depended on internet connection.
    @staticmethod
    def internet_on():
        try:
            urlopen('http://216.58.192.142', timeout=1)
            return True
        except urllib2.URLError as err:
            return False

    # Connected to Arduino for serial communication. This method will hang/wait until it can make the connection,
    # so make sure not to call this method if you aren't testing with an Arduino connected (see __init__ above)
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

    # Send serial command to Arduino. If the last command that we've sent is the same as the one we're trying to send
    # now, then ignore it since the Arduino already has the up-to-date command. Note that there's a commented out
    # section of this code that made it so that even if the command was a duplicate of the last send one,
    # it would still send the command as long as a certain time period had passed. Depending on how the Arduino code
    # worked this may have been necessary, but we did not need it.
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

    # Call this when closing this openCV process. It will stop the WebcamVideoStream thread, close all openCV
    # windows, and close the SerialPort as long as it exists (if we're connected to an Arduino).
    def cleanup_resources(self):
        # TODO change the order of stream release and stop() - see what happens
        self.vs.stop()
        # self.vs.stream.release() # TODO this should be called but is throwing errors - it works as-is though
        cv2.destroyAllWindows()  # Not necessary since I do it at the end of every method
        if self.serialPort is not None:  # Close serialPort if it exists
            self.serialPort.close()

    # Send an email with optional attachments specified (see usage in this file)
    @staticmethod
    def send_mail(send_from: str, subject: str, text: str, send_to: list, files=None):
        username = 'UmdJetson2@gmail.com'
        password = 'jetson1994'
        msg = MIMEMultipart()
        msg['From'] = send_from
        msg['To'] = ', '.join(send_to)
        msg['Subject'] = subject

        msg.attach(MIMEText(text))

        for f in files or []:
            with open(f, "rb") as fil:
                ext = f.split('.')[-1:]
                attachedfile = MIMEApplication(fil.read(), _subtype=ext)
                attachedfile.add_header(
                    'content-disposition', 'attachment', filename=basename(f))
            msg.attach(attachedfile)

        smtp = smtplib.SMTP(host="imap.gmail.com", port=587)
        smtp.starttls()
        smtp.login(username, password)
        smtp.sendmail(send_from, send_to, msg.as_string())
        smtp.close()

    # Countdown timer and then take a photo using a frame from the WebcamVideoStream. The photo is sent to the emails
    # of the group members. Note that the quality of the image can likely be increased by tuning the webcam settings
    # (I did not because it was not necessary).
    def take_photo(self, cvQueue: Queue):

        # Creating a window for later use
        cv2.namedWindow('result')
        cv2.resizeWindow('result', 600, 600)

        # Use the below stuff to do the countdown timer
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
            timeDiff = int(currTime - startTime)  # seconds

            if timeDiff < waitTimeSec:  # Not time to take the photo yet
                # Put time countdown on the frame
                cv2.putText(frame, strSec[timeDiff], (250, 300), self.font, 5, (200, 255, 155), 4, cv2.LINE_AA)
                cv2.imshow("result", frame)
            else:  # Time's up - take the photo and do post-processing
                cv2.imshow("result", frame)
                time.sleep(1)  # Freeze the screen so that it's obvious that the photo was taken

                # Save the image in the current working directory
                cv2.imwrite(filename='saved_img_' + str(self.imageNumber) + '.jpg', img=frame)

                cv2.destroyAllWindows()  # Close the CV windows

                # Send email
                username = 'UmdJetson2@gmail.com'
                password = 'jetson1994'
                sendList = ['nikhilu@terpmail.umd.edu',
                            'ndkoenigsmark@gmail.com',
                            'huy1994@gmail.com']
                filesToSend = [os.path.abspath('saved_img_' + str(self.imageNumber) + '.jpg')]
                # if self.internet_on():
                self.send_mail(send_from=username, subject="test", text="text", send_to=sendList, files=filesToSend)
                # else:
                #    print("Internet is not on - did not send email")
                self.imageNumber += 1  # Increment for next possible photo
                break

            # Close application on 'q' key press or if new stuff added to queue
            key = cv2.waitKey(1) & 0xFF
            if (key == ord("q")) or (not cvQueue.empty()):
                # We've been requested to leave ...
                # Don't destroy everything - just destroy cv2 windows ... webcam still runs
                cv2.destroyAllWindows()
                break

    # Utilizes the person/eyeball_follow algorithms to move to the requested april tag and reach a desired distance
    # from the tag. This function is called multiple times (once for each april tag we want to go to). The variables
    # isFirstUse and isLastUse are used to avoid setting and resetting the webcam's gain and exposure rapidly back
    # and forth (this creates a problem). By using these variables we set it only on the first april tag and reset it
    # back to the original specifications on the last tag.
    def april_following(self, desiredTag, desiredDistance, cvQueue: Queue, isFirstUse, isLastUse):

        # Fast-fail. If there is something on the cvQueue then that means we need to respond to it. There are
        # multiple calls of april_following(...) being made in succession in a for-loop to get the robot to a
        # destiantion. We want to quickly exit from each of the calls in this situation.
        if not cvQueue.empty():
            return

        # Tune the webcam to better see april tags while robot is moving
        # (compensating for motion blur). Restore settings when done.
        # These settings can be played with to create the best effect (along with other settings if you want)
        if isFirstUse:
            self.WebcamVideoStreamObject.stream.set(cv2.CAP_PROP_EXPOSURE, 0.5)
            self.WebcamVideoStreamObject.stream.set(cv2.CAP_PROP_GAIN, 1)

        # Frame is considered to be 600x600 (after resize) (actually it's like 600x400)
        # Below are variables to set what we consider center and in-range (these numbers are in pixels)
        radiusInRangeLowerBound, radiusInRangeUpperBound = desiredDistance - 10, desiredDistance + 10
        centerRightBound, centerLeftBound = 400, 200
        radiusTooCloseLowerLimit = 250

        # When turning to search for the desiredTag, we specify time to turn, and time to wait after each semi-turn.
        # Note that these two variables are NO LONGER USED! By adjusting the exposure to reduce the effects of motion
        # blur, we no longer have to do this turn-and-stop manuever to search for tags around us. Just rotating works
        # fine.
        searchingTimeToTurn = 0.3  # seconds
        searchingTimeToHalt = 1.0  # seconds

        # Creating a window for later use
        cv2.namedWindow('result')
        cv2.resizeWindow('result', 600, 600)

        # Variables to 'smarten' the following procedure. See their usage below.
        objectSeenOnce = False  # Object has never been seen before
        leftOrRightLastSent = None  # Keep track of whether we sent left or right last
        firstTimeObjectNotSeen = None  # Holds timestamp (in seconds) of the first time we haven't been able to see
        # the tag. We don't want to instantly start freaking out and turning around looking for the tag because it's
        # very possible it was lost in some bad frame, so we wait some X number of seconds before looking around (
        # this is what this timestamp is used for).

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

        # TODO delete this block when done (not necessary to so we kept it - just thought removing extra details
        #  would speed up performance)
        start = time.time()
        num_frames = 0
        inPosition = False
        numHalts = 0

        while True:

            # Grab frame - break if we don't get it (some unknown error occurred)
            frame = self.vs.read()
            if frame is None:
                break

            # TODO delete this block when done (same as above TODO)
            end = time.time()
            seconds = end - start
            num_frames += 1
            fps = 0 if (seconds == 0) else num_frames / seconds

            frame = imutils.resize(frame, width=600)
            # frame = cv2.filter2D(frame, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])) # Sharpen image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Use grayscale image for detection
            res = det.detect(gray)  # Run the image through the apriltag detector and get the results

            commandString = None  # Stores string to print on the screen (the current command to execute)

            # Check if the desiredTag is visible
            tagObject = None
            for r in res:
                if r.tag_id == desiredTag:
                    tagObject = r

            if tagObject is None:  # We don't see the tag that we're looking for

                # Don't see the tag? Possibly just bad frame, lets wait 2 seconds and then start turning

                numHalts += 1
                # TODO delete all the numHalt tracking stuff (this was to keep track and lessen the
                #  effects of motion blur ... we kept it since it didn't affect performance).

                if firstTimeObjectNotSeen is None:
                    firstTimeObjectNotSeen = time.time()
                    self.send_serial_command(Direction.STOP, b'h')
                    commandString = "STOP"
                else:
                    secondsOfNoTag = time.time() - firstTimeObjectNotSeen
                    if secondsOfNoTag > 2:  # Haven't seen our tag for more than 2 seconds
                        if leftOrRightLastSent is not None:
                            if leftOrRightLastSent == Direction.RIGHT:
                                self.send_serial_command(Direction.RIGHT, b'r')
                                commandString = "SEARCHING: GO RIGHT"
                            elif leftOrRightLastSent == Direction.LEFT:
                                self.send_serial_command(Direction.LEFT, b'l')
                                commandString = "SEARCHING: GO LEFT"
                        else:  # variable hasn't been set yet (seems unlikely), but default to left
                            self.send_serial_command(Direction.LEFT, b'r')
                            commandString = "DEFAULT SEARCHING: GO RIGHT"

                        # We've sent the command now wait half a second and then send a halt (WE DON"T NEED THIS ANYMORE)
                        # time.sleep(searchingTimeToTurn)
                        # self.send_serial_command(Direction.STOP, b'h');
                        # time.sleep(searchingTimeToHalt)

                    else:  # Keep waiting - 2 seconds haven't elapsed
                        self.send_serial_command(Direction.STOP, b'h')
                        commandString = "STOP"

            else:  # We see the desired tag!

                # Reset firstTimeObjectNotSeen to None for the next time we can't find the tag
                if firstTimeObjectNotSeen is not None:
                    firstTimeObjectNotSeen = None

                # Set objectSeenOnce to True if isn't already
                if not objectSeenOnce:
                    objectSeenOnce = True

                # Get the corners and draw a minimally enclosing circle of it
                # and get the x/y/radius information of that circle to use for our navigation
                corners = np.array(tagObject.corners, dtype=np.float32).reshape((4, 2, 1))

                cornersList = []
                for c in corners:
                    cornersList.append([int(x) for x in c])

                cornersList = np.array(cornersList, dtype=np.int32)  # Turn the list into a numpy array
                ((x, y), radius) = cv2.minEnclosingCircle(cornersList)
                M = cv2.moments(cornersList)

                # Grab the desired information...
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                filteredPtsRadius = [radius]
                filteredPtsX = [center[0]]
                filteredPtsY = [center[1]]

                # Draw circle and center point on the frame
                cv2.circle(frame, (int(x), int(y)), int(filteredPtsRadius[0]), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                # Determine what command to send to the Arudino (motors)
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

            # Show FPS and number of halts (this stuff will be on the frame regardless of whether we see our desired
            # tag or not) (TODO delete this stuff later if we don't want it)
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
                self.send_serial_command(Direction.STOP, b'h');
                # Restore webcam settings
                if not inPosition or (inPosition and isLastUse):
                    # Reset the webcam to its original exposure and gain
                    self.WebcamVideoStreamObject.stream.set(cv2.CAP_PROP_EXPOSURE, self.originalExposure)
                    self.WebcamVideoStreamObject.stream.set(cv2.CAP_PROP_GAIN, self.originalGain)
                    # Activate auto_exposure (which is what the webcam starts out with by default but we mess it up
                    # by changing the exposure manually).
                    subprocess.check_call("v4l2-ctl -d /dev/video1 -c exposure_auto=3", shell=True)
                cv2.destroyAllWindows()
                break

    # Used to priortize the coordinate information provided by closer tags as opposed to the farther aways ones (
    # which would likely have less reliable information).
    @staticmethod
    def calc_weight(p1, p2):
        max_weight = 150
        dist = np.linalg.norm(p1 - p2)
        return max_weight / dist

    # Get an estimate of our current x and z coorindates and return them so they can be passed through to the chat
    # server and disseminated to the other teams (for the distress signal).
    def get_coordinates(self, cvQueue: Queue):

        # To get the coordinate, we rotate on our axis some X number of times to form images that compose a complete
        # 360 degree view of our surroundings. We use each image (as long as there are april tags in it) to get a (x,
        # z) coordinate value, and then we choose which (x,z) coordinate to return based off of which we deem the
        # most correct/reliable (this decision is shown in the code below)

        # When turning to search for the desiredTag, we specify time to turn, and time to wait after each semi-turn.
        # We do this because we want a stable photo/shot at each
        searchingTimeToTurn = 0.5  # seconds
        searchingTimeToHalt = 2.0  # seconds

        # Note that refine_pose is set to True (takes more work/processing but hopefully gets better coordinates)
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
        numIterations = 10

        while True:
            # Rotate camera by going left by some amount
            self.send_serial_command(Direction.LEFT, b'l')
            time.sleep(searchingTimeToTurn)
            self.send_serial_command(Direction.STOP, b'h')
            time.sleep(searchingTimeToHalt)

            # Now lets read the frame (while the robot is halted so that image is clean)
            frame = self.vs.read()
            if frame is None:
                print("ERROR - frame read a NONE")
                break

            # Use grayscale image for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = det.detect(gray)

            # Check how many tags we see... if it's 0 then ignore this frame and move on to capturing the next frame
            numTagsSeen = len(res)
            print("\nNumber of tags seen", numTagsSeen)  # TODO remove

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

                    cornersList = np.array(cornersList, dtype=np.int32)  # Turn into numpy array (openCV wants this)

                    # Draw circle around tag using its corners & get radius of that tag
                    ((x, y), radius) = cv2.minEnclosingCircle(cornersList)
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
                # Now get the average pose across the tags and get the largest tag radius that we saw.
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
                    coordinateToReturn = (
                        int(coordinateToReturn[0]), int(coordinateToReturn[1]))  # This stays regardless
                else:
                    coordinateToReturn = (0, -1)  # TODO set to some default outside the door

                # Cleanup and return
                cv2.destroyAllWindows()
                print("Value to return:")  # TODO remove
                print(coordinateToReturn)  # TODO remove
                return coordinateToReturn
            else:  # Still have iterations to go, increment the value
                iterationNumber += 1

            # Q to quit
            key = cv2.waitKey(1) & 0xFF
            if (key == ord("q")) or (not cvQueue.empty()):
                self.send_serial_command(Direction.STOP, b'h')
                cv2.destroyAllWindows()
                break

    # Follow a person (represented by a green folder). If the second argument (run_py_eyes) is set to True then we'll
    # create eyeballs using PyGame that follow the user (folder) as it moves around.
    def person_following(self, run_py_eyes, cvQueue: Queue):

        # Frame is considered to be 600x600 (after resize)
        # Below are variables to set what we consider center and in-range
        radiusInRangeLowerBound, radiusInRangeUpperBound = 80, 120
        centerRightBound, centerLeftBound = 400, 200
        radiusTooCloseLowerLimit = 250

        # Creating a window for later use
        cv2.namedWindow('result')
        cv2.resizeWindow('result', 600, 600)

        # Variables to 'smarten' the following procedure (see usage below)
        objectSeenOnce = False  # Object has never been seen before
        leftOrRightLastSent = None  # Keep track of whether we sent left or right last

        # TODO delete this block when done
        start = time.time()
        num_frames = 0

        # PyEyes Setup... Note that I've done some performance tinkering with pygame. Instead of redrawing the entire
        # frame on each iteration, I only turn the previously drawn pupils of the last frame white (to match the
        # background) and draw the new pupils. I also enable some performance enhancements and disable some unneeded
        # functionality. This kept out frame rate at a reliable level.
        if run_py_eyes:
            screen = pygame.display.set_mode((1024, 600), DOUBLEBUF)
            screen.set_alpha(None)  # Not needed, so set it to this for performance improvement
            surface = pygame.display.get_surface()
            # Draw the eyeballs (without pupils) and white background that we'll use for the rest of the process
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

            # TODO delete this block when done (if you want)
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
            mask = cv2.erode(mask, None, iterations=2)  # TODO: these were 3 or 5 before (more small blob removal)
            mask = cv2.dilate(mask, None, iterations=2)

            # find contours in the mask and initialize the current (x, y) center of the ball
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            commandString = None

            # Only proceed if at least one contour was found
            # If nothing is found, then look around OR send the STOP command to halt movement (depends on situation)
            if len(cnts) == 0:
                # If we haven't seen the object before, then we'll stay halted until we see one. If we HAVE seen the
                # object before, then we'll move in the direction (left or right) that we did most recently
                if not objectSeenOnce:
                    self.send_serial_command(Direction.STOP, b'h')
                    commandString = "STOP"
                else:  # Object has been seen before
                    if leftOrRightLastSent is not None:
                        if leftOrRightLastSent == Direction.RIGHT:
                            self.send_serial_command(Direction.RIGHT, b'r')
                            commandString = "SEARCHING: GO RIGHT"
                        elif leftOrRightLastSent == Direction.LEFT:
                            self.send_serial_command(Direction.LEFT, b'l')
                            commandString = "SEARCHING: GO LEFT"
                    else:  # variable hasn't been set yet (seems unlikely), but default to left
                        self.send_serial_command(Direction.LEFT, b'l')
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

                    #  draw the circle on the frame TODO consider removing this eventually - could speed things up (barely)
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
                # If our coordinates are out of bounds for the eyes, then cap them at their correct bounds
                if leftx < 106: leftx = 106
                if leftx > 406: leftx = 406
                if lefty < 150: lefty = 150
                if lefty > 450: lefty = 450
                # rightx = leftx + 400 + 112 - width
                # if rightx < 568:
                #     rightx = 568
                # if rightx > 968:
                #     rightx = 968

                # Note that the eyes could be made to get a little crossed eyed (close together) when you get very
                # close to the robot. It's not hard to do, but I didn't include it here (that's why the above lines
                # are commented out).

                # Draw left pupil
                rects.append(pygame.draw.circle(surface, self.RGB_BLACK, (leftx, lefty), self.EYEBALL_RADIUS, 0))
                # Draw right pupil
                rects.append(
                    pygame.draw.circle(surface, self.RGB_BLACK, (leftx + 500 + 12, lefty), self.EYEBALL_RADIUS, 0))
                # Update the display so our changes show up
                pygame.display.update(rects)
                # Save the left and right pupil circles so that we can remove them on the next iteration (instead of
                # clearing the whole display and redrawing it all (which is expensive))
                rects = [pygame.draw.circle(surface, self.RGB_WHITE, (leftx, lefty), self.EYEBALL_RADIUS, 0),
                         pygame.draw.circle(surface, self.RGB_WHITE, (leftx + 500 + 12, lefty), self.EYEBALL_RADIUS, 0)]

            # Close application on 'q' key press, or if the queue is not empty (there's some command to respond to).
            key = cv2.waitKey(1) & 0xFF
            if (key == ord("q")) or (not cvQueue.empty()):
                self.send_serial_command(Direction.STOP, b'h')
                # We've been requested to leave ...
                # Don't destroy everything - just destroy cv2 windows ... webcam still runs
                cv2.destroyAllWindows()
                if run_py_eyes:
                    pygame.display.quit()
                    pygame.quit()
                break


# Main driver that listens to the queue and initiates actions accordingly (this is called externally by our weaver
# class)
def run(cvQueue: Queue):
    # Initialize an object for the class - this will connect to the webcam and serial port and begin grabbing frames
    cvObject = OpenCVController()

    while True:
        if not cvQueue.empty():  # If there's something in the queue...

            commandFromQueue = cvQueue.get()
            cvQueue.task_done()

            if commandFromQueue == "terminate":
                cvObject.cleanup_resources()
                print("Terminate OpenCV")
                return
            elif commandFromQueue == "halt":
                cvObject.send_serial_command(Direction.STOP, b'h')
                print("Sent halt command")
            elif commandFromQueue == "getCoordinates":
                print("got command getCoordinates")
                x, z = cvObject.get_coordinates(cvQueue)
                print("sending coordinates")
                cvQueue.put(x)
                cvQueue.put(z)
                cvQueue.join()
                print("nonblocking")
            elif commandFromQueue == "personFollow":
                cvObject.person_following(False, cvQueue)
            elif commandFromQueue == "photo":
                cvObject.take_photo(cvQueue)
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

                print("getting my location")
                x_cord, z_cord = cvObject.get_coordinates(cvQueue)
                # See aprilTags class (in our codebase) and our final document for an explanation of what this does (
                # essentially creates a list of tags to go to and the desied distance for each tag - we'll pass each
                # of these steps into our april_following method one-by-one).
                first_tag = aprilTags.getClosestTag(x_cord, z_cord, True)

                print("first tag is " + str(first_tag))
                print("last tag is " + str(final_target_tag_number))
                cvObject.april_following(first_tag[0], first_tag[1], cvQueue, True, False)
                i = 0
                end_index = aprilTags.endOptions[final_target_tag_number]
                for target_pair in aprilTags.aprilTargets:
                    print("going to:" + str(target_pair[0]))
                    cvObject.april_following(target_pair[0], target_pair[1], cvQueue, False, False)
                    if i == end_index:
                        break
                    i += 1
                print("going to " + str(final_target_tag_number))
                cvObject.april_following(final_target_tag_number, final_target_tag_radius, cvQueue, False, True)

            elif commandFromQueue == "halt":
                pass


# The below is used when running openCVController from the command line (python3 openCVController). This' used to
# test functionality.
if __name__ == "__main__":
    classObject = OpenCVController()
    cvQueue = Queue()
    # classObject.runningPersonFollowing = True
    # classObject.april_following(22, 80,  cvQueue)
    # print("here?")
    classObject.april_following(41, 80, cvQueue, True, True)
    # classObject.person_following(True,  cvQueue)
    # print(str(classObject.get_coordinates(cvQueue)))
    # classObject.cleanup_resources()
    # classObject.take_photo(cvQueue)
    # exit()

    # cvQueue = Queue()
    # cvQueue.put("personFollow")
    # run(cvQueue)
