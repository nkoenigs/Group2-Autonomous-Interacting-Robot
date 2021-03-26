#!/usr/bin/python3

from tkinter import *
import serial
import serial.tools.list_ports              	# for listing serial ports
import sys              	# command line lib
import time
import glob
import os
 

#Decide which COM port to open
ser = None
ard_con = 2
if os.path.exists('/dev/ttyACM0') == True:
   ard_con = 0
   ser = serial.Serial('/dev/ttyACM0',115200,timeout = 1)
if os.path.exists('/dev/ttyACM1') == True:
   ard_con = 1
   ser = serial.Serial('/dev/ttyACM1',115200,timeout = 1)
if ard_con != 2:
   print(serialPort.read(11).decode('UTF-8'))

#Function to run when button is click
def upButtonClick():	
    textBox.configure(state="normal")
    textBox.insert(END, "Up\n")
    textBox.see(END)
    textBox.configure(state="disabled")
def downButtonClick():	
    textBox.configure(state="normal")
    textBox.insert(END, "Down\n")
    textBox.see(END)
    textBox.configure(state="disabled")
def leftButtonClick():	
    textBox.configure(state="normal")
    textBox.insert(END, "Left\n")
    textBox.see(END)
    textBox.configure(state="disabled")
def rightButtonClick():	
    textBox.configure(state="normal")
    textBox.insert(END, "Right\n")
    textBox.see(END)
    textBox.configure(state="disabled")
def stopButtonClick():	
    textBox.configure(state="normal")
    textBox.insert(END, "Stop\n")
    textBox.see(END)
    textBox.configure(state="disabled")

#Create & Configure root
root = Tk()
Grid.rowconfigure(root, 0, weight=1)
Grid.columnconfigure(root, 0, weight=1)
 
#Create textbox for display
textBox = Text(root, state="disabled")
textBox.grid(row=0, column=0, sticky=E+W)
 
#Create & Configure frame
frame=Frame(root)
frame.grid(row=1, column=0, sticky=E+W)
 
#Prepare for buttons layout on the screen
for row_index in range(8):
    Grid.rowconfigure(frame, row_index, weight=1)
    for col_index in range(5):
        Grid.columnconfigure(frame, col_index, weight=1)
 
#Up button setup
upButton = Button(frame, text="Up", command=upButtonClick, width=20, height=5)    	
upButton.grid(row=6, column=3)
 
#Down button setup
downButton = Button(frame, text="Down", command=downButtonClick, width=20, height=5)    	
downButton.grid(row=8, column=3)

#Left button setup
leftButton = Button(frame, text="Left", command=leftButtonClick, width=20, height=5)    	
leftButton.grid(row=7, column=2)

#Right button setup
rightButton = Button(frame, text="Right", command=rightButtonClick, width=20, height=5)    	
rightButton.grid(row=7, column=4)

#Stop button setup
stopButton = Button(frame, text="Stop", command=stopButtonClick, width=20, height=5)    	
stopButton.grid(row=7, column=3)

root.geometry("800x480")
root.title("The one and only one awesome software")
root.mainloop()

