# Group2-Autonomous-Interacting-Robot
group 2 master

to run first call ./ngrok http 5000
then update alexa endpoint to new URL
run python3 weaverv2.py

if you force quit weaver killall python3 processes

Weaver is main + Alexa backed 
Cv controller does everything needing opencv
Heartbeatcontorller is bonus fun thing 
Chat handles robot to robot comms
April is functions and constants for swarming

@other groups
Feel free to take weaver, 
and look at how to integrate other
Scripts as processes and threads.

For every get call task_done once or join will fail to sync
