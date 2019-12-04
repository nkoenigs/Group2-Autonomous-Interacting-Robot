import socket
import sys
import errno
import threading
import time

my_username = "WoodyRobot"

HEADER_LENGTH = 10

IP = "10.104.29.62"
PORT = 1234

def create_header(strLen, headLen):
    result = "{}".format(strLen)
    resultLen = len(result)
    if resultLen < headLen:
        for x in range(headLen - resultLen) :
            result = result + " "
    return result

# Create a socket
# socket.AF_INET - address family, IPv4, some otehr possible are AF_INET6, AF_BLUETOOTH, AF_UNIX
# socket.SOCK_STREAM - TCP, conection-based, socket.SOCK_DGRAM - UDP, connectionless, datagrams, socket.SOCK_RAW - raw IP packets
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to a given ip and port
client_socket.connect((IP, PORT))

# Set connection to non-blocking state, so .recv() call won;t block, just return some exception we'll handle
client_socket.setblocking(False)

# Prepare username and header and send them
# We need to encode username to bytes, then count number of bytes and prepare header of fixed size, that we encode to bytes as well
username = my_username.encode('utf-8')
username_header = create_header(len(username), HEADER_LENGTH).encode('utf-8')
client_socket.send(username_header + username)


def send(x, z):
    # Encode message to bytes, prepare header and convert to bytes, like for username above, then send    
    message = "{} > distress: {},{}".format(my_username, x, z).encode('utf-8')
    message_header = create_header(len(message), HEADER_LENGTH).encode('utf-8')
    client_socket.send(message_header + message)


#used to be receive
def run(chatQueue):        
    goal_x = 0
    goal_z = 0	
    recent_x = 0
    recent_z = 0    
    while True:        
        try:
            # Now we want to loop over received messages (there might be more than one) and print them
            while True:
                # if queue is empty we check for distress signal
                # if queue is not empty we have several case such as: terminate, send(x,z)                    
                if not chatQueue.empty():
                    item = chatQueue.get()   
                    chatQueue.task_done()                 
                    if item == "terminate":
                        client_socket.close()                            
                        print("Terminate chat controller")
                        return
                    if item == "sendDistress":                        
                        x = chatQueue.get()
                        chatQueue.task_done()
                        z = chatQueue.get()
                        chatQueue.task_done()
                        send(x,z)


                # Receive our "header" containing username length, it's size is defined and constant
                username_header = client_socket.recv(HEADER_LENGTH)

                # If we received no data, server gracefully closed a connection, for example using socket.close() or socket.shutdown(socket.SHUT_RDWR)
                #if not len(username_header):
                    #print('Connection closed by the server')
                    #sys.exit()

                # Convert header to int value
                username_length = int(username_header.decode('utf-8').strip())

                # Receive and decode username
                username = client_socket.recv(username_length).decode('utf-8')

                # Now do the same for message (as we received username, we received whole message, there's no need to check if it has any length)
                message_header = client_socket.recv(HEADER_LENGTH)
                message_length = int(message_header.decode('utf-8').strip())
                message = client_socket.recv(message_length).decode('utf-8')

                # Print message                
                if(message[0] == 'd'):
                    m = message.split()                                    
                    if len(m) == 2:
                        m = m[1].split(',')                        
                        goal_x = float(m[0])
                        goal_z = float(m[1])                        
                    if len(m) == 3:
                        goal_x = float(m[1].split(',')[0])
                        goal_z = float(m[2])                                            
                    #advoid duplicate                                                         
                    if goal_x != recent_x and goal_z != recent_z:                        
                        recent_x = goal_x
                        recent_z = goal_z                    
                        print("Received Coordinate: ",goal_x,",",goal_z)
                        chatQueue.put("receivedDistress")
                        chatQueue.put(goal_x)
                        chatQueue.put(goal_z)
                        chatQueue.join()
                        while not chatQueue.empty():
                            pass #keep looping until the value is taken

        except IOError as e:
            # This is normal on non blocking connections - when there are no incoming data error is going to be raised
            # Some operating systems will indicate that using AGAIN, and some using WOULDBLOCK error code
            # We are going to check for both - if one of them - that's expected, means no incoming data, continue as normal
            # If we got different error code - something happened
            if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
                print('Reading error: {}'.format(str(e)))
                sys.exit()

            # We just did not receive anything
            continue

        except Exception as e:
            # Any other exception - something happened, exit
            print('Reading error: '.format(str(e)))
            sys.exit()
