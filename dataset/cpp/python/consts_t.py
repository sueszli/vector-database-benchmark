SERVER = 'server'
CLIENT = 'client'

LEFT_CLIENT = 'left'
RIGHT_CLIENT = 'right'

## -- Socket consts -- ##
SOCKET_TIMEOUT = 5
HEADER_LENGTH = 24
# IP = "192.168.20.9"		# server IP address
IP = "192.168.20.9"
# IP = "127.0.0.1"
PORT = 1234 

CHUNK_SIZE = 4096
REC_T = 1			# default recording time
REC_T_MAX = 5		# maximum recording time
CALIB_T = 10  # number of calibration images to take
CALIB_IMG_DELAY = 1 # seconds between each image
STREAM_MAX = 60 # maximum time for a stream
# CALIB_T = 10		# 10 seconds for calibration to take images

## -- Camera consts -- ##
FRAMERATE = 90

LEARNING_RATE = 0.15
BACKGROUND_RATE = 30

## -- LED consts -- ##
LED_F_MAX = 60		# max LED frequency
LED_F_MIN = 0.5		# min LED frequency
R_LED_F = 0			# default LED frequency
G_LED_F = 0
R_LED_PIN = 24		# red LED pin
G_LED_PIN = 23		# green LED pin

## -- message type definitions -- ##
TYPE_STR = "text"
TYPE_VAR = "var"

TYPE_REC = "record"
TYPE_CAP = "capture"
TYPE_IMG = "img"
TYPE_BALLS = "balls"
TYPE_DONE = "done"
TYPE_SHUTDOWN = "shutdown"
TYPE_STREAM = "stream"

## -- State definitions -- ##
STATE_IDLE = "idle"
STATE_RECORDING = "recording"
STATE_CAPTURING = "capturing"
STATE_STOP = "stop"
STATE_SHUTDOWN = "shutdown"
STATE_CALIBRATION = "calibration"

## -- Calibration consts -- ##
sensor_size = (3.68, 2.76)	# size of the image sensor on the camera
square_size = 23.4E-3		# size of squares on the chessboard
pattern_size = (9, 6)  		# number of points (where the black and white intersects)
MIN_PATTERNS = 25

## -- Filename consts -- ##
STEREO_CALIB_F = 'stereo_calib.npy'
LEFT_CALIB_F = 'calib_L.npy'
RIGHT_CALIB_F = 'calib_R.npy'