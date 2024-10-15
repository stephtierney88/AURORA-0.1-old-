import datetime
import os
import random
import sys
import threading
import time
from io import BytesIO
import base64
import keyboard
import numpy as np
import openai
import pyautogui
import pydub
from pydub import AudioSegment
from pydub.playback import play
from pyscreeze import screenshot
import requests
import schedule
import sounddevice as sd
import tiktoken
from PIL import Image
from PIL import ImageGrab
from PIL import ImageDraw
from PIL import ImageFont # ImageFilter
import random
from threading import Lock
from threading import RLock
from scipy.io.wavfile import write
import tempfile
import ffmpeg
from getpass import getpass
import json
import io
import math
import queue
import re
import ctypes
from ctypes import wintypes
import winsound
from pygame import mixer
from PIL import Image, ImageDraw, ImageFont
from colorama import Fore, Style
import threading  # Ensure lock is available

API_KEY=                                                                                                                                                                                                                                                                                                                "sk-proj-SECRETKEY"
POWER_WORD = ""
REQUIRE_POWER_WORD = False
chat_history = []
tasks_list = []
apriority_tasks = []
notable_inventory = []
party_status = []
recent_context_events = []
pinned_immediate_actions = []
pinned_short_term_actions = []
recent_executed_actions = []
significant_past_tasks = []
inference_counter = 0
model_interval = 9999  # Switch to 'o1-preview' every 3 inferences
hierarchical_tasks = []  # Global variable to store the hierarchical tasks


CONTEXT_LENGTH = 25192  # or whatever the max token count for GPT-4 is
disable_commands = False  # Global boolean variable to track whether command processing is currently disabled
show_user_text = True
show_ai_text = True
hide_ai_commands = False
hide_user_commands = False
tokenizer = tiktoken.get_encoding("cl100k_base")
token_limit = 18100  # Set this to whatever limit you want
token_counter = 0  # This will keep track of the tokens used so far
character_name = "Aurora"
last_interaction_time = time.time()
init_handoff_in_progress = False
enable_unimportant_messages = True
enable_important_messages = True
global ADD_UMSGS_TO_HISTORY
global ADD_IMSGS_TO_HISTORY
ADD_UMSGS_TO_HISTORY = True
ADD_IMSGS_TO_HISTORY = True
global IMGS_DECAY # Decay time in minutes for important messages
global UMSGS_DECAY # Decay time in minutes for unimportant messages
global CHECK_UMSGS_DECAY
global CHECK_IMSGS_DECAY
global userName
global aiName
# Global variable to enable or disable model responses
ENABLE_MODEL_RESPONSES = True  # Set to False to disable model responses
INIT_MODE = 'test'  # Options: 'load', 'skip', 'test'
#load will load from init.txt file, test will use the test init prompt, skip will void the pinned init message.
# Define a default or test init prompt
test_init_prompt = "This is a test init prompt."

init_prompt = ""  # Initialize to an empty string
SEND_TO_MODEL = False  # Control sending text to the model
# Load a TrueType font with the specified size

# Customize the font size
font_size = 23
# Ensure this path points to a valid TrueType font file on your system
FONT_PATH = "times.ttf"  # Times New Roman font file, change to the correct path if needed
try:
    font = ImageFont.truetype(FONT_PATH, font_size)
except IOError:
    print("Font file not found. Falling back to default font.")
    font = ImageFont.load_default()  # Fallback to default font if TrueType font not found
CHECK_UMSGS_DECAY = True  # Initially set to True to enable decay checks for unimportant messages
CHECK_IMSGS_DECAY = True  # Initially set to True to enable decay checks for important messages
Always_ = False
hide_input = None
# Global variable to store the last executed command
global last_command
last_command = None
screen_width, screen_height = pyautogui.size()
MAX_IMAGES_IN_HISTORY = 0  #was 15 Global variable for the maximum number of images to retain
#image_detail =  "high"   # "low" or depending on your requirement
image_detail =  "low"   #was low, "high" or depending on your requirement
latest_image_detail = "low"  #was "high" for the latest image, "low" for older images
# Define the High_Detail global variable
global High_Detail
High_Detail = 0  #was 7  Initialize to 0 or set as needed  pos for recent high detail, neg for last
global Recent_images_High
Recent_images_High= 0 #was 7

image_timestamp = None
last_key = None  # Initialize the global variable
mouse_position = {"x": 0, "y": 0}  # Initialize as a dictionary

global queued_user_input
queued_user_input = queue.Queue()

global MAX_IMPORTANT_MESSAGES
global MAX_UNIMPORTANT_MESSAGES
MAX_IMPORTANT_MESSAGES = 100  # Example value
MAX_UNIMPORTANT_MESSAGES = 100  # Example value
IMGS_DECAY = 9999 # Setting the default decay time to 3 minutes for important messages
UMSGS_DECAY = 3.55 # Setting the default decay time to 3 minutes for unimportant messages
time_interval = 0.35 # Time interval between screenshots (in seconds)

cursor_size = 20  # Size of the cursor representation
enable_human_like_click = True
circle_duration = 0.5  # Duration for the circular motion in seconds
circle_radius = 10  # Radius of the circular motion

text_count = 0
image_or_other_count = 0
# Default values defined at the beginning of your script
default_scroll_amount = 100  # Default scroll amount
default_double_click_speed = 0.5  # Default double-click speed

#script_directory = os.path.dirname(os.path.abspath(__file__))
script_directory = 'C:\\Users\\thebeast\\OneDrive\\Desktop\\AURORA'
# Set the sampling frequency (fs) to 48000 and channels to 4
fs = 48000
channels = 2

# Set the default device (verify that device 20 is indeed the correct device)
sd.default.device = 1

lock = RLock()
mouse_position_lock = RLock()
last_key_lock = RLock()
image_history_lock = RLock()
#queued_input_lock = RLock() #RLock? 


enable_colored_tile_grid = False  # Set to True if you want to enable it
enable_unit_labels = False        # Set to True if you want to enable TU/RU/LU/DU labels


# Parameters
threshold = 16.11  
audio_pause_duration =  0.7  #1.8  this is for audio
#sampling_rate = 44100 
sampling_rate = fs 
audio_buffer = []


# Alias for Style.RESET_ALL
RESET = Style.RESET_ALL
# Initialize the mixer
mixer.init()

# Load different sounds for different stages
chimeswav = "C:\Windows\Media\chimes.wav"
chordwav = "C:\Windows\Media\chord.wav"
alarm3 = "C:\Windows\Media\Alarm03.wav"
alarm10 = "C:\Windows\Media\Alarm10.wav"
tada_wav = "C:\Windows\Media\tada.wav"
speech_off_wav = "C:\Windows\Media\Speech Off.wav"
speech_on_wav = "C:\Windows\Media\Speech On.wav"
speech_sleep_wav = "C:\Windows\Media\Speech Sleep.wav"
winringin_wav = "C:\Windows\Media\Windows Ringin.wav"
winunlock_wav= "C:\Windows\Media\Windows Unlock.wav"

mixer.music.set_volume(0.8)  # Set global volume, adjust as needed

def ftime():
    return datetime.datetime.now().strftime('%M:%S.%f')[:-4]

# Color map to store each function's assigned color
func_colors = {}

# List of distinct bright colors
color_list = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.LIGHTRED_EX, Fore.LIGHTGREEN_EX, Fore.LIGHTYELLOW_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTMAGENTA_EX, Fore.LIGHTCYAN_EX]

# Function to assign colors to functions in a stable manner
def get_color(func_name):
    if func_name not in func_colors:
        # Assign the next available color from the list
        color = color_list[len(func_colors) % len(color_list)]
        func_colors[func_name] = color
    return func_colors[func_name]




# Time logger decorator
def time_logger(func):
    def wrapper(*args, **kwargs):
        start_time = ftime()
        color = get_color(func.__name__)
        print(f"{color}Starting function '{func.__name__}' at {start_time}{Style.RESET_ALL}")
        
        result = func(*args, **kwargs)
        
        end_time = ftime()
        print(f"{color}Function '{func.__name__}' completed at {end_time}{Style.RESET_ALL}")
        
        # Calculate the time difference (convert to datetime for calculation)
        start_dt = datetime.datetime.strptime(start_time, '%M:%S.%f')
        end_dt = datetime.datetime.strptime(end_time, '%M:%S.%f')
        time_diff = end_dt - start_dt
        
        print(f"{color}Time taken for '{func.__name__}': {time_diff}{Style.RESET_ALL}")
        return result
    return wrapper



# Additional global variables
AUTO_PROMPT_INTERVAL = 9  # DEPRECIATED..  Auto-prompt every 30 seconds, but you can change this value # Default auto message depreciated in favor of send_screenshot
enable_auto_prompt = False
auto_prompt_message = "Auto Prompt."  # Default auto message depreciated in favor of send_screenshot which also has text... 


# Function to run the audio stream

def run_audio_stream():
    with sd.InputStream(callback=audio_callback):
        sd.sleep(100000000)  # This will keep the audio stream open indefinitely. Adjust as needed.





last_print_time = time.time()


def audio_callback(indata, frames, time_info, status):
    global last_print_time
    global queued_user_input
    volume_norm = np.linalg.norm(indata) * 10

    if volume_norm < threshold and SEND_TO_MODEL:
        if len(audio_buffer) > sampling_rate * audio_pause_duration:  
            if not ENABLE_MODEL_RESPONSES:
                print("Model responses are disabled. Skipping Whisper API call.")
                audio_buffer.clear()
                return
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=script_directory) as temp_file:
                audio_data = np.array(audio_buffer, dtype=np.float32)
                audio_segment = AudioSegment(
                    audio_data.tobytes(),
                    frame_rate=sampling_rate,
                    sample_width=audio_data.dtype.itemsize,
                    channels=2
                )

                # Adjust the sample width to a valid value (e.g., 2)
                audio_segment = audio_segment.set_sample_width(2)

                write(temp_file.name, sampling_rate, audio_data.astype('float32'))
                response = openai.Audio.transcribe(
                    api_key=API_KEY,
                    model="whisper-1",
                    file=open(temp_file.name, "rb")
                )
            print("My Response: ", response)     

            # Assuming the transcribed text is in `response['text']`
            transcribed_text = response['text']

            # Check if sending text to model is enabled
            if SEND_TO_MODEL:
                # Put the transcribed text into the queue
                queued_user_input.put(transcribed_text)
            else:
                print("Sending text to model is disabled. Ignoring transcribed text.")

            audio_buffer.clear()  
    else:
        audio_buffer.extend(indata.tolist())





# Start the audio stream in a separate thread
audio_thread = threading.Thread(target=run_audio_stream)
audio_thread.daemon = True
audio_thread.start()

def threshold_check(current_token_count, total_tokens, threshold_percentage):
    threshold = total_tokens * threshold_percentage / 100
    return current_token_count > threshold




# Define missing handle types
HCURSOR = ctypes.c_void_p
HICON = ctypes.c_void_p
HBITMAP = ctypes.c_void_p
HDC = ctypes.c_void_p
HGDIOBJ = ctypes.c_void_p

# Windows structures and constants (as before)
class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long),
                ("y", ctypes.c_long)]

# Define necessary Windows structures
class CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hCursor", HCURSOR),
        ("ptScreenPos", wintypes.POINT),
    ]

class ICONINFO(ctypes.Structure):
    _fields_ = [
        ("fIcon", wintypes.BOOL),
        ("xHotspot", wintypes.DWORD),
        ("yHotspot", wintypes.DWORD),
        ("hbmMask", HBITMAP),
        ("hbmColor", HBITMAP),
    ]

class BITMAP(ctypes.Structure):
    _fields_ = [
        ("bmType", wintypes.LONG),
        ("bmWidth", wintypes.LONG),
        ("bmHeight", wintypes.LONG),
        ("bmWidthBytes", wintypes.LONG),
        ("bmPlanes", wintypes.WORD),
        ("bmBitsPixel", wintypes.WORD),
        ("bmBits", ctypes.c_void_p),
    ]

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]

class RGBQUAD(ctypes.Structure):
    _fields_ = [
        ("rgbBlue", wintypes.BYTE),
        ("rgbGreen", wintypes.BYTE),
        ("rgbRed", wintypes.BYTE),
        ("rgbReserved", wintypes.BYTE),
    ]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", RGBQUAD * 1),  # Adjust the array size if needed
    ]



# Constants
BI_RGB = 0
DIB_RGB_COLORS = 0
SRCCOPY = 0x00CC0020
# Define the argument types and return types for the Windows API functions

# For PatBlt
ctypes.windll.gdi32.PatBlt.argtypes = [HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
ctypes.windll.gdi32.PatBlt.restype = wintypes.BOOL


# For GetObjectW
ctypes.windll.gdi32.GetObjectW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
ctypes.windll.gdi32.GetObjectW.restype = ctypes.c_int

# For SelectObject
ctypes.windll.gdi32.SelectObject.argtypes = [HDC, HGDIOBJ]
ctypes.windll.gdi32.SelectObject.restype = HGDIOBJ

# For DeleteObject
ctypes.windll.gdi32.DeleteObject.argtypes = [HGDIOBJ]
ctypes.windll.gdi32.DeleteObject.restype = wintypes.BOOL

# For GetDC
ctypes.windll.user32.GetDC.argtypes = [wintypes.HWND]
ctypes.windll.user32.GetDC.restype = HDC

# For ReleaseDC
ctypes.windll.user32.ReleaseDC.argtypes = [wintypes.HWND, HDC]
ctypes.windll.user32.ReleaseDC.restype = wintypes.INT

# For CreateCompatibleDC
ctypes.windll.gdi32.CreateCompatibleDC.argtypes = [HDC]
ctypes.windll.gdi32.CreateCompatibleDC.restype = HDC

# For DeleteDC
ctypes.windll.gdi32.DeleteDC.argtypes = [HDC]
ctypes.windll.gdi32.DeleteDC.restype = wintypes.BOOL

# For GetDIBits
ctypes.windll.gdi32.GetDIBits.argtypes = [
    HDC,             # hdc
    HBITMAP,         # hbm
    wintypes.UINT,   # start
    wintypes.UINT,   # cLines
    ctypes.c_void_p, # lpvBits
    ctypes.POINTER(BITMAPINFO),  # lpbi
    wintypes.UINT    # usage
]
ctypes.windll.gdi32.GetDIBits.restype = wintypes.INT

# GetDC
ctypes.windll.user32.GetDC.argtypes = [wintypes.HWND]
ctypes.windll.user32.GetDC.restype = HDC

# ReleaseDC
ctypes.windll.user32.ReleaseDC.argtypes = [wintypes.HWND, HDC]
ctypes.windll.user32.ReleaseDC.restype = wintypes.INT

# CreateCompatibleDC
ctypes.windll.gdi32.CreateCompatibleDC.argtypes = [HDC]
ctypes.windll.gdi32.CreateCompatibleDC.restype = HDC

# DeleteDC
ctypes.windll.gdi32.DeleteDC.argtypes = [HDC]
ctypes.windll.gdi32.DeleteDC.restype = wintypes.BOOL

# CreateCompatibleBitmap
ctypes.windll.gdi32.CreateCompatibleBitmap.argtypes = [HDC, ctypes.c_int, ctypes.c_int]
ctypes.windll.gdi32.CreateCompatibleBitmap.restype = HBITMAP

# SelectObject
ctypes.windll.gdi32.SelectObject.argtypes = [HDC, HGDIOBJ]
ctypes.windll.gdi32.SelectObject.restype = HGDIOBJ

# DeleteObject
ctypes.windll.gdi32.DeleteObject.argtypes = [HGDIOBJ]
ctypes.windll.gdi32.DeleteObject.restype = wintypes.BOOL

# DrawIconEx
ctypes.windll.user32.DrawIconEx.argtypes = [HDC, ctypes.c_int, ctypes.c_int, HICON, ctypes.c_int, ctypes.c_int, ctypes.c_uint, wintypes.HBRUSH, ctypes.c_uint]
ctypes.windll.user32.DrawIconEx.restype = wintypes.BOOL

# SetBkMode
ctypes.windll.gdi32.SetBkMode.argtypes = [HDC, ctypes.c_int]
ctypes.windll.gdi32.SetBkMode.restype = ctypes.c_int

# GetDIBits
ctypes.windll.gdi32.GetDIBits.argtypes = [
    HDC,             # hdc
    HBITMAP,         # hbmp
    wintypes.UINT,   # uStartScan
    wintypes.UINT,   # cScanLines
    ctypes.c_void_p, # lpvBits
    ctypes.POINTER(BITMAPINFO),  # lpbi
    wintypes.UINT    # uUsage
]
ctypes.windll.gdi32.GetDIBits.restype = wintypes.INT

# GetCursorInfo
ctypes.windll.user32.GetCursorInfo.argtypes = [ctypes.POINTER(CURSORINFO)]
ctypes.windll.user32.GetCursorInfo.restype = wintypes.BOOL

# GetIconInfo
ctypes.windll.user32.GetIconInfo.argtypes = [HCURSOR, ctypes.POINTER(ICONINFO)]
ctypes.windll.user32.GetIconInfo.restype = wintypes.BOOL


# Initialize the lock for thread safety
lock = threading.RLock()  #Rlock or lock here?

#def get_cursor():
#    # Initialize CURSORINFO
#    cursor_info = CURSORINFO()
#    cursor_info.cbSize = ctypes.sizeof(CURSORINFO)
#    if not ctypes.windll.user32.GetCursorInfo(ctypes.byref(cursor_info)):
#        return None, None, None
#
#    hicon = cursor_info.hCursor
#    if not hicon:
#        return None, None, None
#
#    # Get ICONINFO
#    icon_info = ICONINFO()
#    if not ctypes.windll.user32.GetIconInfo(hicon, ctypes.byref(icon_info)):
#        return None, None, None
#
#    # Get cursor position
#    x, y = cursor_info.ptScreenPos.x, cursor_info.ptScreenPos.y
#
#    # Get cursor hotspot
#    hotspot = (icon_info.xHotspot, icon_info.yHotspot)
#
#    # Convert HBITMAP to PIL Image
#    if icon_info.hbmColor:
#        cursor_image = hbitmap_to_pil_image(icon_info.hbmColor)
#    else:
#        cursor_image = hbitmap_to_pil_image(icon_info.hbmMask)
#        if cursor_image is not None:
#            # Convert mask image to RGBA
#            cursor_image = cursor_image.convert('RGBA')
#            # Apply a default color, e.g., black
#            cursor_image_data = cursor_image.getdata()
#            new_data = []
#            for item in cursor_image_data:
#                if item == 0:
#                    new_data.append((0, 0, 0, 0))  # Transparent
#                else:
#                    new_data.append((0, 0, 0, 255))  # Black
#            cursor_image.putdata(new_data)
#
#    if cursor_image is None:
#        return None, None, None
#
#    # Clean up GDI objects
#    ctypes.windll.gdi32.DeleteObject(icon_info.hbmColor)
#    ctypes.windll.gdi32.DeleteObject(icon_info.hbmMask)
#
#    return cursor_image, hotspot, (x, y)

#def get_cursor():
#    # Initialize CURSORINFO
#    cursor_info = CURSORINFO()
#    cursor_info.cbSize = ctypes.sizeof(CURSORINFO)
#    if not ctypes.windll.user32.GetCursorInfo(ctypes.byref(cursor_info)):
#        return None, None, None
#
#    hicon = cursor_info.hCursor
#    if not hicon:
#        return None, None, None
#
#    # Get cursor position
#    x, y = cursor_info.ptScreenPos.x, cursor_info.ptScreenPos.y
#
#    # Get ICONINFO
#    icon_info = ICONINFO()
#    if not ctypes.windll.user32.GetIconInfo(hicon, ctypes.byref(icon_info)):
#        return None, None, None
#
#    hotspot = (icon_info.xHotspot, icon_info.yHotspot)
#
#    # Get cursor size
#    cursor_w = ctypes.windll.user32.GetSystemMetrics(13)  # SM_CXCURSOR = 13
#    cursor_h = ctypes.windll.user32.GetSystemMetrics(14)  # SM_CYCURSOR = 14
#
#    # Create a compatible DC and bitmap
#    hdc = ctypes.windll.user32.GetDC(None)
#    if not hdc:
#        print("GetDC failed.")
#        return None, None, None
#
#    mem_dc = ctypes.windll.gdi32.CreateCompatibleDC(hdc)
#    if not mem_dc:
#        print("CreateCompatibleDC failed.")
#        ctypes.windll.user32.ReleaseDC(None, hdc)
#        return None, None, None
#
#    hbitmap = ctypes.windll.gdi32.CreateCompatibleBitmap(hdc, cursor_w, cursor_h)
#    if not hbitmap:
#        print("CreateCompatibleBitmap failed.")
#        ctypes.windll.gdi32.DeleteDC(mem_dc)
#        ctypes.windll.user32.ReleaseDC(None, hdc)
#        return None, None, None
#
#    old_obj = ctypes.windll.gdi32.SelectObject(mem_dc, hbitmap)
#    if not old_obj:
#        print("SelectObject failed.")
#        ctypes.windll.gdi32.DeleteObject(hbitmap)
#        ctypes.windll.gdi32.DeleteDC(mem_dc)
#        ctypes.windll.user32.ReleaseDC(None, hdc)
#        return None, None, None
#
#    # Fill the background with transparency (optional)
#    ctypes.windll.gdi32.SetBkMode(mem_dc, 1)  # TRANSPARENT
#
#    # Draw the cursor into the memory DC
#    if not ctypes.windll.user32.DrawIconEx(
#        mem_dc,
#        0,
#        0,
#        hicon,
#        cursor_w,
#        cursor_h,
#        0,
#        None,
#        0x0003  # DI_NORMAL
#    ):
#        print("DrawIconEx failed.")
#        ctypes.windll.gdi32.SelectObject(mem_dc, old_obj)
#        ctypes.windll.gdi32.DeleteObject(hbitmap)
#        ctypes.windll.gdi32.DeleteDC(mem_dc)
#        ctypes.windll.user32.ReleaseDC(None, hdc)
#        return None, None, None
#
#    # Prepare bitmap info
#    bmi = BITMAPINFO()
#    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
#    bmi.bmiHeader.biWidth = cursor_w
#    bmi.bmiHeader.biHeight = -cursor_h  # Negative for top-down DIB
#    bmi.bmiHeader.biPlanes = 1
#    bmi.bmiHeader.biBitCount = 32
#    bmi.bmiHeader.biCompression = BI_RGB
#
#    buf_size = cursor_w * cursor_h * 4
#    buffer = (ctypes.c_byte * buf_size)()
#
#    res = ctypes.windll.gdi32.GetDIBits(
#        mem_dc,
#        hbitmap,
#        0,
#        cursor_h,
#        buffer,
#        ctypes.byref(bmi),
#        DIB_RGB_COLORS
#    )
#    if res == 0:
#        print("GetDIBits failed.")
#        ctypes.windll.gdi32.SelectObject(mem_dc, old_obj)
#        ctypes.windll.gdi32.DeleteObject(hbitmap)
#        ctypes.windll.gdi32.DeleteDC(mem_dc)
#        ctypes.windll.user32.ReleaseDC(None, hdc)
#        return None, None, None
#
#    # Create PIL Image from buffer
#    image = Image.frombuffer('RGBA', (cursor_w, cursor_h), buffer, 'raw', 'BGRA', 0, 1)
#
#    # Clean up
#    ctypes.windll.gdi32.SelectObject(mem_dc, old_obj)
#    ctypes.windll.gdi32.DeleteObject(hbitmap)
#    ctypes.windll.gdi32.DeleteDC(mem_dc)
#    ctypes.windll.user32.ReleaseDC(None, hdc)
#    ctypes.windll.gdi32.DeleteObject(icon_info.hbmColor)
#    ctypes.windll.gdi32.DeleteObject(icon_info.hbmMask)
#
#    return image, hotspot, (x, y)

def get_cursor():
    # Initialize CURSORINFO
    cursor_info = CURSORINFO()
    cursor_info.cbSize = ctypes.sizeof(CURSORINFO)
    if not ctypes.windll.user32.GetCursorInfo(ctypes.byref(cursor_info)):
        print("GetCursorInfo failed.")
        return None, None, None

    hicon = cursor_info.hCursor
        # Get cursor position
    x, y = cursor_info.ptScreenPos.x, cursor_info.ptScreenPos.y
    cursor_position = (x, y)

    if not hicon:
        print("No cursor handle.")
        return None, None, cursor_position

    # Get cursor position
    x, y = cursor_info.ptScreenPos.x, cursor_info.ptScreenPos.y

    # Get ICONINFO to retrieve hotspot
    icon_info = ICONINFO()
    if not ctypes.windll.user32.GetIconInfo(hicon, ctypes.byref(icon_info)):
        print("GetIconInfo failed.")
        return None, None, None

    hotspot = (icon_info.xHotspot, icon_info.yHotspot)

    # Get cursor size
    cursor_w = ctypes.windll.user32.GetSystemMetrics(13)  # SM_CXCURSOR = 13
    cursor_h = ctypes.windll.user32.GetSystemMetrics(14)  # SM_CYCURSOR = 14

    # Create a compatible DC and bitmap
    hdcScreen = ctypes.windll.user32.GetDC(None)
    hdcMem = ctypes.windll.gdi32.CreateCompatibleDC(hdcScreen)
    hbmp = ctypes.windll.gdi32.CreateCompatibleBitmap(hdcScreen, cursor_w, cursor_h)
    ctypes.windll.gdi32.SelectObject(hdcMem, hbmp)

    # Fill the background with white color to ensure visibility
    brush = ctypes.windll.gdi32.GetStockObject(0)  # WHITE_BRUSH = 0
    rect = RECT(0, 0, cursor_w, cursor_h)
    ctypes.windll.user32.FillRect(hdcMem, ctypes.byref(rect), brush)

    # Draw the cursor into the memory DC
    if not ctypes.windll.user32.DrawIconEx(
        hdcMem,
        0,
        0,
        hicon,
        cursor_w,
        cursor_h,
        0,
        None,
        0x0003  # DI_NORMAL
    ):
        print("DrawIconEx failed.")
        ctypes.windll.gdi32.DeleteObject(hbmp)
        ctypes.windll.gdi32.DeleteDC(hdcMem)
        ctypes.windll.user32.ReleaseDC(None, hdcScreen)
        return None, None, None

    # Prepare bitmap info
    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = cursor_w
    bmi.bmiHeader.biHeight = -cursor_h  # Negative for top-down DIB
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = BI_RGB

    buf_size = cursor_w * cursor_h * 4
    buffer = (ctypes.c_byte * buf_size)()

    res = ctypes.windll.gdi32.GetDIBits(
        hdcMem,
        hbmp,
        0,
        cursor_h,
        buffer,
        ctypes.byref(bmi),
        DIB_RGB_COLORS
    )
    if res == 0:
        print("GetDIBits failed.")
        ctypes.windll.gdi32.DeleteObject(hbmp)
        ctypes.windll.gdi32.DeleteDC(hdcMem)
        ctypes.windll.user32.ReleaseDC(None, hdcScreen)
        return None, None, None

    # Create PIL Image from buffer
    image = Image.frombuffer('RGBA', (cursor_w, cursor_h), buffer, 'raw', 'BGRA', 0, 1)

    # Clean up
    ctypes.windll.gdi32.DeleteObject(hbmp)
    ctypes.windll.gdi32.DeleteDC(hdcMem)
    ctypes.windll.user32.ReleaseDC(None, hdcScreen)
    ctypes.windll.gdi32.DeleteObject(icon_info.hbmColor)
    ctypes.windll.gdi32.DeleteObject(icon_info.hbmMask)

    # For debugging purposes, save the cursor image
    # image.save('cursor_debug.png')

    return image, hotspot, cursor_position


# Define RECT structure
class RECT(ctypes.Structure):
    _fields_ = [
        ('left', ctypes.c_long),
        ('top', ctypes.c_long),
        ('right', ctypes.c_long),
        ('bottom', ctypes.c_long),
    ]

# Define FillRect function
ctypes.windll.user32.FillRect.argtypes = [HDC, ctypes.POINTER(RECT), wintypes.HBRUSH]
ctypes.windll.user32.FillRect.restype = ctypes.c_int

# Define GetStockObject function
ctypes.windll.gdi32.GetStockObject.argtypes = [ctypes.c_int]
ctypes.windll.gdi32.GetStockObject.restype = wintypes.HGDIOBJ

# Constants for stock objects
WHITE_BRUSH = 0

def hbitmap_to_pil_color_cursor(icon_info):
    hdc = ctypes.windll.user32.GetDC(None)
    mem_dc = ctypes.windll.gdi32.CreateCompatibleDC(hdc)
    old_bitmap = ctypes.windll.gdi32.SelectObject(mem_dc, icon_info.hbmColor)

    bmp = BITMAP()
    ctypes.windll.gdi32.GetObjectW(icon_info.hbmColor, ctypes.sizeof(BITMAP), ctypes.byref(bmp))

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = bmp.bmWidth
    bmi.bmiHeader.biHeight = -bmp.bmWidth  # Negative indicates top-down bitmap
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = BI_RGB

    buf_size = bmp.bmWidth * bmp.bmHeight * 4
    buffer = (ctypes.c_byte * buf_size)()

    res = ctypes.windll.gdi32.GetDIBits(
        mem_dc,
        icon_info.hbmColor,
        0,
        bmp.bmHeight,
        buffer,
        ctypes.byref(bmi),
        DIB_RGB_COLORS
    )
    if res == 0:
        print("GetDIBits failed for color cursor.")
        ctypes.windll.gdi32.SelectObject(mem_dc, old_bitmap)
        ctypes.windll.gdi32.DeleteDC(mem_dc)
        ctypes.windll.user32.ReleaseDC(None, hdc)
        return None

    image = Image.frombuffer('RGBA', (bmp.bmWidth, bmp.bmHeight), buffer, 'raw', 'BGRA', 0, 1)

    ctypes.windll.gdi32.SelectObject(mem_dc, old_bitmap)
    ctypes.windll.gdi32.DeleteDC(mem_dc)
    ctypes.windll.user32.ReleaseDC(None, hdc)

    return image

def hbitmap_to_pil_monochrome_cursor(icon_info):
    hdc = ctypes.windll.user32.GetDC(None)
    mem_dc = ctypes.windll.gdi32.CreateCompatibleDC(hdc)

    # Get mask bitmap dimensions
    bmp = BITMAP()
    ctypes.windll.gdi32.GetObjectW(icon_info.hbmMask, ctypes.sizeof(BITMAP), ctypes.byref(bmp))
    width = bmp.bmWidth
    height = bmp.bmHeight // 2  # The mask bitmap contains both AND and XOR masks, so we divide by 2

    # Create a bitmap for the AND mask
    and_mask_bmp = ctypes.windll.gdi32.CreateBitmap(width, height, 1, 1, None)
    ctypes.windll.gdi32.SelectObject(mem_dc, and_mask_bmp)
    ctypes.windll.gdi32.BitBlt(mem_dc, 0, 0, width, height, None, 0, 0, 0x00CC0020)  # SRCCOPY

    # Get the AND mask bits
    and_bmi = BITMAPINFO()
    and_bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    and_bmi.bmiHeader.biWidth = width
    and_bmi.bmiHeader.biHeight = -height
    and_bmi.bmiHeader.biPlanes = 1
    and_bmi.bmiHeader.biBitCount = 1
    and_bmi.bmiHeader.biCompression = BI_RGB

    and_buf_size = width * height // 8
    and_buffer = (ctypes.c_byte * and_buf_size)()

    res = ctypes.windll.gdi32.GetDIBits(
        mem_dc,
        icon_info.hbmMask,
        0,
        height,
        and_buffer,
        ctypes.byref(and_bmi),
        DIB_RGB_COLORS
    )
    if res == 0:
        print("GetDIBits failed for monochrome cursor AND mask.")
        ctypes.windll.gdi32.DeleteObject(and_mask_bmp)
        ctypes.windll.gdi32.DeleteDC(mem_dc)
        ctypes.windll.user32.ReleaseDC(None, hdc)
        return None

    # Now get the XOR mask
    xor_mask_bmp = ctypes.windll.gdi32.CreateBitmap(width, height, 1, 1, None)
    ctypes.windll.gdi32.SelectObject(mem_dc, xor_mask_bmp)
    ctypes.windll.gdi32.BitBlt(mem_dc, 0, 0, width, height, None, 0, height, 0x00CC0020)  # SRCCOPY

    # Get the XOR mask bits
    xor_bmi = BITMAPINFO()
    xor_bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    xor_bmi.bmiHeader.biWidth = width
    xor_bmi.bmiHeader.biHeight = height
    xor_bmi.bmiHeader.biPlanes = 1
    xor_bmi.bmiHeader.biBitCount = 1
    xor_bmi.bmiHeader.biCompression = BI_RGB

    xor_buf_size = width * height // 8
    xor_buffer = (ctypes.c_byte * xor_buf_size)()

    res = ctypes.windll.gdi32.GetDIBits(
        mem_dc,
        icon_info.hbmMask,
        height,
        height,
        xor_buffer,
        ctypes.byref(xor_bmi),
        DIB_RGB_COLORS
    )
    if res == 0:
        print("GetDIBits failed for monochrome cursor XOR mask.")
        ctypes.windll.gdi32.DeleteObject(and_mask_bmp)
        ctypes.windll.gdi32.DeleteObject(xor_mask_bmp)
        ctypes.windll.gdi32.DeleteDC(mem_dc)
        ctypes.windll.user32.ReleaseDC(None, hdc)
        return None

    # Combine the masks to create the final image
    and_image = Image.frombytes('1', (width, height), and_buffer, 'raw')
    xor_image = Image.frombytes('1', (width, height), xor_buffer, 'raw')

    # Create the final image
    final_image = Image.new('RGBA', (width, height))

    for x in range(width):
        for y in range(height):
            and_pixel = and_image.getpixel((x, y))
            xor_pixel = xor_image.getpixel((x, y))

            if and_pixel == 0:
                # Transparent pixel
                final_image.putpixel((x, y), (0, 0, 0, 0))
            else:
                if xor_pixel == 0:
                    # Black pixel
                    final_image.putpixel((x, y), (0, 0, 0, 255))
                else:
                    # White pixel
                    final_image.putpixel((x, y), (255, 255, 255, 255))

    # Clean up
    ctypes.windll.gdi32.DeleteObject(and_mask_bmp)
    ctypes.windll.gdi32.DeleteObject(xor_mask_bmp)
    ctypes.windll.gdi32.DeleteDC(mem_dc)
    ctypes.windll.user32.ReleaseDC(None, hdc)

    return final_image


def hbitmap_to_pil_image(hbitmap):
    if not hbitmap:
        return None

    # Get bitmap information
    bmp = BITMAP()
    res = ctypes.windll.gdi32.GetObjectW(hbitmap, ctypes.sizeof(BITMAP), ctypes.byref(bmp))
    if res == 0:
        print("GetObjectW failed.")
        return None

    # Print bitmap information
    print(f"Bitmap Type: {bmp.bmType}")
    print(f"Bitmap Width: {bmp.bmWidth}")
    print(f"Bitmap Height: {bmp.bmHeight}")
    print(f"Bitmap WidthBytes: {bmp.bmWidthBytes}")
    print(f"Bitmap Planes: {bmp.bmPlanes}")
    print(f"Bitmap BitsPixel: {bmp.bmBitsPixel}")

    # Calculate bytes per pixel
    bytes_per_pixel = max(bmp.bmBitsPixel // 8, 1)  # Ensure at least 1 byte per pixel

    # Prepare buffer
    buf_size = bmp.bmWidthBytes * bmp.bmHeight
    buffer = (ctypes.c_byte * buf_size)()

    # Create compatible DC
    hdc = ctypes.windll.user32.GetDC(None)
    mem_dc = ctypes.windll.gdi32.CreateCompatibleDC(hdc)
    if not mem_dc:
        print("CreateCompatibleDC failed.")
        ctypes.windll.user32.ReleaseDC(None, hdc)
        return None

    # Select bitmap into DC
    old_bitmap = ctypes.windll.gdi32.SelectObject(mem_dc, hbitmap)
    if not old_bitmap:
        print("SelectObject failed.")
        ctypes.windll.gdi32.DeleteDC(mem_dc)
        ctypes.windll.user32.ReleaseDC(None, hdc)
        return None

    # Set up bitmap info
    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = bmp.bmWidth
    bmi.bmiHeader.biHeight = bmp.bmHeight  # Positive for bottom-up DIB
    bmi.bmiHeader.biPlanes = bmp.bmPlanes
    bmi.bmiHeader.biBitCount = bmp.bmBitsPixel
    bmi.bmiHeader.biCompression = BI_RGB
    bmi.bmiHeader.biSizeImage = buf_size

    # Retrieve the bitmap bits
    res = ctypes.windll.gdi32.GetDIBits(
        mem_dc,
        hbitmap,
        0,
        bmp.bmHeight,
        buffer,
        bmi,
        DIB_RGB_COLORS
    )

    if res == 0:
        print("GetDIBits failed.")
        ctypes.windll.gdi32.SelectObject(mem_dc, old_bitmap)
        ctypes.windll.gdi32.DeleteDC(mem_dc)
        ctypes.windll.user32.ReleaseDC(None, hdc)
        return None

    # Determine the image mode and raw mode
    if bmp.bmBitsPixel == 32:
        mode = 'RGBA'
        raw_mode = 'BGRA'
    elif bmp.bmBitsPixel == 24:
        mode = 'RGB'
        raw_mode = 'BGR'
    elif bmp.bmBitsPixel == 16:
        mode = 'RGB'
        raw_mode = 'BGR;16'
    elif bmp.bmBitsPixel == 8:
        mode = 'L'
        raw_mode = 'L'
    elif bmp.bmBitsPixel == 4:
        mode = 'P'
        raw_mode = 'P'
    elif bmp.bmBitsPixel == 1:
        mode = '1'
        raw_mode = '1'
    else:
        print(f"Unsupported bit depth: {bmp.bmBitsPixel}")
        ctypes.windll.gdi32.SelectObject(mem_dc, old_bitmap)
        ctypes.windll.gdi32.DeleteDC(mem_dc)
        ctypes.windll.user32.ReleaseDC(None, hdc)
        return None

    print(f"Using mode: {mode}, raw mode: {raw_mode}")

    # Create PIL Image from buffer
    image = Image.frombuffer(
        mode,
        (bmp.bmWidth, bmp.bmHeight),
        buffer,
        'raw',
        raw_mode,
        bmp.bmWidthBytes,
        1
    )

    # Flip the image vertically if needed
    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # Clean up
    ctypes.windll.gdi32.SelectObject(mem_dc, old_bitmap)
    ctypes.windll.gdi32.DeleteDC(mem_dc)
    ctypes.windll.user32.ReleaseDC(None, hdc)

    return image



def is_command(message):
    global REQUIRE_POWER_WORD, POWER_WORD
    if REQUIRE_POWER_WORD:
        if POWER_WORD and message.startswith(POWER_WORD):
            return True
        else:
            return False
    else:

        # Assuming you have a function `is_known_command` that checks for known commands
        return is_known_command(message) 
    
# List of known command prefixes
is_known_command_prefixes = [
       '/*', '*/', 'TOGGLE_POWER_WORD', 'toggle_power_word', 'VKB_CMD:', 'vkb_cmd:', 'CURSORCMD:', 'cursorcmd:',
    'toggle_always', 'TOGGLE_ALWAYS', 'INIT', 'init', 'PIN', 'pin', 'RETRIEVE_HANDOFF', 'retrieve_handoff', 'HANDOFF', 'handoff',
    'RECALL', 'recall', 'CLEAR_NOPIN%', 'clear_nopin%', 'CLEAR_NONPIN', 'clear_nonpin', 'CLEAR%', 'clear%',
    '-ch', '-CH', 'CLEAR', 'clear', 'edit msgs', 'EDIT MSGS', 'REMOVE_MSGS', 'remove_msgs', 'DELETE_MSG:', 'delete_msg:',
    'SAVEPINNEDINIT', 'savepinnedinit', 'move_key','SAVEPINNEDHANDOFF', 'savepinnedhandoff', 'SAVEEXEMPTIONS', 'saveexemptions', 'SAVE PINS', 'save pins',
    'SAVE ALL PINS', 'save all pins', 'HELP_VISIBILITY', 'help_visibility', 'HELP', 'help', 'hide', 'HIDE', 'toggle', 'TOGGLE',
    'TOGGLE_SKIP_COMMANDS', 'toggle_skip_commands', 'DISPLAYHISTORY', 'displayhistory', 'DHISTORY', 'dhistory', 'SAVECH', 'savech', 'VKPAG:', 'vkpag:', 'pyag:', 'PYAG:',
    'shutdown instance', 'SHUTDOWN INSTANCE', 'terminate instance', 'TERMINATE INSTANCE', 'CLEAR ALL MESSAGES', 'clear all messages',
    'CLEAR INIT', 'clear init', 'CLEAR PIN', 'clear pin', 'CLEAR HANDOFF', 'clear handoff', 'TOGGLE AUTO PROMPT', 'toggle auto prompt',     'ADD_GOAL', 'UPDATE_GOAL', 'REMOVE_GOAL',
    'ADD_APRIORITY_GOAL', 'UPDATE_INVENTORY', 'UPDATE_PARTY', 'UPDATE_CONTEXT',
    'UPDATE_GOAL_STATUS', 'COMPLETE_GOAL', 'tasks', 'HELP', '/HELP', '/?', '~help', '/help', '-h', '--help',
    'pyag:', 'CMD', 'terminate_instance', 'SAVE_TASKS', 'LOAD_TASKS'
# ... add any other command identifiers you need
    # Note: Make sure all prefixes are unique and not a subset of another prefix,
    # otherwise, it may cause issues in recognizing commands accurately.
]

# Function to check if a message is a known command
def is_known_command(message):
    if message is None:
        return False
    # Check if the message starts with any of the known command prefixes
        # Check if the message starts with any of the known command prefixes
    for prefix in is_known_command_prefixes:
        if message.startswith(prefix) or message.startswith(f"{prefix}("):
            return True
    
    return False


def encode_image_to_base64(image_path):
    """
    Encodes an image to base64.
    """
    try:
        with Image.open(image_path) as image:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")  # You can change PNG to a different format if needed
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None

def upload_image_and_get_file_id(image_path):
    try:
        response = openai.File.create(
            file=open(image_path, "rb"),
            purpose='chat_completion',
        )
        return response['id']
    except Exception as e:
        print(f"Failed to upload image and obtain file_id: {e}")
        return None


# Function to send prompt to ChatGPT
@time_logger
def send_prompt_to_chatgpt(prompt, role="user", image_path=None, image_timestamp=None, exemption=None, sticky=False):
    global token_counter
    global init_handoff_in_progress
    global text_count
    global image_or_other_count
    global image_detail
    global Recent_images_High
    global inference_counter
    inference_counter += 1

    current_time = datetime.datetime.now()
    timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
    # Ensure the lock and headers are defined at the appropriate place in your code.

        # Check if sending text is disabled
    if not SEND_TO_MODEL and prompt and not image_path:
        print("Sending text to model is disabled. Skipping text prompt.")
        return None
    
    with lock:
        # Headers for the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
    
    if not ENABLE_MODEL_RESPONSES:
        print("Model responses are disabled. Skipping GPT/O1 API call.")
        return None

    # Determine which model to use
    if inference_counter % model_interval == 0:
        model_name = 'o1-preview'
        # Add tasksning instructions
        tasksning_prompt = "tasks: Please review and update your goals, inventory, and suggest next actions with reasoning."
        prompt = tasksning_prompt + "\n" + prompt
    else:
        model_name = 'gpt-4o-mini'

       # if not is_important_message(prompt) and not init_handoff_in_progress:
        #    update_chat_history("user", prompt)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
                # Check for pinned messages
                # Add a default pinned message if none exists
        if not any(entry.get('Exemption') in ['Pinned', 'Handoff', 'Special'] for entry in chat_history):
            default_pinned_message = "Exemption: PINNED:PRIORITY GOALS: [Editable area for priority goals (10)] APRIORITY GOALS: [Editable area for apriority goals] NOTES: [Editable area for notes]"
            update_chat_history("system", {'type': 'text', 'text': default_pinned_message}, exemption='Pinned')


        # Collect only image entries from chat history
        image_entries = [entry for entry in chat_history if entry['content']['type'] == 'image']



        # Add the context message for chat history images
        messages.append({
            "role": "user",
            "content": "Below are images from the recent chat history. Use these as a reference to understand the context and events that have occurred recently. These images will help you loosely guide your decisions and actions moving forward, providing you with a sense of continuity."
        })

        # Instead of just extending the messages with text content, use the provided snippet
        # to correctly format and include images as well.
        # Properly format chat history entries for messages
        for entry in chat_history:

            if entry['content']['type'] == 'image':
                image_detail = entry['content'].get('detail', 'low')
                print(image_detail + " detail in chat history")
                if entry['content'].get('sticky', False):
                    image_detail = "high"
                    
                # If the entry is an image, encode it and add it as an image_url content type
                messages.append({
                    "role": entry['role'],
                    "content": [{"type": "image_url", "image_url":  {"url": f"data:image/png;base64,{entry['content']['data']}","detail": image_detail}
                }]
                })
                #print("chathistory msgs: ")
                #print(messages)
            elif entry['content']['type'] == 'text':
                # If the entry is text, add it as plain text content
                messages.append({
                    "role": entry['role'],
                    "content": entry['content']['text']
                })
        # Add the current prompt with a timestamp to messages
        formatted_prompt = f"[{timestamp}] {prompt}"    
        #Now, you would add the current prompt or image that's being processed.

        
        # Handle the case where both queued user input and an image path exist
        if image_path and queued_user_input:
            # Avoid updating chat history twice. Update only for the image.
            print("Image and queued user input detected")

            # Set the base64 encoding for the image
            base64_image = encode_image_to_base64(image_path)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"User input: {queued_user_input}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": image_detail}}
                ]
            })

            # Update chat history once for both the image and the prompt if exemption is not "queued"
            if exemption != "queued":
                update_chat_history("user", queued_user_input, image_path=image_path, image_timestamp=image_timestamp, sticky=sticky)

        elif image_path:
            # Set the detail level for the latest image
            #image_detail = latest_image_detail
            # If an image path is provided, you process it here as you have done before.
            print(image_path)
            base64_image = encode_image_to_base64(image_path)
            print("success")
            #print(image_detail + " of image path")
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Based on the provided screenshot(s), please use pyag:  ie pyag: move_key(key, tiles) ie pyag:move_key(l,1) to move left one tile in game overworld; pyag:press(u) would move up one selection in game menus or selection screens; pyag:press(r) would move right one selection in game menus or selection screens; pyag:move_key(d,4) #would move down 4 tiles in game overworld. move keys are u,d, l, r for gameboy dpad up, gameboy dpad down, gameboy dpad left, and gameboy dpad right respectively; pyag: press(a) #at title screen/to scroll dialog text boxes, please only use pyag: press(a) by itself and not with multiple commands in single inference though; press(b) to backspace or quick no or cancel; or pyag: hold(moveKey); move keys are u for up, d for down, l for left, right for moving right; pyag: hold(r,1) to move right a few spaces, pyag: press (r) to face right if in game or to navigate one char right in selection or naming screens, or pyag: press (l) to face left or navigate one char left in selection screens, or pyag: press (u) to face up in gameworld #does not move you in gameworld, or pyag press (d) to face down or navigate one down in naming screens or text dialog boxes, press a at menu selections or scroll dialog but do not use multiple commands with pyag: press(a), where gameboy button likely requires a gameboy a press which is the keyboard key a; and then use a comment # to make action notes and to then describe each or any screenshot to your future self, then respond with an appropriate description; controls for game are: Gameboy button Up=U key, Gameboy button Down=D key, Gameboy button Left=L key, Gameboy button Right=R key, Gameboy button Button A=A key, Gameboy button Button B=B key, Button L=K key, Button R=E key  "},  # Use your specific prompt if needed

                    #{"type": "text", "text": "Based on the provided screenshot(s), please describe each or any screenshot then respond with the appropriate command to complete objective of playing Pokemon. If not in Pokemon try clicking on VBA emulator, and then using next response to adust notes in pinned messages of actions taken via edit msgs. If an action is required, use 'VKPAG: [command(args); command(args)]' for interface interactions, or 'edit msgs [timestamp]' to update Pinned messages. If clarification or assistance is needed, feel free to engage in a conversation without a command prefix. Otherwise please do not respond about the image unless absolutely necessary. Responses should be VKPAG or edit msgs"},  # Use your specific prompt if needed
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}",
                    #{"type": "image_url", "image_url":  f"data:image/png;base64,{base64_image}",
                 "detail": image_detail}
            }]
            })



            #print("image path msg: ")
            print(image_detail + " detail in image path message append")
            #print(messages)

            print(f" In Image Path, Prompt: ", prompt)
            print(f"In send_prompt_to_chatgpt, Before user update_chat_history, Image Timestamp: {image_timestamp}")
            # Reset the image detail level for older images

           # image_detail = "low"
                    # Update chat history for the image if exemption is not "queued"
            if exemption != "queued":
                update_chat_history("user", prompt, image_path=image_path, image_timestamp=image_timestamp, sticky=sticky)

        else:
        # Handle text prompts
            print(f" In Else, Prompt: ", prompt)  
                        # If no image, just send the prompt as text
            print("Text prompt only detected")
            if exemption != "queued":
                update_chat_history("user", prompt, exemption=exemption)      
            #update_chat_history("user", prompt, exemption=exemption)
            messages.append({"role": role, "content": formatted_prompt})


        # Construct the payload
        payload = {
            "model": model_name,            
            #"model": "gpt-4o",
            #"model": "gpt-4-vision-preview",
            #"model": "gpt-3.5-turbo-1106",
            "messages": messages,
            "max_tokens": 1937,
            "temperature": 0.5
           #"detail": image_detail  # Add the detail parameter here
        }
    
        # Make the POST request
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        # Check for successful status code
        if response.status_code == 200:
            response_data = response.json()
            ai_response = response_data['choices'][0]['message']['content'].strip()
            

            # Display and update chat history with AI's response
            display_message("assistant", ai_response)
            update_chat_history("assistant", ai_response)
            print("meow")
            #print(chat_history)
            print("NYAN")

            process_assistant_response(ai_response)
            return ai_response
                # Here you check if the AI's response is a command and handle it
           # if is_command(ai_response):
           #     handle_commands(ai_response, is_user=False)
            #else:
                #if not is_important_message(prompt, entry.get("Exemption")): threshold_check

        else:
            # Handle errors
            print(f"Error: {response.status_code}")
            print(f"Message: {response.text}")
            return None

@time_logger
def update_chat_history(role, content, token_count=None, image_path=None, exemption='None', image_timestamp=None, sticky=False ):

    global chat_history
    global enable_unimportant_messages
    global enable_important_messages
    global mouse_position  # make sure to include this global if you want to use it in the function
    global last_key  # make sure to include this global if you want to use it in the function
    global last_command  # Use the global variable here
    global ADD_UMSGS_TO_HISTORY
    global ADD_IMSGS_TO_HISTORY
    print(f"update_chat_history, Image Timestamp: {image_timestamp}")

    important_exemptions = {'Pinned', 'Init', 'Special', 'Handoff'}
    is_important = exemption in important_exemptions

    if not is_important and not enable_unimportant_messages:
        return
    if is_important and not enable_important_messages:
        return

    if not is_important and not ADD_UMSGS_TO_HISTORY:
        return
    if is_important and not ADD_IMSGS_TO_HISTORY:
        return

    current_time = datetime.datetime.now()
    timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
        # Prepend timestamp to the content for all messages
    content_with_timestamp = f"[{timestamp}] {content}"
    #modified_content = content
    # Check if the message is a special type like PINNED, Init, or Special
    #if exemption in ['Pinned', 'Init', 'Handoff', 'Special']:

    # Acquire lock before accessing mouse_position
    with mouse_position_lock:
        try:
            current_position = pyautogui.position()
            mouse_position = {"x": current_position.x, "y": current_position.y}
        except Exception as e:
            print(f"Error getting mouse position: {e}")

    # Prepare the dictionary entry for the chat history
    history_entry = {
        "timestamp": timestamp,
        "role": role,
        "token_count": token_count,
        "Exemption": exemption,
        "last_key": last_key,  # Add this line to include the last key press
        "mouse_position": mouse_position,  # Add this line to include the current mouse position
        "last_command": last_command,  # Add last_command to the history entry
    }

    # If an image path is provided, encode the image and add to the history entry
    if image_path:
        print("inside UCH"+ image_path)
        base64_image = encode_image_to_base64(image_path)
        #history_entry["timestamp"] = image_timestamp
        history_entry["Exemption"] = "Image"  # Set exemption specifically for images
        history_entry["content"] = {
        "type": "image",
        "data": base64_image,
        "detail": image_detail,
        "sticky": sticky  # Add sticky flag
        #"timestamp": image_timestamp  # Include timestamp here
    }
                # Also log the corresponding text as a separate entry
        text_history_entry = {
            "timestamp": timestamp,
            "role": role,
            "token_count": token_count,
            "Exemption": exemption,
            "last_key": last_key,
            "mouse_position": mouse_position,
            "last_command": last_command,
            "content": {"type": "text", "text": content_with_timestamp}
        }

                # Append both the image and the text to the chat history
        chat_history.append(text_history_entry)  # Log the text first
        chat_history.append(history_entry)
        # Iterate over chat history and check for image entries
        for entry in chat_history:
            if entry['content']['type'] == 'image':
                image_data = entry['content']['data']

                # Extract the specified character ranges from the base64 string
                range1 = image_data[50:80]  # Characters 33 to 55
                range2 = image_data[75:100]  # Characters 115 to 119

                # Print the extracted ranges
                print(f"Range 50-80: {range1}")
                #print(f"Range 75-100: {range2}")
                # Counters for image content types and image data entries
        image_content_count = 0
        image_data_count = 0

        # Iterate over chat history to count image content types and image data entries
        for entry in chat_history:
            if entry['content']['type'] == 'image':
                image_content_count += 1
                if 'data' in entry['content'] and entry['content']['data']:
                    image_data_count += 1

        # Print the counts
        print(f"Number of 'content type = image' entries: {image_content_count}")
        print(f"Number of 'content data' entries: {image_data_count}")
        #print(chat_history)        
        manage_image_history(base64_image, image_timestamp=image_timestamp, sticky=sticky)  # Manage the image history

        image_content_count2 = 0
        image_data_count2 = 0
        for entry in chat_history:
            if entry['content']['type'] == 'image':
                image_content_count2 += 1
                if 'data' in entry['content'] and entry['content']['data']:
                    image_data_count2 += 1
    else:
        history_entry["content"] = {"type": "text", "text": content_with_timestamp}
        chat_history.append(history_entry)
        print("ELSE IMAGE")
        #print(chat_history)

    # Append the history entry to the chat history
    last_command = None  # Reset last_command after updating chat history
    # Check and remove messages based on decay time and importance
    if CHECK_UMSGS_DECAY or CHECK_IMSGS_DECAY:
        chat_history = [
            entry for entry in chat_history
            if not (
                CHECK_UMSGS_DECAY 
                and entry.get("Exemption") != "Image"  # Exclude image entries
                and not (entry.get("Exemption") in important_exemptions)
                and (current_time - datetime.datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S')).total_seconds()/60 > UMSGS_DECAY
            ) and not (
                CHECK_IMSGS_DECAY 
                and entry.get("Exemption") != "Image"  # Exclude image entries
                and (entry.get("Exemption") in important_exemptions)
                and (current_time - datetime.datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S')).total_seconds()/60 > IMGS_DECAY
            )
        ]

    print(f"At end of update_chat_history, Image Timestamp: {image_timestamp}")

def construct_hierarchical_tasks_text():
    tasks_text = "**Hierarchical tasks with Immediate Actions:**\n\n"
    for level_num, level in enumerate(hierarchical_tasks, start=1):
        tasks_text += f"- **Level {level_num}**\n\n"
        for idx, item in enumerate(level, start=1):
            if 'command' in item:
                tasks_text += f"  - **Action {idx}**: `{item['command']}`\n"
                tasks_text += f"    - **Priority**: {item.get('priority', 'No Priority')}\n"
                tasks_text += f"    - **Significance**: {item.get('significance', 'No Significance')}\n"
                tasks_text += f"    - **Associated Goal**: {item.get('associated_goal', 'No Goal')}\n"
                tasks_text += f"    - **Reasoning**: {item.get('reasoning', '')}\n"
                if 'notes' in item:
                    tasks_text += f"    - **Notes**: {item['notes']}\n"
            else:
                # Handle placeholders or general suggestions
                tasks_text += f"  - **Action {idx}**: {item.get('description', 'TBD')}\n"
                if 'notes' in item:
                    tasks_text += f"    - **Notes**: {item['notes']}\n"
        tasks_text += "\n"
    return tasks_text

def add_goal_command(args):
    goal_text = args[0].strip()
    priority = args[1].strip() if len(args) > 1 else 'No Priority'
    significance = args[2].strip() if len(args) > 2 else 'No Significance'
    add_goal(goal_text, priority, significance)
    display_message('system', f"Added goal: {goal_text} with priority: {priority} and significance: {significance}")
    print(f"Executing command: ADD_GOAL ")

# Function to process assistant's response
def process_assistant_response(response_text):
    # Split the response into sections
    # Implement parsing logic based on the assistant's response format
    # For simplicity, let's assume the assistant's response includes executed commands and the hierarchical tasks
    executed_commands, tasks_section = split_response_sections(response_text)

    # Handle executed commands
    handle_commands(executed_commands, is_user=False)

    # Update hierarchical tasks
    parse_hierarchical_tasks(tasks_section)

    # Execute immediate actions from the updated tasks
    if hierarchical_tasks and hierarchical_tasks[0]:  # Check if Level 1 actions exist
        immediate_actions = hierarchical_tasks[0]  # Level 1 actions
        execute_immediate_actions(immediate_actions)

    # Update pinned information
    update_pinned_information()


# Function to split the assistant's response into sections
def split_response_sections(response_text):
    # Implement logic to split the response into executed commands and tasks
    # This is a placeholder implementation
    # Adjust this function based on the assistant's response format
    executed_commands = ''
    tasks_section = ''
    lines = response_text.strip().split('\n')
    in_tasks_section = False
    for line in lines:
        if line.strip().startswith('**Hierarchical tasks'):
            in_tasks_section = True
            tasks_section += line + '\n'
        elif in_tasks_section:
            tasks_section += line + '\n'
        else:
            executed_commands += line + '\n'
    return executed_commands.strip(), tasks_section.strip()

def execute_immediate_actions(actions):
    global recent_executed_actions
    for action in actions:
        command = action.get('command')
        if command and not command.startswith('TBD'):
            handle_commands(command, is_user=False)
            # Update recent executed actions
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            recent_executed_actions.append({
                'timestamp': timestamp,
                'action': command,
                'associated_goal': action.get('associated_goal', 'No specific goal')
            })
            # Keep only the last 7 executed actions
            recent_executed_actions = recent_executed_actions[-7:]
        else:
            # Handle placeholders or general suggestions
            continue  # Placeholders are not executed but inform future tasksning

# Define the Goal class
class Goal:
    def __init__(self, text, priority='No Priority', significance='No Significance', status='Incomplete', sub_goals=None):
        self.id = len(tasks_list) + 1
        self.text = text
        self.priority = priority  # e.g., 'High', 'Medium', 'Low'
        self.significance = significance  # e.g., 'Critical', 'Significant', 'Minor'
        self.status = status  # 'Incomplete' or 'Complete'
        self.sub_goals = sub_goals if sub_goals else []  # List of sub-goal objects


# Function to parse the hierarchical tasks
def parse_hierarchical_tasks(tasks_text):
    # Implement parsing logic to extract the hierarchical tasks
    # For simplicity, let's assume we store it in a global variable
    global hierarchical_tasks
    hierarchical_tasks = []
    # Parse the tasks_text and populate hierarchical_tasks
    # Placeholder implementation
    
    level_pattern = re.compile(r'- \*\*Level (\d+)\*\*')
    action_pattern = re.compile(r'- \*\*Action (\d+)\*\*: `(.+?)`')
    priority_pattern = re.compile(r'- \*\*Priority\*\*: (.+)')
    significance_pattern = re.compile(r'- \*\*Significance\*\*: (.+)')
    associated_goal_pattern = re.compile(r'- \*\*Associated Goal\*\*: (.+)')
    reasoning_pattern = re.compile(r'- \*\*Reasoning\*\*: (.+)')
    notes_pattern = re.compile(r'- \*\*Notes\*\*: (.+)')

    current_level = None
    current_action = None

    lines = tasks_text.strip().split('\n')
    for line in lines:
        level_match = level_pattern.match(line)
        if level_match:
            current_level = int(level_match.group(1))
            while len(hierarchical_tasks) < current_level:
                hierarchical_tasks.append([])
            continue

        action_match = action_pattern.match(line)
        if action_match and current_level is not None:
            action_number = int(action_match.group(1))
            command = action_match.group(2)
            current_action = {
                'command': command,
                'priority': 'No Priority',
                'significance': 'No Significance',
                'associated_goal': 'No Goal',
                'reasoning': '',
                'notes': ''
            }
            hierarchical_tasks[current_level - 1].append(current_action)
            continue

        if current_action is not None:
            if priority_match := priority_pattern.match(line):
                current_action['priority'] = priority_match.group(1)
            elif significance_match := significance_pattern.match(line):
                current_action['significance'] = significance_match.group(1)
            elif associated_goal_match := associated_goal_pattern.match(line):
                current_action['associated_goal'] = associated_goal_match.group(1)
            elif reasoning_match := reasoning_pattern.match(line):
                current_action['reasoning'] = reasoning_match.group(1)
            elif notes_match := notes_pattern.match(line):
                current_action['notes'] = notes_match.group(1)

        else:
            # Handle placeholders or general suggestions
            if current_level is not None:
                description_match = re.match(r'- \*\*Action (\d+)\*\*: (.+)', line)
                if description_match:
                    action_number = int(description_match.group(1))
                    description = description_match.group(2)
                    current_action = {
                        'description': description,
                        'notes': ''
                    }
                    hierarchical_tasks[current_level - 1].append(current_action)
                    continue
                elif notes_match := notes_pattern.match(line):
                    current_action['notes'] = notes_match.group(1)                

 # Function to update the pinned information

def update_pinned_information():
    # Remove existing pinned information
    chat_history[:] = [entry for entry in chat_history if entry.get('Exemption') != 'PinnedInfo']

    # Construct the pinned information text
    pinned_text = "=== Pinned Information ===\n\n"

    # Add hierarchical tasks
    pinned_text += construct_hierarchical_tasks_text()

    # Add notable inventory
    if notable_inventory:
        pinned_text += "**Notable Inventory:**\n- " + "\n- ".join(notable_inventory) + "\n\n"

    # Add party status
    if party_status:
        pinned_text += "**Current Party Status:**\n- " + "\n- ".join(party_status) + "\n\n"

    # Add recent context and events
    if recent_context_events:
        pinned_text += "**Recent Context and Events:**\n- " + "\n- ".join(recent_context_events) + "\n\n"

    # Add recent executed actions
    if recent_executed_actions:
        pinned_text += "**Recent Executed Actions:**\n"
        for entry in recent_executed_actions:
            pinned_text += f"- [{entry['timestamp']}] {entry['action']} (Goal: {entry['associated_goal']})\n"
        pinned_text += "\n"

    # Add significant goals achieved
    if significant_past_tasks:
        pinned_text += "**Significant Goals Achieved:**\n"
        for entry in significant_past_tasks:
            pinned_text += f"- [{entry['timestamp']}] {entry['goal']}\n"
        pinned_text += "\n"

    # Create the pinned message entry
    pinned_message_content = {'type': 'text', 'text': pinned_text}
    chat_history.append({
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'role': 'system',
        'token_count': None,
        'Exemption': 'PinnedInfo',
        'content': pinned_message_content
    })

# Function to add a goal
def add_goal(goal_text, priority='No Priority', significance='No Significance'):
    goal = Goal(goal_text, priority, significance)
    tasks_list.append(goal)
    update_pinned_information()

# Function to add an apriority goal
def add_apriority_goal(goal_text):
    apriority_tasks.append(goal_text)
    update_pinned_information()

# Function to update inventory
def update_inventory(inventory_items):
    global notable_inventory
    notable_inventory = inventory_items.split(';')
    update_pinned_information()

# Function to update party status
def update_party_status(party_status_text):
    global party_status
    party_status = party_status_text.split(';')
    update_pinned_information()

# Function to update recent context
def update_recent_context(context_events):
    global recent_context_events
    recent_context_events = context_events.split(';')
    update_pinned_information()

# Function to mark a goal as completed
def mark_goal_as_completed(goal_id):
    for goal in tasks_list:
        if goal.id == goal_id:
            goal.status = 'Complete'
            significant_past_tasks.append({
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'goal': goal.text
            })
            # Keep only the last 5 significant goals
            significant_past_tasks[:] = significant_past_tasks[-5:]
            break
    update_pinned_information()    

# Function to display help message
def display_help():
    help_message = """
    **Available Commands:**

    - **Goal Management Commands**:
        - ADD_GOAL(goal_text; priority; significance): Add a new goal with priority and significance.
        - ADD_APRIORITY_GOAL(goal_text): Add a new open-ended goal.
        - UPDATE_INVENTORY(inventory_items): Update your notable inventory; separate items with semicolons.
        - UPDATE_PARTY(party_status): Update your current party status; provide details for each Pokémon, separated by semicolons.
        - UPDATE_CONTEXT(recent_events): Update recent context and events; separate events with semicolons.
        - COMPLETE_GOAL(goal_id): Mark a goal as completed.

    - **PyAutoGUI Commands**:
        - pyag: command: Execute a pyautogui command.

    - **Other Commands**:
        - tasks: Trigger a tasksning inference.
        - HELP: Display this help message.

    """
    display_message("system", help_message)

@time_logger
def manage_image_history(new_image_base64, image_timestamp=None, sticky=False):
    global chat_history
    global MAX_IMAGES_IN_HISTORY
    global High_Detail
    global Recent_images_High

    # Check if the image already exists in the chat history
    image_already_exists = any(
        entry['content']['type'] == 'image' and entry['content']['data'] == new_image_base64
        for entry in chat_history
    )

    # Add the new image to chat_history only if it's not a duplicate
    if not image_already_exists:
        new_image_entry = {
            "content": {
                "type": "image",
                "data": new_image_base64,
                "detail": "low",  # Default to low detail
                "sticky": sticky  # Add sticky flag
            },
            "timestamp": image_timestamp if image_timestamp else datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        chat_history.append(new_image_entry)

    # Filter to get only image entries
    image_entries = [entry for entry in chat_history if entry['content']['type'] == 'image']

    for entry in image_entries:
        entry['content']['detail'] = "low"  # Reset detail to low

    # Adjust detail levels based on High_Detail
    if isinstance(High_Detail, int):
        if High_Detail > 0:
            # Set the most recent High_Detail number of images to high detail
            for i in range(1, High_Detail + 1):
                if i <= len(image_entries):
                    image_entries[-i]['content']['detail'] = "high"
        elif High_Detail < 0:
            # Set the oldest High_Detail number of images to high detail
            for i in range(-High_Detail):
                if i < len(image_entries):
                    image_entries[i]['content']['detail'] = "high"
    elif isinstance(High_Detail, str) and High_Detail.lower() in ["true", "all", "t", "a"]:
        # Set all images to high detail
        for entry in image_entries:
            entry['content']['detail'] = "high"

    # Ensure sticky images remain high detail
    for entry in image_entries:
        if entry['content'].get('sticky', False):
            entry['content']['detail'] = "high"

        # Adjust detail levels based on Recent_images_High
    for i in range(1, Recent_images_High + 1):
        if i <= len(image_entries):
            image_entries[-i]['content']['detail'] = "high"        

    # If the limit is exceeded, remove the oldest image(s)
    while len(image_entries) > MAX_IMAGES_IN_HISTORY:
        oldest_image_entry = min(image_entries, key=lambda x: x['timestamp'])
        chat_history.remove(oldest_image_entry)
        image_entries.remove(oldest_image_entry)


    # For debugging
    ##print(f"Updated image entries: {image_entries}") 





# Function to handle user input
@time_logger
def user_input_handler():
    global chat_history
    global disable_commands
    global last_interaction_time
    global Always_  # global declaration to access and modify the variable
    global hide_input  # global declaration to access and modify the variable
    global queued_user_input  # Use the global variable for queued input

    while running:
        # Check the elapsed time since the last interaction
        
        elapsed_time = time.time() - last_interaction_time
        
        if enable_auto_prompt and elapsed_time > AUTO_PROMPT_INTERVAL:
            # Reset the last interaction time
            last_interaction_time = time.time()
            
            # Trigger auto prompt
            auto_prompt_response = send_prompt_to_chatgpt(auto_prompt_message)
            #send_prompt_to_chatgpt(auto_prompt_response)
        #user_input = input("") 
        #user_input = getpass(prompt='')
        user_input = get_user_input(Always_, hide_input)  # using the function to get user input
        last_interaction_time = time.time()  # Update the last interaction time

        if user_input.startswith('/*'):
            disable_commands = True
            continue  # Skip the rest of the loop
        elif user_input.startswith('*/'):
            disable_commands = False
            continue  # Skip the rest of the loop

        if disable_commands:
            print("Command processing is currently disabled.")
            continue  # Skip the rest of the loop

        # Move all command handling to the handle_commands function
        if is_command(user_input):
            print("UI is command")
            handle_commands(user_input, is_user=True)
        else:
            #Instead of sending the input to ChatGPT, queue it for the next screenshot
            # Instead of sending the input to ChatGPT, queue it for the next screenshot
            queued_user_input.put(user_input)
            #send_prompt_to_chatgpt(user_input)
            #send_prompt_to_chatgpt(response_text)
            #response_text = send_prompt_to_chatgpt(handoff_prompt)
        if hbuffer:
            handoff_prompt = hbuffer[-1]
            print({hbuffer})
            response_text = send_prompt_to_chatgpt(handoff_prompt)



def get_user_input(Always_=None, hide_input=None):
    if Always_ is not None:
        if Always_:
            hide_input = True
        else:
            hide_input = False
    else:
        hide_input = input("Display input (yes or no)? ").strip().lower() == 'yes'
    
    if hide_input:
        return getpass(prompt='')
    else:
        return input("")



def human_like_click(x_pixel, y_pixel): #now with random effects to better bypass captchas
    # Randomize the circle duration and radius slightly
    circle_duration = random.uniform(0.5, 1.5)
    circle_radius = random.uniform(5, 15)
    
    # Randomize starting angle
    start_angle = random.uniform(0, 2 * math.pi)
    
    start_time = time.time()
    while time.time() - start_time < circle_duration:
        # Increment the angle over time, adding small random noise
        angle = start_angle + ((time.time() - start_time) / circle_duration) * 2 * math.pi + random.uniform(-0.1, 0.1)
        x = x_pixel + math.cos(angle) * circle_radius
        y = y_pixel + math.sin(angle) * circle_radius
        
        # Move to the calculated position with a slight random duration
        pyautogui.moveTo(x, y, duration=random.uniform(0.05, 0.2))
    # Add a slight random delay before the final click
    time.sleep(random.uniform(0.1, 0.3))
    pyautogui.click(x_pixel, y_pixel)

    # Command Dispatch Dictionary





def save_pinned_tasks(file_name="pinned_tasks.json"):
    data_to_save = {
        'goals_list': [goal.__dict__ for goal in tasks_list],
        'apriority_goals': apriority_tasks,
        'notable_inventory': notable_inventory,
        'party_status': party_status,
        'recent_context_events': recent_context_events,
        'hierarchical_tasks': hierarchical_tasks,
        'significant_past_goals': significant_past_tasks,
        'recent_executed_actions': recent_executed_actions
    }
    with open(file_name, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    print(f"Pinned tasks saved to {file_name}")

def load_pinned_tasks(file_name="pinned_tasks.json"):
    global tasks_list, apriority_tasks, notable_inventory, party_status
    global recent_context_events, hierarchical_tasks, significant_past_tasks
    global recent_executed_actions

    if not os.path.exists(file_name):
        print(f"No saved pinned tasks found at {file_name}")
        return

    with open(file_name, 'r') as f:
        data_loaded = json.load(f)

    tasks_list = [Goal(**goal_data) for goal_data in data_loaded.get('goals_list', [])]
    apriority_tasks = data_loaded.get('apriority_goals', [])
    notable_inventory = data_loaded.get('notable_inventory', [])
    party_status = data_loaded.get('party_status', [])
    recent_context_events = data_loaded.get('recent_context_events', [])
    hierarchical_tasks = data_loaded.get('hierarchical_tasks', [])
    significant_past_tasks = data_loaded.get('significant_past_goals', [])
    recent_executed_actions = data_loaded.get('recent_executed_actions', [])

    update_pinned_information()
    print(f"Pinned tasks loaded from {file_name}")

def toggle_power_word(args, is_user):
    global REQUIRE_POWER_WORD
    REQUIRE_POWER_WORD = not REQUIRE_POWER_WORD
    display_message("system", f"POWER_WORD requirement toggled to {'ON' if REQUIRE_POWER_WORD else 'OFF'}.")

def init_command(args, is_user):
    global chat_history
    init_summary = ' '.join(args)
    chat_history = [entry for entry in chat_history if not entry['content'].startswith('Pinned Init summary')]
    update_chat_history('user', f'Pinned Init summary: {init_summary}')
    print(f"Pinned Init summary: {init_summary}")

def pin_command(args, is_user):
    exemption_context = ' '.join(args)
    exemption_type = "Pinned"
    display_message("assistant", f'Exemption: {exemption_context}', include_command=True)
    update_chat_history('assistant', {'type': 'text', 'text': f'Exemption: {exemption_context}'}, exemption=exemption_type)
    print(f"Executing command: PIN | Is user: {is_user}")

def add_goal_command(args, is_user):
    if not args:
        display_message("error", "ADD_GOAL requires at least one argument (goal_text).")
        return
    goal_text = args[0]
    priority = args[1] if len(args) > 1 else 'No Priority'
    significance = args[2] if len(args) > 2 else 'No Significance'
    add_goal(goal_text, priority, significance)
    display_message('system', f"Added goal: {goal_text} with priority: {priority} and significance: {significance}")
    print(f"Executing command: ADD_GOAL | Is user: {is_user}")

def add_apriority_goal_command(args, is_user):
    if not args:
        display_message("error", "ADD_APRIORITY_GOAL requires at least one argument (goal_text).")
        return
    goal_text = args[0]
    add_apriority_goal(goal_text)
    display_message('system', f"Added apriority goal: {goal_text}")
    print(f"Executing command: ADD_APRIORITY_GOAL | Is user: {is_user}")

def update_inventory_command(args, is_user):
    if not args:
        display_message("error", "UPDATE_INVENTORY requires at least one argument (inventory_items).")
        return
    inventory_items = args[0]
    update_inventory(inventory_items)
    display_message('system', f"Updated inventory: {inventory_items}")
    print(f"Executing command: UPDATE_INVENTORY | Is user: {is_user}")

def update_party_command(args, is_user):
    if not args:
        display_message("error", "UPDATE_PARTY requires at least one argument (party_status_text).")
        return
    party_status_text = args[0]
    update_party_status(party_status_text)
    display_message('system', f"Updated party status: {party_status_text}")
    print(f"Executing command: UPDATE_PARTY | Is user: {is_user}")

def update_context_command(args, is_user):
    if not args:
        display_message("error", "UPDATE_CONTEXT requires at least one argument (context_events).")
        return
    context_events = args[0]
    update_recent_context(context_events)
    display_message('system', f"Updated recent context: {context_events}")
    print(f"Executing command: UPDATE_CONTEXT | Is user: {is_user}")

def complete_goal_command(args, is_user):
    if not args:
        display_message("error", "COMPLETE_GOAL requires one argument (goal_id).")
        return
    try:
        goal_id = int(args[0])
        mark_goal_as_completed(goal_id)
        display_message('system', f"Marked goal {goal_id} as completed")
        print(f"Executing command: COMPLETE_GOAL | Is user: {is_user}")
    except ValueError:
        display_message("error", "Invalid goal ID. It must be an integer.")
        print(f"Error executing command: COMPLETE_GOAL | Is user: {is_user}")

def tasks_command(args, is_user):
    # This is a trigger for tasksning inference
    display_message('system', "Triggering tasksning inference")
    # Implement the tasksning logic as needed
    print(f"Executing command: tasks | Is user: {is_user}")

def save_tasks_command(args, is_user):
    file_name = args[0].strip() if args else "pinned_tasks.json"
    save_pinned_tasks(file_name)
    display_message('system', f"Pinned tasks saved to {file_name}")
    print(f"Executing command: SAVE_TASKS | Is user: {is_user}")

def load_tasks_command(args, is_user):
    file_name = args[0].strip() if args else "pinned_tasks.json"
    load_pinned_tasks(file_name)
    display_message('system', f"Pinned tasks loaded from {file_name}")
    print(f"Executing command: LOAD_TASKS | Is user: {is_user}")


def help_command(args, is_user):
    display_help()
    print(f"Executing command: HELP | Is user: {is_user}")



def clear_nopin_percentage_command(args, is_user):
    try:
        percentage = float(args[0]) / 100
        if 0 <= percentage <= 1:
            clear_chat_history_except_pinned(percentage)
            display_message('system', f"Cleared {args[0]}% of non-pinned messages.")
        else:
            display_message('error', "Invalid percentage. Please enter a number between 0 and 100.")
    except (ValueError, IndexError):
        display_message('error', "Invalid command format. Usage: CLEAR_NOPIN%(number)")

def clear_percentage_command(args, is_user):
    try:
        percentage = float(args[0]) / 100
        if 0 <= percentage <= 1:
            clear_chat_history(percentage)
            display_message('system', f"Cleared {args[0]}% of messages.")
        else:
            display_message('error', "Invalid percentage. Please enter a number between 0 and 100.")
    except (ValueError, IndexError):
        display_message('error', "Invalid command format. Usage: CLEAR%(number)")

def clear_all_command(args, is_user):
    global chat_history
    chat_history = [
        entry for entry in chat_history
        if entry.get('Exemption') in ['Pinned', 'Init', 'Handoff']
    ]
    display_message('system', "All messages cleared except pinned messages.")

def edit_msgs_command(args, is_user):
    # Implement the logic to edit messages based on timestamp and new content
    # For this example, we'll assume args contain pairs of timestamp and new content
    if not args or len(args) < 2:
        display_message('error', "EDIT_MSGS requires pairs of timestamp and new content.")
        return
    for i in range(0, len(args), 2):
        timestamp = args[i]
        new_content = args[i + 1] if i + 1 < len(args) else ''
        found = False
        for entry in chat_history:
            if entry['timestamp'] == timestamp:
                entry['content'] = {'type': 'text', 'text': new_content}
                found = True
                break
        if found:
            display_message('system', f"Message with timestamp {timestamp} edited successfully.")
        else:
            display_message('error', f"No message found with timestamp {timestamp}.")

def delete_msg_command(args, is_user):
    timestamp = args[0] if args else None
    if timestamp:
        global chat_history
        chat_history = [entry for entry in chat_history if entry['timestamp'] != timestamp]
        display_message('system', f"Message with timestamp {timestamp} deleted.")
    else:
        display_message('error', "DELETE_MSG command requires a timestamp argument.")

def save_chat_history_command(args, is_user):
    file_name = args[0] if args else 'chat_history'
    save_chat_history_to_file(chat_history, file_name)
    display_message('system', f"Chat history saved to {file_name}.")

def display_history_command(args, is_user):
    if chat_history:
        for entry in chat_history:
            timestamp = entry.get('timestamp', 'No timestamp')
            role = entry.get('role', 'No role')
            content = entry.get('content', {})
            if isinstance(content, dict):
                message = content.get('text', 'No content')
            else:
                message = str(content)
            print(f"{timestamp} - {role}: {message}")
    else:
        display_message('system', "No messages in the chat history.")

def save_pins_command(args, is_user):
    pinned_entries = [
        entry['content']['text'] for entry in chat_history
        if entry.get('Exemption') == 'Pinned'
    ]
    file_name = args[0] if args else 'pinned_entries.txt'
    write_content_to_file('\n'.join(pinned_entries), file_name)
    display_message('system', f"Pinned entries saved to {file_name}.")

def save_all_pins_command(args, is_user):
    all_entries = [f"{entry['role']}: {entry['content']}" for entry in chat_history]
    file_name = args[0] if args else 'all_entries.txt'
    write_content_to_file('\n'.join(all_entries), file_name)
    display_message('system', f"All entries saved to {file_name}.")

def toggle_auto_prompt_command(args, is_user):
    global enable_auto_prompt
    enable_auto_prompt = not enable_auto_prompt
    status = 'enabled' if enable_auto_prompt else 'disabled'
    display_message('system', f"Auto-prompting has been {status}.")

def toggle_always_command(args, is_user):
    global Always_, hide_input
    if len(args) == 2:
        Always_ = args[0].lower() == 'on'
        hide_input = args[1].lower() == 'true'
        display_message('system', f"Always_ set to {Always_}, hide_input set to {hide_input}")
    else:
        display_message('error', "Invalid number of parameters. Usage: TOGGLE_ALWAYS(on|off, true|false)")

def toggle_image_detail_command(args, is_user):
    global image_detail
    image_detail = 'high' if image_detail == 'low' else 'low'
    display_message('system', f"Global image_detail toggled to: {image_detail}")

def toggle_latest_image_detail_command(args, is_user):
    global latest_image_detail
    latest_image_detail = 'high' if latest_image_detail == 'low' else 'low'
    display_message('system', f"Global latest_image_detail toggled to: {latest_image_detail}")

def set_high_detail_command(args, is_user):
    global High_Detail
    try:
        High_Detail = int(args[0])
        display_message('system', f"Global High_Detail set to: {High_Detail}")
    except (ValueError, IndexError):
        display_message('error', "Invalid value for High_Detail. Must be an integer.")

def set_max_images_in_history_command(args, is_user):
    global MAX_IMAGES_IN_HISTORY
    try:
        MAX_IMAGES_IN_HISTORY = int(args[0])
        display_message('system', f"Global MAX_IMAGES_IN_HISTORY set to: {MAX_IMAGES_IN_HISTORY}")
    except (ValueError, IndexError):
        display_message('error', "Invalid value for MAX_IMAGES_IN_HISTORY. Must be an integer.")

def set_image_detail_command(args, is_user):
    global image_detail
    try:
        detail = args[0].strip().lower()
        if detail in ["low", "high"]:
            image_detail = detail
            display_message('system', f"Global image_detail set to: {image_detail}")
        else:
            display_message('error', "Invalid value for image_detail. Must be 'low' or 'high'.")
    except IndexError:
        display_message('error', "No value provided for image_detail.")

def set_latest_image_detail_command(args, is_user):
    global latest_image_detail
    try:
        detail = args[0].strip().lower()
        if detail in ["low", "high"]:
            latest_image_detail = detail
            display_message('system', f"Global latest_image_detail set to: {latest_image_detail}")
        else:
            display_message('error', "Invalid value for latest_image_detail. Must be 'low' or 'high'.")
    except IndexError:
        display_message('error', "No value provided for latest_image_detail.")

def hide_user_text_command(args, is_user):
    global show_user_text
    show_user_text = not show_user_text
    status = 'displayed' if show_user_text else 'hidden'
    display_message('system', f"User text will now be {status}.")

def hide_ai_text_command(args, is_user):
    global show_ai_text
    show_ai_text = not show_ai_text
    status = 'displayed' if show_ai_text else 'hidden'
    display_message('system', f"AI text will now be {status}.")

def hide_ai_commands_command(args, is_user):
    global hide_ai_commands
    hide_ai_commands = not hide_ai_commands
    status = 'displayed' if not hide_ai_commands else 'hidden'
    display_message('system', f"AI commands will now be {status}.")

def hide_user_commands_command(args, is_user):
    global hide_user_commands
    hide_user_commands = not hide_user_commands
    status = 'displayed' if not hide_user_commands else 'hidden'
    display_message('system', f"User commands will now be {status}.")


def toggle_unimportant_messages_command(args, is_user):
    global enable_unimportant_messages
    enable_unimportant_messages = not enable_unimportant_messages
    status = 'enabled' if enable_unimportant_messages else 'disabled'
    display_message('system', f"Unimportant messages have been {status}.")

def toggle_important_messages_command(args, is_user):
    global enable_important_messages
    enable_important_messages = not enable_important_messages
    status = 'enabled' if enable_important_messages else 'disabled'
    display_message('system', f"Important messages have been {status}.")

def toggle_add_umsgs_to_history_command(args, is_user):
    global ADD_UMSGS_TO_HISTORY
    ADD_UMSGS_TO_HISTORY = not ADD_UMSGS_TO_HISTORY
    status = 'enabled' if ADD_UMSGS_TO_HISTORY else 'disabled'
    display_message('system', f"Adding unimportant messages to history is now {status}.")

def toggle_add_imsgs_to_history_command(args, is_user):
    global ADD_IMSGS_TO_HISTORY
    ADD_IMSGS_TO_HISTORY = not ADD_IMSGS_TO_HISTORY
    status = 'enabled' if ADD_IMSGS_TO_HISTORY else 'disabled'
    display_message('system', f"Adding important messages to history is now {status}.")


def toggle_umsgs_decay_check_command(args, is_user):
    global CHECK_UMSGS_DECAY
    CHECK_UMSGS_DECAY = not CHECK_UMSGS_DECAY
    status = 'ON' if CHECK_UMSGS_DECAY else 'OFF'
    display_message('system', f"Unimportant messages decay check toggled to {status}.")

def toggle_imsgs_decay_check_command(args, is_user):
    global CHECK_IMSGS_DECAY
    CHECK_IMSGS_DECAY = not CHECK_IMSGS_DECAY
    status = 'ON' if CHECK_IMSGS_DECAY else 'OFF'
    display_message('system', f"Important messages decay check toggled to {status}.")

@time_logger
def handle_command_toggle(command):
    global disable_commands
    if command == '/*':
        disable_commands = True
        display_message("system", "Command processing is now disabled.")
        return True  # Indicate that a toggle command was processed
    elif command == '*/':
        disable_commands = False
        display_message("system", "Command processing is now enabled.")
        return True  # Indicate that a toggle command was processed
    return False  # Indicate that no toggle command was processed


def handle_commands(command_input, is_user=True, exemption=None, is_assistant=False):
    global last_command
    global disable_commands  # Add this since we're accessing it

    last_command = command_input

    commands = command_input.split(';')
    for command in commands:
        command = command.strip()
        if not command:
            continue

        # Handle block comments using handle_command_toggle
        if handle_command_toggle(command):
            # If the command was a toggle command, return from handle_commands
            return

        if disable_commands:
            display_message("system", "Command processing is currently disabled.")
            return  # Or use 'continue' if you want to skip this command but process others

        # Extract command and arguments
        if '(' in command and ')' in command:
            cmd_name, args_str = command.split('(', 1)
            args_str = args_str.rstrip(')')
            args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
        else:
            cmd_name = command
            args = []

            # Check if it's a known command
        if not is_known_command(cmd_name):
            # If it's not a known command, simply log it or pass without displaying an error
            #print(f"Received non-command response: '{command_input}'")  # Debug log (optional)
            continue    

        # Handle pyautogui commands separately
        if cmd_name.startswith('pyag:'):
            pyag_command = cmd_name[5:]
            handle_pyautogui_command(pyag_command, args)
            continue

        # Lookup the command in the command dispatch dictionary
        handler = command_dispatch.get(cmd_name.strip().upper())

        if handler:
            try:
                handler(args, is_user)
            except Exception as e:
                display_message("error", f"Error executing command '{cmd_name}': {e}")
        else:
            display_message("error", f"Unknown command: {cmd_name}")


  
@time_logger
def handle_pyautogui_command(cmd, args):
    if cmd in ["press", "hold"]:
        key = args[0].strip().lower()
        if key in pyautogui.KEYBOARD_KEYS:
            if cmd == "press":
                pyautogui.press(key)
                display_message("system", f"Key pressed: {key}")
            elif cmd == "hold":
                duration = float(args[1]) if len(args) > 1 else None
                pyautogui.keyDown(key)
                if duration:
                    pyautogui.sleep(duration)
                    pyautogui.keyUp(key)
                display_message("system", f"Key held for {duration} seconds: {key}")
        else:
            display_message("error", f"Invalid key name: {key}")     
    elif cmd == "move_key":
        try:
            key = args[0].strip().lower()
            tiles = int(args[1].strip())
            move_key(key, tiles)
        except (ValueError, IndexError):
            display_message("error", "Invalid arguments for move_key command. Usage: move_key(key, tiles)")
          
    elif cmd == "move":
        try:
            x, y = map(int, args)
            pyautogui.moveTo(x, y)
            display_message("system", f"Cursor moved to ({x}, {y})")
        except ValueError:
            display_message("error", "Invalid coordinates for move command")       
    elif cmd == "drag":
        x, y, duration = map(float, args)
        pyautogui.dragTo(x, y, duration)
        display_message("system", f"Dragged to ({x}, {y}) in {duration} seconds")      
    elif cmd.startswith("scroll"):
        direction_units = cmd.split('_')
        units = int(args[0]) if args else 1
        if "down" in direction_units:
            pyautogui.scroll(-units)
        else:
            pyautogui.scroll(units)
        display_message("system", f"Scrolled {'down' if 'down' in direction_units else 'up'} {units} units")       
    elif cmd == "release":
        key = args[0]
        pyautogui.keyUp(key)
        display_message("system", f"Key released: {key}")      
    elif cmd == "hotkey":
        pyautogui.hotkey(*args)
        display_message("system", f"Hotkey executed: {'+'.join(args)}")        
    elif cmd == "type":
        text = args[0]
        pyautogui.write(text)
        display_message("system", f"Text typed: {text}")       
    elif cmd == "multi_press":
        keys = [key.strip().lower() for key in args if key.strip().lower() in pyautogui.KEYBOARD_KEYS]
        if keys:
            pyautogui.hotkey(*keys)
            display_message("system", f"Simultaneously pressed keys: {', '.join(keys)}")
        else:
            display_message("error", "Invalid keys for multi_press command")       
    elif cmd == "multi_hold":
        keys = [key.strip().lower() for key in args if key.strip().lower() in pyautogui.KEYBOARD_KEYS]
        if keys:
            for key in keys:
                pyautogui.keyDown(key)
            display_message("system", f"Keys held down: {', '.join(keys)}")
        else:
            display_message("error", "Invalid keys for multi_hold command")        
    elif cmd == "multi_release":
        keys = [key.strip().lower() for key in args if key.strip().lower() in pyautogui.KEYBOARD_KEYS]
        if keys:
            for key in keys:
                pyautogui.keyUp(key)
            display_message("system", f"Keys released: {', '.join(keys)}")
        else:
            display_message("error", "Invalid keys for multi_release command")         
    elif cmd == "click" or cmd == "leftclick":
        count = int(args[0]) if args and args[0].isdigit() else 1
        for _ in range(count):
            pyautogui.click()
        display_message("system", f"Executed {count} click(s)")        
    elif cmd == "doubleclick":
        count = int(args[0]) if args and args[0].isdigit() else 1
        for _ in range(count):
            pyautogui.doubleClick()
        display_message("system", f"Executed {count} double click(s)")        
    elif cmd == "hold_click":
        duration = float(args[0]) if args else 1.0
        pyautogui.mouseDown()
        pyautogui.sleep(duration)
        pyautogui.mouseUp()
        display_message("system", f"Held click for {duration} seconds")        
    elif cmd == "screenshot" and len(args) == 2:
        x, y = map(int, args)
        screenshot = pyautogui.screenshot()
        screenshot.save(f'screenshot_{x}_{y}.png')
        display_message("system", f"Screenshot saved as screenshot_{x}_{y}.png")       
    elif len(args) == 3:
        cmd, x, y = args
        x, y = map(int, [x, y])
        if cmd.lower() in ["double_click", "doubleclick", "click"]:
            pyautogui.doubleClick(x, y)
        elif cmd.lower() in ["right_click", "rightclick"]:
            pyautogui.rightClick(x, y)
        display_message("system", f"Executed cursor command: {cmd} at ({x}, {y})")

        
def pyautogui_move(args):
    if len(args) < 2:
        display_message("error", "move command requires at least two arguments (x, y).")
        return
    try:
        x = int(args[0].strip())
        y = int(args[1].strip())
        duration = float(args[2].strip()) if len(args) > 2 else 0
        pyautogui.moveTo(x, y, duration=duration)
        display_message("system", f"Moved cursor to ({x}, {y}) over {duration} seconds.")
    except ValueError:
        display_message("error", "Invalid arguments for move command. x and y must be integers.")

def pyautogui_drag(args):
    if len(args) < 2:
        display_message("error", "drag command requires at least two arguments (x, y).")
        return
    try:
        x = int(args[0].strip())
        y = int(args[1].strip())
        duration = float(args[2].strip()) if len(args) > 2 else 0
        button = args[3].strip().lower() if len(args) > 3 else 'left'
        pyautogui.dragTo(x, y, duration=duration, button=button)
        display_message("system", f"Dragged cursor to ({x}, {y}) over {duration} seconds with {button} button.")
    except ValueError:
        display_message("error", "Invalid arguments for drag command. x and y must be integers.")

def pyautogui_scroll_up(args):
    amount = int(args[0].strip()) if args else 1
    pyautogui.scroll(amount)
    display_message("system", f"Scrolled up by {amount} units.")

def pyautogui_scroll_down(args):
    amount = int(args[0].strip()) if args else 1
    pyautogui.scroll(-amount)
    display_message("system", f"Scrolled down by {amount} units.")

def pyautogui_release(args):
    if not args:
        display_message("error", "release command requires at least one argument (key).")
        return
    key = args[0].strip().lower()
    if key in pyautogui.KEYBOARD_KEYS:
        pyautogui.keyUp(key)
        display_message("system", f"Key released: {key}")
    else:
        display_message("error", f"Invalid key name: {key}")

def pyautogui_hotkey(args):
    if not args:
        display_message("error", "hotkey command requires at least one argument.")
        return
    keys = [key.strip() for key in args if key.strip().lower() in pyautogui.KEYBOARD_KEYS]
    if not keys:
        display_message("error", "Invalid keys for hotkey command.")
        return
    pyautogui.hotkey(*keys)
    display_message("system", f"Hotkey pressed: {' + '.join(keys)}")

def pyautogui_type(args):
    if not args:
        display_message("error", "type command requires at least one argument (text).")
        return
    text = args[0]
    interval = float(args[1].strip()) if len(args) > 1 else 0.0
    pyautogui.typewrite(text, interval=interval)
    display_message("system", f"Typed text: {text}")

def pyautogui_multi_press(args):
    if not args:
        display_message("error", "multi_press command requires at least one argument (keys).")
        return
    keys = [key.strip() for key in args if key.strip().lower() in pyautogui.KEYBOARD_KEYS]
    if not keys:
        display_message("error", "Invalid keys for multi_press command.")
        return
    pyautogui.press(keys)
    display_message("system", f"Keys pressed: {', '.join(keys)}")

def pyautogui_multi_hold(args):
    if not args:
        display_message("error", "multi_hold command requires at least one argument (keys).")
        return
    keys = [key.strip() for key in args if key.strip().lower() in pyautogui.KEYBOARD_KEYS]
    if not keys:
        display_message("error", "Invalid keys for multi_hold command.")
        return
    for key in keys:
        pyautogui.keyDown(key)
    display_message("system", f"Keys held down: {', '.join(keys)}")

def pyautogui_multi_release(args):
    if not args:
        display_message("error", "multi_release command requires at least one argument (keys).")
        return
    keys = [key.strip() for key in args if key.strip().lower() in pyautogui.KEYBOARD_KEYS]
    if not keys:
        display_message("error", "Invalid keys for multi_release command.")
        return
    for key in keys:
        pyautogui.keyUp(key)
    display_message("system", f"Keys released: {', '.join(keys)}")

def pyautogui_click(args):
    try:
        x = int(args[0].strip()) if len(args) > 0 else None
        y = int(args[1].strip()) if len(args) > 1 else None
    except ValueError:
        display_message("error", "Invalid coordinates for click command. x and y must be integers.")
        return
    button = args[2].strip().lower() if len(args) > 2 else 'left'
    pyautogui.click(x=x, y=y, button=button)
    display_message("system", f"Clicked at ({x}, {y}) with {button} button.")

def pyautogui_double_click(args):
    try:
        x = int(args[0].strip()) if len(args) > 0 else None
        y = int(args[1].strip()) if len(args) > 1 else None
    except ValueError:
        display_message("error", "Invalid coordinates for doubleclick command. x and y must be integers.")
        return
    button = args[2].strip().lower() if len(args) > 2 else 'left'
    pyautogui.doubleClick(x=x, y=y, button=button)
    display_message("system", f"Double clicked at ({x}, {y}) with {button} button.")

def pyautogui_hold_click(args):
    try:
        x = int(args[0].strip()) if len(args) > 0 else None
        y = int(args[1].strip()) if len(args) > 1 else None
        duration = float(args[2].strip()) if len(args) > 2 else 0.0
    except ValueError:
        display_message("error", "Invalid arguments for hold_click command. x and y must be integers, duration must be a float.")
        return
    button = args[3].strip().lower() if len(args) > 3 else 'left'
    pyautogui.mouseDown(x=x, y=y, button=button)
    time.sleep(duration)
    pyautogui.mouseUp(button=button)
    display_message("system", f"Hold clicked at ({x}, {y}) with {button} button for {duration} seconds.")

def pyautogui_screenshot(args):
    filename = args[0].strip() if args else 'screenshot.png'
    pyautogui.screenshot(filename)
    display_message("system", f"Screenshot saved as {filename}.")


def handle_pyautogui_command(cmd_name, args):
    handler = pyautogui_dispatch.get(cmd_name.lower())
    if handler:
        try:
            handler(args)
        except Exception as e:
            display_message("error", f"Error executing pyautogui command '{cmd_name}': {e}")
    else:
        display_message("error", f"Unknown pyautogui command: {cmd_name}")



def pyautogui_press(args):
    if not args:
        display_message("error", "press command requires at least one argument (key).")
        return
    key = args[0].strip().lower()
    if key in pyautogui.KEYBOARD_KEYS:
        pyautogui.press(key)
        display_message("system", f"Key pressed: {key}")
    else:
        display_message("error", f"Invalid key name: {key}")



def pyautogui_hold(args):
    if not args:
        display_message("error", "hold command requires at least one argument (key).")
        return
    key = args[0].strip().lower()
    if key in pyautogui.KEYBOARD_KEYS:
        duration = float(args[1]) if len(args) > 1 else None
        pyautogui.keyDown(key)
        if duration:
            time.sleep(duration)
            pyautogui.keyUp(key)
        display_message("system", f"Key held for {duration} seconds: {key}")
    else:
        display_message("error", f"Invalid key name: {key}")




def move_key_command(args):
    if len(args) < 2:
        display_message("error", "move_key command requires two arguments (key, tiles).")
        return
    key = args[0].strip().lower()
    tiles = int(args[1])
    move_key(key, tiles)
    display_message("system", f"Moved key: {key} for {tiles} tiles")

def move_key(key, tiles):
    time.sleep(1.35)  # Wait for 3.5 seconds before executing the command
    
    if 1 <= tiles < 4:
        duration = tiles * 0.175  # Multiplier for tile lengths 1-4
    elif 4 <= tiles <= 8:
        duration = tiles * 0.239  # Multiplier for tile lengths 5-8
    elif 9 <= tiles < 10:
        duration = tiles * 0.245  # Multiplier for tile lengths 9-10
    elif tiles >= 10:
        duration = tiles * 0.265    
    else:
        raise ValueError("Invalid tile range. Please choose a number between 1 and 10.")
    
    print('hihihi')
    handle_pyautogui_command("hold", [key, str(duration)])
    print('yesyesyes')


         
def write_content_to_file(content, file_name):
        with open(file_name, "w") as f:
            f.write(content)
        print(f"Content saved to {file_name}")


def save_chat_history_to_file(chat_history, file_name):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_nametime = f"{file_name}_{current_time}.txt"

    with open(file_nametime, 'w', encoding='utf-8') as file:
        for entry in chat_history:
            # Extract information
            timestamp = entry.get('timestamp', 'No timestamp')
            role = entry.get('role', 'No role')
            content = entry.get('content', {})
            content_type = content.get('type', 'No type')

            # Write basic info
            file.write(f"Timestamp: {timestamp}\nRole: {role}\nType: {content_type}\n")

            # Handle different content types
            if content_type == 'text':
                file.write(f"Message: {content.get('text', 'No text content')}\n\n")
            elif content_type == 'image':
                # For image, either write the data or a placeholder
                image_data = content.get('data', 'No image data')
                file.write(f"Image Data: {image_data[50:80]}... [truncated]\n\n")  # Truncate for preview
            else:
                file.write(f"Other Content: {content}\n\n")

    print(f"Chat history saved to {file_nametime}")



@time_logger
def display_message(role, content, token_count=None, include_command=False):
    global show_user_text, show_ai_text, hide_ai_commands, hide_user_commands, REQUIRE_POWER_WORD, POWER_WORD

    token_count_str = f" (Token count: {token_count}/{CONTEXT_LENGTH})" if token_count else ""

    # Determine if content is a dictionary (i.e., possibly containing image data)
    if isinstance(content, dict):
        content_type = content.get('type')
        if content_type == 'text':
            # It's a text message
            formatted_content = content.get('text', '[Missing text]')
        elif content_type == 'image':
            # It's an image. Handle accordingly (e.g., display a placeholder or decode image)
            formatted_content = '[Image content]'
        else:
            formatted_content = '[Unknown content type]'
    else:
        # Content is not a dictionary, assume it's a text string
        formatted_content = content

    role_prefix = {
        "user": "User",
        "assistant": "Assistant",
        "system": "System"
    }.get(role, "Unknown")

    if role == "user" and (not show_user_text or (hide_user_commands and is_command)):
        return
    elif role == "assistant" and (not show_ai_text or (hide_ai_commands and is_command)):
        return

    print(f"{role_prefix}: {formatted_content}{token_count_str}")


command_dispatch = {
    'TOGGLE_POWER_WORD': toggle_power_word,
    'INIT': init_command,
    'PIN': pin_command,
    'ADD_GOAL': add_goal_command,
    'ADD_APRIORITY_GOAL': add_apriority_goal_command,
    'UPDATE_INVENTORY': update_inventory_command,
    'UPDATE_PARTY': update_party_command,
    'UPDATE_CONTEXT': update_context_command,
    'COMPLETE_GOAL': complete_goal_command,
    'tasks': tasks_command,
    'HELP': help_command,
    'SAVE_TASKS': save_tasks_command,
    'LOAD_TASKS': load_tasks_command,
    'CLEAR_NOPIN%': clear_nopin_percentage_command,
    'CLEAR%': clear_percentage_command,
    'CLEAR': clear_all_command,
    'EDIT_MSGS': edit_msgs_command,
    'DELETE_MSG': delete_msg_command,
    'SAVECH': save_chat_history_command,
    'DHISTORY': display_history_command,
    'SAVE_PINS': save_pins_command,
    'SAVE_ALL_PINS': save_all_pins_command,
    'TOGGLE_AUTO_PROMPT': toggle_auto_prompt_command,
    'TOGGLE_ALWAYS': toggle_always_command,
    'TOGGLE_IMAGE_DETAIL': toggle_image_detail_command,
    'TOGGLE_LATEST_IMAGE_DETAIL': toggle_latest_image_detail_command,
    'SET_HIGH_DETAIL': set_high_detail_command,
    'SET_MAX_IMAGES_IN_HISTORY': set_max_images_in_history_command,
    'SET_IMAGE_DETAIL': set_image_detail_command,
    'SET_LATEST_IMAGE_DETAIL': set_latest_image_detail_command,
    'HIDE_USER_TEXT': hide_user_text_command,
    'HIDE_AI_TEXT': hide_ai_text_command,
    'HIDE_AI_COMMANDS': hide_ai_commands_command,
    'HIDE_USER_COMMANDS': hide_user_commands_command,
    'TOGGLE_UMSGS': toggle_unimportant_messages_command,
    'TOGGLE_IMSGS': toggle_important_messages_command,
    'TOGGLE_ADD_UMSGS_TO_HISTORY': toggle_add_umsgs_to_history_command,
    'TOGGLE_ADD_IMSGS_TO_HISTORY': toggle_add_imsgs_to_history_command,
    'TOGGLE_UMSGS_DECAY_CHECK': toggle_umsgs_decay_check_command,
    'TOGGLE_IMSGS_DECAY_CHECK': toggle_imsgs_decay_check_command,
    # Include any other command mappings as needed
}


# PyAutoGUI Command Dispatch Dictionary
pyautogui_dispatch = {
    'press': pyautogui_press,
    'hold': pyautogui_hold,
    'move_key': move_key_command,
    'move': pyautogui_move,
    'drag': pyautogui_drag,
    'scroll_up': pyautogui_scroll_up,
    'scroll_down': pyautogui_scroll_down,
    'release': pyautogui_release,
    'hotkey': pyautogui_hotkey,
    'type': pyautogui_type,
    'multi_press': pyautogui_multi_press,
    'multi_hold': pyautogui_multi_hold,
    'multi_release': pyautogui_multi_release,
    'click': pyautogui_click,
    'leftclick': pyautogui_click,  # Alias for click
    'doubleclick': pyautogui_double_click,
    'hold_click': pyautogui_hold_click,
    'screenshot': pyautogui_screenshot,
    # Include any other pyautogui command mappings as needed
}

class ChatMessage:
    def __init__(self, role, content, token_count, timestamp, file_id=None, image_path=None, last_command=None, mouse_position=None, last_key=None):
        global userName, aiName
        name = aiName if role.lower() == 'assistant' else userName
        self.name = name if name else ""
        self.role = role
        self.content = content
        self.token_count = token_count
        self.timestamp = timestamp
        self.file_id = file_id  # Temporal reference to the image
        self.image_path = image_path  # Permanent reference to the image
        self.last_command = last_command or 'N/A'
        self.mouse_position = mouse_position or 'N/A'
        self.last_key = last_key or 'N/A'

    def format_for_chatgpt(self):
        # Adjust formatting to include both file_id and image_path
        image_info = ""
        if self.file_id:
            image_info += f" FI {self.file_id}"
        if self.image_path:
            image_info += f" IP {self.image_path}"
        return f"{self.role} {self.name} {self.content}{image_info} TC {self.token_count} {self.last_command} {self.last_key}: {self.mouse_position} {self.timestamp}"

def is_important_message(content, exemption):
    # Define a set of important exemption values
    important_exemptions = {'Pinned', 'Init', 'Special', 'Handoff'}

    # Check if the exemption value is in the set of important exemptions
    return exemption in important_exemptions


def is_unimportant_message(content, exemption=None):
    # Check if the message content or its exemption status is unimportant
    return not is_important_message(content, exemption)






def count_tokens_in_history(chat_history):
    # This will join all the 'text' from entries where the type is 'text'
    text = " ".join([entry['content']['text'] for entry in chat_history if entry['content']['type'] == 'text'])
    tokens_used = len(list(tokenizer.encode(text)))
    return f"{tokens_used} out of {CONTEXT_LENGTH} available"






def clear_chat_history_except_pinned(percentage_to_clear):
    # Filter out the important messages
    important_msgs = [entry for entry in chat_history if entry['content'].startswith(('Pinned Init summary', 'Pinned Handoff context', 'Exemption'))]
    non_important_msgs = [entry for entry in chat_history if not entry['content'].startswith(('Pinned Init summary', 'Pinned Handoff context', 'Exemption'))]

    # Randomly select non-important messages to remove based on the percentage
    num_to_remove = int(len(non_important_msgs) * percentage_to_clear)
    msgs_to_remove = random.sample(non_important_msgs, num_to_remove)
    
    # Filter out the selected messages to remove
    chat_history[:] = [msg for msg in chat_history if msg not in msgs_to_remove] + important_msgs

    print(f"Removed {num_to_remove} non-important messages. Important messages are preserved.")


# Define clear_chat_history function with percentage if over limit
def clear_chat_history(percentage):
    global chat_history

    # Number of chats to keep
    num_chats_to_keep = int(len(chat_history) * (1-percentage))

    pinned_entries = [entry for entry in chat_history if entry['content'].startswith('Pinned Init summary') or entry['content'].startswith('Pinned Handoff context')]
    exempted_entries = [entry for entry in chat_history if entry['content'].startswith('Exemption')]

    # Concatenate the lists and sort them by their index in the original chat_history
    exempted_and_pinned_entries = sorted(pinned_entries + exempted_entries, key=chat_history.index)

    # If the number of chats to keep is less than the length of the exempted_and_pinned_entries list
    # Keep only the most recent exempted and pinned entries
    if num_chats_to_keep < len(exempted_and_pinned_entries):
        chat_history = exempted_and_pinned_entries[-num_chats_to_keep:]
    else:
        non_exempted_entries = [entry for entry in chat_history if entry not in exempted_and_pinned_entries]
        chat_history = non_exempted_entries[-(num_chats_to_keep-len(exempted_and_pinned_entries)):] + exempted_and_pinned_entries

    print(f"Chat history cleared, {int(percentage*100)}% chats are removed, only most recent {num_chats_to_keep} entries, pinned summaries, and exemptions remain.")



def check_and_save_chat_history():
    """
    Check the token count in the chat history, and if it reaches the defined limit,
    save the chat history to a file.
    """
    global token_counter

    # Check if the token counter exceeds the limit
    if token_counter > token_limit:
        print("Token limit reached. Saving chat history to file...")
        #save_chat_history_to_file()  # Save the chat history to a text file
        save_chat_history_to_file(chat_history, 'chat_history')
        token_counter = 0  # Reset the token counter


def clear_percentage_except_pinned_and_exempt(command: str):
    try:
        percentage_to_clear = float(command[6:].strip()) / 100
        if 0 <= percentage_to_clear <= 1:
            
            # Separate out pinned/init/exempted entries and others
            pinned_and_exempted_entries = [entry for entry in chat_history if entry['content'].startswith('Pinned Init summary') or entry['content'].startswith('Pinned Handoff context') or entry['content'].startswith('Exemption')]
            other_entries = [entry for entry in chat_history if entry not in pinned_and_exempted_entries]

            # Calculate the number of entries to keep
            num_to_keep = int(len(other_entries) * (1 - percentage_to_clear))
            kept_entries = other_entries[:num_to_keep]

            # Merge the kept entries with the pinned/init/exempted ones
            chat_history[:] = pinned_and_exempted_entries + kept_entries

            print(f"Cleared {percentage_to_clear*100}% of chat history excluding pinned and exempted entries.")
        else:
            print("Invalid percentage. Please enter a number between 0 and 100.")
    except ValueError:
        print("Invalid command format. Expected format is 'CLEAR%{number}'.")


def add_pinned_init_summary(init_summary: str):
    if any(entry['content'].startswith(f'Pinned Init summary: {init_summary}') for entry in chat_history):
        print("Handoff summary already in chat history.")
        return

    chat_history.append({'role': 'user', 'content': f'Pinned Init summary: {init_summary}'})
    print(f"Pinned Init summary: {init_summary}")

def retrieve_and_add_handoff(handoff_filename: str):
    handoff_filepath = f"{logging_folder}/{handoff_filename}"
    
    if not os.path.exists(handoff_filepath):
        print(f"No such file exists: {handoff_filepath}")
        return

    with open(handoff_filepath, "r") as file:
        handoff_summary = file.read().strip()

    if any(entry['content'].startswith(f'Pinned Handoff context: {handoff_summary}') for entry in chat_history):
        print("Handoff summary already in chat history.")
        return

    chat_history.append({'role': 'assistant', 'content': f'Pinned Handoff context: {handoff_summary}'})
    print(f"Pinned Handoff context from file: {handoff_summary}")

def clear_non_pinned_entries():
    chat_history[:] = [entry for entry in chat_history if entry['content'].startswith('Pinned Init summary') or entry['content'].startswith('Pinned Handoff context') or entry['content'].startswith('Exemption')]
    print("Chat history cleared, only pinned summaries and exemptions remain.")




# Function to check if daily summary has been completed
def daily_summary_completed(date):
    filename = f"{logging_folder}/daily_summary_{date}.txt"
    return os.path.exists(filename)

# Function to save daily summary
def save_daily_summary():
    date = datetime.date.today().strftime("%Y-%m-%d")
    if not daily_summary_completed(date):
        summary_prompt = "Summarize the important events and points from Aurora's perspective today."
        daily_summary_text = send_prompt_to_chatgpt(summary_prompt)
        with open(f"{logging_folder}/daily_summary_{date}.txt", "w") as log_file:
            log_file.write(daily_summary_text)
        print("Daily summary saved.")

# Function to check if it's time to save the daily summary
def check_daily_summary_time():
    current_time = datetime.datetime.now().time()
    daily_summary_time = datetime.time(hour=19)  # 7 PM
    return current_time >= daily_summary_time

def daily_summary():
    current_time = datetime.datetime.now()
    if current_time.hour == 19:  # 7 pm
        summary = send_prompt_to_chatgpt("Summarize today's important events and points from Aurora's perspective.")
        send_prompt_to_chatgpt(summary)
        with open(f"{logging_folder}/daily_summary_{current_time.strftime('%Y-%m-%d')}.txt", "w") as log_file:
            log_file.write(summary)
        print("Daily summary saved.")

# Function to check for handoff from previous day
def check_previous_handoff():
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    handoff_filename = f"{logging_folder}/handoff_{yesterday.strftime('%Y-%m-%d')}.txt"
    if os.path.exists(handoff_filename):
        with open(handoff_filename, "r") as handoff_file:
            handoff_text = handoff_file.read()
        response_text = send_prompt_to_chatgpt(handoff_text)
        send_prompt_to_chatgpt(response_text)

# Function to perform shutdown procedure for the current ChatGPT instance and exit the program
def shutdown_chatgpt_instance_and_exit():
    global running
    date = datetime.date.today().strftime("%Y-%m-%d")
    if not daily_summary_completed(date):
        save_daily_summary()
    handoff_summary = f"Handoff {date}: " + hbuffer[-1]  # Save the last summary in the buffer
    with open(f"{logging_folder}/handoff_{date}.txt", "w") as log_file:
        log_file.write(handoff_summary)
    print("Handoff summary saved.")
    # Exit the program
    print("Exiting the program.")
    sys.exit()

def terminate_instance():
    daily_summary()
    handoff_summary = send_prompt_to_chatgpt("Create a handoff summary for the next instance.")
    send_prompt_to_chatgpt(handoff_summary)
    with open(f"{logging_folder}/Handoff_{datetime.datetime.now().strftime('%Y-%m-%d')}.txt", "w") as log_file:
        log_file.write(handoff_summary)
    print("Handoff summary saved.")
    # You can use sys.exit() to exit the program
    sys.exit()


# Function to restart ChatGPT instance
def restart_chatgpt_instance():
    # Send init_prompt and handoff_prompt to the new instance
    response_text = send_prompt_to_chatgpt(init_prompt)
    send_prompt_to_chatgpt(response_text)
    if hbuffer:
        handoff_prompt = hbuffer[-1]
        response_text = send_prompt_to_chatgpt(handoff_prompt)
    
    # Send init_prompt and handoff_prompt to the new instance
    response_text = send_prompt_to_chatgpt(init_prompt)
  

def check_for_previous_handoff():
    today = datetime.datetime.now().date()
    yesterday = today - datetime.timedelta(days=1)
    handoff_filename = f"{logging_folder}/Handoff_{yesterday.strftime('%Y-%m-%d')}.txt"
    if os.path.exists(handoff_filename):
        with open(handoff_filename, "r") as handoff_file:
            handoff_text = handoff_file.read()
        response_text = send_prompt_to_chatgpt(handoff_text)
        send_prompt_to_chatgpt(response_text)


def recall_previous_summary(character_name):
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    handoff_file = f"{logging_folder}/handoff_{yesterday.strftime('%Y-%m-%d')}.txt"
    
    if os.path.exists(handoff_file):
        with open(handoff_file, "r") as f:
            handoff_text = f.read().strip()
        print("Handoff summary:", handoff_text)
    else:
        print("No handoff summary found for the previous day.")
    
    init_file = f"{character_name}_init.txt"
    if os.path.exists(init_file):
        with open(init_file, "r") as f:
            init_text = f.read().strip()
        print("Initial profile summary:", init_text)
    else:
        print("No initial profile summary found.")

def listen_to_keyboard():
    global last_key
    while True:
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_DOWN:
            with lock:
                last_key = event.name





# Function to add a grid with color-coded tiles and labeled coordinates




#def draw_cursor(draw, cursor_position, cursor_size):
#    # Define colors
#    outer_color = "black"
#    inner_color = "white"
#    large_circle_color = "red"
#    medium_circle_color = "red"
#    small_circle_color = "blue"
#
#    # Outer rectangle (black outline)
#    outer_rectangle = [cursor_position.x - 1, cursor_position.y - 1, cursor_position.x + cursor_size + 1, cursor_position.y + cursor_size + 1]
#    draw.rectangle(outer_rectangle, outline=outer_color, fill=outer_color)
#
#    # Inner rectangle (white cursor)
#    inner_rectangle = [cursor_position.x, cursor_position.y, cursor_position.x + cursor_size, cursor_position.y + cursor_size]
#    draw.rectangle(inner_rectangle, outline=inner_color, fill=inner_color)
#
#    # Draw an 'X' inside the rectangle
#    draw.line([cursor_position.x, cursor_position.y, cursor_position.x + cursor_size, cursor_position.y + cursor_size], fill=outer_color)
#    draw.line([cursor_position.x, cursor_position.y + cursor_size, cursor_position.x + cursor_size, cursor_position.y], fill=outer_color)
#
#    # Draw the red circle around the cursor
#    large_radius = cursor_size + 3  # Adjust the radius as needed
#    large_circle_bounds = [cursor_position.x - large_radius, cursor_position.y - large_radius, cursor_position.x + cursor_size + large_radius, cursor_position.y + cursor_size + large_radius]
#    draw.ellipse(large_circle_bounds, outline=large_circle_color, width=2)
#
#    # Calculate the center of the square
#    center_x = cursor_position.x + cursor_size / 2
#    center_y = cursor_position.y + cursor_size / 2
#
#    # Draw the medium red circle that the box fits into perfectly
#    medium_radius = (cursor_size / 2) * (2 ** 0.5)  # sqrt(2) times the half size of the square
#    medium_circle_bounds = [center_x - medium_radius, center_y - medium_radius, center_x + medium_radius, center_y + medium_radius]
#    draw.ellipse(medium_circle_bounds, outline=medium_circle_color, width=2)
#
#    # Draw the smaller blue circle that circumscribes the square reticle
#    small_radius = cursor_size / 2
#    small_circle_bounds = [center_x - small_radius, center_y - small_radius, center_x + small_radius, center_y + small_radius]
#    draw.ellipse(small_circle_bounds, outline=small_circle_color, width=2)
#
#    # Add cursor coordinates with background
#    screen_width, screen_height = draw.im.size
#    text_color = "white"
#    background_color = (0, 128, 0)  # Greenish background color
#    background_opacity = 128
#
#    # Customize the font size
#    font_size = 23
#    # Load a TrueType font with the specified size
#    try:
#        font = ImageFont.truetype(FONT_PATH, font_size)
#    except IOError:
#        print("Font file not found. Falling back to default font.")
#        font = ImageFont.load_default()  # Fallback to default font if TrueType font not found
#
#    # Calculate position for text
#    shift_x, shift_y = 5, 20
#    text_position = (cursor_position.x + shift_x, cursor_position.y + shift_y)
#    if cursor_position.x + shift_x + 100 > screen_width:
#        shift_x = -100  # Adjust shift to place text on the left
#    if cursor_position.y + shift_y + 20 > screen_height:
#        shift_y = -30  # Adjust shift to place text above
#
#    cursor_text = f"({cursor_position.x}, {cursor_position.y})"
#    draw_text_with_background(draw, text_position, cursor_text, font, text_color, background_color, background_opacity, shift_x, shift_y)


# Function to add the colored tile grid with only directional labels
@time_logger
def add_colored_tile_grid_v3(image, center_tile, tile_size, colors_x, colors_y):
    
    draw = ImageDraw.Draw(image)
    
        # Calculate half-tile offsets for proper grid alignment
    half_tile_offset = (tile_size[0] // 2, tile_size[1] // 2)
    # Apply a half-tile shift to center the grid
    x_offset = -half_tile_offset[0]
    y_offset = -half_tile_offset[1]

    # Loop through the grid around the center tile
    for i in range(-5, 6):  # Adjust the range for the grid size (5 tiles in each direction)
        for j in range(-5, 6):
            # Calculate the position of the current tile
            tile_x = center_tile[0] + i * tile_size[0] + x_offset
            tile_y = center_tile[1] + j * tile_size[1] + y_offset

            # Determine the color for X and Y distances
            color_x = colors_x[min(abs(i), len(colors_x) - 1)]
            color_y = colors_y[min(abs(j), len(colors_y) - 1)]

            # Draw the tile with the appropriate colors
            draw_tile_with_colors(draw, tile_x, tile_y, tile_size, color_x, color_y)

            # Label only the direct horizontal and vertical tiles (1-5 tiles away)
            if i == 0 and j != 0:  # Vertical tiles (Up/Down)
                label = f"{abs(j)}T" + ("U" if j < 0 else "D")
                draw_text_with_background(draw, (tile_x + 5, tile_y - 15), label, font)
            elif j == 0 and i != 0:  # Horizontal tiles (Left/Right)
                label = f"{abs(i)}T" + ("L" if i < 0 else "R")
                draw_text_with_background(draw, (tile_x + 5, tile_y - 15), label, font)



# Helper function to draw the tiles with X/Y axis colors
@time_logger
def draw_tile_with_colors(draw, x, y, tile_size, color_x, color_y):
    # Draw the left and right borders (X axis)
    draw.line([(x, y), (x, y + tile_size[1])], fill=color_x, width=3)  # Left
    draw.line([(x + tile_size[0], y), (x + tile_size[0], y + tile_size[1])], fill=color_x, width=3)  # Right

    # Draw the top and bottom borders (Y axis)
    draw.line([(x, y), (x + tile_size[0], y)], fill=color_y, width=3)  # Top
    draw.line([(x, y + tile_size[1]), (x + tile_size[0], y + tile_size[1])], fill=color_y, width=3)  # Bottom


@time_logger
def draw_custom_cursor(draw, cursor_position, cursor_size):
    # Define colors
    outer_color = "black"
    inner_color = "white"
    large_circle_color = "red"
    medium_circle_color = "red"
    small_circle_color = "blue"

    # Outer rectangle (black outline)
    outer_rectangle = [
        cursor_position.x - 1,
        cursor_position.y - 1,
        cursor_position.x + cursor_size + 1,
        cursor_position.y + cursor_size + 1
    ]
    draw.rectangle(outer_rectangle, outline=outer_color, fill=outer_color)

    # Inner rectangle (white cursor)
    inner_rectangle = [
        cursor_position.x,
        cursor_position.y,
        cursor_position.x + cursor_size,
        cursor_position.y + cursor_size
    ]
    draw.rectangle(inner_rectangle, outline=inner_color, fill=inner_color)

    # Draw an 'X' inside the rectangle
    draw.line(
        [
            (cursor_position.x, cursor_position.y),
            (cursor_position.x + cursor_size, cursor_position.y + cursor_size)
        ],
        fill=outer_color
    )
    draw.line(
        [
            (cursor_position.x, cursor_position.y + cursor_size),
            (cursor_position.x + cursor_size, cursor_position.y)
        ],
        fill=outer_color
    )

    # Draw the red circles
    center_x = cursor_position.x + cursor_size / 2
    center_y = cursor_position.y + cursor_size / 2

    large_radius = cursor_size + 3
    large_circle_bounds = [
        center_x - large_radius,
        center_y - large_radius,
        center_x + large_radius,
        center_y + large_radius
    ]
    draw.ellipse(large_circle_bounds, outline=large_circle_color, width=2)

    medium_radius = (cursor_size / 2) * (2 ** 0.5)
    medium_circle_bounds = [
        center_x - medium_radius,
        center_y - medium_radius,
        center_x + medium_radius,
        center_y + medium_radius
    ]
    draw.ellipse(medium_circle_bounds, outline=medium_circle_color, width=2)

    small_radius = cursor_size / 2
    small_circle_bounds = [
        center_x - small_radius,
        center_y - small_radius,
        center_x + small_radius,
        center_y + small_radius
    ]
    draw.ellipse(small_circle_bounds, outline=small_circle_color, width=2)

class CursorPosition:
    def __init__(self, x, y):
        self.x = x
        self.y = y

@time_logger
def draw_cursor(draw, cursor_position, cursor_size, native_cursor=False, font=None, cursor_image=None,):
    if not native_cursor:
        # Draw the custom cursor
        draw_custom_cursor(draw, cursor_position, cursor_size)
    if native_cursor:
        x=cursor_position[0]
        y=cursor_position[1]
        CursorPosition.x=x
        CursorPosition.y=y
        if cursor_image is None or cursor_image.getbbox() is None:
            draw_custom_cursor(draw, pyautogui.position(), cursor_size)
            # After getting the cursor image
        #if cursor_image:
        #    cursor_image.save('cursor_debug.png')
            
    # Always draw the cursor coordinates label
    draw_cursor_label(draw, pyautogui.position(), font)

# Helper function to draw text with background
@time_logger
def draw_text_with_background(draw, position, text, font, text_color="white", background_color=(0, 0, 0), background_opacity=128, shift_x=5, shift_y=20):
    # Calculate text size
    text_bbox = draw.textbbox(position, text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Set padding around the text
    padding = 4

    # Define the background rectangle position with shift
    background_position = (
        position[0] - padding + shift_x,
        position[1] - padding + shift_y,
        position[0] + text_width + padding + shift_x,
        position[1] + text_height + padding + shift_y
    )

    # Create a transparent image for the background
    background_image = Image.new('RGBA', draw.im.size, (0, 0, 0, 0))
    background_draw = ImageDraw.Draw(background_image)
    background_color_with_opacity = (background_color[0], background_color[1], background_color[2], background_opacity)
    background_draw.rectangle(background_position, fill=background_color_with_opacity)

    # Extract the alpha channel from the background image
    alpha = background_image.split()[3]

    # Composite the transparent background onto the original image using paste with a mask
    bounding_box = (0, 0, draw.im.size[0], draw.im.size[1])
    draw.im.paste(background_image.im, bounding_box, alpha.im)

    # Draw the text over the background rectangle
    draw.text((position[0] + shift_x, position[1] + shift_y), text, fill=text_color, font=font)

@time_logger
def draw_cursor_label(draw, cursor_position, font_size=12, shift_x=5, shift_y=20):
    text_color = "white"
    background_color = (0, 128, 0)  # Greenish background
    background_opacity = 128
    # Customize the font size
    font_size = 23
        # Load the font with the specified size
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except IOError:
        print("Font file not found. Falling back to default font.")
        font = ImageFont.load_default()

    cursor_text = f"({cursor_position.x}, {cursor_position.y})"
    text_position = (cursor_position.x + shift_x, cursor_position.y + shift_y)

    draw_text_with_background(
        draw,
        text_position,
        cursor_text,
        font,
        text_color,
        background_color,
        background_opacity,
        shift_x,
        shift_y
    )    

# Function to add a dot grid with labeled coordinates
@time_logger
def add_dot_grid_with_labels(image, grid_interval, key_points):
    draw = ImageDraw.Draw(image)

    # Draw dots and coordinates on the key points
    for x in range(0, image.size[0], grid_interval):
        for y in range(0, image.size[1], grid_interval):
            draw.ellipse((x-4, y-4, x+4, y+4), fill="red", outline="red")  # Draw the larger red dot
            if (x == 0 or y == 0) or (x, y) in key_points:  # Label only top row, left column, and key points
                coordinate_label = f"{x},{y}"
                draw_text_with_background(draw, (x + 5, y - 15), coordinate_label, font, background_opacity=128)





# Function to add grids, tiles, and labels
@time_logger
def add_grids_and_labels(screenshot, cursor_position, current_last_key):
    draw = ImageDraw.Draw(screenshot)  # This creates the 'draw' object to work with the screenshot

    add_dot_grid_with_labels(screenshot, grid_interval=150, key_points=[
        (150, 150), (300, 750), (450, 150), 
        (900, 150), (1200, 150),
        (1050, 750), (1350, 300), (1350, 600),
        (1350, 750)
    ])

    if enable_colored_tile_grid:
        # Continue with your grid and tile drawing
        add_colored_tile_grid_v3(screenshot, center_tile=(719, 444), tile_size=(162, 98),
                                 colors_x=['blue', 'red', 'orange', 'yellow', 'purple', 'black'],
                                 colors_y=['blue', 'red', 'orange', 'yellow', 'purple', 'black'])


    # Customize the font size
    font_size = 23
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except IOError:
        print("Font file not found. Falling back to default font.")
        font = ImageFont.load_default()

    # Draw text with background
    text_position = (40, 55)  # Position of the text
    text_info = f"Cursor Position: {cursor_position} | Last Key: {current_last_key} | Timestamp: {image_timestamp}"
    draw_text_with_background(draw, text_position, text_info, font, background_opacity=128, shift_x=5, shift_y=20)


# Function to handle sending the screenshot and user input
def handle_queued_input(screenshot_file_path, image_timestamp):
    global queued_user_input

    combined_input_list = []
    while not queued_user_input.empty():
        combined_input_list.append(queued_user_input.get())

    if combined_input_list and SEND_TO_MODEL:
        combined_input = ' '.join(combined_input_list)
        prompt = f"user input: {combined_input}"
        send_prompt_to_chatgpt(prompt, role="user", image_path=screenshot_file_path, image_timestamp=image_timestamp)
    else:
        # Decide whether to send the screenshot without text or skip sending
        if SEND_TO_MODEL:
            # Send a default prompt with the screenshot
            send_prompt_to_chatgpt("System: Screenshot taken. Please provide instructions...", role="user", image_path=screenshot_file_path, image_timestamp=image_timestamp)
        ###else:
            # If not sending text to the model, perhaps send only the image
            #send_prompt_to_chatgpt("", role="user", image_path=screenshot_file_path, image_timestamp=image_timestamp)
            ###print ('no queue')


    # Use the function and provide a path to save the screenshot
    #capture_screenshot_with_cursor_info('screenshot_info.png')
# Function to take a screenshot
@time_logger
def take_screenshot():
    global last_key  # Ensure you are referring to the global variable updated by the listener thread
    global image_timestamp
    global queued_user_input


    # Initialize variables to None
    cursor_image = None
    cursor_position = None
    screenshot = None  # Initialize screenshot to None

    # Check if the third option for pyautogui native screenshot is enabled
    if screenshot_options.get("native_cursor_screenshot"):
        # Capture entire screen with cursor using pyautogui.screenshot()
        #screenshot = pyautogui.screenshot()
        # Replace pyautogui.screenshot() with ImageGrab.grab()
        screenshot = ImageGrab.grab()
        mixer.music.load(speech_off_wav)
        mixer.music.play()
        ###print(f"Screenshot mode: {screenshot.mode}")
        ###print(f"Cursor image mode: {cursor_image.mode if cursor_image else 'None'}")
        if cursor_image is not None:
            if cursor_image.mode != screenshot.mode:
                cursor_image = cursor_image.convert(screenshot.mode)
        ###else:
           ### print("Cursor image is None.")

        # Get the mouse cursor's current position
        #cursor_position = pyautogui.position()

        # Get the cursor image, hotspot, and position
        cursor_image, hotspot, cursor_position = get_cursor()
        
        if cursor_image is not None:
            ###print("Have Cursor image")
            # Calculate where to paste the cursor image on the screenshot
            x = cursor_position[0] - hotspot[0]
            y = cursor_position[1] - hotspot[1]

            # Paste the cursor image onto the screenshot with transparency
            screenshot.paste(cursor_image, (x, y), cursor_image)
        else:
            print("Could not capture cursor image.")

        # Ensure thread-safe access to `last_key`
        with lock:
            current_last_key = last_key

        # Draw additional elements if needed
        draw = ImageDraw.Draw(screenshot)

        # Add grid, tiles, and text like before
        add_grids_and_labels(screenshot, cursor_position, current_last_key)
        draw_cursor(draw, cursor_position, cursor_size, True, 23, cursor_image)  # Optional custom cursor drawing
        
        # Draw cursor label only
        #draw = ImageDraw.Draw(screenshot)
        try:
            font = ImageFont.truetype(FONT_PATH, 23)
        except IOError:
            print("Font file not found. Falling back to default font.")
            font = ImageFont.load_default()
    # First option: Capture current window (if implemented)
    elif screenshot_options["current_window"]:
        # Code to capture the current window snapshot
        # This can be platform-specific and needs to be implemented
        # You need to implement this option correctly or remove it if not used.
        # For now, let's set it to the full-screen screenshot as a placeholder:
        screenshot = ImageGrab.grab()  # Temporary fallback
        pass

    # Second option: Capture entire screen using ImageGrab
    elif screenshot_options["entire_screen"]:
        screenshot = ImageGrab.grab()
        print(f"Screenshot mode: {screenshot.mode}")
        #print(f"Cursor image mode: {cursor_image.mode if cursor_image else 'None'}")
        #if cursor_image.mode != screenshot.mode:
        #    cursor_image = cursor_image.convert(screenshot.mode)
        # Get the mouse cursor's current position
        cursor_position = pyautogui.position()

        # Use the global last_key variable instead of reading the event here
        with lock:
            current_last_key = last_key

        # Draw a representation of the cursor on the screenshot
        draw = ImageDraw.Draw(screenshot)
        draw_cursor(draw, cursor_position, cursor_size)

        # Add grid, tiles, and text like before
        add_grids_and_labels(screenshot, cursor_position, current_last_key)

    # Save the screenshot, regardless of which option was used
    current_time = datetime.datetime.now()
    image_timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
    timestamp = int(time.time())
    screenshot_file_path = f"{logging_folder}/{timestamp}.png"
    screenshot.save(screenshot_file_path)
    print(f"Screenshot saved at {screenshot_file_path} at {image_timestamp}")
    #winsound.Beep(1000, 500)  # Frequency = 1000Hz, Duration = 500ms
    mixer.music.load(chimeswav)
    mixer.music.play()
    # Handle user input and send the screenshot
    handle_queued_input(screenshot_file_path, image_timestamp)



def initiate_and_handoff():
    global init_handoff_in_progress
    init_handoff_in_progress = True

    try:
        # Check if the init prompt is already in the chat history
        if not any(entry['content'].startswith(f'Pinned Init summary: {init_prompt}') for entry in chat_history):
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
           # init_message = f'Pinned Init summary: {init_prompt}'
            # Update the chat history with the init message and the Init exemption
            #update_chat_history("system", init_message, exemption='Init')
            send_prompt_to_chatgpt(init_prompt, role="system", exemption='Init')
            # Optionally process the response_text further as needed
    except openai.error.InvalidRequestError as e:
        print(f"Error during initiation: {e}")



    init_handoff_in_progress = False
    #handoff_to_chatgpt()  #still need to work on this

def handoff_to_chatgpt():
    while hbuffer:
        try:
            handoff_prompt = hbuffer.pop(0)  # Using a queue mechanism to prevent potential infinite loops
            response_text = send_prompt_to_chatgpt(handoff_prompt)
            send_prompt_to_chatgpt(response_text)
        except openai.error.InvalidRequestError as e:
            print(f"Error during handoff: {e}")
        if any(entry['content'].startswith(f'Pinned Handoff context: {handoff_prompt}') for entry in chat_history):
            print("Handoff summary already in chat history.")
            return

    


# Function to run the scheduled tasks
def run_scheduled_tasks():
    while running:
        schedule.run_pending()
        time.sleep(1)

# Read the init prompt from a file
#init_file = f"{character_name}_init.txt"
#with open(init_file, "r") as f:
#    init_prompt = f.read().strip()

if INIT_MODE == 'load':
    # Load init prompt from the file
    init_file = f"{character_name}_init.txt"
    try:
        with open(init_file, "r") as f:
            init_prompt = f.read().strip()
        print(f"Init prompt loaded from {init_file}")
        initiate_and_handoff()
    except FileNotFoundError:
        print(f"Init file {init_file} not found. Proceeding without an init prompt.")
elif INIT_MODE == 'test':
    # Use the test init prompt
    init_prompt = test_init_prompt
    print("Using test init prompt.")
    initiate_and_handoff()
elif INIT_MODE == 'skip':
    # Skip loading the init prompt
    print("Skipping init prompt loading.")
else:
    print(f"Unknown INIT_MODE: {INIT_MODE}. Proceeding without an init prompt.")


# Read the handoff summary from a file
handoff_file = f"{character_name}_handoff.txt"
if os.path.exists(handoff_file):
    with open(handoff_file, "r") as f:
        handoff_text = f.read().strip()
    response_text = send_prompt_to_chatgpt(handoff_text)
    send_prompt_to_chatgpt(response_text)      #Fixxxxxxxxxxxxxxxxxxxxxx   meeeee


# Function to monitor and restart threads if necessary
def thread_watchdog(threads):
    while running:
        for thread_name, thread_obj in threads.items():
            if not thread_obj.is_alive():
                print(f"Thread {thread_name} seems to have crashed. Restarting...")
                new_thread = threading.Thread(target=thread_obj._target)
                new_thread.daemon = True
                new_thread.start()
                threads[thread_name] = new_thread
        time.sleep(5)  # Adjust the sleep time as necessary

# Global variable to control the main loop
running = True




# Set up screenshot options
screenshot_options = {
    "current_window": False,
    "entire_screen": False,
    "native_cursor_screenshot": True  # Set to True by default for native cursor screenshots
}

# Set up buffer and logging folder
screenbuffer = []
hbuffer= []
logging_folder = "screenshots"
if not os.path.exists(logging_folder):
    os.makedirs(logging_folder)












#
## Schedule the screenshot taking function
schedule.every(time_interval).seconds.do(take_screenshot)
#
## Schedule daily summary
schedule.every().day.at("19:00").do(save_daily_summary)
#


# Dictionary to keep track of threads
threads = {
    "user_input_thread": threading.Thread(target=user_input_handler),
    "listen_to_keyboard": threading.Thread(target=listen_to_keyboard),
    "scheduled_tasks_thread": threading.Thread(target=run_scheduled_tasks)

}

# Start the threads and store them in the dictionary
for thread_name, thread_obj in threads.items():
    thread_obj.daemon = True
    thread_obj.start()

# Create a thread for the watchdog function and start it
watchdog_thread = threading.Thread(target=thread_watchdog, args=(threads,))
watchdog_thread.daemon = True
watchdog_thread.start()

# Main loop to keep the program running
while running:
    pass
