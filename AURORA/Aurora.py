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

API_KEY=                                                                                                                                                                                                                                                                                                                "sk-proj-SECRETKEY"
POWER_WORD = ""
REQUIRE_POWER_WORD = False
chat_history = []
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
MAX_IMAGES_IN_HISTORY = 15  # Global variable for the maximum number of images to retain
#image_detail =  "high"   # "low" or depending on your requirement
image_detail =  "low"   # "high" or depending on your requirement
latest_image_detail = "high"  # "high" for the latest image, "low" for older images
# Define the High_Detail global variable
global High_Detail
High_Detail = 7  # Initialize to 0 or set as needed  pos for recent high detail, neg for last
global Recent_images_High
Recent_images_High= 7

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
time_interval = 6.5 # Time interval between screenshots (in seconds)

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

# Parameters
threshold = 16.11  
pause_duration =  1.33  #1.8  
#sampling_rate = 44100 
sampling_rate = fs 
audio_buffer = []


# Additional global variables
AUTO_PROMPT_INTERVAL = 9  # Auto-prompt every 30 seconds, but you can change this value # Default auto message depreciated in favor of send_screenshot
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

    if volume_norm < threshold:
        if len(audio_buffer) > sampling_rate * pause_duration:  
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

            # Put the transcribed text into the queue
            queued_user_input.put(transcribed_text)

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
    'CLEAR INIT', 'clear init', 'CLEAR PIN', 'clear pin', 'CLEAR HANDOFF', 'clear handoff', 'TOGGLE AUTO PROMPT', 'toggle auto prompt'
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
def send_prompt_to_chatgpt(prompt, role="user", image_path=None, image_timestamp=None, exemption=None, sticky=False):
    global token_counter
    global init_handoff_in_progress
    global text_count
    global image_or_other_count
    global image_detail
    global Recent_images_High

    current_time = datetime.datetime.now()
    timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
    # Ensure the lock and headers are defined at the appropriate place in your code.
    with lock:
        # Headers for the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
    
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
           # messages.append({
           #     "role": "user",
           #     "content": [
           #         {"type": "text", "text": "What’s in this image please be short and concise? What images & text are in this conversation so far?"},
           #         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}, "detail": latest_image_detail}
           #     ]
           # })

            #messages.append({
            #    "role": "user",
            #    "content": {
            #        {"type": "text", "text": "What’s in this image please be short and concise? What images & text are in this conversation so far?"},  # Use your specific prompt if needed
            #        #{"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}",
            #        #{"type": "image_url", "image_url":  f"data:image/png;base64,{base64_image}",
            #     #"detail": latest_image_detail}
            #
            #}})
            #messages.append({
            #    "role": "user",
            #    "content": {
            #        "type": "text",
            #        "text": "What’s in this image please be short and concise? What images & text are in this conversation so far?"
            #    }
            #})
            #messages.append({
            #    "role": "user",
            #    # Content should be an object, not a list
            #    "content": {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}", "detail": latest_image_detail}  # Changed: made image_url a key-value pair directly
            #})
            print(f" In Image Path, Prompt: ", prompt)
            print(f"In send_prompt_to_chatgpt, Before user update_chat_history, Image Timestamp: {image_timestamp}")
            # Reset the image detail level for older images

            # Add the image data to chat history using update_chat_history
            #update_chat_history("user", prompt, image_path=image_path, image_timestamp=image_timestamp, sticky=sticky)
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
            "model": "gpt-4o-mini",            
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


                # Here you check if the AI's response is a command and handle it
            if is_command(ai_response):
                handle_commands(ai_response, is_user=False)
            else:
                if not is_important_message(prompt, entry.get("Exemption")):
                    #update_chat_history("assistant", ai_response)

                     # Token counting and threshold checking
                     token_count = count_tokens_in_history(chat_history)
                     current_token_count = int(token_count.split()[0])
                     token_counter = current_token_count
                     print(f"Current token count in chat history: {token_count}")
                     if threshold_check(current_token_count, CONTEXT_LENGTH, 75):
                         warning_message = "Warning: Approaching token limit and context length"
                         display_message("system", warning_message)
                         #chat_history.append({"role": "system", "content": warning_message})
 
            return ai_response
        else:
            # Handle errors
            print(f"Error: {response.status_code}")
            print(f"Message: {response.text}")
            return None

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
    #    # Format the message to include the timestamp
    #    content_with_timestamp = f"Timestamp: {timestamp}] {content}"
    #else:
    #    content_with_timestamp = content

    # Update mouse position from pyautogui with error handling
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
            queued_user_input = user_input
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

# Function for human-like click
#def human_like_click(x_pixel, y_pixel):
#    start_time = time.time()
#    while time.time() - start_time < circle_duration:
#        angle = ((time.time() - start_time) / circle_duration) * 2 * math.pi
#        x = x_pixel + math.cos(angle) * circle_radius
#        y = y_pixel + math.sin(angle) * circle_radius
#        pyautogui.moveTo(x, y, duration=0.1)
#    pyautogui.click(x_pixel, y_pixel)

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

def handle_commands(command_input, is_user=True, exemption=None):
    global chat_history
    global disable_commands
    global REQUIRE_POWER_WORD
    global show_user_text
    global show_ai_text
    global hide_ai_commands
    global hide_user_commands
    global token_counter
    global enable_unimportant_messages
    global enable_important_messages
    global ADD_UMSGS_TO_HISTORY
    global ADD_IMSGS_TO_HISTORY
    global Always_
    global hide_input
    global image_detail
    global latest_image_detail
    global High_Detail
    global MAX_IMAGES_IN_HISTORY

    if command_input.startswith('/*'):
        disable_commands = True
        display_message("system", "Command processing is now disabled.")
        return
    elif command_input.startswith('*/'):
        disable_commands = False
        display_message("system", "Command processing is now enabled.")
        return

    if disable_commands:
        display_message("system", "Command processing is currently disabled.")
        return

    if command_input is None:
        return

    global last_command
    last_command = command_input
    command = None

    try:
        commands, *comment = command_input.split('#')
        commands = commands.split(';')
        for command in commands:
            command = command.strip()
            if not command:
                continue

            try:
                if command.startswith('pyag:'):
                    pyag_commands = command[5:].split(';')
                    for pyag_command in pyag_commands:
                        pyag_command = pyag_command.strip()
                        if not pyag_command or pyag_command.startswith('#'):
                            continue

                        if '(' in pyag_command and ')' in pyag_command:
                            cmd, args_part = pyag_command.split('(', 1)
                            args_part = args_part.rstrip(')')
                            args = args_part.split(',') if args_part else []
                            cmd = cmd.strip().lower()
                        else:
                            args = []
                            cmd = pyag_command.lower()

                        handle_pyautogui_command(cmd, args)
                    #update_chat_history('system', command_input)
                    continue

                if '(' in command and ')' in command:
                    cmd, args_part = command.split('(', 1)
                    args_part = args_part.rstrip(')')
                    args = args_part.split(',') if args_part else []
                    cmd = command.lower()
                else:
                    args = []
                    cmd = command.lower()

                if cmd == "toggle_image_detail":
                    image_detail = "high" if image_detail == "low" else "low"
                    display_message("system", f"Global image_detail toggled to: {image_detail}")
                    #update_chat_history('system', f"Global image_detail toggled to: {image_detail}")

                elif cmd == "toggle_latest_image_detail":
                    latest_image_detail = "high" if latest_image_detail == "low" else "low"
                    display_message("system", f"Global latest_image_detail toggled to: {latest_image_detail}")
                    #update_chat_history('system', f"Global latest_image_detail toggled to: {latest_image_detail}")

                elif cmd == "set_high_detail":
                    try:
                        value = int(args[0])
                        High_Detail = value
                        display_message("system", f"Global High_Detail set to: {High_Detail}")
                        #update_chat_history('system', f"Global High_Detail set to: {High_Detail}")
                    except ValueError:
                        display_message("error", "Invalid value for High_Detail. Must be an integer.")

                elif cmd == "set_max_images_in_history":
                    try:
                        value = int(args[0])
                        MAX_IMAGES_IN_HISTORY = value
                        display_message("system", f"Global MAX_IMAGES_IN_HISTORY set to: {MAX_IMAGES_IN_HISTORY}")
                        #update_chat_history('system', f"Global MAX_IMAGES_IN_HISTORY set to: {MAX_IMAGES_IN_HISTORY}")
                    except ValueError:
                        display_message("error", "Invalid value for MAX_IMAGES_IN_HISTORY. Must be an integer.")

                elif cmd == "set_image_detail":
                    try:
                        detail = args[0].strip().lower()
                        if detail in ["low", "high"]:
                            image_detail = detail
                            display_message("system", f"Global image_detail set to: {image_detail}")
                            #update_chat_history('system', f"Global image_detail set to: {image_detail}")
                        else:
                            display_message("error", "Invalid value for image_detail. Must be 'low' or 'high'.")
                    except IndexError:
                        display_message("error", "No value provided for image_detail.")

                elif cmd == "set_latest_image_detail":
                    try:
                        detail = args[0].strip().lower()
                        if detail in ["low", "high"]:
                            latest_image_detail = detail
                            display_message("system", f"Global latest_image_detail set to: {latest_image_detail}")
                            #update_chat_history('system', f"Global latest_image_detail set to: {latest_image_detail}")
                        else:
                            display_message("error", "Invalid value for latest_image_detail. Must be 'low' or 'high'.")
                    except IndexError:
                        display_message("error", "No value provided for latest_image_detail.")



                else:
                    display_message("error", f"Invalid command format: {command}")
                    continue

                #update_chat_history('system', command_input)

            except Exception as e:
                display_message("system", f"Output: {command}. Error: {str(e)}.")
                continue

        if command_input.lower() in ("edit msgs"):
            edit_commands = command_input[9:].split(';')
            for edit_command in edit_commands:
                try:
                    components = edit_command.strip().split(' ')
                    timestamp = components[-1]
                    new_content = ' '.join(components[:-1]).strip()

                    found = False
                    for entry in chat_history:
                        if entry['timestamp'] == timestamp:
                            current_exemption = entry.get('Exemption', 'None')
                            entry['Exemption'] = current_exemption
                            current_time = datetime.datetime.now()
                            new_timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
                            new_token_count = len(new_content.split())

                            entry['content'] = {'type': 'text', 'text': f"Edited: {new_content} {new_token_count} {new_timestamp}"}
                            entry['token_count'] = new_token_count
                            entry['timestamp'] = new_timestamp
                            entry['Exemption'] = current_exemption
                            found = True
                            break

                    if found:
                        print(f"Message with timestamp {timestamp} edited successfully!")
                        #update_chat_history('system', command_input)
                    else:
                        print(f"No message found with timestamp {timestamp}.")
                except ValueError:
                    print(f"Invalid edit command format: {edit_command}. Correct format is 'new content timestamp'")

        if command_input.startswith('toggle_always'):
            params = command_input.split()[1:]
            if len(params) == 2:
                Always_ = params[0].lower() == 'on'
                hide_input = params[1].lower() == 'true'
            else:
                print("Invalid number of parameters. Usage: toggle_always on|off true|false")

            #update_chat_history('system', command_input)

        if command_input.startswith("TOGGLE_POWER_WORD"):
            REQUIRE_POWER_WORD = not REQUIRE_POWER_WORD
            display_message("system", f"POWER_WORD requirement toggled to {'ON' if REQUIRE_POWER_WORD else 'OFF'}.")
            return

        if command_input.startswith('INIT'):
            chat_history = [entry for entry in chat_history if not entry['content'].startswith('Pinned Init summary')]
            init_summary = command_input[5:]
            update_chat_history('user', f'Pinned Init summary: {init_summary}')
            print(f"Pinned Init summary: {init_summary}")
            update_chat_history('system', command_input)

        if command_input.startswith("PIN"):
            exemption_context = command_input[4:].strip()
            exemption_type = "Pinned"
            display_message("assistant", f'Exemption: {exemption_context}', include_command=True)
            update_chat_history('assistant', {'type': 'text', 'text': f'Exemption: {exemption_context}'}, exemption=exemption_type)
            update_chat_history('system', command_input)

        if command_input.startswith("REMOVE_MSGS"):
            try:
                _, time_range = command_input.split("REMOVE_MSGS", 1)
                start_time_str, _, end_time_str = time_range.strip().partition(" to ")
                start_time = datetime.datetime.strptime(start_time_str.strip(), '%Y-%m-%d %H:%M:%S')
                end_time = datetime.datetime.strptime(end_time_str.strip(), '%Y-%m-%d %H:%M:%S')

                messages_to_remove = [msg for msg in chat_history if start_time <= datetime.datetime.strptime(msg['timestamp'], '%Y-%m-%d %H:%M:%S') <= end_time]

                if not messages_to_remove:
                    print("No messages found within the provided time range.")
                    return

                chat_history = [msg for msg in chat_history if msg not in messages_to_remove]
                print(f"Removed {len(messages_to_remove)} messages from the chat history.")
                update_chat_history('system', command_input)
            except Exception as e:
                print(f"Error processing the REMOVE_MSGS command: {e}")

        if command_input.startswith('RETRIEVE_HANDOFF'):
            handoff_filename = command_input.split('_')[2].strip()
            if not handoff_filename:
                print("No handoff file specified.")
                return
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

        if command_input.startswith("HANDOFF"):
            handoff_summary = command_input[7:].strip()
            hbuffer.append(handoff_summary)
            timestamp = int(time.time())
            with open(f"{logging_folder}/handoff_{timestamp}.txt", "w") as log_file:
                log_file.write(handoff_summary)
            update_chat_history('assistant', f'Pinned Handoff context: {handoff_summary}')
            all_entries = [f"{entry['role']}: {entry['content']}" for entry in chat_history]
            write_content_to_file("\n".join(all_entries), "all_entries.txt")
            print("Handoff summary saved.")
            restart_chatgpt_instance()

        if command_input.lower() in ("TOGGLE AUTO PROMPT"):
            global enable_auto_prompt
            enable_auto_prompt = not enable_auto_prompt
            if enable_auto_prompt:
                display_message("system", "Auto-prompting has been enabled.")
            else:
                display_message("system", "Auto-prompting has been disabled.")

        if command_input.lower() in ("shutdown instance"):
            shutdown_chatgpt_instance_and_exit()

        if command_input.lower() in ('terminate instance'):
            terminate_instance()

        if command_input.startswith("RECALL"):
            recall_previous_summary(character_name)
            update_chat_history('system', command_input)

        if command_input.lower() in ("clear_nopin%"):
            try:
                percentage_to_clear = float(command_input[12:].strip()) / 100
                if 0 <= percentage_to_clear <= 1:
                    clear_chat_history_except_pinned(percentage_to_clear)
                    update_chat_history('system', command_input)
                else:
                    print("Invalid percentage. Please enter a number between 0 and 100.")
            except ValueError:
                print("Invalid command format. Expected format is 'CLEAR_NOPIN%{number}'.")

        if command_input.startswith("CLEAR%"):
            try:
                percentage_to_clear = float(command_input[6:].strip()) / 100
                if 0 <= percentage_to_clear <= 1:
                    clear_chat_history(percentage_to_clear)
                    update_chat_history('system', command_input)
                else:
                    print("Invalid percentage. Please enter a number between 0 and 100.")
            except ValueError:
                print("Invalid command format. Expected format is 'CLEAR%{number}'.")

        if command_input.lower() in ("CLEAR ALL MESSAGES"):
            chat_history = [
                entry for entry in chat_history
                if 'text' in entry['content'] and
                   (entry['content']['text'].startswith('Pinned Init summary') or
                    entry['content']['text'].startswith('Pinned Handoff context'))
            ]
            print("Chat history cleared, only pinned summaries remain.")
            update_chat_history('system', command_input)

        if command_input.lower() in ("CLEAR INIT"):
            chat_history = [
                entry for entry in chat_history
                if 'text' in entry['content'] and
                   not entry['content']['text'].startswith('Pinned Init summary')
            ]
            print("Chat history cleared, Init summaries removed.")
            update_chat_history('system', command_input)

        if command_input.lower() in ("CLEAR PIN"):
            chat_history = [entry for entry in chat_history if entry.get('Exemption') != 'Pinned']
            print("Pinned messages cleared.")
            update_chat_history('system', command_input)

        if command_input.lower() in ("CLEAR HANDOFF"):
            chat_history = [entry for entry in chat_history if not entry['content'].startswith('Pinned Handoff context')]
            print("Chat history cleared, Handoff summaries removed.")

        if command_input.startswith("DELETE_MSG:"):
            timestamp = command_input[11:].strip()
            chat_history = [entry for entry in chat_history if entry['timestamp'] != timestamp]
            print(f"Message with timestamp {timestamp} deleted successfully!")
            update_chat_history('system', command_input)

        if command_input.startswith('SAVEPINNEDINIT'):
            pinned_init_summaries = [entry['content']['text'][19:] for entry in chat_history if isinstance(entry['content'], dict) and entry['content']['text'].startswith('Pinned Init summary')]
            write_content_to_file("\n".join(pinned_init_summaries), "pinned_init_summaries.txt")
            update_chat_history('system', command_input)

        if command_input.startswith('SAVEPINNEDHANDOFF'):
            pinned_handoff_contexts = [entry['content'][22:] for entry in chat_history if entry['content'].startswith('Pinned Handoff context')]
            write_content_to_file("\n".join(pinned_handoff_contexts), "pinned_handoff_contexts.txt")
            update_chat_history('system', command_input)

        if command_input.startswith('SAVEEXEMPTIONS'):
            exemptions = [entry['content'][10:] for entry in chat_history if entry['content'].startswith('Exemption')]
            write_content_to_file("\n".join(exemptions), "exemptions.txt")
            update_chat_history('system', command_input)

        if command_input.startswith('SAVE PINS'):
            pinned_entries = [entry['content'] for entry in chat_history if entry['content'].startswith('Pinned Init summary') or entry['content'].startswith('Exemption') or entry['content'].startswith('Pinned Handoff context')]
            write_content_to_file("\n".join(pinned_entries), "pinned_entries.txt")
            update_chat_history('system', command_input)

        if command_input.startswith('SAVE ALL PINS'):
            all_entries = [f"{entry['role']}: {entry['content']}" for entry in chat_history]
            write_content_to_file("\n".join(all_entries), "all_entries.txt")
            update_chat_history('system', command_input)

        if command_input == 'HELP_VISIBILITY':
            display_message("system", """
            HIDE - Toggle the visibility of user text.
            HIDE_AI - Toggle the visibility of AI responses.
            HIDE_AI_COMMANDS - Toggle the visibility of commands given by the AI.
            HIDE_USER_COMMANDS - Toggle the visibility of commands issued by the user.
            TOGGLE_IMSGS - Toggle visibility of important messages by all, and prevents them from appending chat history
                important messages being PINS, INITS, HANDOFF.
            TOGGLE_UMSGS - Toggle visibility of unimportant by all, and prevents them from appending chat history
            """)
            update_chat_history('system', command_input)

        if command_input.startswith(('HELP', '/HELP', '/?', '~help', '/help', '-h', '--help', '/man', '-man', '--man', '-manual', '--manual', '/manual', '-manual', '/info', '-info', '--info', '-?', '--?')):
            display_message("system", """
            Here is the list of available commands and their descriptions:

            -/? or HELP 

            - "/* escape command
              "*/" ends escape command
                            
            - HIDE_USER_TEXT: Toggle the visibility of user text. 
              Usage: POWER_WORD HIDE_USER_TEXT

            - HIDE_AI_TEXT: Toggle the visibility of AI responses.
              Usage: POWER_WORD HIDE_AI_TEXT

            - HIDE_AI_COMMANDS: Toggle the visibility of commands given by the AI.
              Usage: POWER_WORD HIDE_AI_COMMANDS

            - HIDE_USER_COMMANDS: Toggle the visibility of commands issued by the user.
              Usage: POWER_WORD HIDE_USER_COMMANDS

            - TOGGLE_IMSGS: Toggle visibility of important messages by all, and prevents them from appending to chat history.
              Important messages include PINS, INITS, HANDOFF.
              Usage: POWER_WORD TOGGLE_IMSGS

            - TOGGLE_UMSGS: Toggle visibility of unimportant messages by all, and prevents them from appending to chat history.
              Usage: POWER_WORD TOGGLE_UMSGS

            - VKB_CMD: Execute a series of keyboard commands. Commands should be separated by a semicolon.
              Example: POWER_WORD VKB_CMD: hold_A,0.5; B+SHIFT; C

            - CURSORCMD: Execute a series of cursor commands. Commands should be separated by a semicolon.
              Example: POWER_WORD CURSORCMD: move,100,200; click,300,400

            - INIT: Clears any existing 'Pinned Init summary' from the chat history and adds a new one.
              Usage: INIT {your summary here}

            - RETRIEVE_HANDOFF: Retrieves a handoff summary from a specified file and adds it to the chat history, if not already present.
              Usage: RETRIEVE_HANDOFF_{filename}
                        
            - PIN: Add a new exemption context.
              Example: POWER_WORD PIN This is an exemption context.

            - HANDOFF: Pin a handoff context.
              Example: POWER_WORD HANDOFF This is a handoff context.

            - CLEAR_NOPIN%: Clears a specified percentage of messages that are not pinned from the chat history.
              Usage: CLEAR_NOPIN%{number}

            - CLEAR_NON_PINNED: Clears all messages that are not pinned (not starting with 'Pinned Init summary', 'Pinned Handoff context', or 'Exemption') from the chat history.
              Usage: CLEAR_NON_PINNED

            - CLEAR%: Clears a specified percentage of messages from the chat history.
              Usage: CLEAR%{number}

            - CLEARALL: Clears all messages except those that start with 'Pinned Init summary' or 'Pinned Handoff context' from the chat history.
              Usage: CLEARALL

            - EDIT_MSGS: Allows you to edit one or multiple messages based on their timestamps.
              Usage: EDIT_MSGS timestamp~|~|~new content;timestamp~|~|~new content

            - REMOVE_MSGS: Removes multiple messages from the chat history based on a start and end timestamp range.
              Usage: REMOVE_MSGS {start timestamp} to {end timestamp}

            - DELETE_MSG: Deletes a specific message from the chat history based on its timestamp.
              Usage: DELETE_MSG:{timestamp}

            - DISPLAY_HISTORY: Displays the entire chat history with timestamps and roles.
              Usage: DISPLAY_HISTORY
            """)
            update_chat_history('system', command_input)

        if command_input.lower() == 'hide user text':
            show_user_text = not show_user_text
            if show_user_text:
                display_message("system", "User text will now be displayed.")
            else:
                display_message("system", "User text will now be hidden.")

        if command_input.lower() == 'hide ai text':
            show_ai_text = not show_ai_text
            if show_ai_text:
                display_message("system", "AI text will now be displayed.")
            else:
                display_message("system", "AI text will now be hidden.")

        if command_input.lower() == 'hide ai commands':
            hide_ai_commands = not hide_ai_commands
            if hide_ai_commands:
                display_message("system", "AI commands will now be hidden.")
            else:
                display_message("system", "AI commands will now be displayed.")

        if command_input.lower() == 'hide user commands':
            hide_user_commands = not hide_user_commands
            if hide_user_commands:
                display_message("system", "User commands will now be hidden.")
            else:
                display_message("system", "User commands will now be displayed.")

        if command_input.lower() in ('toggle unimportant messages'):
            enable_unimportant_messages = not enable_unimportant_messages
            return

        if command_input.lower() in ('toggle important messages'):
            enable_important_messages = not enable_important_messages
            return

        if command_input.lower() in ('toggle add umsgs to history'):
            ADD_UMSGS_TO_HISTORY = not ADD_UMSGS_TO_HISTORY
            return

        if command_input.lower() in ('toggle add imsgs to history'):
            ADD_IMSGS_TO_HISTORY = not ADD_IMSGS_TO_HISTORY
            return

        if command_input.lower() in ('toggle unmsgs decay check'):
            CHECK_UMSGS_DECAY = not CHECK_UMSGS_DECAY
            print(f"Unimportant messages decay check toggled to {'ON' if CHECK_UMSGS_DECAY else 'OFF'}.")

        if command_input.lower() in ('toggle imsgs decay check'):
            CHECK_IMSGS_DECAY = not CHECK_IMSGS_DECAY
            print(f"Important messages decay check toggled to {'ON' if CHECK_IMSGS_DECAY else 'OFF'}.")

        if not enable_unimportant_messages and is_unimportant_message(command_input, exemption=exemption):
            return

        if not enable_important_messages and is_important_message(command_input, exemption=exemption):
            return

        if command_input.startswith('TOGGLE_SKIP_COMMANDS'):
            SKIP_ADDING_COMMANDS_TO_CHAT_HISTORY = not SKIP_ADDING_COMMANDS_TO_CHAT_HISTORY
            display_message("system", f"Skipping adding commands to chat history: {SKIP_ADDING_COMMANDS_TO_CHAT_HISTORY}")
            return

        if command_input.startswith("SAVECH"):
            save_chat_history_to_file(chat_history, 'chat_history')
            display_message("system", "Chat history saved to chat_history.txt.")
            return

        if command_input.startswith("DHISTORY"):
            if chat_history:
                for entry in chat_history:
                    timestamp = entry.get('timestamp', 'No timestamp')
                    role = entry.get('role', 'No role')
                    token_count = entry.get('token_count', 'N/A')
                    last_key = entry.get('last_key', 'N/A')
                    mouse_position = entry.get('mouse_position', 'N/A')
                    last_command = entry.get('last_command', 'N/A')

                    content = entry.get('content', {})
                    message = 'No content'

                    if isinstance(content, dict):
                        content_type = content.get('type')
                        if content_type == 'text':
                            message = content.get('text', 'No text content')[:15] + '...'
                        elif content_type == 'image':
                            message = f"Image Data: {content.get('data', '')[:30]}..."
                        else:
                            message = f'[Other content type: {content_type}]'
                    else:
                        message = str(content)[:80] + '...' if content else 'Empty content'
                    print(f"{timestamp} - {role}: {message}")
                    print(f"Token Count: {token_count}, Last Key: {last_key}, Mouse Position: {mouse_position}, Last Command: {last_command}")
            else:
                print("No messages in the chat history.")
    except Exception as e:
        display_message("system", f"Failed to execute command: {command}. Error: {str(e)}.")
        role = "assistant" if not is_user else "user"
        #if not (is_user and hide_user_commands) or (not is_user and hide_ai_commands):
            #display_message(role, command_input)

        token_count = count_tokens_in_history(chat_history)
        tokens_in_message = len(list(tokenizer.encode(command_input)))
        token_counter = tokens_in_message
        print(f"Current token count in chat history else1: {token_count}")
        if token_counter > 0.85 * token_limit and token_counter < token_limit:
            print("Warning: Approaching token limit! Chat history will be saved.")
        elif token_counter >= token_limit:
            print("Token limit reached! Chat history saved and non-pinned messages will be cleared.")
            clear_percentage_except_pinned_and_exempt("CLEAR%70")
            token_counter = 0
    else:
        role = "assistant" if not is_user else "user"
        #if not (is_user and hide_user_commands) or (not is_user and hide_ai_commands):
        #    display_message(role, command_input)

        token_count = count_tokens_in_history(chat_history)
        tokens_in_message = len(list(tokenizer.encode(command_input)))
        token_counter = tokens_in_message
        print(f"Current token count in chat history e2: {token_count}")
        if token_counter > 0.85 * token_limit and token_counter < token_limit:
            print("Warning: Approaching token limit! Chat history will be saved.")
        elif token_counter >= token_limit:
            save_chat_history_to_file(chat_history, 'chat_history')
            clear_percentage_except_pinned_and_exempt("CLEAR%70")
            token_counter = 0

  

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

#def save_chat_history_to_file(chat_history, file_name):
#        # Get the current time and format it for the file name
#    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#    # Construct the file name with the current time
#    file_nametime = f"{file_name}_{current_time}.txt"
#    with open(file_nametime, 'w') as file:
#        for entry in chat_history:
#            file.write(str(entry) + '\n')
#    print(f"Chat history saved to {file_name}")        
#def save_chat_history_to_file(chat_history, file_name):
#    """
#    Saves the chat history to a file, including details for each entry.
#    """
#    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#    file_nametime = f"{file_name}_{current_time}.txt"
#
#    with open(file_nametime, 'w') as file:
#        for entry in chat_history:
#            timestamp = entry.get('timestamp', 'No timestamp')
#            role = entry.get('role', 'No role')
#            token_count = entry.get('token_count', 'N/A')
#            last_key = entry.get('last_key', 'N/A')
#            mouse_position = entry.get('mouse_position', 'N/A')
#            last_command = entry.get('last_command', 'N/A')
#
#            content = entry.get('content', {})
#            message = 'No content'
#
#            if isinstance(content, dict):
#                content_type = content.get('type')
#                if content_type == 'text':
#                    message = content.get('text', 'No text content')
#                if content_type == 'image':
#                    image_data = content.get('data', '')
#                    message = f"Image Data: {image_data}" if image_data else "Image Data: Missing or empty 'data' key"
#                else:
#                    message = f'[Other content type: {content_type}]'
#
#            file.write(f"{timestamp} - {role}: {message}\n\n")
#            file.write(f"Token Count: {token_count}, Last Key: {last_key}, Mouse Position: {mouse_position}, Last Command: {last_command}\n")
#
#    print(f"Chat history saved to {file_nametime}")

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
            response_text = send_prompt_to_chatgpt(init_prompt, role="system", exemption='Init')
            # Optionally process the response_text further as needed
    except openai.error.InvalidRequestError as e:
        print(f"Error during initiation: {e}")



    init_handoff_in_progress = False
    handoff_to_chatgpt()

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




##def save_chat_history_to_file_old():
 #   """
 #   Save the current chat history to a file.
 #   The filename will be formatted as 'chat_history_PrimaryAIName_date_time.txt',
 #   where 'PrimaryAIName' is replaced with the name of the primary AI,
 #   'date' is replaced with the current date, and 'time' is replaced with the current time.
 #   """
 #   # Get the current date and time
 #   now = datetime.datetime.now()
 #   
 #   # Format the date and time as strings
 #   date_str = now.strftime('%Y-%m-%d')
 #   time_str = now.strftime('%H-%M-%S')
 #   
 #   # Create the filename
 #   filename = f"chat_history_Aurora_{date_str}_{time_str}.txt"
 #   
 #   # Write the chat history to the file
 #   with open(filename, 'w') as file:
 #       for entry in chat_history:
 #           role = entry['role']
 #           content = entry['content']
 #           file.write(f"{role}: {content}\n")
 #               
 #   print(f"Chat history saved to {filename}")

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



## Function to add a grid with labeled coordinates
#def add_grid_to_screenshot(image, grid_interval):
#    draw = ImageDraw.Draw(image)
#
#    # Customize the font size
#    font_size = 23
#
#    # Load a TrueType font with the specified size
#    try:
#        font = ImageFont.truetype(FONT_PATH, font_size)
#    except IOError:
#        print("Font file not found. Falling back to default font.")
#        font = ImageFont.load_default()  # Fallback to default font if TrueType font not found
#
#    # Draw the grid and label intersections
#    for x in range(0, screen_width, grid_interval):
#        for y in range(0, screen_height, grid_interval):
#            draw.line([(x, 0), (x, screen_height)], fill="blue")
#            draw.line([(0, y), (screen_width, y)], fill="blue")
#            coordinate_label = f"{x},{y}"
#            #draw.text((x + 2, y), coordinate_label, fill="red", font=font)
#            #draw_text_with_background(draw, (x + 5, y - 15), coordinate_label, font)

# Function to add a dot grid with labeled coordinates
#def add_grid_to_screenshot2(image, grid_interval):
#    print("Starting add_grid_to_screenshot2 function")
#    draw = ImageDraw.Draw(image)
#
#    # Customize the font size
#    font_size = 23
#
#    # Load a TrueType font with the specified size
#    try:
#        font = ImageFont.truetype(FONT_PATH, font_size)
#    except IOError:
#        print("Font file not found. Falling back to default font.")
#        font = ImageFont.load_default()  # Fallback to default font if TrueType font not found
#
#    screen_width, screen_height = image.size
#
#    # Define key points to label
#    key_points = [
#        (150, 150), (450, 150), (150, 450), (450, 450),
#        (900, 150), (1200, 150), (900, 450), (1200, 450),
#        (600, 750), (1050, 750), (1350, 300), (1350, 600),
#         (1350, 750), (750, 300)
#    ]
#
#    # Draw dots and coordinates with background boxes on the top row, left column, and key points
#    # Draw larger dots and coordinates with background boxes on the top row, left column, and key points
#    for x in range(0, screen_width, grid_interval):
#        for y in range(0, screen_height, grid_interval):
#            draw.ellipse((x-4, y-4, x+4, y+4), fill="red", outline="red")  # Draw the larger dot
#            if (x == 0 or y == 0) or (x, y) in key_points:  # Top row, left column, or key points
#                coordinate_label = f"{x},{y}"
#                draw_text_with_background(draw, (x + 5, y - 15), coordinate_label, font, background_opacity=128, shift_x=5, shift_y=20)
#    print("Finished add_grid_to_screenshot2 function")


# Function to add a grid with color-coded tiles and labeled coordinates


# Function to add the colored tile grid with only directional labels
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


#def add_grid_to_screenshot3p0(image, grid_interval, center_tile, tile_size, colors_x, colors_y):
#    print("Starting add_grid_to_screenshot2 function")
#    draw = ImageDraw.Draw(image)
#
#    # Customize the font size
#    font_size = 23
#
#    # Load a TrueType font with the specified size
#    try:
#        font = ImageFont.truetype(FONT_PATH, font_size)
#    except IOError:
#        print("Font file not found. Falling back to default font.")
#        font = ImageFont.load_default()  # Fallback to default font if TrueType font not found
#
#    screen_width, screen_height = image.size
#
#    # Calculate half-tile offsets for proper grid alignment
#    half_tile_offset = (tile_size[0] // 2, tile_size[1] // 2)
#    
#    # Apply a half-tile shift to center the grid
#    x_offset = -half_tile_offset[0]
#    y_offset = -half_tile_offset[1]
#
#    # Loop through the grid around the center tile, applying the colors based on X and Y distance
#    for i in range(-5, 6):  # Adjust the range as necessary for the grid size
#        for j in range(-5, 6):
#            # Calculate the position of the current tile with offsets for the half-tile shift
#            tile_x = center_tile[0] + i * tile_size[0] + x_offset
#            tile_y = center_tile[1] + j * tile_size[1] + y_offset
#
#            # Determine the color based on the X and Y distance from the center
#            color_x = colors_x[min(abs(i), len(colors_x) - 1)]  # Use column color (X axis) for sides
#            color_y = colors_y[min(abs(j), len(colors_y) - 1)]  # Use row color (Y axis) for top/bottom
#
#            # Draw the tile with the appropriate colors (colored borders based on X/Y distance)
#            draw_tile_with_colors(draw, tile_x, tile_y, tile_size, color_x, color_y)
#
#            # Optionally, label the grid coordinates if you want
#            coordinate_label = f"{tile_x},{tile_y}"
#            draw_text_with_background(draw, (tile_x + 5, tile_y - 15), coordinate_label, font)
#
#    print("Finished add_grid_to_screenshot2 function")

#def draw_tile_with_colors(draw, x, y, tile_size, color_x, color_y):
#    # Draw the floor (left and right) for the column (X)
#    draw.line([(x, y), (x, y + tile_size[1])], fill=color_x, width=5)  # Left (floor)
#    draw.line([(x + tile_size[0], y), (x + tile_size[0], y + tile_size[1])], fill=color_x, width=5)  # Right (ceiling)
#    
#    # Draw the top and bottom for the row (Y)
#    draw.line([(x, y), (x + tile_size[0], y)], fill=color_y, width=5)  # Top
#    draw.line([(x, y + tile_size[1]), (x + tile_size[0], y + tile_size[1])], fill=color_y, width=5)  # Bottom
# Helper function to draw the tiles with X/Y axis colors
def draw_tile_with_colors(draw, x, y, tile_size, color_x, color_y):
    # Draw the left and right borders (X axis)
    draw.line([(x, y), (x, y + tile_size[1])], fill=color_x, width=3)  # Left
    draw.line([(x + tile_size[0], y), (x + tile_size[0], y + tile_size[1])], fill=color_x, width=3)  # Right

    # Draw the top and bottom borders (Y axis)
    draw.line([(x, y), (x + tile_size[0], y)], fill=color_y, width=3)  # Top
    draw.line([(x, y + tile_size[1]), (x + tile_size[0], y + tile_size[1])], fill=color_y, width=3)  # Bottom


def draw_cursor(draw, cursor_position, cursor_size):
    # Define colors
    outer_color = "black"
    inner_color = "white"
    large_circle_color = "red"
    medium_circle_color = "red"
    small_circle_color = "blue"

    # Outer rectangle (black outline)
    outer_rectangle = [cursor_position.x - 1, cursor_position.y - 1, cursor_position.x + cursor_size + 1, cursor_position.y + cursor_size + 1]
    draw.rectangle(outer_rectangle, outline=outer_color, fill=outer_color)

    # Inner rectangle (white cursor)
    inner_rectangle = [cursor_position.x, cursor_position.y, cursor_position.x + cursor_size, cursor_position.y + cursor_size]
    draw.rectangle(inner_rectangle, outline=inner_color, fill=inner_color)

    # Draw an 'X' inside the rectangle
    draw.line([cursor_position.x, cursor_position.y, cursor_position.x + cursor_size, cursor_position.y + cursor_size], fill=outer_color)
    draw.line([cursor_position.x, cursor_position.y + cursor_size, cursor_position.x + cursor_size, cursor_position.y], fill=outer_color)

    # Draw the red circle around the cursor
    large_radius = cursor_size + 3  # Adjust the radius as needed
    large_circle_bounds = [cursor_position.x - large_radius, cursor_position.y - large_radius, cursor_position.x + cursor_size + large_radius, cursor_position.y + cursor_size + large_radius]
    draw.ellipse(large_circle_bounds, outline=large_circle_color, width=2)

    # Calculate the center of the square
    center_x = cursor_position.x + cursor_size / 2
    center_y = cursor_position.y + cursor_size / 2

    # Draw the medium red circle that the box fits into perfectly
    medium_radius = (cursor_size / 2) * (2 ** 0.5)  # sqrt(2) times the half size of the square
    medium_circle_bounds = [center_x - medium_radius, center_y - medium_radius, center_x + medium_radius, center_y + medium_radius]
    draw.ellipse(medium_circle_bounds, outline=medium_circle_color, width=2)

    # Draw the smaller blue circle that circumscribes the square reticle
    small_radius = cursor_size / 2
    small_circle_bounds = [center_x - small_radius, center_y - small_radius, center_x + small_radius, center_y + small_radius]
    draw.ellipse(small_circle_bounds, outline=small_circle_color, width=2)

    # Add cursor coordinates with background
    screen_width, screen_height = draw.im.size
    text_color = "white"
    background_color = (0, 128, 0)  # Greenish background color
    background_opacity = 128

    # Customize the font size
    font_size = 23
    # Load a TrueType font with the specified size
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except IOError:
        print("Font file not found. Falling back to default font.")
        font = ImageFont.load_default()  # Fallback to default font if TrueType font not found

    # Calculate position for text
    shift_x, shift_y = 5, 20
    text_position = (cursor_position.x + shift_x, cursor_position.y + shift_y)
    if cursor_position.x + shift_x + 100 > screen_width:
        shift_x = -100  # Adjust shift to place text on the left
    if cursor_position.y + shift_y + 20 > screen_height:
        shift_y = -30  # Adjust shift to place text above

    cursor_text = f"({cursor_position.x}, {cursor_position.y})"
    draw_text_with_background(draw, text_position, cursor_text, font, text_color, background_color, background_opacity, shift_x, shift_y)

#def draw_text_with_background(draw, position, text, font, text_color="white", background_color="black", background_opacity=58, shift_x=5, shift_y=20):
#    # Calculate text size
#    text_size = draw.textsize(text, font=font)
#    
#    # Set padding around the text
#    padding = 4
#
#    # Define the background rectangle position with shift
#    background_position = (
#        position[0] - padding + shift_x,
#        position[1] - padding + shift_y,
#        position[0] + text_size[0] + padding + shift_x,
#        position[1] + text_size[1] + padding + shift_y
#    )
#
#    # Draw the background rectangle directly on the draw object with transparency
#    draw.rectangle(background_position, fill=(0, 0, 0, background_opacity))
#
#    # Draw the text over the background rectangle
#    draw.text((position[0] + shift_x, position[1] + shift_y), text, fill=text_color, font=font)

#def draw_text_with_background(draw, position, text, font, text_color="white", background_color=(0, 0, 0), background_opacity=128, shift_x=5, shift_y=20):
#    # Calculate text size using textbbox
#    text_bbox = draw.textbbox(position, text, font=font)
#    text_width = text_bbox[2] - text_bbox[0]
#    text_height = text_bbox[3] - text_bbox[1]
#    
#    # Set padding around the text
#    padding = 4
#
#    # Define the background rectangle position with shift
#    background_position = (
#        position[0] - padding + shift_x,
#        position[1] - padding + shift_y,
#        position[0] + text_width + padding + shift_x,
#        position[1] + text_height + padding + shift_y
#    )

    # Create a transparent image for the background
#    background_image = Image.new('RGBA', draw.im.size, (0, 0, 0, 0))
#    background_draw = ImageDraw.Draw(background_image)
#    background_color_with_opacity = (background_color[0], background_color[1], background_color[2], background_opacity)
#    background_draw.rectangle(background_position, fill=background_color_with_opacity)
#
#    # Extract the alpha channel from the background image
#    alpha = background_image.split()[3]
#
#    # Composite the transparent background onto the original image using paste with a mask
#    bounding_box = (0, 0, draw.im.size[0], draw.im.size[1])
#    draw.im.paste(background_image.im, bounding_box, alpha.im)
#
#    # Draw the text over the background rectangle
#    draw.text((position[0] + shift_x, position[1] + shift_y), text, fill=text_color, font=font)
# Helper function to draw text with background
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

# Function to add a dot grid with labeled coordinates
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
def add_grids_and_labels(screenshot, cursor_position, current_last_key):
    # Continue with your grid and tile drawing
    add_colored_tile_grid_v3(screenshot, center_tile=(719, 444), tile_size=(162, 98),
                             colors_x=['blue', 'red', 'orange', 'yellow', 'purple', 'black'],
                             colors_y=['blue', 'red', 'orange', 'yellow', 'purple', 'black'])

    add_dot_grid_with_labels(screenshot, grid_interval=150, key_points=[
        (150, 150), (300, 750), (450, 150), 
        (900, 150), (1200, 150),
        (1050, 750), (1350, 300), (1350, 600),
        (1350, 750)
    ])

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

    if combined_input_list:
        combined_input = ' '.join(combined_input_list)
        prompt = f"user input: {combined_input}"
        send_prompt_to_chatgpt(prompt, role="user", image_path=screenshot_file_path, image_timestamp=image_timestamp)
    else:
        send_prompt_to_chatgpt("System: Screenshot taken. Please provide instructions...", role="user", image_path=screenshot_file_path, image_timestamp=image_timestamp)


   # if queued_user_input:
   #     prompt = f"user input: {queued_user_input}"
   #     send_prompt_to_chatgpt(prompt, role="user", image_path=screenshot_file_path, image_timestamp=image_timestamp)
   #     queued_user_input = None
   # else:
   #     send_prompt_to_chatgpt("System: Screenshot taken. Please provide instructions...", role="user", image_path=screenshot_file_path, image_timestamp=image_timestamp)
#def take_screenshotOld():
#    global last_key  # Ensure you are referring to the global variable updated by the listener thread
#    global image_timestamp
#    global queued_user_input
#
#
#    
#    # Check if the third option for pyautogui native screenshot is enabled
#    if screenshot_options.get("native_cursor_screenshot"):
#        # Capture entire screen with cursor using pyautogui.screenshot()
#        screenshot = pyautogui.screenshot()
#
#        # Get the mouse cursor's current position
#        cursor_position = pyautogui.position()
#
#        # Ensure thread-safe access to `last_key`
#        with lock:
#            current_last_key = last_key
#
#        # Draw additional elements if needed
#        draw = ImageDraw.Draw(screenshot)
#        draw_cursor(draw, cursor_position, cursor_size)  # Optional custom cursor drawing
#
#        # Add grid, tiles, and text like before
#        add_grids_and_labels(screenshot, cursor_position, current_last_key)
#
#
#    if screenshot_options["current_window"]:
#        # Code to capture the current window snapshot
#        # Depending on the platform, you might need additional libraries and code
#        pass
#    
#    if screenshot_options["entire_screen"]:
#        # Capture entire screen
#        screenshot = ImageGrab.grab()
#    
#        # Get the mouse cursor's current position
#        cursor_position = pyautogui.position()
#
#        # Use the global last_key variable instead of reading the event here
#        with lock:  # Ensure thread-safe access to last_key
#            current_last_key = last_key
#
#        # Draw a representation of the cursor on the screenshot
#        draw = ImageDraw.Draw(screenshot)
#
#        
#        # Draw the enhanced cursor
#        draw_cursor(draw, cursor_position, cursor_size)
#
#        #draw.rectangle([cursor_position.x, cursor_position.y, cursor_position.x + cursor_size, cursor_position.y + cursor_size], outline="red")
#
#        current_time = datetime.datetime.now()
#        image_timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
#        timestamp = int(time.time())
#        # You can also draw the cursor position and last key pressed on the screenshot for visibility
#        text_position = (40, 55)  # Position of the text
#        # Add the grid with labeled coordinates
#        #add_grid_to_screenshot(screenshot, grid_interval=150)  # Set grid_interval as needed
#        # Draw the grid with labeled coordinates
#        text_info = f"Cursor Position: {cursor_position} | Last Key: {current_last_key} | Timestamp: {image_timestamp}"
#
#        # Add the colored tile grid with directional labels
#        add_colored_tile_grid_v3(screenshot, center_tile=(719, 444), tile_size=(162, 98),
#                                 colors_x=['blue', 'red', 'orange', 'yellow', 'purple', 'black'],
#                                 colors_y=['blue', 'red', 'orange', 'yellow', 'purple', 'black'])
#
#        # Add the red dot grid with labels
#     #   add_dot_grid_with_labels1111(screenshot, grid_interval=150, key_points=[
#     #       (150, 150), (450, 150), (150, 450), (450, 450),
#     #       (900, 150), (1200, 150), (900, 450), (1200, 450),
#     #       (600, 750), (1050, 750), (1350, 300), (1350, 600),
#     #       (1350, 750), (750, 300)
#     #   ])
#
#        # Add the red dot grid with labels
#        add_dot_grid_with_labels(screenshot, grid_interval=150, key_points=[
#            (150, 150), (300, 750), (450, 150), 
#            (900, 150), (1200, 150),
#            (1050, 750), (1350, 300), (1350, 600),
#            (1350, 750)
#        ])
#
#        # Customize the font size
#        font_size = 23
#
#        # Load a TrueType font with the specified size
#        try:
#            font = ImageFont.truetype(FONT_PATH, font_size)
#        except IOError:
#            print("Font file not found. Falling back to default font.")
#            font = ImageFont.load_default()  # Fallback to default font if TrueType font not found
#
#        # Draw text with background
#        draw_text_with_background(draw, text_position, text_info, font, background_opacity=128, shift_x=5, shift_y=20)
#        #draw.text(text_position, f"Cursor Pos: {cursor_position} | Last Key: {current_last_key} | Timestamp: {image_timestamp} ", fill="white")
#
#        screenshot_file_path = f"{logging_folder}/{timestamp}.png"
#        screenshot.save(screenshot_file_path)
#        print(f"Screenshot Timestamp: {image_timestamp}")
#        # Now, we send the prompt and include the screenshot file path
#        #send_prompt_to_chatgpt("Here's a screenshot of my entire screen. Lets play Pokemon Blue in the VBA emulator. :3 We will play with individual button presses. Please simply reply to screenshots with only a command. VKPAG:click for example. If you need help, or have questions then ask away, but always be concise. with one or a short series set of individual button presses in response to each image. images will only update after a response is obtain from here. Display is 1920 X 1080 Landscape. ie click on folder/app called at mouse position x, y via move,x,y; leftclick; x,y select open keys in VBA emulator are Up: Up Arrow Please pretend like there is a parser listening for key commands and mouse commands the format from handle_commands of VKPAG:(PYAUTOGUI KEYBOARD/mouse commands, arguments);  VKPAG:hold_shift,1.5;move(100,200); rightClick(100,200); click(100,200)", screenshot_file_path)
#        
#        ##send_prompt_to_chatgpt("please respond with usually pyag: commands #notes structure ie at title screen or certain menues selections try  please use pyag:  ie pyag: press(a) #at title screen/to scroll dialog text boxes, please do not use press(a) with multiple commands in a single inference but multiple move commands ie pyag hold(u,0.5); pyag: hold(r, 1) is fine; or pyag: hold (moveKey,1) #one is duration; move keys are u for moving up, d for moving down, l for moving left, r for moving right; pyag: hold(r,1) to move right, or pyag: hold(l,1) to move left, or pyag: hold(u,1) to move up, or pyag hold(d,1) to move down; press a at title screen/to scroll dialog text boxes, or pyag: press (r) to face right, or pyag: press (l) to face left, or pyag: press (u) to face up, or pyag press (d) to face down,  #Notes: quotes aren't necessary around the key command and the keyboard a key happens to be the Gameboy a button and click won't do much except inside vba window focus on window '", role="user", image_path= screenshot_file_path, image_timestamp=image_timestamp)  #image_timestamp might be buggy
#        
#        #auto_prompt_response = send_prompt_to_chatgpt("meow ")
#        #send_prompt_to_chatgpt(auto_prompt_response)
#       # send_prompt_to_chatgpt("auto prompt:", screenshot_file_path)
# # Include the queued user input when sending the screenshot
#        if queued_user_input:
#            # Update chat history with the queued input, but set an exemption flag so it won't be added again.
#            update_chat_history("user", queued_user_input, exemption="queued")
#
#            # Send the input along with the screenshot, without adding it to the chat history again
#
#            prompt = f"user input: {queued_user_input}"  # Use your desired format for combining input and screenshot
#            send_prompt_to_chatgpt(prompt, role="user", image_path=screenshot_file_path, image_timestamp=image_timestamp)
#            queued_user_input = None  # Clear the queued input after sending
#        else:
#            send_prompt_to_chatgpt("System: Screenshot taken. Please provide instructions for parser to execute actions and comment notes thoroughly for future self or to communicate to user or user input.", role="user", image_path=screenshot_file_path, image_timestamp=image_timestamp)
#    
    # Use the function and provide a path to save the screenshot
    #capture_screenshot_with_cursor_info('screenshot_info.png')
# Function to take a screenshot
def take_screenshot():
    global last_key  # Ensure you are referring to the global variable updated by the listener thread
    global image_timestamp
    global queued_user_input

    # Check if the third option for pyautogui native screenshot is enabled
    if screenshot_options.get("native_cursor_screenshot"):
        # Capture entire screen with cursor using pyautogui.screenshot()
        screenshot = pyautogui.screenshot()

        # Get the mouse cursor's current position
        cursor_position = pyautogui.position()

        # Ensure thread-safe access to `last_key`
        with lock:
            current_last_key = last_key

        # Draw additional elements if needed
        draw = ImageDraw.Draw(screenshot)
        #draw_cursor(draw, cursor_position, cursor_size)  # Optional custom cursor drawing

        # Add grid, tiles, and text like before
        add_grids_and_labels(screenshot, cursor_position, current_last_key)

    # First option: Capture current window (if implemented)
    elif screenshot_options["current_window"]:
        # Code to capture the current window snapshot
        # This can be platform-specific and needs to be implemented
        pass

    # Second option: Capture entire screen using ImageGrab
    elif screenshot_options["entire_screen"]:
        screenshot = ImageGrab.grab()

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
    print(f"Screenshot saved at {screenshot_file_path}")

    # Handle user input and send the screenshot
    handle_queued_input(screenshot_file_path, image_timestamp)
# Function to run the scheduled tasks
def run_scheduled_tasks():
    while running:
        schedule.run_pending()
        time.sleep(1)

# Read the init prompt from a file
init_file = f"{character_name}_init.txt"
with open(init_file, "r") as f:
    init_prompt = f.read().strip()



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
    "current_window": True,
    "entire_screen": True,
}

# Set up buffer and logging folder
screenbuffer = []
hbuffer= []
logging_folder = "screenshots"
if not os.path.exists(logging_folder):
    os.makedirs(logging_folder)







initiate_and_handoff()




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
