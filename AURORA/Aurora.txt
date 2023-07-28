import openai
import time
from PIL import ImageGrab
import os
import keyboard
import pyautogui
import threading
import sys
import datetime
import requests
from io import BytesIO
from PIL import Image
import schedule

character_name = "Aurora"

# Function to send prompt to ChatGPT
def send_prompt_to_chatgpt(prompt, image_path=None):
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    
    if image_path is not None:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        messages.append({"role": "user", "content": {"image": image_data}})
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=100000,
        n=1,
        temperature=0.5,
    )

    return response.choices[0].message['content'].strip()


# Read the init prompt from a file
init_file = f"{character_name}_init.txt"
with open(init_file, "r") as f:
    init_prompt = f.read().strip()


# Define process_chatgpt_response function before calling it
def process_chatgpt_response(response_text):
    if response_text.startswith("CMD:"):
        command = response_text[4:]
        if len(command) == 1:
            keyboard.press_and_release(command)
            print(f"Executed command: {command}")
	else:
            pyautogui.write(command)
            print(f"Typed string: {command}")
    elif response_text.startswith("HANDOFF"):
        handoff_summary = response_text[7:] 
        buffer.append(handoff_summary)
        timestamp = int(time.time())
        with open(f"{logging_folder}/handoff_{timestamp}.txt", "w") as log_file:
            log_file.write(handoff_summary)
        print("Handoff summary saved.")
        restart_chatgpt_instance() 
    elif response_text.startswith("CURSORCMD:"):
        command = response_text[10:]
        x, y = map(int, command.split(','))  # Assuming coordinates are comma-separated
        pyautogui.click(x, y)
        print(f"Moved cursor and clicked at ({x}, {y})")
    elif response_text.startswith("terminate_instance"):
        shutdown_chatgpt_instance_and_exit()
    else:
        print("ChatGPT response:", response_text)


def initiate_and_handoff():
    response_text = send_prompt_to_chatgpt(init_prompt)
    process_chatgpt_response(response_text)
    if buffer:
        handoff_prompt = buffer[-1]
        response_text = send_prompt_to_chatgpt(handoff_prompt)

# Read the handoff summary from a file
handoff_file = f"{character_name}_handoff.txt"
if os.path.exists(handoff_file):
    with open(handoff_file, "r") as f:
        handoff_text = f.read().strip()
    response_text = send_prompt_to_chatgpt(handoff_text)
    process_chatgpt_response(response_text)

# Global variable to control the main loop
running = True

# Set up OpenAI API client
openai.api_key = "API KEY HERE"

# Initialize ChatGPT API endpoint
endpoint_url = "https://api.openai.com/v1/chat/completions/gpt-4"

# Time interval between screenshots (in seconds)
time_interval = 25.5

# Set up screenshot options
screenshot_options = {
    "current_window": True,
    "entire_screen": True,
}

# Set up buffer and logging folder
buffer = []
logging_folder = "screenshots"
if not os.path.exists(logging_folder):
    os.makedirs(logging_folder)



# Send init prompt to ChatGPT
response_text = send_prompt_to_chatgpt(init_prompt)
print("ChatGPT response:", response_text)

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
        process_chatgpt_response(summary)
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
        process_chatgpt_response(response_text)

# Function to perform shutdown procedure for the current ChatGPT instance and exit the program
def shutdown_chatgpt_instance_and_exit():
    global running
    date = datetime.date.today().strftime("%Y-%m-%d")
    if not daily_summary_completed(date):
        save_daily_summary()
    handoff_summary = f"Handoff {date}: " + buffer[-1]  # Save the last summary in the buffer
    with open(f"{logging_folder}/handoff_{date}.txt", "w") as log_file:
        log_file.write(handoff_summary)
    print("Handoff summary saved.")
    # Exit the program
    print("Exiting the program.")
    sys.exit()

def terminate_instance():
    daily_summary()
    handoff_summary = send_prompt_to_chatgpt("Create a handoff summary for the next instance.")
    process_chatgpt_response(handoff_summary)
    with open(f"{logging_folder}/Handoff_{datetime.datetime.now().strftime('%Y-%m-%d')}.txt", "w") as log_file:
        log_file.write(handoff_summary)
    print("Handoff summary saved.")
    # You can use sys.exit() to exit the program
    sys.exit()


# Function to restart ChatGPT instance
def restart_chatgpt_instance():
    # Send init_prompt and handoff_prompt to the new instance
    response_text = send_prompt_to_chatgpt(init_prompt)
    process_chatgpt_response(response_text)
    if buffer:
        handoff_prompt = buffer[-1]
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
        process_chatgpt_response(response_text)


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


initiate_and_handoff()
# Function to handle user input
def user_input_handler():
    while running:
        user_input = input("Type your message, 'recall' to read previous summaries, or 'terminate_instance' to end: ")
        if user_input.lower() == "terminate_instance":
            terminate_instance()
        elif user_input.lower() == "recall":
            recall_previous_summary(character_name)
        else:
            response_text = send_prompt_to_chatgpt(user_input)
            process_chatgpt_response(response_text)
            if buffer:
                handoff_prompt = buffer[-1]
                response_text = send_prompt_to_chatgpt(handoff_prompt)

# Create a thread for user input handling and start it
#user_input_thread = threading.Thread(target=lambda: user_input_handler(input("Type your message, 'recall' to read previous summaries, or 'terminate_instance' to end: ")))
user_input_thread = threading.Thread(target=user_input_handler)
user_input_thread.daemon = True
user_input_thread.start()

# Function to take a screenshot
def take_screenshot():
    if screenshot_options["current_window"]:
        # Code to capture the current window snapshot
        # Depending on the platform, you might need additional libraries and code
        pass

    if screenshot_options["entire_screen"]:
        # Capture entire screen
        screenshot = ImageGrab.grab()
        timestamp = int(time.time())
        screenshot_file_path = f"{logging_folder}/{timestamp}.png"
        screenshot.save(screenshot_file_path)
        send_prompt_to_chatgpt("Here is a screenshot:", screenshot_file_path)


# Function to run the scheduled tasks
def run_scheduled_tasks():
    while running:
        schedule.run_pending()
        time.sleep(1)

# Schedule the screenshot taking function
schedule.every(time_interval).seconds.do(take_screenshot)

# Schedule daily summary
schedule.every().day.at("19:00").do(save_daily_summary)

# Create a thread for running scheduled tasks and start it
scheduled_tasks_thread = threading.Thread(target=run_scheduled_tasks)
scheduled_tasks_thread.daemon = True
scheduled_tasks_thread.start()

# Main loop to keep the program running
while running:
    pass
