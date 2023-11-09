# Imports
import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch
from ovos_tts_plugin_mimic3_server import Mimic3ServerTTSPlugin
import openai
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
import pygame
import threading
import json
import requests
from urllib.parse import quote_plus
import pickle
import os.path
import datetime
import json
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Constants
MODEL_CHOICES = ["tiny", "base", "small", "medium", "large"]
DEFAULT_MIC_LINUX = 'pulse'
DEFAULT_ENERGY_THRESHOLD = 1000
DEFAULT_RECORD_TIMEOUT = 2
DEFAULT_PHRASE_TIMEOUT = 3
SAMPLE_RATE = 16000
VOICE = "en_US/hifi-tts_low#92"
openai.api_key = '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'
weather_apikey= 'MNaEN7r1qnw8vLF3551XjNtZJ0mkmUKa'
SCOPES = ['https://www.googleapis.com/auth/calendar']


# Utilities
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_current_weather(location: str) -> dict:
    """
    Queries the Tomorrow.io weather API for real-time weather data for a given location.
    
    Args:
    location (str): The location for which to retrieve weather data.
    
    Returns:
    dict: A dictionary containing the weather data.
    """
    api_key = "MNaEN7r1qnw8vLF3551XjNtZJ0mkmUKa"
    location_encoded = quote_plus(location)
    url = f"https://api.tomorrow.io/v4/weather/realtime?location={location_encoded}&units=metric&apikey={api_key}"

    headers = {"accept": "application/json"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check for HTTP errors and raise exception if any
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return {}
    except Exception as err:
        print(f"An error occurred: {err}")
        return {}

    weather_data = response.json()
    return str(weather_data)

def get_google_calendar_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('calendar', 'v3', credentials=creds)
    return service

def create_calendar_event(date_str, start_time_str, end_time_str, summary, description=None, location=None):
    """Create an event on the user's Google Calendar.
    
    Args:
        date_str (str): The date of the event in the format 'mm/dd/yyyy'.
        start_time_str (str): The start time of the event in 'HH:MM' format (24-hour).
        end_time_str (str): The end time of the event in 'HH:MM' format (24-hour).
        summary (str): The summary or title of the event.
        description (str, optional): The description of the event. Defaults to None.
        location (str, optional): The location of the event. Defaults to None.
        
    Returns:
        str: A JSON string with the created event's details or an error message.
    """
    try:
        # Parse the date and time strings into a datetime object
        start_datetime = datetime.datetime.strptime(f"{date_str} {start_time_str}", '%m/%d/%Y %H:%M')
        end_datetime = datetime.datetime.strptime(f"{date_str} {end_time_str}", '%m/%d/%Y %H:%M')
    except ValueError as e:
        return json.dumps({'error': 'Date or time format error: ' + str(e)})

    event_body = {
        'summary': summary,
        'location': location,
        'description': description,
        'start': {
            'dateTime': start_datetime.isoformat(),
            'timeZone': 'UTC',
        },
        'end': {
            'dateTime': end_datetime.isoformat(),
            'timeZone': 'UTC',
        }
    }

    try:
        service = get_google_calendar_service()
        event = service.events().insert(calendarId='primary', body=event_body).execute()
    except (HttpError, OSError) as api_error:
        return json.dumps({'error': 'Failed to create event: ' + str(api_error)})

    # Return a JSON string with the created event's details
    return json.dumps(event)

def get_next_n_calendar_events(number_of_calendar_events_to_fetch):
    """Gets the next n events on the user's calendar, including only the summary,
    start, end, organizer, and optionally description information, and returns
    a JSON string of this data."""
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    try:
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
    except (OSError, pickle.UnpicklingError, RefreshError) as auth_error:
        return json.dumps({'error': 'Authentication error: ' + str(auth_error)})

    try:
        service = build('calendar', 'v3', credentials=creds)
        # Call the Calendar API
        now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        events_result = service.events().list(calendarId='primary', timeMin=now,
                                              maxResults=number_of_calendar_events_to_fetch, singleEvents=True,
                                              orderBy='startTime').execute()
    except (HttpError, OSError) as api_error:
        return json.dumps({'error': 'API request error: ' + str(api_error)})

    events = events_result.get('items', [])

    # Filter the events to only include the summary, start, end, organizer, and description
    filtered_events = []
    for event in events:
        filtered_event = {
            'summary': event.get('summary', 'No summary provided'),
            'start': event['start'].get('dateTime', event['start'].get('date')),
            'end': event['end'].get('dateTime', event['end'].get('date')),
            'organizer': event.get('organizer', {}).get('email', 'No organizer email')
        }
        # Only add description if it exists
        if 'description' in event:
            filtered_event['description'] = event['description']
        
        filtered_events.append(filtered_event)

    # Convert the filtered events to a JSON string
    events_json_str = json.dumps(filtered_events, indent=4)
    return events_json_str

def get_calendar_events_for_day_x(date_str):
    """Fetches calendar events for the specified date."""
    try:
        day = datetime.datetime.strptime(date_str, '%m/%d/%y')
    except ValueError as e:
        return json.dumps({'error': 'Date format error: ' + str(e)})

    timeMin = day.isoformat() + 'Z'
    timeMax = (day + datetime.timedelta(days=1)).isoformat() + 'Z'

    try:
        service = get_google_calendar_service()
        events_result = service.events().list(calendarId='primary', timeMin=timeMin, timeMax=timeMax,
                                              singleEvents=True, orderBy='startTime').execute()
    except (HttpError, OSError) as api_error:
        return json.dumps({'error': 'API request error: ' + str(api_error)})

    events = events_result.get('items', [])

    # Filter and structure the events data as required
    filtered_events = []
    for event in events:
        filtered_event = {
            'summary': event.get('summary', 'No summary provided'),
            'start': event['start'].get('dateTime', event['start'].get('date')),
            'end': event['end'].get('dateTime', event['end'].get('date')),
            'organizer': event.get('organizer', {}).get('email', 'No organizer email')
        }
        if 'description' in event:
            filtered_event['description'] = event['description']
        filtered_events.append(filtered_event)

    # Convert the filtered events to a JSON string
    events_json_str = json.dumps(filtered_events, indent=4)
    return events_json_str

def text_to_audio(text, filename):
    tt = Mimic3ServerTTSPlugin()
    tt.get_tts(text, filename, voice=VOICE)

def play_audio(filename):
    def _play():
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() == True:
            pass

    thread = threading.Thread(target=_play)
    thread.start()



# Smart Mirror Class
class SmartMirror:
    def __init__(self, args):
        self.args = args
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = args.energy_threshold
        self.recorder.dynamic_energy_threshold = False
        self.audio_model = whisper.load_model(self.get_model_name())
        self.last_sample = bytes()
        self.data_queue = Queue()
        self.transcription = ['']
        self.message_list = [{"role": "system", "content": "You are a helpful assistant in a smart mirror, named Claudia."}]
        # Add the current UTC date to the content of the system message
        self.message_list[0]["content"] += " Today's date is " + datetime.datetime.utcnow().strftime('%Y-%m-%d')

    def get_model_name(self):
        model = self.args.model
        if self.args.model != "small" and not self.args.non_english:
            model = model + ".en"
        return model

    def get_microphone(self):
        if 'linux' in platform:
            mic_name = self.args.default_microphone
            if not mic_name or mic_name == 'list':
                print("Available microphone devices are: ")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    print(f"Microphone with name \"{name}\" found")
                exit(0)
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        return sr.Microphone(sample_rate=SAMPLE_RATE, device_index=index)
        return sr.Microphone(sample_rate=SAMPLE_RATE)

    def record_callback(self, _, audio: sr.AudioData):
        data = audio.get_raw_data()
        self.data_queue.put(data)

    # GPT-4 Communication
    def get_gpt4_response(self):
        functions = [
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city to query, e.g 'West Lafayette'",
                        },
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "get_next_n_calendar_events",
                "description": "Get next n calendar events",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number_of_calendar_events_to_fetch": {
                            "type": "integer",
                            "description": "Number of calendar events to fetch, when unspecified fetch 10 calendar events",
                        },
                    },
                    "required": ["number_of_calendar_events_to_fetch"],
                }
            },
            {
                "name": "create_calendar_event",
                "description": "Create an event on the user's Google Calendar at a specified date and time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date_str": {
                            "type": "string",
                            "description": "The date of the event in the format 'mm/dd/yyyy'"
                        },
                        "start_time_str": {
                            "type": "string",
                            "description": "The start time of the event in 'HH:MM' format (24-hour)"
                        },
                        "end_time_str": {
                            "type": "string",
                            "description": "The end time of the event in 'HH:MM' format (24-hour)"
                        },
                        "summary": {
                            "type": "string",
                            "description": "The summary or title of the event"
                        },
                        "description": {
                            "type": "string",
                            "description": "The description of the event (optional)"
                        },
                        "location": {
                            "type": "string",
                            "description": "The location of the event (optional)"
                        }
                    },
                    "required": ["date_str", "start_time_str", "end_time_str", "summary"]
                }
            },
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            max_tokens=1000,
            functions=functions,
            function_call="auto",
            messages=self.message_list
        )
        response_message = response["choices"][0]["message"]

        if response_message.get("function_call"):
            function_name = response_message["function_call"]["name"]
            function_args = json.loads(response_message["function_call"]["arguments"])

            def call_function_with_args(function, args):
                return function(**args)

            available_functions = {
                "get_current_weather": get_current_weather,
                "get_next_n_calendar_events": get_next_n_calendar_events,
                "get_calendar_events_for_day_x": get_calendar_events_for_day_x,
                "create_calendar_event": create_calendar_event,
            }

            function_to_call = available_functions.get(function_name)
            if function_to_call:
                function_response = call_function_with_args(function_to_call, function_args)

                self.message_list.extend([
                    response_message,
                    {
                        "role": "function",
                        "name": function_name,
                        "content": function_response,
                    }
                ])
                print(self.message_list)
                second_response = openai.ChatCompletion.create(
                    model="gpt-4-0613",
                    messages=self.message_list,
                )
                self.message_list.append(second_response["choices"][0]["message"])
                print(self.message_list)
                return second_response["choices"][0]["message"]["content"]
            else:
                raise ValueError(f"Function {function_name} not available")
        else:
            self.message_list.append(response_message)
            return response_message["content"]
    
    def main_loop(self):
        self.phrase_time = None
        source = self.get_microphone()
        with source:
            self.recorder.adjust_for_ambient_noise(source)
        self.recorder.listen_in_background(source, self.record_callback, phrase_time_limit=self.args.record_timeout)
        print("Model loaded.\nListening...")
        while True:
            self.process_audio(source)
            sleep(0.25)
    
    def recognize_intent_and_answer(self, text):
        prompt = {"role": "user", "content": text}
        self.message_list.append(prompt)
        get_gpt4_response = self.get_gpt4_response()
        return get_gpt4_response
        
    def process_audio(self, source):
        try:
            now = datetime.datetime.utcnow()
            if not self.data_queue.empty():
                phrase_complete = False
                if self.phrase_time and now - self.phrase_time > datetime.timedelta(seconds=self.args.phrase_timeout):
                    self.last_sample = bytes()
                    phrase_complete = True
                self.phrase_time = now
                while not self.data_queue.empty():
                    data = self.data_queue.get()
                    self.last_sample += data
                audio_data = sr.AudioData(self.last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())
                temp_file = NamedTemporaryFile().name
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())
                result = self.audio_model.transcribe(temp_file, fp16=torch.cuda.is_available(), task='translate')
                text = result['text'].strip().lower()
                if "claudia" in text:
                    gpt4_response = self.recognize_intent_and_answer(text)
                    audio_filename = f"response_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}.wav"
                    text_to_audio(gpt4_response, audio_filename)
                    print(f"GPT-4 Response saved to {audio_filename}")
                    play_audio(audio_filename)
                if phrase_complete:
                    self.transcription.append(text)
                else:
                    self.transcription[-1] = text
                clear_console()
                for line in self.transcription:
                    print(line)
                print('', end='', flush=True)
        except KeyboardInterrupt:
            self.finalize()

    def finalize(self):
        print("\n\nTranscription:")
        for line in self.transcription:
            print(line)
        exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use", choices=MODEL_CHOICES)
    parser.add_argument("--non_english", action='store_true')
    parser.add_argument("--energy_threshold", default=DEFAULT_ENERGY_THRESHOLD, type=int)
    parser.add_argument("--record_timeout", default=DEFAULT_RECORD_TIMEOUT, type=float)
    parser.add_argument("--phrase_timeout", default=DEFAULT_PHRASE_TIMEOUT, type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default=DEFAULT_MIC_LINUX, type=str)
    args = parser.parse_args()
    smart_mirror = SmartMirror(args)
    smart_mirror.main_loop()
