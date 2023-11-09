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



# Constants
MODEL_CHOICES = ["tiny", "base", "small", "medium", "large"]
DEFAULT_MIC_LINUX = 'pulse'
DEFAULT_ENERGY_THRESHOLD = 1000
DEFAULT_RECORD_TIMEOUT = 2
DEFAULT_PHRASE_TIMEOUT = 3
SAMPLE_RATE = 16000
VOICE = "en_US/hifi-tts_low#92"
openai.api_key = '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'

# Utilities
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# GPT-4 Communication
def get_gpt4_response(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        max_tokens=1000,
        temperature=0.7,
        messages=messages
    )
    return response["choices"][0]["message"]["content"]

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
        self.messages = [{"role": "system", "content": "You are an AI assistant who is talking to a barbershop representative. Your goal is to schedule a hair appointment for 2pm on Thursday on behalf of your client Alisher."}]
        
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

    def main_loop(self):
        self.phrase_time = None
        source = self.get_microphone()
        with source:
            self.recorder.adjust_for_ambient_noise(source)
        self.recorder.listen_in_background(source, self.record_callback, phrase_time_limit=self.args.record_timeout)
        print("Model loaded.\n")
        while True:
            self.process_audio(source)
            sleep(0.25)
    
    def openai_response(self, text):
        self.messages.append({"role": "user", "content": text})  # Add user message
        response = get_gpt4_response(self.messages)
        self.messages.append({"role": "assistant", "content": response})  # Add assistant message
        
        
    def process_audio(self, source):
        try:
            now = datetime.utcnow()
            if not self.data_queue.empty():
                phrase_complete = False
                if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.args.phrase_timeout):
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
                text = result['text'].strip()
                
                self.openai_response(text)
                audio_filename = f"response_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.wav"
                print(self.messages[-1]['content'])
                text_to_audio(self.messages[-1]['content'], audio_filename)
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
