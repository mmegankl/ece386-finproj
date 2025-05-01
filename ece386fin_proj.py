#imports go here
import torch
import sys
import time
import sounddevice as sd
import numpy as np
import numpy.typing as npt
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline
from ollama import chat
from pydantic import BaseModel
import sys
from typing import Optional

# function declarations
def initialize():
    #need to initialize GPIO, mic, models, etc.

    #pin
    '''Prints 'UP' or 'DOWN' based on edges on Jetson pin #29'''
    import Jetson.GPIO as GPIO

    # Init as digital input
    my_pin = 29
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    GPIO.setup(my_pin, GPIO.IN)  # digital input

    print('Starting Demo! Move pin 29 between 0V and 3.3V')

    while True:
        GPIO.wait_for_edge(my_pin, GPIO.RISING)
        print('UP!')
        GPIO.wait_for_edge(my_pin, GPIO.FALLING)
        print('down')
        return 0

def build_whisper_pipeline(model_id: str, torch_dtype: torch.dtype, device: str) -> Pipeline:
    """Creates a Hugging Face automatic-speech-recognition pipeline on the given device."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

def wait_for_rising_edge():
    #rising edge for pin 29
    return 0

def start_recording():
    #recording from mic
    return 0

def transcribe_audio():
    # Get model as argument, default to "distil-whisper/distil-medium.en" if not given
    model_id = sys.argv[1] if len(sys.argv) > 1 else "distil-whisper/distil-medium.en"
    print("Using model_id {model_id}")
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device {device}.")

    print("Building model pipeline...")
    pipe = build_pipeline(model_id, torch_dtype, device)
    print(type(pipe))
    print("Done")

    print("Recording...")
    audio = record_audio()
    print("Done")

    print("Transcribing...")
    start_time = time.time_ns()
    speech = pipe(audio)
    end_time = time.time_ns()
    print("Done")

    print(speech)
    print(f"Transcription took {(end_time-start_time)/1000000000} seconds")

def llm_processor(transcribed_text):
    weather_input = 0
    return weather_input

def get_weather(weather_input):
    #call on wttr.in
    url = "https://wtter.in/{weather_input}"
    weather = requests.get(url)
    if response.status_coide == 200:
        return response.text
    else:
        return None
    
def print_weather(response.text, weather_input):
    print(f"The weather in {weather_input} is:")
    
