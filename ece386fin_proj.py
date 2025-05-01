# imports go here
import torch
import sounddevice as sd
import numpy as np
import numpy.typing as npt
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline
from ollama import chat
from typing import Optional
import Jetson.GPIO as GPIO
import requests


def build_whisper_pipeline(
    model_id: str, torch_dtype: torch.dtype, device: str
) -> Pipeline:
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


# 4 second recording
def record_audio(duration_seconds: int = 4) -> npt.NDArray:
    """Record duration_seconds of audio from default microphone.
    Return a single channel numpy array."""
    sample_rate = 16000  # Hz
    samples = int(duration_seconds * sample_rate)
    # Will use default microphone; on Jetson this is likely a USB WebCam
    audio = sd.rec(samples, samplerate=sample_rate, channels=1, dtype=np.float32)
    # Blocks until recording complete
    sd.wait()
    # Model expects single axis
    return np.squeeze(audio, axis=1)


def llm_processor(transcribed_text):
    weather_input = 0
    return weather_input


def get_weather(weather_input):
    # call on wttr.in
    url = "https://wtter.in/{weather_input}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None


def print_weather(text, weather_input):
    print(f"The weather in {weather_input} is:")


if __name__ == "__main__":
    # init GPIO
    # Init as digital input
    my_pin = 29
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    GPIO.setup(my_pin, GPIO.IN)  # digital input

    # build Whisper pipeline
    # Get model as argument, default to "distil-whisper/distil-medium.en" if not given
    model_id = "distil-whisper/distil-medium.en"
    print("Using model_id {model_id}")
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device {device}.")

    print("Building model pipeline...")
    pipe = build_whisper_pipeline(model_id, torch_dtype, device)
    print(type(pipe))
    print("Done")

    # wait for GPIO rising edge
    while True:
        GPIO.wait_for_edge(my_pin, GPIO.RISING)
        print("UP!")

        print("Recording...")
        audio = record_audio()
        print("Done")

        print("Transcribing...")
        speech = pipe(audio)
        print("Done")

        print(speech)
