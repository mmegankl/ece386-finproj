# imports go here
import torch
import sounddevice as sd
import numpy as np
import numpy.typing as npt
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline
from ollama import chat
from ollama import Client
from typing import Optional
import Jetson.GPIO as GPIO
import requests
import os


# copied directly from ICE3
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


# 10 second recording; subject to change
def record_audio(duration_seconds: int = 10) -> npt.NDArray:
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


# LLM integration
LLM_MODEL: str = "gemma3:27b"  # Change this to be the model you want
client: Client = Client(
    host="http://ai.dfec.xyz:11434"  # Change this to be the URL of your LLM
)

few_shot_prompt: str = """
Given a weather query, return a formatted string that includes the location for the wttr.in API. This API takes the following formats:

1. Cities
2. 3-letter airport codes

If a popular attraction/geographical location is mentioned, include a tilde ('~') before the word. 
Anytime the city is more than one word, replace the spaces between the words with '+' and capitalize all words. 
When requesting the weather at or near an airport mentioned, output the three-letter airport code in all lowercase.

Examples:

Input: What is the weather in Vegas?
Output: Las+Vegas

Input: What's the weather near the Eiffel Tower?
Output: ~Eiffel+Tower

Input: Please give me the weather at Honolulu International Airport.
Output: hnl

Input: Please give me the weather at the airport in San Franscisco.
Output: sfo

Input: I'm at JFK right now. What's the weather?
Output: New+York+City

Input: What's the weather in New York City?
Output: New+York+City

Input: Forecast for Rio Rancho?
Output: Rio+Rancho


Please note how New York City and Las Vegas are each two or more words, so the output includes a '+' where the space between the words should be.
These two examples mentioned above are not the only instances where a '+' is necessary. This rule applies to ANY city that is more than one word.
Therefore, do not give me an output with any spaces. There are '+' instead of spaces in the output.

"""


# TODO: define llm_parse_for_wttr()
def llm_parse_for_wttr(prompt: str) -> str:

    prompt = prompt["text"]
    print(prompt)
    response = client.chat(
        messages=[
            {
                "role": "system",
                "content": few_shot_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model=LLM_MODEL,
    )

    print(response)
    output = response["message"]["content"].strip()  # used AI for this line

    return output


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
    # while True:
    # audio recording and transcription
    GPIO.wait_for_edge(my_pin, GPIO.RISING)
    print("UP!")

    print("Recording...")
    audio = record_audio()
    print("Done")

    print("Transcribing...")
    speech = pipe(audio)
    print("Done")

    print(speech)

    # LLM
    place = llm_parse_for_wttr(speech)
    print(place)

    url = f"curl wttr.in/{place}"
    print(url)

    # output
    os.system(url)
