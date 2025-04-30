#imports go here

# function declarations
def initialize():
    #need to initialize GPIO, mic, models, etc.
    return 0

def build_whisper_pipeline():
    #setup whistper model
    return 0

def wait_for_rising_edge():
    #rising edge for pin 29
    return 0

def start_recording():
    #recording from mic
    return 0

def transcribe_audio():
    #use whisper to transcribe to text
    transcribed_text = 0
    return transcribed_text

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
    