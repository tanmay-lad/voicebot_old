import base64
import json

import asyncio
import websockets
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os

from elevenlabs.client import ElevenLabs
from elevenlabs import stream
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()


class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=os.getenv(
            "GROQ_API_KEY"))  # mixtral-8x7b-32768, llama3-70b-8192, llama2-70b-4096
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", openai_api_key=os.getenv("OPENAI_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125", openai_api_key=os.getenv("OPENAI_API_KEY"))

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load the system prompt from a file
        with open('system_prompt_hotel_booking_hin.txt', 'r') as file:
            system_prompt = file.read().strip()

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)  # Add user message to memory

        start_time = time.time()

        # Go get the response from the LLM
        response = self.conversation.invoke({"text": text})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['text']}")
        return response['text']


class TextToSpeech:
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    # model = "eleven_turbo_v2"  # Example model name, change as needed
    model = "eleven_multilingual_v2"  # Example model name, change as needed
    voice_id = '21m00Tcm4TlvDq8ikWAM'  # Change this as required
    # voice_id = 'Monika Sogam - Hindi Modulated'  # Change this as required

    async def speak(self, text):
        start_time = time.time()  # Record the time before sending the request
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id={self.model}"

        async with websockets.connect(uri) as websocket:

            await websocket.send(json.dumps({
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                "xi_api_key": self.elevenlabs_api_key,
            }))

            async def listen():
                """Listen to the websocket for audio data and stream it."""
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        if data.get("audio"):
                            yield base64.b64decode(data["audio"])
                        elif data.get('isFinal'):
                            break
                    except websockets.exceptions.ConnectionClosed:
                        print("Connection closed")
                        break

            listen_task = asyncio.create_task(stream(listen()))

            # Send LLM response to eleven labs for tts
            llm_response = {
                "text": text,
                "try_trigger_generation": True
            }
            await websocket.send(json.dumps(llm_response))

            # Send EOS message with an empty string instead of a single space as mentioned in the documentation
            eos_message = {
                "text": ""
            }
            await websocket.send(json.dumps(eos_message))

            await listen_task
            end_time = time.time()  # Record the time before sending the request
            ttts = int((end_time - start_time) * 1000)  # Calculate the time for tts to complete
            print(f"TTS Completion Time (TTTS): {ttts} ms\n")


class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)


transcript_collector = TranscriptCollector()


async def get_transcript(callback):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient("", config)

        dg_connection = deepgram.listen.asynclive.v("1")
        print("Listening...")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript

            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                # Check if the full_sentence is not empty before printing
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="hi",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=500,  # 300, Time in milliseconds of silence to wait for before finalizing speech
            smart_format=True,
        )

        await dg_connection.start(options)

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return


class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Start with voicebot introduction
        llm_response = self.llm.process("Please introduce yourself")

        tts = TextToSpeech()
        await tts.speak(llm_response)

        # Loop indefinitely until "goodbye" is detected
        while True:
            await get_transcript(handle_full_sentence)

            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                break

            llm_response = self.llm.process(self.transcription_response)

            tts = TextToSpeech()
            await tts.speak(llm_response)
            # asyncio.get_event_loop().run_until_complete(tts.speak(llm_response))

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""


if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())