# Voice bot + Call analytics

This is a project showing a bot that uses Text-To-Speech, Speech-To-Text, and a language model to have a conversation with a user.

This demo is set up to use [Deepgram](www.deepgram.com) for STT, [Groq](https://groq.com/) for LLM and [ElevanLabs](https://elevenlabs.io/) for TTS.

The files in `building_blocks` are the isolated components if you'd like to inspect them. Some old code files are archived in `old_files` folder.

**System prompt guide:**
- for `dg_groq_ell_ws_stream.py`:
  - Change the prompts in `main()`
- for `old_files` folder:
  - Refer to individual prompt files based on use cases
  - Change filename for `system_prompt_usecase.txt` in `QuickAgent.py` file at 'line 40'

**Conversation guide:**
- Guide for user to ask questions for different use cases

**Main code (flow):**
- `def main()` defined system prompts and conversation_history -> `llm_tts` is called first for bot to introduce itself -> process below is followed
- Transcript will be generated in sentence form using `class TranscriptCollector` and `stt()` in `class ConversationManager`
- Microphone will be kept switched on throughout and transripts collected will get added to the `sentence_queue` in `on_message()`
- `process_sentence_queue()` takes customer queries and calls `llm_tts()` function
- `llm_tts()` will generate a response stream in chunks -> `text_iterator()` yields these chunks as output -> `TextToSpeech().speak()` is called to give audio output -> audio output is generated in chunks while ensuring flow using `text_chunker()` function which identifies splitters in the stream and stores in buffers

**Call analytics:**
- Refer to `call_recording` files and analytics codes
- `analytics_dg`: Deepgram takes audio files as input; need to assess intent and sentiment analysis output
- `analytics_llm`: LLM takes json file as input and runs analysis based on prompt 

```
Run command: python3 dg_groq_ell_ws_stream.py
```
