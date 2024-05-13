# Quick Voice bot demo

This is a alpha demo showing a bot that uses Text-To-Speech, Speech-To-Text, and a language model to have a conversation with a user.

This demo is set up to use [Deepgram](www.deepgram.com) for STT + TTS and [Groq](https://groq.com/) the LLM.

The files in `building_blocks` are the isolated components if you'd like to inspect them

System prompt guide:
- Refer to individual prompt files based on use cases
- Change filename for `system_prompt_usecase.txt` in `QuickAgent.py` file at 'line 40'

```
Run command: python3 QuickAgent.py
```