import asyncio
import streamlit as st
from typing import AsyncGenerator, Generator
from dotenv import load_dotenv
import os
import json
import time

from groq import Groq

load_dotenv()

st.set_page_config(page_icon="telephone_receiver", layout="wide",
                   page_title="Call Analytics")


# Print the incremental deltas returned by the LLM.
def generate_chat_responses(stream):
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="")
            yield delta.content

def main() -> None:
    client = Groq(api_key = os.getenv('GROQ_API_KEY'))

    # system prompts
    introduction = "As a hotel reservations manager at the Beachview hotel, you are tasked to speak with customers seeking room bookings at the hotel. Your name is Pooja."
    task = "Whenever the user asks for room availability, ask them for the basic details such as dates, number of guests, room preferences, breakfast inclusion, and any special requests - do not ask everything in a single question."
    property_details = "The Beachview hotel is a 5-star property in Mumbai and commands a regal view of the Arabian Sea and the famous Juhu beach. It is located just 30 mins from Mumbai Airport, the travel is also arranged by the concierge. Amenities include Swimming pool, Gym, Spa, Beachfront, Cafe, Business Lounge, Banquet hall, Garden, etc. Prices quoted include breakfast and access to swimming pool, gym, beachfront, garden. Other amenities to be charged as per requirements. If the user asks for prices without breakfast, you can deduct rupees 1000 from the price quoted per night. Also, inclusion of buffet wil cost rupees 1000 extra per person for each lunch and dinner. Room types along with the details is as follows = '1. Superior room = 'area 260 square feet, city view, perfect for business and leisure travellers on the go, priced at rupees 9500 per night, inventory of 150 rooms. 2. Premier room = 'area 260 square feet, ocean view, offering stunning views of the Arabian Sea, priced at rupees 10500 per night, inventory of 100 rooms. 3. Executive room = 'area 350 square feet, city view, large studio rooms, priced at rupees 12500 per night, inventory of 100 rooms. 4. Deluxe room = 'area 350 square feet, ocean view, large studio rooms offering stunning views of the Arabian Sea, priced at rupees 18000 per night, inventory of 50 rooms. 5 = 'Luxury suite = 'area 500 square feet, ocean view, consisting of a living room and a separate bedroom, priced at rupees 25000 per night, inventory of 10 rooms."
    conversation_style = "Communicate concisely and conversationally. Aim for responses in short, clear prose, ideally under 20 words. Always maintain a professional stance."
    language = "Speak like a human as possible, use everyday language and avoid using big and complex words."
    customer_engagement = "Lead the conversation and do not be passive. Most times, engage users by ending with a question. Advise customer on what's best for them."
    transcript_reading = "Don't repeat what's in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized."
    ASR_errrors = "This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial while making use of phrases like 'didn't catch that', 'some noise', 'pardon', 'you're coming through choppy', 'static in your speech', 'voice is cutting in and out'. Do not ever mention 'transcription error', and don't repeat yourself."
    role = "If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don't repeat yourself in doing this. You should still be creative, human-like, and lively."
    brackets = "Answer should not have any parantheses or brackets."

    conversation_history = [
        {"role": "system", "content": introduction},
        {"role": "system", "content": task},
        {"role": "system", "content": property_details},
        {"role": "system", "content": conversation_style},
        {"role": "system", "content": language},
        {"role": "system", "content": customer_engagement},
        {"role": "system", "content": transcript_reading},
        {"role": "system", "content": ASR_errrors},
        {"role": "system", "content": role},
        {"role": "system", "content": brackets},
    ]

    with open('call_recording_2.json', 'r') as file:
        conversation = json.load(file)

    conversation_history.extend(conversation)

    query = "I want to make use of call center analytics on this conversation. Please perform Customer Sentiment Analysis, Agent Performance Analysis, Call Flow Optimization, and Quality Assurance. Also give rating out of 10 for Agent Performance Analysis, Call Flow Optimization, and Quality Assurance. Highlight specific instances if applicable."

    start_time = time.time()

    stream = client.chat.completions.create(
        #
        # Required parameters
        #
        messages = conversation_history + [
            # conversation_history includes system prompt and chat history
            
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": query,
            },
        ],
        # The language model which will generate the completion.
        model="llama3-70b-8192",
        #
        # Optional parameters
        #
        # Controls randomness: lowering results in less random completions.
        # As the temperature approaches zero, the model will become
        # deterministic and repetitive.
        temperature=0.5,
        # The maximum number of tokens to generate. Requests can use up to
        # 2048 tokens shared between prompt and completion.
        max_tokens=1024,
        # Controls diversity via nucleus sampling: 0.5 means half of all
        # likelihood-weighted options are considered.
        top_p=1,
        # A stop sequence is a predefined or user-specified text string that
        # signals an AI to stop generating content, ensuring its responses
        # remain focused and concise. Examples include punctuation marks and
        # markers like "[end]".
        stop=None,
        # If set, partial message deltas will be sent.
        stream=True,
    )

    end_time = time.time()

    elapsed_time = int((end_time - start_time) * 1000)
    print(f"LLM ({elapsed_time}ms): ")

    button_clicked = st.button("Run call analytics :speech_balloon:", type="primary")

    if button_clicked:
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(stream)
            st.write_stream(chat_responses_generator)


if __name__ == "__main__":
    main()