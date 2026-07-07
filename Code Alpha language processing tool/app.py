import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS
import os

st.title("🌍 AI Language Translation Tool")

text = st.text_area("Enter Text")

languages = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Arabic": "ar",
    "Japanese": "ja",
    "Chinese": "zh-CN"
}

source = st.selectbox("Source Language", languages.keys())
target = st.selectbox("Target Language", languages.keys())

if st.button("Translate"):

    translated = GoogleTranslator(
        source=languages[source],
        target=languages[target]
    ).translate(text)

    st.success("Translated Text")

    st.write(translated)

    st.download_button(
        "Download Translation",
        translated,
        file_name="translation.txt"
    )

    if st.button("Generate Speech"):

        tts = gTTS(translated, lang=languages[target])

        tts.save("voice.mp3")

        audio_file = open("voice.mp3", "rb")

        st.audio(audio_file.read())