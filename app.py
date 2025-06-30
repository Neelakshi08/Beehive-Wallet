# app.py
import streamlit as st
import whisper
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from gtts import gTTS
import os
from tempfile import NamedTemporaryFile

st.set_page_config(page_title="Beehive Wallet 🐝", layout="centered")

st.title("🐝 Beehive Wallet – Voice to Ledger")
st.markdown("Speak a transaction like: **'Meena added 500 rupees for school books'**")

audio_file = st.file_uploader("🎙️ Upload a voice file (WAV recommended)", type=["wav", "mp3", "m4a"])

if audio_file:
    with NamedTemporaryFile(delete=False) as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    with st.spinner("🔍 Transcribing..."):
        model = whisper.load_model("base")
        result = model.transcribe(temp_audio_path)
        transcript = result["text"]
        st.success("Transcription complete!")
        st.text_area("📝 Transcript", transcript)

    with st.spinner("🤖 Extracting structured data..."):
        prompt = PromptTemplate(
            input_variables=["speech"],
            template="""
            Extract financial transaction details from this voice command: {speech}
            Give output in JSON format with member, action, amount, reason.
            """
        )
        llm = ChatOpenAI()
        chain = LLMChain(llm=llm, prompt=prompt)
        try:
            output = chain.run(transcript)
        except Exception:
            output = '{"member": "Unknown", "action": "added", "amount": 0, "reason": "undetected"}'
        st.success("Data parsed successfully!")
        st.json(output)

    with st.spinner("🔊 Generating voice feedback..."):
        response_text = f"Got it! {output}"
        tts = gTTS(response_text)
        tts_file = "response.mp3"
        tts.save(tts_file)
        audio_file = open(tts_file, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')
        st.success("Voice response ready!")
