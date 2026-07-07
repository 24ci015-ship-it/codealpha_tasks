import streamlit as st
import subprocess
import os

st.set_page_config(
    page_title="AI Music Generator",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 AI Music Generator")
st.write("Generate new music using an AI LSTM model.")

st.markdown("---")

if st.button("🎼 Generate Music"):

    with st.spinner("Generating music... Please wait..."):
        subprocess.run(["python", "generate.py"])

    st.success("✅ Music Generated Successfully!")

    if os.path.exists("output.mid"):

        st.write("### 🎵 Generated MIDI File")

        with open("output.mid", "rb") as file:
            st.download_button(
                label="📥 Download output.mid",
                data=file,
                file_name="output.mid",
                mime="audio/midi"
            )

        st.info("Your browser may not play MIDI files directly. Download the file and open it using VLC, MuseScore, or any MIDI player.")

    else:
        st.error("output.mid was not created.")