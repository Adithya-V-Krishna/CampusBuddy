# type: ignore
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from utils import initialize_models, process_audio_and_respond, get_custom_css

# Page configuration
st.set_page_config(
    page_title="CampusBuddy",
    page_icon="🎓",
    layout="wide"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Initialize models and cache them
@st.cache_resource
def get_models():
    return initialize_models()

models = get_models()

def main():
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">🎓 CampusBuddy: Malayalam Voice Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask your questions in Malayalam about our college!</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Audio Recording Section
    st.markdown('<h2 class="section-header">🎤 Record Your Question</h2>', unsafe_allow_html=True)
    
    # Audio recorder
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e84545",
        neutral_color="#6aa84f",
        icon_size="2x"
    )

    if audio_bytes:
        # Process the audio with spinner indicators
        with st.spinner("Transcribing your question..."):
            with st.spinner("Generating response..."):
                malayalam_question, malayalam_response, audio_path = process_audio_and_respond(audio_bytes, models)

        # Add to chat history
        st.session_state.chat_history.append({
            "question": malayalam_question,
            "response": malayalam_response,
            "audio_path": audio_path
        })

    # Chat History Display
    if st.session_state.chat_history:
        st.markdown('<h2 class="section-header">💬 Conversation History</h2>', unsafe_allow_html=True)
        for entry in reversed(st.session_state.chat_history):
            with st.container():
                st.markdown('<div class="chat-message">', unsafe_allow_html=True)
                st.markdown("**Your Question:**")
                st.write(entry["question"])
                st.markdown("**Assistant's Response:**")
                st.write(entry["response"])
                st.audio(entry["audio_path"], format='audio/mp3')
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()