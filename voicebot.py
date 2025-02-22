# type: ignore
import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from openai import OpenAI
import tempfile
import io
import soundfile as sf
from audio_recorder_streamlit import audio_recorder
import torch 
from aksharamukha import transliterate 
import torchaudio 

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="CampusBuddy",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS remains the same as previous version
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-title {
        font-size: 3rem !important;
        font-weight: bold !important;
        margin-bottom: 0.5rem !important;
        text-align: center;
    }
    .subtitle {
        font-size: 1.5rem !important;
        color: #666;
        margin-bottom: 2rem;
        text-align: center;
    }
    .section-header {
        font-size: 2rem !important;
        font-weight: bold !important;
        margin: 2rem 0 1rem 0;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        background-color: rgba(255, 255, 255, 0.1);
    }
    .record-button {
        margin: 1rem 0;
    }
    .stAudio {
        margin-top: 1rem;
    }
    div.element-container div.stMarkdown {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize models and database
@st.cache_resource
def initialize_models():
    # Initialize LLM and embeddings
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
    llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')
    vector_store = Chroma(
        collection_name="SCT_college_data",
        embedding_function=embeddings_model,
        persist_directory="chroma_db",
    )
    
    # Initialize Silero TTS model
    tts_model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='indic',
        speaker='v4_indic'
    )
    
    return embeddings_model, llm, vector_store, tts_model

embeddings_model, llm, vector_store, tts_model = initialize_models()
retriever = vector_store.as_retriever(search_kwargs={'k': 5})

def transcribe_audio_with_whisper(audio_bytes):
    client = OpenAI()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name
        
    with open(temp_audio_path, "rb") as audio_file:
        transcript = client.audio.translations.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    os.unlink(temp_audio_path)
    return transcript

def translate_to_malayalam(english_text):
    translation_prompt = f"Translate the following text from English to Malayalam:\n\n{english_text}"
    response = llm.invoke(translation_prompt)
    return response.content

def convert_text_to_malayalam_audio(malayalam_text):
    # Convert Malayalam text to Roman script
    roman_text = transliterate.process('Malayalam', 'ISO', malayalam_text)
    
    # Generate audio using Silero TTS
    audio = tts_model.apply_tts(roman_text, 
            speaker='malayalam_female',
            sample_rate=24000,
            put_accent=True,
            put_yo=True)
    
    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        torchaudio.save(temp_audio.name, audio.unsqueeze(0), 24000)
        return temp_audio.name

    
def process_audio_and_respond(audio_bytes):
    with st.spinner("Transcribing your question..."):
        english_text = transcribe_audio_with_whisper(audio_bytes)
        
    with st.spinner("Translating to Malayalam..."):
        malayalam_question = translate_to_malayalam(english_text)
        
    docs = retriever.invoke(english_text)
    knowledge = "".join(doc.page_content + "\n\n" for doc in docs)
    
    rag_prompt = f"""
    You are a helpful assistant answering questions about the college based on the provided knowledge.
    Be precise and concise in your answers.
    
    Question: {english_text}
    Knowledge: {knowledge}
    """
    
    with st.spinner("Generating response..."):
        response = llm.invoke(rag_prompt)
        malayalam_response = translate_to_malayalam(response.content)
        audio_path = convert_text_to_malayalam_audio(malayalam_response)
    
    return malayalam_question, malayalam_response, audio_path

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
        # Process the audio and get response
        malayalam_question, malayalam_response, audio_path = process_audio_and_respond(audio_bytes)
        
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