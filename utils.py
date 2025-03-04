# type: ignore
import os
import tempfile
import io
import torch
import torchaudio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
from aksharamukha import transliterate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_models():
    """Initialize all models and vector store"""
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
    
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    
    return {
        'embeddings_model': embeddings_model,
        'llm': llm,
        'vector_store': vector_store,
        'tts_model': tts_model,
        'retriever': retriever
    }

def transcribe_audio_with_whisper(audio_bytes):
    """Transcribe audio using OpenAI Whisper API"""
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

def translate_to_malayalam(english_text, llm):
    """Translate English text to Malayalam using LLM"""
    translation_prompt = f"Translate the following text from English to Malayalam:\n\n{english_text}"
    response = llm.invoke(translation_prompt)
    return response.content

def convert_text_to_malayalam_audio(malayalam_text, tts_model):
    """Convert Malayalam text to audio using Silero TTS"""
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

def process_audio_and_respond(audio_bytes, models):
    """Process audio input and generate response"""
    llm = models['llm']
    tts_model = models['tts_model']
    retriever = models['retriever']
    
    # Transcribe audio to English
    english_text = transcribe_audio_with_whisper(audio_bytes)
    
    # Translate question to Malayalam
    malayalam_question = translate_to_malayalam(english_text, llm)
    
    # Retrieve relevant documents
    docs = retriever.invoke(english_text)
    knowledge = "".join(doc.page_content + "\n\n" for doc in docs)
    
    # Generate response using RAG
    rag_prompt = f"""
    You are a helpful assistant answering questions about the college based on the provided knowledge.
    Be precise and concise in your answers.
    
    Question: {english_text}
    Knowledge: {knowledge}
    """
    
    response = llm.invoke(rag_prompt)
    
    # Translate response to Malayalam and convert to audio
    malayalam_response = translate_to_malayalam(response.content, llm)
    audio_path = convert_text_to_malayalam_audio(malayalam_response, tts_model)
    
    return malayalam_question, malayalam_response, audio_path

def get_custom_css():
    """Return custom CSS for the Streamlit app"""
    return """
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
    """