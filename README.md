# CampusBuddy: Malayalam Voice Assistant

## Project Overview
A voice-enabled AI assistant for college information using Malayalam language support.

## Prerequisites
- Python 3.10
- OpenAI API Key
- Supported Operating Systems: macOS, Linux, Windows

## Setup Instructions

1. Clone the repository
2. Create virtual environment
```bash
python3.10 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
- Create `.env` file
- Add `OPENAI_API_KEY=your_key_here`

5. Prepare document database
```bash
python ingest.py
```

6. Run the application
```bash
streamlit run app.py
```

## Troubleshooting
- Ensure OpenAI API key is valid
- Check internet connectivity
- Verify Python and library versions

## Libraries Used
- Streamlit
- LangChain
- OpenAI
- Silero TTS
- Aksharamukha Transliteration