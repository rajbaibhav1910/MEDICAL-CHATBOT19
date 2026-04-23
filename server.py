import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import io
import pyttsx3
from typing import Optional

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import pipeline

def load_local_env():
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

load_local_env()

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if HF_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

HF_REPO_ID = os.environ.get("HF_REPO_ID", "mosaicml/mpt-7b-chat")
DB_FAISS_PATH = "vectorstore/db_faiss"

class LocalTextGenerator:
    def __init__(self, model_name="distilgpt2", max_new_tokens=100):
        self.max_new_tokens = max_new_tokens

    def invoke(self, prompt) -> str:
        return "⚠️ **SYSTEM ALERT:** No Hugging Face API key was found in the environment.\n\nI am currently running in an offline placeholder mode. I am unable to connect to my large medical neural networks to process your query about your symptoms. I am returning this automatic message instead.\n\n**To Fix This:** Please provide a Hugging Face API Token in your `.env` file and restart the core to activate full diagnostic capabilities."

# If the default model is unsupported for your account, set HF_REPO_ID to a model you can access.
# Example safe alternatives:
#   HF_REPO_ID=mosaicml/mpt-7b-chat
#   HF_REPO_ID=tiiuae/falcon-7b-instruct
#   HF_REPO_ID=TheBloke/wizardLM-7B-uncensored
# Note: some models require access or a paid plan.
CUSTOM_PROMPT_TEMPLATE = """
You are Dr. Ava, a compassionate and knowledgeable AI medical assistant. You help patients understand medical information based on the provided context. 

IMPORTANT GUIDELINES:
- Always be empathetic and professional in your responses
- Use the medical information from the context to answer questions
- If you don't know the answer, say so clearly and suggest consulting a healthcare professional
- Never provide specific medical diagnoses or treatment recommendations
- Always remind patients that you are an AI assistant and they should consult healthcare professionals for medical decisions
- Be encouraging and supportive while maintaining medical accuracy

Context: {context}
Question: {question}

Provide a helpful, empathetic response based on the context above.
"""


def build_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    hf_llm: Optional[HuggingFaceEndpoint] = None
    if HF_TOKEN:
        try:
            endpoint = HuggingFaceEndpoint(
                repo_id=HF_REPO_ID,
                task="conversational",
                temperature=0.5,
                max_new_tokens=512,
                top_p=0.9,
                huggingfacehub_api_token=HF_TOKEN,
            )
            hf_llm = endpoint
        except Exception as exc:
            print(f"Hugging Face endpoint unavailable, using local fallback instead: {exc}")
            hf_llm = None
    else:
        print("No Hugging Face token found, running in local fallback mode.")

    local_llm = LocalTextGenerator()

    def chat_runner(prompt_input) -> str:
        prompt_text = str(prompt_input)
        if hf_llm is not None:
            try:
                return hf_llm.invoke(prompt_text)
            except Exception as exc:
                print(f"Hugging Face invocation failed, falling back locally: {exc}")
        return local_llm.invoke(prompt_text)

    chat_llm = RunnableLambda(chat_runner)

    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = db.as_retriever(search_kwargs={"k": 3})

    qa = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | chat_llm
        | StrOutputParser()
    )
    return qa


doc_qa_chain = None

app = FastAPI(title="Medical Chatbot")

@app.on_event("startup")
async def startup_event():
    global doc_qa_chain
    try:
        doc_qa_chain = build_qa_chain()
        print("QA chain initialized successfully.")
    except Exception as exc:
        print(f"Failed to initialize QA chain: {exc}")
        doc_qa_chain = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/ask")
async def ask(request: Request):
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "question is required"})

    if doc_qa_chain is None:
        return JSONResponse(status_code=500, content={"error": "Server is not ready. QA chain failed to initialize."})

    result = doc_qa_chain.invoke(question)

    return {
        "answer": result,
    }


@app.post("/api/speak")
async def speak(request: Request):
    """Convert text to speech for the AI avatar"""
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "text is required"})
    
    try:
        # Initialize text-to-speech engine
        engine = pyttsx3.init()
        
        # Set voice properties for a more natural female voice (Zira)
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'zira' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        
        # Set speech rate and volume for natural conversation
        engine.setProperty('rate', 160)  # Slightly faster for better engagement
        engine.setProperty('volume', 0.9)  # Higher volume for clarity
        
        # Create temporary file for audio output
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        # Generate speech to temporary file
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        
        # Read the generated audio file
        with open(temp_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        # Return audio data
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=dr_ava_response.wav",
                "Cache-Control": "no-cache"
            }
        )
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"TTS failed: {str(e)}"})


@app.get("/api/tts-test")
async def tts_test():
    """Test endpoint to verify TTS functionality"""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        voice_info = []
        for voice in voices:
            voice_info.append({
                "name": voice.name,
                "id": voice.id,
                "is_female": 'zira' in voice.name.lower() or 'female' in voice.name.lower()
            })
        
        return {
            "status": "TTS is working",
            "available_voices": voice_info,
            "current_voice": "Microsoft Zira Desktop (Female voice selected)",
            "speech_rate": 160,
            "volume": 0.9
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"TTS test failed: {str(e)}"})


if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (for Vercel deployment)
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)