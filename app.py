import streamlit as st
import os
import re
import asyncio
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound,CouldNotRetrieveTranscript
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai

# Gemini setup
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Ensure event loop exists (for async)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Gemini LLM (cached)
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

#Streamlit page setup
st.set_page_config(page_title="YouTube RAG Chatbot", layout="centered")
st.title("📺 YouTube BOT")

#Load local CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("app.css")

#Sidebar instructions
with st.sidebar:
    st.header("📖 How to Use")

    st.markdown("""
    1. **Paste a YouTube Video Link** 🎥  
       - Supports Hindi + English.
       - Make sure it has subtitles enabled.
    
    2. **Ask a Question** 💬  
       - Type a question about the video.
       - Example: *"What is explained in the video?"*

    3. **Get Answers Instantly** ⚡  
       - The bot will retrieve and answer using transcript.

    ---
    🧠 Powered by Gemini + FAISS  
    💡 Tip: Use short and clear questions.
    """)


# Chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Extract YouTube video ID
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

# Get transcript
def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            # Try to find manually uploaded English transcript
            transcript = transcript_list.find_transcript(['en'])
        except NoTranscriptFound:
            # If not found, try auto-generated English
            transcript = transcript_list.find_generated_transcript(['en'])

        return " ".join(chunk["text"] for chunk in transcript.fetch())

    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
        return "Transcript not available for this video."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "Transcript not available for this video."

# Chunking
def split_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([text])

# Embedding
def embed_chunks(chunks):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.from_documents(chunks, embed_model)

# Prompt template
template = """
You are a helpful assistant.
Use only the context from the transcript to answer.
If the answer is not in the context, say "Sorry, I don't know.and if u can answer
respond in the same language as the context provided."

Context:
{context}

Conversation:
{question}
"""
prompt = PromptTemplate.from_template(template)
parser = StrOutputParser()

#  UI Inputs
video_url = st.text_input("🎥 Enter a YouTube video link:")
user_input = st.chat_input("💬 Ask something about the video")

if video_url:
    video_id = extract_video_id(video_url)

    with st.spinner("📄 Fetching transcript..."):
        transcript = get_transcript(video_id)

    if not transcript:
        st.error("Transcript not available for this video.")
        st.stop()
    else:
        with st.spinner("🔎 Building knowledge base..."):
            chunks = split_into_chunks(transcript)
            vector_store = embed_chunks(chunks)

            st.write("📑 Sample chunk:", chunks[0].page_content[:300])
            st.write("📊 Total chunks created:", len(chunks))


            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

            llm = get_llm()

        #User query handling
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            past_user_inputs = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"][:-1]
            generated_responses = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "assistant"]

            retrieved_docs = retriever.invoke(user_input)


            st.write("📍 Retrieved Docs:", retrieved_docs)



            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

            #Build conversation history
            conversation_history = ""
            for u, a in zip(past_user_inputs, generated_responses):
                conversation_history += f"User: {u}\nAssistant: {a}\n"
            conversation_history += f"User: {user_input}\n"

            final_prompt = f"""
            You are a helpful assistant. Use only the following context to answer.
            If the context is insufficient, say "I don't know".

            Context:
            {context_text}

            Conversation History:
            {conversation_history}
            Assistant:"""

            with st.spinner("💬 Thinking..."):
                try:



                    st.subheader("📋 Final Prompt Sent to LLM")
                    st.code(final_prompt, language="markdown")



                    raw_response = llm.invoke(final_prompt)
                    answer=raw_response.content
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])


# Footer HTML only (CSS handled in app.css)
st.markdown("""
<div class="footer">
    Created by <a href="https://github.com/MihirChandra04" target="_blank">@MihirChandra</a>
</div>
""", unsafe_allow_html=True)

