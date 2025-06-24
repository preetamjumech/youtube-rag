import os
import re
import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nomic import NomicEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def extract_video_id(url: str) -> str:
    patterns = [
        r"youtu\.be/([^?&]+)",
        r"youtube\.com/watch\?v=([^?&]+)",
        r"youtube\.com/embed/([^?&]+)",
        r"youtube\.com/v/([^?&]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url.strip()

def fetch_transcript(video_id: str, language: str = "en") -> str:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript
    except TranscriptsDisabled:
        return ""

def split_transcript(transcript: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([transcript])

def create_vector_store(chunks):
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
    return FAISS.from_documents(chunks, embeddings)

def get_retriever(vector_store, k: int = 4):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})

def retrieve_context(retriever, question: str) -> str:
    docs = retriever.invoke(question)
    if isinstance(docs, list):
        return "\n".join(doc.page_content for doc in docs)
    return docs.page_content

def build_prompt(context: str, question: str, history: list) -> str:
    history_text = ""
    for turn in history:
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    prompt = PromptTemplate(
        template=f"""
        🤖 You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {history_text}
        Context:
        {context}
        Question: {question}
        """,
        input_variables=[]
    )
    return prompt.format()

def get_llama_answer(prompt: str) -> str:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        timeout=None,
        max_retries=2,
    )
    response = llm.invoke(prompt)
    return response.content

def get_thumbnail_url(url: str) -> str:
    video_id = extract_video_id(url)
    if video_id:
        return f"https://img.youtube.com/vi/{video_id}/0.jpg"
    return ""

def gradio_qa(url, question, history=[]):
    video_id = extract_video_id(url)
    transcript = fetch_transcript(video_id)
    if not transcript:
        return "❌ Transcript not found or unavailable.", history
    chunks = split_transcript(transcript)
    vector_store = create_vector_store(chunks)
    retriever = get_retriever(vector_store)
    context_text = retrieve_context(retriever, question)
    prompt = build_prompt(context_text, question, history)
    answer = get_llama_answer(prompt)
    history.append({"user": question, "assistant": answer})
    return answer, history

def update_thumbnail(url):
    thumb_url = get_thumbnail_url(url)
    if thumb_url:
        return thumb_url
    return None

with gr.Blocks() as demo:
    gr.Markdown("## 🎬 YouTube Transcript QA 🤖\nPaste a YouTube URL and ask your question below!")

    with gr.Row():
        url_box = gr.Textbox(label="🔗 YouTube Video URL", placeholder="Paste the full YouTube video URL here")
        thumbnail = gr.Image(label="Video Thumbnail", interactive=False, show_label=True, width=180, height=120)
    question_box = gr.Textbox(label="❓ Your Question", placeholder="Ask a question about the video transcript")
    submit_btn = gr.Button("🚀 Ask")
    output = gr.Textbox(label="🤖 Assistant's Answer")
    state = gr.State([])

    url_box.change(fn=update_thumbnail, inputs=url_box, outputs=thumbnail)
    submit_btn.click(
        fn=gradio_qa,
        inputs=[url_box, question_box, state],
        outputs=[output, state]
    )

demo.launch()