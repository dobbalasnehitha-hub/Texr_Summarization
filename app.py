import streamlit as st
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
st.set_page_config(page_title="Text Summarizer",page_icon="📝",layout="wide")
MODELS={
    "DistilBART(faster,lighter)":
    "sshleifer/distilbart-cnn-12-6",
    "BART larger(better quality)":
    "facebook/bart-large-cnn",
}
LENGTHS={
    "short":(30,80),
    "Medium":(60,150),
    "long":(100,250),
}
CHUNK_WORDS=600
@st.cache_resource(show_spinner="loading model(first run downloads it)...")
def load_model(model_id:str):
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    model=AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer,model
def chunk_text(text:str,size:int=CHUNK_WORDS):
    words=text.split()
    return[" ".join(words[i:i+size])for i in range(0,len(words),size)]
def summarize_chunk(text,tokenizer,model,min_len,max_len):
    inputs=tokenizer(text,return_tensors="pt",
    truncation=True,max_length=1024)
    ids=model.generate(
    **inputs,
    min_length=min_len,
    max_length=max_len,
    num_beams=4,
    length_penalty=2.0,
    no_repeat_ngram_size=3,
    early_stopping=True,
    )
    return tokenizer.decode(ids[0],skip_special_tokens=True)
def summarize(text,tokenizer,model,min_len,max_len):
    chunks=chunk_text(text)
    progress=st.progress(0.0,text="summarizing...")
    parts=[]
    for i,chunk in enumerate(chunks,1):
        words=len(chunk.split())
        parts.append(summarize_chunk(chunk,tokenizer,
        model,min(min_len,words//2),max_len))
        progress.progress(i/len(chunks),text=f"summarized chunk {i}/{len(chunks)}")
    progress.empty()
    combined=" ".join(parts)
    if len(chunks)>1 and len(combined.split())> CHUNK_WORDS:
        combined=summarize_chunk(combines,
        tokenizer,
        model,
        min_len,
        max_len*2)
    return combined
def read_pdf(file):
    reader=PdfReader(file)
    return "\n".join(page.extract_text() or " " for page in reader.pages)
st.sidebar.header("settings")
model_name=st.sidebar.selectbox("model",list(MODELS.keys()))
length=st.sidebar.radio("summary length",list(LENGTHS.keys()),index=1)
st.title("📝 text summarizer")
st.write("paste text or upload a file(.txt/.pdf) to get a concise summary.")
source=st.radio("input type",["paste text","upload file"],horizontal=True)
text=""
if source=="paste text":
    text=st.text_area("your text",height=300,placeholder="paste an article,notes,or report here...")
else:
    file=st.file_uploader("upload.txt or .pdf",type=["txt","pdf"])
    if file:
        text=read_pdf(file) if file.name.lower().endswith(".pdf")else file.read().decode("utf-8",errors="ignore")
        with st.expander("preview extracted text"):
            st.write(text[:3000]+("..." if len(text)>3000 else""))
if st.button("summarize",type="primary"):
    word_count=len(text.split())
    if word_count<50:
        st.warning("please provide at least 50 words of text.")
    else:
        tokenizer,model=load_model(MODELS[model_name])
        min_len,max_len=LENGTHS[length]
        summary=summarize(text,tokenizer,model,min_len,max_len)
        col1,col2=st.columns(2)
        with col1:
            st.subheader("original")
            st.caption(f"{word_count}words")
            st.write(text[:5000]+("..." if len (text)>5000 else ""))
        with col2:
            st.subheader("summary")
            s_count=len(summary.split())
            st.caption(f"{s_count}words . {100-s_count*100 //word_count}%shorter")
            st.success(summary)
            st.download_button("⬇️ download summary",summary,"summary.txt","text/plain")
