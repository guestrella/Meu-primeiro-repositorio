# Meu-primeiro-repositorio
Professor Denilson
from rag import criar_base_vetorial, responder_pergunta

st.set_page_config(
    page_title="Chatbot RAG - Manual da Panela",
    page_icon="🍲"
)

st.title("🍲 Chatbot do Manual da Panela")

st.write(
    "Faça perguntas sobre o manual da panela. "
    "O sistema responderá apenas com informações presentes no PDF."
)

if "vectorstore" not in st.session_state:
    with st.spinner("Carregando manual e criando índice..."):
        st.session_state.vectorstore = criar_base_vetorial(
            "data/manual_panela.pdf"
        )

pergunta = st.text_input("Digite sua pergunta:")

if st.button("Perguntar"):

    if pergunta.strip() == "":
        st.warning("Digite uma pergunta.")
    else:
        with st.spinner("Buscando resposta..."):
            resposta = responder_pergunta(
                pergunta,
                st.session_state.vectorstore
            )

        st.subheader("Resposta")
        st.write(resposta)
        
SyntaxError: multiple statements found while compiling a single statement
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def criar_base_vetorial(pdf_path):

    loader = PyPDFLoader(pdf_path)
    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documentos)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


def responder_pergunta(pergunta, vectorstore):

    documentos = vectorstore.similarity_search(
        pergunta,
        k=3
    )

    contexto = "\n\n".join(
        [doc.page_content for doc in documentos]
    )

    prompt = f"""
Você é um assistente especializado em responder perguntas
sobre um manual de panela elétrica.

IMPORTANTE:
- Responda APENAS usando as informações do contexto.
- Não invente informações.
- Se a resposta não estiver no contexto, responda exatamente:

"Não encontrei essa informação no manual."

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta:
"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )

    resposta = llm.invoke(prompt)

    return resposta.content
SyntaxError: multiple statements found while compiling a single statement
streamlit
langchain
langchain-community
langchain-google-genai
faiss-cpu
sentence-transformers
pypdf
python-dotenv
SyntaxError: multiple statements found while compiling a single statement
venv/
.env

__pycache__/
*.pyc

.streamlit/

.faiss/
