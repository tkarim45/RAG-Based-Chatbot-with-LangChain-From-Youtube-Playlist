# RAG-Based Chatbot with LangChain from YouTube Playlist

This project implements a Retrieval-Augmented Generation (RAG) based chatbot using LangChain, designed to answer questions about neural networks and machine learning concepts from educational YouTube content.

## 🎯 Project Overview

The chatbot combines the power of large language models with retrieval-based context augmentation to provide accurate, context-aware responses. It processes YouTube video transcripts and uses them as a knowledge base to answer user queries.

### Key Features

- YouTube playlist content processing
- Audio transcription using Whisper
- Semantic search capabilities
- RAG-based response generation
- Source attribution for answers

## 🛠️ Technical Architecture

The system consists of several key components:

1. **Data Collection**

   - Downloads audio from YouTube videos
   - Transcribes audio using Whisper model
   - Processes transcripts into structured documents

2. **Document Processing**

   - Splits documents into manageable chunks
   - Maintains metadata (video source, title)
   - Creates embeddings for semantic search

3. **Vector Storage**

   - Uses Chroma as the vector database
   - Stores document embeddings for efficient retrieval
   - Enables semantic similarity search

4. **RAG Pipeline**
   - Retrieves relevant context based on user queries
   - Generates responses using LLM (Mistral-7B)
   - Provides source attribution for answers

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- GPU support (recommended) for faster processing

### Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd RAG-Based-Chatbot-with-LangChain-From-Youtube-Playlist
```

2. Install required packages:

```bash
pip install langchain openai chromadb tiktoken sentence_transformers langchainhub langchain-community tqdm yt-dlp whisper --upgrade openai-whisper
```

3. Set up environment variables:

```bash
export HUGGINGFACE_API_TOKEN="your-huggingface-token"
```

### Usage

1. **Data Preparation**

   - The system will process YouTube videos from a specified playlist
   - Transcripts are generated and stored automatically

2. **Running the Chatbot**
   - Open the Jupyter notebook
   - Follow the step-by-step implementation
   - Ask questions about neural networks and machine learning

## 🔧 Components

### 1. Document Processing

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=200,
    add_start_index=True
)
```

### 2. Embedding Generation

```python
bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)
```

### 3. Vector Store

```python
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=bge_embeddings
)
```

### 4. RAG Chain

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

## 📚 Example Usage

```python
# Ask a question
response = rag_chain.invoke("What is Gradient Descent?")

# Get response with sources
response = rag_chain_with_source.invoke("How does backpropagation work?")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Thanks to the LangChain community
- Special thanks to 3Blue1Brown for the educational content used in this project
