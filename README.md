
# Agentic RAG Chatbot with LangGraph

A sophisticated AI chatbot built using LangGraph framework with Retrieval Augmented Generation (RAG), featuring comprehensive evaluation metrics, session management, and production-ready deployment.

## 🚀 Features

### Core Functionality
- **Agentic RAG with LangGraph**: Intelligent routing between direct responses and knowledge base retrieval
- **Vector Database**: Chroma DB for efficient document storage and retrieval
- **Advanced Embeddings**: E5-large embeddings from Hugging Face for superior semantic understanding
- **LLM Integration**: ChatGPT 4o for high-quality responses
- **Session Management**: Multiple concurrent sessions with unique chat histories
- **Chat History**: Maintains top 5 recent messages per session in stack format

### UI & Deployment
- **Streamlit UI**: Interactive web interface with session switching and document management
- **FastAPI Backend**: RESTful API for scalable deployment
- **Docker Support**: Containerized deployment for easy scaling
- **SQLite Database**: Lightweight database for telemetry, feedback, and session storage

### Evaluation & Monitoring
- **DeepEval Integration**: Comprehensive evaluation using multiple metrics:
  - Answer Relevancy
  - Faithfulness
  - Context Precision
  - Context Recall
  - Context Relevancy
  - Hallucination Detection
  - BLEU Score
  - ROUGE Score
- **Telemetry Logging**: Complete transaction tracking with query, retrieved documents, prompts, and responses
- **Feedback System**: User feedback collection with transaction ID linking
- **Metrics Dashboard**: Real-time evaluation metrics monitoring

## 📋 Requirements

- Python 3.8+
- OpenAI API Key
- 8GB+ RAM recommended
- Docker (optional, for containerized deployment)

## 🛠️ Installation

### Method 1: Automatic Setup

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd agentic_chatbot
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Configure environment:**
   ```bash
   # Edit .env file and add your OpenAI API key
   nano .env
   ```

3. **Run the application:**
   ```bash
   # Terminal 1: Start FastAPI backend
   ./run_fastapi.sh

   # Terminal 2: Start Streamlit UI
   ./run_streamlit.sh
   ```

4. **Access the application:**
   - Streamlit UI: http://localhost:8501
   - FastAPI Docs: http://localhost:8000/docs

### Method 2: Docker Deployment

1. **Set environment variable:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

2. **Run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   - Streamlit UI: http://localhost:8501
   - FastAPI API: http://localhost:8000

### Method 3: Manual Installation

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('punkt')"
   ```

3. **Setup database:**
   ```bash
   python -c "from database.models import create_tables; create_tables()"
   ```

4. **Configure environment:**
   ```bash
   cp .env.template .env
   # Edit .env and add your OpenAI API key
   ```

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   FastAPI API   │────│  LangGraph Agent│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │  SQLite Database│    │   Chroma VectorDB│
                       └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │ DeepEval Metrics│
                       └─────────────────┘
```

### LangGraph Workflow

```
[User Query] → [Router Node] → [Decision]
                    │              │
                    ├─[Direct]─────→ [Direct Response] → [End]
                    │
                    └─[Retrieval]─→ [RAG Node] → [Generation] → [End]
```

### Database Schema

#### Tables:
- **telemetry_logs**: Transaction ID, query, retrieved documents, complete prompt, response
- **feedback_logs**: Transaction ID, query, LLM response, user feedback
- **chat_sessions**: Session management with unique IDs
- **chat_history**: Message history per session (top 5 maintained)
- **evaluation_metrics**: DeepEval metric scores per transaction

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Required
OPENAI_API_KEY=your-openai-api-key-here

# Optional (with defaults)
EMBEDDING_MODEL=intfloat/e5-large-v2
CHROMA_PERSIST_DIRECTORY=./chroma_db
DATABASE_URL=sqlite:///./chatbot.db
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_PORT=8501
```

## 📡 API Endpoints

### Chat Endpoints
- `POST /chat` - Send message to chatbot
- `POST /feedback` - Submit user feedback
- `GET /sessions` - List all chat sessions
- `GET /sessions/{session_id}/history` - Get chat history
- `POST /sessions` - Create new session
- `DELETE /sessions/{session_id}` - Delete session

### Knowledge Base
- `POST /documents` - Add document to vector store
- `GET /documents/count` - Get document count

### Evaluation
- `GET /evaluation/{transaction_id}` - Get evaluation metrics

## 🧪 Evaluation Metrics

The system automatically evaluates each response using DeepEval framework:

1. **Answer Relevancy** - How relevant is the response to the query
2. **Faithfulness** - How faithful is the response to retrieved context
3. **Context Precision** - Precision of retrieved context
4. **Context Recall** - Recall of retrieved context
5. **Context Relevancy** - Relevancy of retrieved context
6. **Hallucination** - Detection of hallucinated content
7. **BLEU Score** - Text similarity metric
8. **ROUGE Score** - Text overlap metric

## 🔄 Session Management

- **Multiple Sessions**: Create and switch between different chat sessions
- **Unique IDs**: Each session has a unique identifier
- **Chat History**: Maintains last 5 messages per session in stack format
- **Persistent Storage**: Sessions and history stored in SQLite database

## 📊 Monitoring & Telemetry

### Transaction Tracking
Each interaction generates a unique transaction ID that links:
- Original query
- Retrieved documents (top 5)
- Complete prompt sent to LLM
- Generated response
- Evaluation metrics
- User feedback (if provided)

### Feedback System
- Users can provide thumbs up/down feedback
- Feedback linked to specific transaction IDs
- Supports additional text feedback
- Stored for future model improvements

## 🚀 Production Deployment

### Docker Production
```bash
# Build production image
docker build -t agentic-chatbot .

# Run with environment variables
docker run -p 8000:8000 -p 8501:8501 \
  -e OPENAI_API_KEY=your-key \
  -v $(pwd)/data:/app/data \
  agentic-chatbot
```

### Cloud Deployment
The application is ready for deployment on:
- AWS ECS/EKS
- Google Cloud Run
- Azure Container Instances
- Railway
- Render
- Heroku

## 🔍 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   ```bash
   # Verify your API key is set
   echo $OPENAI_API_KEY
   ```

2. **Port Already in Use**
   ```bash
   # Kill processes on ports 8000/8501
   lsof -ti:8000 | xargs kill -9
   lsof -ti:8501 | xargs kill -9
   ```

3. **Database Connection Error**
   ```bash
   # Recreate database
   rm chatbot.db
   python -c "from database.models import create_tables; create_tables()"
   ```

4. **Chroma DB Issues**
   ```bash
   # Reset vector database
   rm -rf chroma_db
   ```

## 📝 Usage Examples

### Adding Documents
```python
import requests

# Add a document to the knowledge base
response = requests.post("http://localhost:8000/documents", 
    json={"content": "Your document content here"})
```

### Querying the Chatbot
```python
import requests

# Send a message
response = requests.post("http://localhost:8000/chat",
    json={"message": "What is artificial intelligence?"})
print(response.json())
```

### Getting Evaluation Metrics
```python
import requests

# Get metrics for a transaction
response = requests.get(f"http://localhost:8000/evaluation/{transaction_id}")
print(response.json())
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangGraph** - For the agent framework
- **OpenAI** - For GPT-4o language model
- **Chroma** - For vector database
- **Hugging Face** - For E5-large embeddings
- **DeepEval** - For evaluation metrics
- **Streamlit** - For the user interface
- **FastAPI** - For the backend API

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation at `/docs`

---

Built with ❤️ using LangGraph, OpenAI, Chroma DB, and DeepEval
