# Backend - Optimized RAG System

This directory contains the production-ready RAG (Retrieval-Augmented Generation) system with conversation context and insight-driven graph generation.

## 📁 File Structure

```
backend/
├── optimized_rag.py      # Main RAG system with Flask API
├── insight_graphs.py     # Graph generation system
└── graphs/              # Generated visualization files (auto-created)
```

## 🚀 Main Components

### `optimized_rag.py`
- **Primary RAG system** with precomputed embeddings for fast search
- **Conversation context** - maintains chat history across interactions
- **Graph integration** - automatically generates visualizations
- **Flask API endpoints** for frontend communication
- **RoBERTa large model** (1024D) for high-quality embeddings

### `insight_graphs.py`  
- **InsightGraphGenerator class** for creating data visualizations
- **5 graph types**: sentiment evolution, platform analysis, heatmaps, correlations, content analysis
- **Base64 image encoding** for easy frontend display
- **Intelligent insights** extracted from social media data

### `graphs/`
- **Auto-generated directory** for storing graph files
- **Automatic cleanup** - files managed by the system
- **Base64 encoded images** returned in API responses

## 🛠️ Key Features

✅ **Fast Search**: Precomputed 1024D embeddings for instant retrieval  
✅ **Conversation Memory**: Maintains context across multiple queries  
✅ **Auto-Graphs**: Generates insights when ≥3 results found  
✅ **Multi-Platform**: Reddit + YouTube social media data  
✅ **Smart Filtering**: Relevance thresholds and quality controls  
✅ **Error Handling**: Graceful fallbacks and robust error management  

## 🔧 Dependencies

Core packages installed in virtual environment:
- `flask` - Web API framework
- `sentence-transformers` - RoBERTa large embedding model  
- `matplotlib` + `seaborn` - Graph generation
- `pandas` + `numpy` - Data processing
- `google-generativeai` - Gemini AI integration
- `scikit-learn` - Similarity calculations

## 🎯 Usage

The system automatically loads preprocessed embeddings on startup and provides:
- `/api/chat` - Main conversation endpoint with graph generation
- `/api/conversation/clear` - Reset conversation history  
- `/api/conversation/history` - View conversation state

Graphs generate automatically when queries return sufficient relevant results (minimum 3 social media posts/comments).