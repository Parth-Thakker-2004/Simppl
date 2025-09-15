# RAG Conversational Agent - Setup & Run Instructions

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Git** (if you want to clone/manage the repository)
3. **Internet connection** for downloading models and accessing Gemini AI

## Installation Steps

### 1. Install Python Dependencies

Open Command Prompt in the project root directory and run:

```cmd
pip install flask flask-cors sentence-transformers google-generativeai scikit-learn beautifulsoup4 requests numpy pandas
```

### 2. Verify Data Files

Ensure you have the following preprocessed JSON files in the `data/` directory:
- `reddit_joined_data.json` (Reddit posts with comments)
- `youtube_joined_data.json` (YouTube posts with comments)

If you don't have these files, you need to run the data preprocessing first with your CSV files.

### 3. API Key Configuration

The Gemini AI API key is already configured in the code:
- API Key: `AIzaSyCvWysEM6_RGTf3jPtlZLEAMPbaZjUBjwY`

## Running the Application

### Step 1: Start the Backend Server

1. Open Command Prompt in the project root directory
2. Navigate to the backend folder:
   ```cmd
   cd backend
   ```
3. Start the Flask server:
   ```cmd
   python simple_rag.py
   ```

You should see output like:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

**Important**: The first time you run this, it will:
- Load and process your social media data
- Generate embeddings (this may take 2-5 minutes)
- Cache the embeddings for future use

### Step 2: Open the Frontend

1. Open a web browser
2. Navigate to the `frontend` folder in your file explorer
3. Double-click on `index.html` to open it in your browser

Alternatively, you can open `index.html` directly in your browser by entering the file path.

### Step 3: Initialize the System

1. When the frontend loads, it will automatically attempt to initialize the RAG system
2. You'll see a loading message while the system initializes
3. Once ready, you can start chatting with the AI agent

## System Features

### Smart Caching
- **No Re-initialization**: The system remembers your data and embeddings
- **Automatic Detection**: Only rebuilds if data files change
- **Fast Startup**: Subsequent runs load from cache (< 30 seconds)

### Conversational AI
- **Context-Aware**: Uses your social media data for relevant responses
- **Online Sources**: Fetches current news and information
- **Source Attribution**: Shows which posts/comments informed the response

### User Interface
- **Claude-like Interface**: Clean, modern chat experience
- **Dark/Light Themes**: Toggle between themes
- **Real-time Responses**: Streaming-like message display
- **Source Links**: Click to see original social media sources

## Usage Tips

### Asking Questions
- Ask about trends in your social media data
- Request analysis of user sentiment or engagement
- Inquire about specific topics mentioned in posts/comments
- Ask for current news related to topics in your data

### Example Questions
- "What are the main topics people are discussing?"
- "Show me posts with high engagement about technology"
- "What's the sentiment around [specific topic]?"
- "Find recent news about topics mentioned in the comments"

### Troubleshooting

**Backend won't start:**
- Check if all dependencies are installed: `pip list`
- Verify Python version: `python --version`
- Ensure data files exist in the `data/` folder

**Frontend not connecting:**
- Make sure backend is running on http://127.0.0.1:5000
- Check browser console for error messages
- Try refreshing the page

**Slow initial startup:**
- This is normal for first-time setup (embedding generation)
- Subsequent runs will be much faster due to caching
- Don't interrupt the process during initialization

**Memory issues:**
- The system is optimized for large datasets
- If you encounter memory errors, restart the backend
- Cache files help reduce memory usage on restarts

## File Structure

```
Simppl/
├── backend/
│   └── simple_rag.py          # Main backend server
├── data/
│   ├── reddit_joined_data.json # Reddit data
│   ├── youtube_joined_data.json # YouTube data
│   └── rag_embeddings.pkl     # Cached embeddings (auto-generated)
├── frontend/
│   ├── index.html             # Main UI
│   ├── style.css              # Styling and themes
│   └── script.js              # Frontend logic
└── SETUP_INSTRUCTIONS.md     # This file
```

## Performance Notes

- **First Run**: 2-5 minutes (embedding generation)
- **Subsequent Runs**: 10-30 seconds (loads from cache)
- **Query Response**: 2-10 seconds (depends on complexity)
- **Memory Usage**: ~500MB-1GB (varies with data size)

## Support

If you encounter issues:
1. Check that all dependencies are installed
2. Verify data files are in the correct location
3. Ensure internet connectivity for Gemini AI and news sources
4. Restart both backend and frontend if needed

The system is designed to be robust and handle various data sizes efficiently while avoiding unnecessary re-initialization of the RAG pipeline.