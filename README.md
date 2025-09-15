# MediaMeld Journalism Media Analysis Platform

This guide will help you run all components of the MediaMeld platform.

## Project Purpose and Overview

MediaMeld is a specialized tool designed for journalists and media analysts to gain comprehensive insights across both traditional news sources and social media platforms. Using AI-powered analysis, it allows users to:

1. **Cross-platform media analysis**: Compare how stories are covered in professional journalism versus social media discussions
2. **Detect emerging narratives**: Identify trending topics and audience sentiment across different platforms
3. **Source verification**: Trace information to its source across multiple platforms
4. **Generate research-quality reports**: Export professional, citation-rich PDFs suitable for academic or journalistic use

The platform leverages Retrieval-Augmented Generation (RAG) technology to provide context-aware analysis by pulling relevant information from a database of news articles and social media posts.

### System Architecture

MediaMeld consists of two main components:

1. **Python Backend (RAG Service)** – Loads, embeds, and retrieves data from CSV files
2. **Vanilla JavaScript Frontend** – User interface for journalistic analysis and PDF export

## Setup Instructions

### 1. Prepare the Data Folder

1. Create a folder named `data` in the root of this project.
2. Add your input CSV files (such as Reddit and YouTube data) into the `data` folder.
   - Example: `data/comments Data Dump - Reddit.csv`, `data/posts Data Dump - Youtube.csv`, etc.
3. The application will use these CSV files as input for processing and analysis.

> **Note:** The `data` folder is excluded from version control via `.gitignore`. You must add your own CSV files locally.

### 2. Install Python Dependencies

```bash
cd backend
pip install flask sentence-transformers pandas numpy torch
```

### 3. Start the Python Backend

```bash
cd backend
python optimized_rag.py
```

This will start the backend service (by default on http://localhost:5000).

### 4. Open the Frontend

1. Open `frontend/index.html` in your web browser (double-click or use a local server if needed).
2. The MediaMeld interface allows you to:
   - Ask questions about social media and news data
   - View sources and data insights
   - Export research-style PDF reports

## Troubleshooting

- **Data not loading:** Ensure CSV files are in the correct `data` folder
- **Backend errors:** Check that all Python dependencies are installed and the backend is running
- **Frontend not working:** Make sure you are opening `frontend/index.html` in a modern browser

## Data Structure

The system expects CSV files in the `data` folder, such as:
- comments Data Dump - Reddit.csv
- comments Data Dump - Youtube.csv
- posts Data Dump - Reddit.csv
- posts Data Dump - Youtube.csv

You may use your own CSVs, but they should match the expected format for the backend to process them.

## Technical Implementation Details

### How It Works

1. **Data Processing**: The system ingests and processes social media posts and comments from Reddit and YouTube, along with news articles from various publishers.

2. **Semantic Embedding**: Using sentence-transformers, the content is converted into vector embeddings that represent the semantic meaning of each text chunk.

3. **Query Processing**: When a user asks a question, the system:
   - Converts the query to a vector embedding
   - Retrieves the most semantically relevant content from the database
   - Uses a Gemini Large Language Model to generate a comprehensive answer
   - Cites all sources used to form the response

4. **Source Attribution**: All information is presented with proper citation, including direct links to the original content where available.

5. **PDF Generation**: MediaMeld can produce research-quality PDF documents with proper academic formatting, including:
   - Title page
   - Abstract
   - Introduction
   - Methodology
   - Results
   - Discussion
   - Conclusion
   - References

### Use Cases

- **Journalism Research**: Compare public discourse with traditional media narratives
- **Media Analysis**: Track how stories evolve across platforms
- **Academic Study**: Research digital communication patterns with proper citation
- **Content Creation**: Develop well-sourced articles based on cross-platform analysis
- **Trend Monitoring**: Identify emerging topics and public sentiment

### Why This Matters

This tool bridges the gap between traditional journalism and social media discourse, providing a comprehensive view of information ecosystems that is increasingly crucial in today's fragmented media landscape. By combining automated analysis with proper source attribution, it helps maintain journalistic integrity while expanding the scope of research beyond what would be manually feasible.