# Simppl Journalism Media Analysis Platform

This guide will help you run all components of the Simppl platform.

## Overview

Simppl consists of two main components:

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
python simple_rag.py
```

This will start the backend service (by default on http://localhost:5000).

### 4. Open the Frontend

1. Open `frontend/index.html` in your web browser (double-click or use a local server if needed).
2. The interface allows you to:
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