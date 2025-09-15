# Simppl Social Media Analysis Platform

This guide will help you run all components of the Simppl platform.

## Overview

Simppl consists of three main components:

1. **Python RAG Service** - Backend for data loading, embedding, and retrieval
2. **Node.js Server** - API proxy and server for the React frontend
3. **React Frontend** - User interface for journalists

## Setup Instructions

### 1. Install Dependencies

#### Python RAG Service

```bash
# Navigate to the backend folder
cd backend

# Install Python dependencies
pip install flask sentence-transformers pandas numpy torch
```

#### Node.js Server

```bash
# Navigate to the root folder
cd ..

# Install Node.js dependencies
npm install
```

#### React Frontend

```bash
# Navigate to the frontend folder
cd frontend

# Install React dependencies
npm install
```

### 2. Start the Services

The services need to be started in the correct order:

#### Step 1: Start the Python RAG Service

```bash
# Navigate to the backend folder
cd backend

# Start the Python RAG service
python simple_rag.py
```

This will start the RAG service on http://localhost:8000. The service will automatically load data from the CSV files in the data folder.

#### Step 2: Start the Node.js Server

Open a new terminal window:

```bash
# Navigate to the root folder
cd <path-to-simppl>

# Start the Node.js server
npm start
```

This will start the Node.js server on http://localhost:3001.

#### Step 3: Start the React Frontend

Open a new terminal window:

```bash
# Navigate to the frontend folder
cd frontend

# Start the React development server
npm start
```

This will start the React development server on http://localhost:3000.

## Using the Application

1. Open your browser and navigate to http://localhost:3000
2. The interface allows journalists to:
   - Ask questions about social media data
   - View data visualizations based on the results
   - See related news articles
   - Export reports for publication

## Troubleshooting

If you encounter any issues:

1. **Data not loading**: Ensure CSV files are in the correct location (data folder)
2. **Connection errors**: Verify all three services are running
3. **Node.js server errors**: Check that the RAG service is running on port 8000
4. **UI not rendering charts**: Check browser console for errors

## Data Structure

The system expects four CSV files in the data folder:
- comments Data Dump - Reddit.csv
- comments Data Dump - Youtube.csv
- posts Data Dump - Reddit.csv
- posts Data Dump - Youtube.csv