#!/bin/bash
echo "=== Social Media RAG Setup ==="
echo ""
echo "To enable AI-powered analysis, you need a free Gemini API key:"
echo "1. Visit: https://ai.google.dev/"
echo "2. Click 'Get API Key' and sign in with Google"
echo "3. Create a new API key (free tier available)"
echo "4. Copy the key and paste it below"
echo ""
read -p "Enter your Gemini API Key (or press Enter to skip): " api_key

if [ ! -z "$api_key" ]; then
    echo "GEMINI_API_KEY=$api_key" > backend/.env
    echo "✅ API key saved to backend/.env"
else
    echo "⚠️  Skipping API key setup. System will run in offline mode."
fi

echo ""
echo "Starting services..."
echo "1. Python RAG Service (port 8000)"
echo "2. Node.js Backend (port 5000)" 
echo "3. React Frontend (port 3000)"
echo ""
echo "Once all services start, open: http://localhost:3000"
echo ""
