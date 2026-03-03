#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — Run this once to scaffold and initialise everything
# Usage: bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit on error

echo "================================================"
echo "  Question Classifier — Project Setup"
echo "================================================"

# ── 1. Create directory structure ────────────────────────────────────────────
echo ""
echo "📁 Creating project structure..."

mkdir -p backend/src
mkdir -p backend/models       # ← place your .pt file here
mkdir -p backend/data         # ← auto-populated on first API start
mkdir -p frontend/src
mkdir -p notebooks

echo "✅ Directory structure created"

# ── 2. Backend virtual environment ───────────────────────────────────────────
echo ""
echo "🐍 Setting up Python virtual environment..."

cd backend
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "✅ Backend environment ready"
cd ..

# ── 3. Frontend dependencies ─────────────────────────────────────────────────
echo ""
echo "⚛️  Installing frontend dependencies..."

cd frontend
npm install --silent

# Create .env from example if not present
if [ ! -f .env ]; then
  cp .env.example .env
  echo "✅ Created frontend/.env from .env.example"
fi

cd ..

# ── 4. Git initialisation ─────────────────────────────────────────────────────
echo ""
echo "🔧 Initialising git repository..."

git init
git add .
git commit -m "Initial commit: Question Classifier full-stack app"

echo "✅ Git repository initialised"

# ── 5. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo "  ✅ Setup complete!"
echo "================================================"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Copy your model weights:"
echo "     cp /path/to/q4_prefix_tuning_best.pt backend/models/"
echo ""
echo "  2. Start the backend:"
echo "     cd backend && source venv/bin/activate"
echo "     uvicorn app:app --reload --port 8000"
echo ""
echo "  3. Start the frontend (new terminal):"
echo "     cd frontend && npm run dev"
echo ""
echo "  4. Push to GitHub:"
echo "     git remote add origin https://github.com/YOUR_USERNAME/question-classifier.git"
echo "     git push -u origin main"
echo ""
echo "  API docs: http://localhost:8000/docs"
echo "  Frontend: http://localhost:5173"
echo "================================================"
