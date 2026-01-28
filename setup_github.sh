#!/bin/bash
# GitHub Setup Script for Render Deployment

echo "🚀 Stock Analysis API - GitHub Setup"
echo "===================================="
echo ""

# Prompt for GitHub username
read -p "Enter your GitHub username: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo "❌ GitHub username is required"
    exit 1
fi

echo ""
echo "📝 Setting up GitHub repository..."
echo ""

# Add remote
git remote add origin "https://github.com/$GITHUB_USERNAME/stock-analysis-api.git"

echo "✅ Remote added: https://github.com/$GITHUB_USERNAME/stock-analysis-api.git"
echo ""
echo "📤 Pushing code to GitHub..."
echo ""

# Push to GitHub
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Code successfully pushed to GitHub!"
    echo ""
    echo "📋 Next Steps:"
    echo "   1. Go to: https://render.com/"
    echo "   2. Sign in with GitHub"
    echo "   3. Click 'New +' → 'Web Service'"
    echo "   4. Connect repository: stock-analysis-api"
    echo "   5. Configure:"
    echo "      - Name: stock-analysis-api"
    echo "      - Environment: Python 3"
    echo "      - Build Command: pip install -r requirements.txt"
    echo "      - Start Command: python api_server.py"
    echo "      - Instance Type: Free"
    echo "   6. Click 'Create Web Service'"
    echo ""
    echo "🌐 After deployment, you'll get a URL like:"
    echo "   https://stock-analysis-api.onrender.com"
    echo ""
    echo "📱 Update your mobile app config.js with that URL!"
else
    echo ""
    echo "❌ Push failed. Please check:"
    echo "   1. Repository exists at https://github.com/$GITHUB_USERNAME/stock-analysis-api"
    echo "   2. You have push access"
    echo "   3. You're logged into GitHub (run: gh auth login)"
fi
