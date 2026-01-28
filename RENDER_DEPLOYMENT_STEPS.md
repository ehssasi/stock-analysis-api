# 🚀 Render Deployment - Step by Step

Your code is ready to deploy! Follow these steps:

---

## Step 1: Create GitHub Repository (2 minutes)

1. **Go to**: https://github.com/new

2. **Fill in details**:
   - Repository name: `stock-analysis-api`
   - Description: "Stock Analysis API for mobile app"
   - Privacy: **Private** (recommended) or Public
   - **DO NOT** check "Initialize with README" (you already have files)

3. **Click**: "Create repository"

---

## Step 2: Push Your Code to GitHub (1 minute)

Your code is already committed! Now push it:

### Option A: Using the script (Easiest)
```bash
cd "/mnt/c/Users/111806/OneDrive - Grundfos/AI Solutions/LearningCoding"
bash setup_github.sh
```

The script will ask for your GitHub username and handle everything.

### Option B: Manual commands
```bash
cd "/mnt/c/Users/111806/OneDrive - Grundfos/AI Solutions/LearningCoding"

# Replace YOUR_USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR_USERNAME/stock-analysis-api.git
git push -u origin main
```

If prompted for credentials:
- Username: Your GitHub username
- Password: Use a **Personal Access Token** (not your password)
  - Create token at: https://github.com/settings/tokens
  - Permissions needed: `repo` (full control)

---

## Step 3: Deploy on Render (3 minutes)

1. **Go to**: https://render.com/

2. **Sign up/Login**: Click "Get Started" and **sign in with GitHub**

3. **Create Web Service**:
   - Click **"New +"** button (top right)
   - Select **"Web Service"**

4. **Connect Repository**:
   - Click **"Connect a repository"**
   - If first time: Authorize Render to access your GitHub
   - Find and select: `stock-analysis-api`
   - Click **"Connect"**

5. **Configure Service**:
   ```
   Name: stock-analysis-api
   Region: Any (Oregon/Frankfurt/Singapore - choose closest to you)
   Branch: main
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: python api_server.py
   Instance Type: Free
   ```

6. **Click**: "Create Web Service"

7. **Wait for deployment**:
   - Render will install dependencies (1-2 minutes)
   - You'll see logs in real-time
   - Status will change from "Building" → "Live"

---

## Step 4: Get Your URL

Once deployed (status shows "Live"):

1. Copy your URL from the top of the page:
   ```
   https://stock-analysis-api.onrender.com
   ```
   (Your actual URL will have a unique identifier)

2. **Test it**:
   ```bash
   curl https://stock-analysis-api.onrender.com/health
   ```

   Should return:
   ```json
   {
     "status": "healthy",
     "timestamp": "2026-01-28T...",
     "version": "1.0.0"
   }
   ```

---

## Step 5: Update Mobile App

Edit `StockAnalysisApp/config.js`:

```javascript
// Replace this line:
export const API_BASE_URL = 'http://YOUR_COMPUTER_IP:5000';

// With your Render URL:
export const API_BASE_URL = 'https://stock-analysis-api.onrender.com';
```

Save the file.

---

## Step 6: Test Mobile App

```bash
cd StockAnalysisApp
npm start
```

Scan QR code with Expo Go app and test with any stock symbol!

Your app now works from **anywhere in the world**! 🌍

---

## ⚠️ Important Notes

### Free Tier Limitations
- App "sleeps" after 15 minutes of inactivity
- First request after sleep takes ~30 seconds (cold start)
- Subsequent requests are fast

### Keep It Awake (Optional)
To prevent sleeping:
1. Sign up at: https://uptimerobot.com/ (free)
2. Add monitor: Your Render URL + `/health`
3. Check interval: 5 minutes
4. Your app stays awake 24/7!

### Viewing Logs
On Render dashboard:
- Click your service name
- Go to "Logs" tab
- See real-time API requests and errors

---

## 🐛 Troubleshooting

### Push to GitHub fails
- **Check**: Repository exists at `https://github.com/YOUR_USERNAME/stock-analysis-api`
- **Check**: You're logged into GitHub
- **Try**: Create a Personal Access Token at https://github.com/settings/tokens

### Build fails on Render
- **Check**: `requirements.txt` is correct (already verified ✓)
- **Check**: Build logs for specific error
- **Usually**: Render will auto-detect Python and use correct version

### Mobile app can't connect
- **Check**: URL in `config.js` is correct (must be HTTPS)
- **Check**: No trailing slash in URL
- **Wait**: First request may take 30-60 seconds (cold start)

### "Application Error" on Render
- **Check**: Logs tab on Render dashboard
- **Usually**: First request loads data, takes 20-30 seconds
- **Retry**: Refresh after 30 seconds

---

## ✅ Success Checklist

- [ ] GitHub repository created
- [ ] Code pushed to GitHub
- [ ] Render web service created
- [ ] Service status shows "Live"
- [ ] Health endpoint works: `curl https://your-url/health`
- [ ] `config.js` updated with Render URL
- [ ] Mobile app tested with Render backend
- [ ] App works from anywhere (test on cellular data!)

---

## 🎉 You're Done!

Your Stock Analysis API is now:
- ✅ Deployed to cloud
- ✅ Accessible from anywhere
- ✅ Running 24/7 (free tier)
- ✅ No IP configuration needed
- ✅ Professional hosting

**Total time**: ~5-10 minutes

**Cost**: $0/month (Free tier: 750 hours/month)

---

## 📝 Next Steps (Optional)

1. **Custom Domain**: Add your own domain in Render settings
2. **Environment Variables**: Add API keys if needed (Render → Environment)
3. **Upgrade**: Paid plan ($7/mo) for always-on, no cold starts
4. **Monitoring**: Set up UptimeRobot for 24/7 uptime

---

Need help? Check Render docs: https://render.com/docs
