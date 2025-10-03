# 🚀 Railway Deployment Guide - Brain Stroke Prediction System

This guide will walk you through deploying your Brain Stroke Risk Prediction System to Railway with SQLite database.

## 📋 Prerequisites

Before deploying, ensure you have:
- ✅ GitHub account with your project repository
- ✅ Railway account (free tier available)
- ✅ All code committed and pushed to GitHub

## 💤 Railway Sleep/Shutdown Policy

**Important:** Railway free tier has the following limitations:
- **Sleep after inactivity**: Apps sleep after 30 minutes of no requests
- **Monthly usage limit**: 500 hours per month on free tier
- **Auto-wake**: Apps automatically wake up when accessed
- **Data persistence**: SQLite data survives sleep cycles but NOT redeployments

For college project demonstrations, this is perfect since:
- ✅ App wakes up instantly when you access it
- ✅ Great for presentations and demos
- ✅ Zero cost for typical usage patterns
- ✅ Data persists during sleep periods

## 🎯 Quick Deployment (5 Minutes)

### Step 1: Create Railway Account
1. Go to [Railway.app](https://railway.app)
2. Sign up with your GitHub account
3. Authorize Railway to access your repositories

### Step 2: Deploy Your Project
1. Click **"New Project"** in Railway dashboard
2. Select **"Deploy from GitHub repo"**
3. Choose your `brain-stroke-prediction` repository
4. Railway will automatically detect your project

### Step 3: Configure SQLite (No Database Setup Needed!)
1. Your app is already configured to use SQLite automatically
2. No additional database service required
3. Demo data will be created automatically on first run
4. Perfect for college projects and demonstrations

### Step 4: Configure Environment Variables
In your Railway project settings, add these environment variables:

```bash
FLASK_ENV=production
SECRET_KEY=your-super-secure-secret-key-change-this
JWT_SECRET_KEY=your-jwt-secret-key-change-this
PYTHONPATH=/app/backend:/app/ml-model
FORCE_SQLITE=true
```

### Step 5: Deploy & Test
1. Railway automatically deploys your code
2. Wait for deployment to complete (2-3 minutes)
3. Test your deployment at the generated Railway URL

## 🔧 Detailed Setup Instructions

### Database Configuration

Your app uses SQLite automatically with these benefits:

1. **No Setup Required**: SQLite file is created automatically
2. **Demo Data**: Sample data is seeded on first run
3. **Fast Performance**: Perfect for demonstrations
4. **Zero Cost**: No database service charges
5. **Instant Login**: Use demo@strokeprediction.com / demo123

### Environment Variables Explained

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `FLASK_ENV` | Flask environment | Yes | `production` |
| `SECRET_KEY` | Flask secret key | Yes | `your-secret-key-here` |
| `JWT_SECRET_KEY` | JWT signing key | Yes | `your-jwt-secret-here` |
| `PYTHONPATH` | Python module search path | Yes | `/app/backend:/app/ml-model` |
| `FORCE_SQLITE` | Force SQLite usage | Yes | `true` |
| `PORT` | Application port | Auto-set | `5000` |

### Custom Domain (Optional)

To use a custom domain:

1. In Railway project settings, go to **"Domains"**
2. Click **"Add Domain"**
3. Enter your domain name
4. Configure your DNS provider to point to Railway

## 🌐 Frontend Deployment Options

### Option 1: Vercel (Recommended)
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy frontend
cd frontend
vercel --prod
```

### Option 2: Netlify
1. Connect your GitHub repository to Netlify
2. Set build command: `npm run build`
3. Set publish directory: `build`
4. Deploy automatically on git push

### Option 3: Railway (Same Platform)
You can also deploy your frontend on Railway:
1. Create a new Railway service for frontend
2. Configure build command: `cd frontend && npm run build`
3. Configure start command: `cd frontend && npm start`

## 🔄 Environment Variables for Frontend

Update your frontend's API endpoint:

```javascript
// In your React app, update API base URL
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-railway-app-url.railway.app'
  : 'http://localhost:5000';
```

Add to your frontend `.env` file:
```bash
REACT_APP_API_URL=https://your-railway-app-url.railway.app
```

## 🔍 Testing Your Deployment

### 1. Health Check
```bash
curl https://your-app-url.railway.app/
```

Expected response:
```json
{
  "status": "healthy",
  "service": "Brain Stroke Risk Prediction API",
  "version": "2.0.0",
  "database": "connected",
  "ml_service": "available",
  "environment": "production"
}
```

### 2. API Information
```bash
curl https://your-app-url.railway.app/api/info
```

### 3. User Registration Test
```bash
curl -X POST https://your-app-url.railway.app/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test User",
    "email": "test@example.com",
    "password": "testpass123"
  }'
```

### 4. Database Check
Your app automatically creates demo data on first run:
- **Email**: `demo@strokeprediction.com`
- **Password**: `demo123`
- **Sample Predictions**: 2-3 example predictions pre-loaded
- **Instant Demo**: Perfect for presentations

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. Build Fails
**Error**: Python dependencies not found
**Solution**: Ensure `requirements.txt` is in the backend folder
```bash
# Check if requirements.txt exists
ls backend/requirements.txt

# If missing, create it
cd backend
pip freeze > requirements.txt
```

#### 2. SQLite Database Error
**Error**: `database is locked` or `no such table`
**Solution**: Check file permissions and initialization
```bash
# SQLite file is created automatically on first run
# Demo data is seeded during startup
```

#### 3. Module Import Error
**Error**: `ModuleNotFoundError: No module named 'ml_service'`
**Solution**: Add PYTHONPATH environment variable
```bash
PYTHONPATH=/app/backend:/app/ml-model
```

#### 4. Port Binding Error
**Error**: `Port already in use`
**Solution**: Use Railway's PORT environment variable
```python
# In your app.py
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)
```

### Deployment Logs

To view deployment logs:
1. Go to your Railway project dashboard
2. Click on your service
3. Go to **"Deployments"** tab
4. Click on latest deployment to view logs

### Database Management

SQLite database management:
1. **Local Development**: Use SQLite browser or CLI tools
2. **Production**: Database resets on redeployment (expected behavior)
3. **Demo Data**: Automatically recreated on each deployment
4. **Backup**: Not needed for demo purposes

## 📊 Database Initialization

Your app automatically initializes the database on first run. To manually initialize:

```python
# SSH into Railway container (if needed)
python backend/init_db.py init
python backend/init_db.py seed
```

## 🔒 Security Considerations

### Production Security Checklist

- ✅ **Environment Variables**: Never commit secrets to git
- ✅ **HTTPS**: Railway provides SSL certificates automatically  
- ✅ **Database**: SQLite is file-based and secure
- ✅ **CORS**: Configure allowed origins for your frontend
- ✅ **Rate Limiting**: Consider adding rate limiting for production
- ⚠️ **Data Persistence**: Remember data resets on redeployment

### Security Environment Variables
```bash
# Strong secret keys (generate with: python -c "import secrets; print(secrets.token_urlsafe(32))")
SECRET_KEY=your-randomly-generated-secret-key
JWT_SECRET_KEY=your-randomly-generated-jwt-key

# CORS configuration
CORS_ORIGINS=https://your-frontend-domain.com,https://your-app.vercel.app
```

## 💰 Cost Estimates

### Railway Pricing (as of 2024)
- **Free Tier**: 500 hours per month + $5 credit
- **SQLite**: $0 (no database service charges)
- **Perfect for College Projects**: Completely free for typical usage
- **Sleep Mode**: App sleeps after 30min inactivity (free tier)

### Cost Optimization
- ✅ **Zero Database Cost**: SQLite is free
- ✅ **Efficient Sleep**: Only uses hours when active
- ✅ **College Friendly**: Perfect for presentations and demos
- ✅ **Auto Wake**: Instant wake-up when accessed

## 📈 Monitoring & Maintenance

### Built-in Monitoring
Railway provides:
- **Deployment History**: Track all deployments
- **Metrics**: CPU, memory, and network usage
- **Logs**: Real-time application logs
- **Health Checks**: Automatic uptime monitoring

### Custom Health Checks
Your app includes health check endpoints:
- `/` - General health status
- `/api/info` - API and database status

## 🚀 Advanced Configuration

### Custom Build Commands
Create `railway.json` for advanced configuration:
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "cd backend && python app.py",
    "healthcheckPath": "/",
    "healthcheckTimeout": 300
  }
}
```

### Database Backups
Railway Pro includes automatic database backups:
- Daily automatic backups
- Point-in-time recovery
- Manual backup creation

## 🎯 Going Live Checklist

Before making your app public:

- [ ] Test all API endpoints
- [ ] Verify SQLite database creation
- [ ] Test user registration/login
- [ ] Test ML prediction functionality
- [ ] Check frontend-backend integration
- [ ] Verify environment variables
- [ ] Test with demo credentials (demo@strokeprediction.com/demo123)
- [ ] Test app wake-up from sleep
- [ ] Verify demo data creation
- [ ] Test on mobile devices

## 📞 Support Resources

### Railway Support
- **Documentation**: [docs.railway.app](https://docs.railway.app)
- **Discord**: Railway Community Discord
- **GitHub**: Issues and discussions

### Project Support
- **GitHub Issues**: Report bugs in your repository
- **Documentation**: This deployment guide
- **Local Testing**: Always test locally before deploying

---

## 🎉 Congratulations!

Your Brain Stroke Risk Prediction System is now deployed to Railway with PostgreSQL! 

**Your Live URLs:**
- **API**: `https://your-app-name.railway.app`
- **Health Check**: `https://your-app-name.railway.app/`
- **API Docs**: `https://your-app-name.railway.app/api/info`

**Demo Credentials (Auto-created):**
- **Email**: `demo@strokeprediction.com`
- **Password**: `demo123`
- **Features**: Pre-loaded with sample predictions for instant demo

**Sleep Behavior:**
- **Sleeps**: After 30 minutes of inactivity
- **Wakes**: Instantly when you access the URL
- **Data**: Survives sleep, resets on redeployment
- **Perfect**: For college presentations! 🎓✨