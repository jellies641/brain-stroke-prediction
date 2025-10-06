# Brain Stroke Prediction - Frontend

🎨 **React Frontend for Brain Stroke Risk Prediction System**

## Overview

This is the React frontend for the Brain Stroke Risk Prediction application. It provides a modern, responsive user interface for stroke risk assessment powered by machine learning.

## Features

- 🎯 **Modern UI**: Built with Material-UI (MUI) components
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- 🔐 **Authentication**: User registration, login, and profile management
- 📊 **Interactive Dashboard**: Visualize prediction history and statistics
- 🤖 **ML Integration**: Real-time stroke risk predictions
- 📈 **Data Visualization**: Charts and graphs using Recharts
- 🎨 **Beautiful Forms**: Form validation with Formik and Yup
- 🔔 **Notifications**: Toast notifications for user feedback

## Technology Stack

- **React** 18.2.0
- **Material-UI (MUI)** 5.15.15
- **React Router** 6.8.1
- **Axios** for API calls
- **Formik & Yup** for form handling
- **Recharts** for data visualization
- **React Toastify** for notifications

## Railway Deployment Guide

### 1. Prerequisites

- Railway account ([railway.app](https://railway.app))
- Backend API deployed (https://web-production-52e4b.up.railway.app)

### 2. Deploy to Railway

#### Option A: Direct Deployment (Recommended)

1. **Fork/Clone this repository**
   ```bash
   git clone https://github.com/your-username/brain-stroke-prediction.git
   ```

2. **Go to Railway Dashboard**
   - Visit [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"

3. **Select Repository**
   - Choose your forked repository
   - **IMPORTANT**: Set root directory to `/frontend`

4. **Configure Environment Variables**
   - Go to Variables tab in Railway
   - Add this environment variable:
     ```
     REACT_APP_API_URL=https://web-production-52e4b.up.railway.app
     ```

5. **Deploy**
   - Railway will automatically detect this as a React app
   - Build process: `npm install` → `npm run build`
   - Serve process: Static file serving

#### Option B: Railway CLI

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login to Railway**
   ```bash
   railway login
   ```

3. **Initialize and Deploy**
   ```bash
   cd frontend
   railway init
   railway up
   ```

4. **Set Environment Variables**
   ```bash
   railway variables set REACT_APP_API_URL=https://web-production-52e4b.up.railway.app
   ```

### 3. Custom Domain (Optional)

1. Go to your Railway project settings
2. Click "Domains"
3. Add custom domain or use Railway-provided URL

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `REACT_APP_API_URL` | Backend API URL | Yes | `https://web-production-52e4b.up.railway.app` |
| `NODE_ENV` | Environment | No | `production` |
| `GENERATE_SOURCEMAP` | Generate source maps | No | `false` |

## Local Development

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Create Environment File
```bash
# Create .env.local file
echo "REACT_APP_API_URL=http://localhost:5000" > .env.local
```

### 3. Start Development Server
```bash
npm start
```

The app will be available at `http://localhost:3000`

## Build for Production

```bash
npm run build
```

This creates an optimized production build in the `build/` folder.

## API Integration

The frontend connects to the backend API at:
- **Production**: `https://web-production-52e4b.up.railway.app`
- **Local Dev**: `http://localhost:5000`

### API Endpoints Used

- `POST /api/auth/signup` - User registration
- `POST /api/auth/login` - User login  
- `GET /api/auth/validate` - Token validation
- `POST /api/predict` - Stroke risk prediction
- `GET /api/history` - Prediction history
- `GET /api/statistics` - User statistics

## Key Features

### 🏠 Homepage
- Hero section with call-to-action
- Feature highlights
- Statistics display
- Risk level explanations

### 🔐 Authentication
- User registration/login
- JWT token management
- Profile management
- Password reset

### 🎯 Prediction System
- Interactive prediction form
- Real-time risk assessment
- Results visualization
- Recommendation display

### 📊 Dashboard
- Prediction history
- Risk statistics
- Data visualization charts
- Export capabilities

### 📱 Responsive Design
- Mobile-first approach
- Touch-friendly interface
- Optimized for all screen sizes

## Folder Structure

```
frontend/
├── public/
│   ├── index.html
│   └── manifest.json
├── src/
│   ├── components/       # Reusable components
│   ├── contexts/         # React contexts
│   ├── pages/           # Page components
│   ├── services/        # API services
│   ├── config/          # Configuration files
│   ├── App.js           # Main app component
│   └── index.js         # Entry point
├── package.json
└── README_FRONTEND.md
```

## Deployment Checklist

- [ ] Backend API is deployed and accessible
- [ ] Environment variable `REACT_APP_API_URL` is set
- [ ] Railway project is configured with correct root directory (`/frontend`)
- [ ] Build completes successfully
- [ ] Frontend can communicate with backend API
- [ ] All features work in production environment

## Troubleshooting

### Build Fails
- Check Node.js version (requires 18.x)
- Verify all dependencies are installed
- Check for TypeScript/ESLint errors

### API Connection Issues
- Verify `REACT_APP_API_URL` environment variable
- Check backend API is accessible
- Verify CORS is enabled on backend

### Railway Deployment Issues
- Ensure root directory is set to `/frontend`
- Check build logs for errors
- Verify environment variables are set

## Support

For issues related to:
- **Frontend bugs**: Create GitHub issue
- **API issues**: Check backend repository
- **Railway deployment**: Check Railway documentation

## License

MIT License - see LICENSE file for details.

---

**🚀 Ready to deploy your frontend to Railway!**

1. Set root directory to `/frontend`
2. Add `REACT_APP_API_URL` environment variable
3. Deploy and enjoy! ✨