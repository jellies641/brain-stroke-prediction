const express = require('express');
const path = require('path');
const app = express();

const port = process.env.PORT || 3000;

// Serve static files from the React build directory
app.use(express.static(path.join(__dirname, 'build')));

// Handle React routing, return all requests to React app
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

app.listen(port, '0.0.0.0', () => {
  console.log(`🚀 React Frontend Server running on port ${port}`);
  console.log(`🌐 Frontend URL: http://localhost:${port}`);
  console.log(`📡 Backend API: ${process.env.REACT_APP_API_URL || 'Not configured'}`);
});
