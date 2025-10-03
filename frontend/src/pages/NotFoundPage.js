import React from 'react';
import {
  Box,
  Container,
  Typography,
  Button,
  Card,
  CardContent,
  Grid
} from '@mui/material';
import {
  Home as HomeIcon,
  ArrowBack as BackIcon,
  Search as SearchIcon,
  Help as HelpIcon
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const NotFoundPage = () => {
  const navigate = useNavigate();

  const handleGoHome = () => {
    navigate('/');
  };

  const handleGoBack = () => {
    navigate(-1);
  };

  const quickLinks = [
    {
      title: 'Start Risk Assessment',
      description: 'Complete your stroke risk evaluation',
      path: '/predict',
      icon: <SearchIcon />
    },
    {
      title: 'View Dashboard',
      description: 'Access your personal dashboard',
      path: '/dashboard',
      icon: <HomeIcon />
    },
    {
      title: 'Get Help',
      description: 'Find answers to common questions',
      path: '/contact',
      icon: <HelpIcon />
    }
  ];

  return (
    <Container maxWidth="md" sx={{ py: 8 }}>
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        {/* Large 404 Display */}
        <Typography
          variant="h1"
          sx={{
            fontSize: { xs: '8rem', md: '12rem' },
            fontWeight: 'bold',
            background: 'linear-gradient(45deg, #1976d2, #42a5f5)',
            backgroundClip: 'text',
            textFillColor: 'transparent',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            mb: 2
          }}
        >
          404
        </Typography>

        {/* Brain Icon */}
        <Typography
          variant="h2"
          sx={{
            fontSize: { xs: '3rem', md: '4rem' },
            mb: 2
          }}
        >
          🧠
        </Typography>

        <Typography
          variant="h4"
          component="h1"
          fontWeight="bold"
          sx={{ mb: 2 }}
        >
          Oops! Page Not Found
        </Typography>

        <Typography
          variant="h6"
          color="textSecondary"
          sx={{ mb: 4, maxWidth: 600, mx: 'auto' }}
        >
          The page you're looking for doesn't exist or has been moved.
          Don't worry, let's get you back on track to better health insights.
        </Typography>

        {/* Action Buttons */}
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mb: 6, flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            size="large"
            startIcon={<HomeIcon />}
            onClick={handleGoHome}
            sx={{ px: 4, py: 1.5 }}
          >
            Go to Homepage
          </Button>
          <Button
            variant="outlined"
            size="large"
            startIcon={<BackIcon />}
            onClick={handleGoBack}
            sx={{ px: 4, py: 1.5 }}
          >
            Go Back
          </Button>
        </Box>
      </Box>

      {/* Quick Links */}
      <Box>
        <Typography
          variant="h5"
          fontWeight="bold"
          sx={{ mb: 3, textAlign: 'center' }}
        >
          Quick Links
        </Typography>
        <Grid container spacing={3}>
          {quickLinks.map((link, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Card
                sx={{
                  height: '100%',
                  cursor: 'pointer',
                  transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: 4
                  }
                }}
                onClick={() => navigate(link.path)}
              >
                <CardContent sx={{ textAlign: 'center', p: 3 }}>
                  <Box sx={{ color: 'primary.main', mb: 2 }}>
                    {React.cloneElement(link.icon, { sx: { fontSize: 48 } })}
                  </Box>
                  <Typography variant="h6" fontWeight="bold" sx={{ mb: 1 }}>
                    {link.title}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    {link.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Help Section */}
      <Box sx={{ mt: 6, textAlign: 'center' }}>
        <Typography variant="body1" color="textSecondary" sx={{ mb: 2 }}>
          Still can't find what you're looking for?
        </Typography>
        <Button
          variant="text"
          color="primary"
          onClick={() => navigate('/contact')}
          sx={{ fontSize: '1.1rem' }}
        >
          Contact our support team
        </Button>
      </Box>

      {/* Fun Brain Health Tip */}
      <Card
        sx={{
          mt: 6,
          p: 3,
          backgroundColor: 'primary.light',
          color: 'primary.contrastText'
        }}
      >
        <Typography variant="h6" fontWeight="bold" sx={{ mb: 1 }}>
          💡 Brain Health Tip
        </Typography>
        <Typography variant="body1">
          While you're here, remember that staying mentally active with puzzles, reading,
          and learning new skills can help maintain cognitive function and potentially
          reduce stroke risk!
        </Typography>
      </Card>
    </Container>
  );
};

export default NotFoundPage;
