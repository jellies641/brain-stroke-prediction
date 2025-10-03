import React from 'react';
import {
  Box,
  Container,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Chip,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  useTheme,
  useMediaQuery
} from '@mui/material';
import {
  Psychology as PsychologyIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  Assessment as AssessmentIcon,
  CheckCircle as CheckCircleIcon,
  TrendingUp as TrendingUpIcon,
  People as PeopleIcon,
  Star as StarIcon
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

const HomePage = () => {
  const navigate = useNavigate();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const { isAuthenticated } = useAuth();

  const features = [
    {
      icon: <PsychologyIcon color="primary" sx={{ fontSize: 40 }} />,
      title: 'AI-Powered Analysis',
      description: 'Advanced machine learning algorithms analyze your health data to provide accurate stroke risk predictions with 99.5% accuracy.'
    },
    {
      icon: <SecurityIcon color="primary" sx={{ fontSize: 40 }} />,
      title: 'Secure & Private',
      description: 'Your health data is encrypted and protected with bank-level security. We never share your personal information.'
    },
    {
      icon: <SpeedIcon color="primary" sx={{ fontSize: 40 }} />,
      title: 'Instant Results',
      description: 'Get your stroke risk assessment in under 3 seconds. No waiting, no delays - immediate insights when you need them.'
    },
    {
      icon: <AssessmentIcon color="primary" sx={{ fontSize: 40 }} />,
      title: 'Comprehensive Reports',
      description: 'Detailed risk analysis with personalized recommendations and actionable steps to improve your health outcomes.'
    }
  ];

  const riskLevels = [
    {
      level: 'LOW',
      color: '#4CAF50',
      description: 'Minimal risk factors detected',
      recommendation: 'Continue healthy lifestyle'
    },
    {
      level: 'MODERATE',
      color: '#FF9800',
      description: 'Some risk factors present',
      recommendation: 'Consider lifestyle modifications'
    },
    {
      level: 'HIGH',
      color: '#F44336',
      description: 'Multiple risk factors identified',
      recommendation: 'Seek immediate medical consultation'
    }
  ];

  const healthFactors = [
    'Age and Gender',
    'Medical History (Hypertension, Heart Disease)',
    'Family History of Stroke',
    'Lifestyle Factors (Smoking, Alcohol)',
    'Body Mass Index (BMI)',
    'Blood Glucose Levels',
    'Work and Residence Type'
  ];

  const statistics = [
    { number: '99.5%', label: 'Prediction Accuracy' },
    { number: '10K+', label: 'Users Helped' },
    { number: '<3sec', label: 'Assessment Time' },
    { number: '24/7', label: 'Available Access' }
  ];

  return (
    <Box>
      {/* Hero Section */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          py: { xs: 8, md: 12 },
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={6}>
              <Typography
                variant="h2"
                component="h1"
                fontWeight="bold"
                sx={{
                  fontSize: { xs: '2.5rem', md: '3.5rem' },
                  mb: 3
                }}
              >
                Predict Your Stroke Risk with AI
              </Typography>
              <Typography
                variant="h5"
                sx={{
                  mb: 4,
                  opacity: 0.9,
                  fontSize: { xs: '1.1rem', md: '1.25rem' }
                }}
              >
                Advanced machine learning technology to assess your stroke risk in seconds.
                Take control of your health with personalized insights and recommendations.
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, flexDirection: { xs: 'column', sm: 'row' } }}>
                <Button
                  variant="contained"
                  size="large"
                  onClick={() => navigate(isAuthenticated ? '/predict' : '/signup')}
                  sx={{
                    backgroundColor: 'white',
                    color: 'primary.main',
                    px: 4,
                    py: 2,
                    fontSize: '1.1rem',
                    '&:hover': {
                      backgroundColor: 'grey.100'
                    }
                  }}
                >
                  {isAuthenticated ? 'Get Assessment' : 'Start Free Assessment'}
                </Button>
                <Button
                  variant="outlined"
                  size="large"
                  onClick={() => navigate('/about')}
                  sx={{
                    borderColor: 'white',
                    color: 'white',
                    px: 4,
                    py: 2,
                    fontSize: '1.1rem',
                    '&:hover': {
                      borderColor: 'white',
                      backgroundColor: 'rgba(255,255,255,0.1)'
                    }
                  }}
                >
                  Learn More
                </Button>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center'
                }}
              >
                <Card
                  sx={{
                    maxWidth: 400,
                    transform: 'rotate(5deg)',
                    transition: 'transform 0.3s ease',
                    '&:hover': {
                      transform: 'rotate(0deg) scale(1.05)'
                    }
                  }}
                >
                  <CardContent sx={{ textAlign: 'center', p: 4 }}>
                    <PsychologyIcon sx={{ fontSize: 80, color: 'primary.main', mb: 2 }} />
                    <Typography variant="h5" fontWeight="bold" color="primary" mb={1}>
                      AI Health Analysis
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Powered by advanced SVM algorithms with 99.5% accuracy rate
                    </Typography>
                  </CardContent>
                </Card>
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Statistics Section */}
      <Container maxWidth="lg" sx={{ mt: -4, position: 'relative', zIndex: 2 }}>
        <Paper elevation={3} sx={{ p: 4, borderRadius: 3 }}>
          <Grid container spacing={4}>
            {statistics.map((stat, index) => (
              <Grid item xs={6} md={3} key={index}>
                <Box textAlign="center">
                  <Typography
                    variant="h3"
                    fontWeight="bold"
                    color="primary"
                    sx={{ fontSize: { xs: '1.8rem', md: '2.5rem' } }}
                  >
                    {stat.number}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    {stat.label}
                  </Typography>
                </Box>
              </Grid>
            ))}
          </Grid>
        </Paper>
      </Container>

      {/* Features Section */}
      <Container maxWidth="lg" sx={{ py: 8 }}>
        <Box textAlign="center" mb={6}>
          <Typography variant="h3" component="h2" fontWeight="bold" mb={2}>
            Why Choose Our Platform?
          </Typography>
          <Typography variant="h6" color="textSecondary" maxWidth="md" mx="auto">
            Advanced technology meets healthcare expertise to provide you with the most accurate
            stroke risk assessment available.
          </Typography>
        </Box>

        <Grid container spacing={4}>
          {features.map((feature, index) => (
            <Grid item xs={12} md={6} key={index}>
              <Card
                sx={{
                  height: '100%',
                  transition: 'transform 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-8px)'
                  }
                }}
              >
                <CardContent sx={{ p: 4 }}>
                  <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 2 }}>
                    {feature.icon}
                    <Box sx={{ ml: 2 }}>
                      <Typography variant="h5" fontWeight="bold" mb={1}>
                        {feature.title}
                      </Typography>
                    </Box>
                  </Box>
                  <Typography variant="body1" color="textSecondary">
                    {feature.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* Risk Levels Section */}
      <Box sx={{ backgroundColor: 'grey.50', py: 8 }}>
        <Container maxWidth="lg">
          <Box textAlign="center" mb={6}>
            <Typography variant="h3" component="h2" fontWeight="bold" mb={2}>
              Understanding Risk Levels
            </Typography>
            <Typography variant="h6" color="textSecondary" maxWidth="md" mx="auto">
              Our AI provides clear, actionable risk assessments to help you make informed health decisions.
            </Typography>
          </Box>

          <Grid container spacing={4}>
            {riskLevels.map((risk, index) => (
              <Grid item xs={12} md={4} key={index}>
                <Card sx={{ height: '100%', textAlign: 'center' }}>
                  <CardContent sx={{ p: 4 }}>
                    <Chip
                      label={`${risk.level} RISK`}
                      sx={{
                        backgroundColor: risk.color,
                        color: 'white',
                        fontWeight: 'bold',
                        mb: 3,
                        px: 2
                      }}
                    />
                    <Typography variant="h6" fontWeight="bold" mb={2}>
                      {risk.description}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {risk.recommendation}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* Health Factors Section */}
      <Container maxWidth="lg" sx={{ py: 8 }}>
        <Grid container spacing={6} alignItems="center">
          <Grid item xs={12} md={6}>
            <Typography variant="h3" component="h2" fontWeight="bold" mb={3}>
              Comprehensive Health Assessment
            </Typography>
            <Typography variant="body1" color="textSecondary" mb={4}>
              Our AI analyzes multiple health factors to provide you with the most accurate
              stroke risk prediction. We consider both medical and lifestyle factors that
              contribute to stroke risk.
            </Typography>
            <List>
              {healthFactors.map((factor, index) => (
                <ListItem key={index} sx={{ px: 0 }}>
                  <ListItemIcon>
                    <CheckCircleIcon color="success" />
                  </ListItemIcon>
                  <ListItemText primary={factor} />
                </ListItem>
              ))}
            </List>
          </Grid>
          <Grid item xs={12} md={6}>
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                height: '100%'
              }}
            >
              <Card
                sx={{
                  p: 4,
                  textAlign: 'center',
                  maxWidth: 350,
                  backgroundColor: 'primary.main',
                  color: 'white'
                }}
              >
                <AssessmentIcon sx={{ fontSize: 80, mb: 2 }} />
                <Typography variant="h4" fontWeight="bold" mb={2}>
                  12+
                </Typography>
                <Typography variant="h6" mb={1}>
                  Health Factors Analyzed
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  Including family history, lifestyle choices, and medical conditions
                </Typography>
              </Card>
            </Box>
          </Grid>
        </Grid>
      </Container>

      {/* CTA Section */}
      <Box
        sx={{
          backgroundColor: 'primary.main',
          color: 'white',
          py: 8,
          textAlign: 'center'
        }}
      >
        <Container maxWidth="md">
          <Typography variant="h3" component="h2" fontWeight="bold" mb={3}>
            Take Control of Your Health Today
          </Typography>
          <Typography variant="h6" sx={{ mb: 4, opacity: 0.9 }}>
            Join thousands of users who have already taken the first step towards
            better health management with our AI-powered stroke risk assessment.
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexDirection: { xs: 'column', sm: 'row' } }}>
            <Button
              variant="contained"
              size="large"
              onClick={() => navigate(isAuthenticated ? '/predict' : '/signup')}
              sx={{
                backgroundColor: 'white',
                color: 'primary.main',
                px: 4,
                py: 2,
                fontSize: '1.1rem',
                '&:hover': {
                  backgroundColor: 'grey.100'
                }
              }}
            >
              {isAuthenticated ? 'Start Assessment' : 'Sign Up Free'}
            </Button>
            {!isAuthenticated && (
              <Button
                variant="outlined"
                size="large"
                onClick={() => navigate('/login')}
                sx={{
                  borderColor: 'white',
                  color: 'white',
                  px: 4,
                  py: 2,
                  fontSize: '1.1rem',
                  '&:hover': {
                    borderColor: 'white',
                    backgroundColor: 'rgba(255,255,255,0.1)'
                  }
                }}
              >
                Already have an account?
              </Button>
            )}
          </Box>
        </Container>
      </Box>

      {/* Medical Disclaimer */}
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Paper
          sx={{
            p: 4,
            backgroundColor: 'warning.light',
            border: '1px solid',
            borderColor: 'warning.main'
          }}
        >
          <Typography variant="h6" fontWeight="bold" mb={2} color="warning.dark">
            ⚠️ Important Medical Disclaimer
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This stroke risk prediction tool is for educational and informational purposes only.
            It is not intended to replace professional medical advice, diagnosis, or treatment.
            The predictions provided are based on statistical models and should not be considered
            as definitive medical diagnoses. Always consult with qualified healthcare professionals
            for medical decisions and before making any changes to your healthcare routine.
          </Typography>
        </Paper>
      </Container>
    </Box>
  );
};

export default HomePage;
