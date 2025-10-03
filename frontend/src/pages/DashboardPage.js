import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Avatar,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Paper,
  IconButton,
  Menu,
  MenuItem
} from '@mui/material';
import {
  Assessment as AssessmentIcon,
  History as HistoryIcon,
  Person as PersonIcon,
  TrendingUp as TrendingUpIcon,
  Add as AddIcon,
  MoreVert as MoreVertIcon,
  Visibility as ViewIcon,
  Delete as DeleteIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import axios from 'axios';
import { toast } from 'react-toastify';
import LoadingSpinner from '../components/Common/LoadingSpinner';

const DashboardPage = () => {
  const navigate = useNavigate();
  const { user } = useAuth();

  const [loading, setLoading] = useState(true);
  const [statistics, setStatistics] = useState(null);
  const [recentPredictions, setRecentPredictions] = useState([]);
  const [anchorEl, setAnchorEl] = useState(null);
  const [selectedPrediction, setSelectedPrediction] = useState(null);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const [statsResponse, historyResponse] = await Promise.all([
        axios.get('/statistics'),
        axios.get('/history?page=1&per_page=5')
      ]);

      setStatistics(statsResponse.data);
      setRecentPredictions(historyResponse.data.predictions || []);
    } catch (error) {
      toast.error('Failed to load dashboard data');
      console.error('Dashboard error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleMenuOpen = (event, prediction) => {
    setAnchorEl(event.currentTarget);
    setSelectedPrediction(prediction);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setSelectedPrediction(null);
  };

  const handleViewPrediction = (predictionId) => {
    navigate(`/history?highlight=${predictionId}`);
    handleMenuClose();
  };

  const handleDeletePrediction = async (predictionId) => {
    try {
      await axios.delete(`/history/${predictionId}`);
      toast.success('Prediction deleted successfully');
      fetchDashboardData();
    } catch (error) {
      toast.error('Failed to delete prediction');
    }
    handleMenuClose();
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'LOW': return '#4CAF50';
      case 'MODERATE': return '#FF9800';
      case 'HIGH': return '#F44336';
      default: return '#757575';
    }
  };

  const getRiskIcon = (level) => {
    switch (level) {
      case 'LOW': return <CheckCircleIcon />;
      case 'MODERATE': return <WarningIcon />;
      case 'HIGH': return <ErrorIcon />;
      default: return <InfoIcon />;
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return <LoadingSpinner fullScreen message="Loading your dashboard..." />;
  }

  const quickActions = [
    {
      title: 'New Risk Assessment',
      description: 'Complete a comprehensive stroke risk evaluation',
      icon: <AssessmentIcon sx={{ fontSize: 40 }} />,
      color: 'primary',
      action: () => navigate('/predict')
    },
    {
      title: 'View History',
      description: 'Review your previous assessments and trends',
      icon: <HistoryIcon sx={{ fontSize: 40 }} />,
      color: 'secondary',
      action: () => navigate('/history')
    },
    {
      title: 'Update Profile',
      description: 'Keep your personal information up to date',
      icon: <PersonIcon sx={{ fontSize: 40 }} />,
      color: 'info',
      action: () => navigate('/profile')
    }
  ];

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Welcome Section */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" fontWeight="bold" sx={{ mb: 1 }}>
          Welcome back, {user?.name?.split(' ')[0] || 'User'}! 👋
        </Typography>
        <Typography variant="body1" color="textSecondary">
          Here's an overview of your health assessment activities and recommendations.
        </Typography>
      </Box>

      <Grid container spacing={4}>
        {/* User Summary Card */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ textAlign: 'center', p: 4 }}>
              <Avatar
                sx={{
                  width: 80,
                  height: 80,
                  margin: '0 auto 16px',
                  backgroundColor: 'primary.main',
                  fontSize: '2rem'
                }}
              >
                {user?.name?.charAt(0)?.toUpperCase() || 'U'}
              </Avatar>
              <Typography variant="h6" fontWeight="bold" sx={{ mb: 1 }}>
                {user?.name || 'User'}
              </Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                {user?.email}
              </Typography>
              <Chip
                label={`Member since ${new Date(user?.created_at).getFullYear()}`}
                size="small"
                color="primary"
                variant="outlined"
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Statistics Cards */}
        <Grid item xs={12} md={8}>
          <Grid container spacing={2}>
            <Grid item xs={6} sm={3}>
              <Card>
                <CardContent sx={{ textAlign: 'center', p: 3 }}>
                  <Typography variant="h4" fontWeight="bold" color="primary">
                    {statistics?.total_predictions || 0}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Total Assessments
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Card>
                <CardContent sx={{ textAlign: 'center', p: 3 }}>
                  <Typography variant="h4" fontWeight="bold" color="success.main">
                    {statistics?.risk_distribution?.LOW || 0}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Low Risk
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Card>
                <CardContent sx={{ textAlign: 'center', p: 3 }}>
                  <Typography variant="h4" fontWeight="bold" color="warning.main">
                    {statistics?.risk_distribution?.MODERATE || 0}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Moderate Risk
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Card>
                <CardContent sx={{ textAlign: 'center', p: 3 }}>
                  <Typography variant="h4" fontWeight="bold" color="error.main">
                    {statistics?.risk_distribution?.HIGH || 0}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    High Risk
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Latest Assessment */}
          {statistics?.latest_prediction && (
            <Card sx={{ mt: 2 }}>
              <CardContent>
                <Typography variant="h6" fontWeight="bold" sx={{ mb: 2 }}>
                  Latest Assessment
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  {getRiskIcon(statistics.latest_prediction.risk_level)}
                  <Box>
                    <Typography variant="subtitle1" fontWeight="bold">
                      {statistics.latest_prediction.risk_level} Risk Level
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {formatDate(statistics.latest_prediction.created_at)}
                    </Typography>
                  </Box>
                  <Box sx={{ ml: 'auto' }}>
                    <Chip
                      label={`${(statistics.latest_prediction.probability_score * 100).toFixed(1)}%`}
                      color={
                        statistics.latest_prediction.risk_level === 'LOW' ? 'success' :
                        statistics.latest_prediction.risk_level === 'MODERATE' ? 'warning' : 'error'
                      }
                      size="small"
                    />
                  </Box>
                </Box>
              </CardContent>
            </Card>
          )}
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12}>
          <Typography variant="h5" fontWeight="bold" sx={{ mb: 3 }}>
            Quick Actions
          </Typography>
          <Grid container spacing={3}>
            {quickActions.map((action, index) => (
              <Grid item xs={12} md={4} key={index}>
                <Card
                  sx={{
                    height: '100%',
                    cursor: 'pointer',
                    transition: 'transform 0.2s ease-in-out',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: 4
                    }
                  }}
                  onClick={action.action}
                >
                  <CardContent sx={{ textAlign: 'center', p: 4 }}>
                    <Box sx={{ color: `${action.color}.main`, mb: 2 }}>
                      {action.icon}
                    </Box>
                    <Typography variant="h6" fontWeight="bold" sx={{ mb: 1 }}>
                      {action.title}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {action.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Grid>

        {/* Recent Predictions */}
        {recentPredictions.length > 0 && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                  <Typography variant="h6" fontWeight="bold">
                    Recent Assessments
                  </Typography>
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => navigate('/history')}
                    endIcon={<HistoryIcon />}
                  >
                    View All
                  </Button>
                </Box>
                <List>
                  {recentPredictions.map((prediction, index) => (
                    <React.Fragment key={prediction.id}>
                      <ListItem
                        sx={{
                          px: 0,
                          '&:hover': {
                            backgroundColor: 'action.hover',
                            borderRadius: 1
                          }
                        }}
                      >
                        <ListItemIcon>
                          {React.cloneElement(getRiskIcon(prediction.risk_level), {
                            sx: { color: getRiskColor(prediction.risk_level) }
                          })}
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="subtitle2">
                                {prediction.name}
                              </Typography>
                              <Chip
                                label={prediction.risk_level}
                                size="small"
                                sx={{
                                  backgroundColor: getRiskColor(prediction.risk_level),
                                  color: 'white',
                                  fontWeight: 'bold'
                                }}
                              />
                            </Box>
                          }
                          secondary={
                            <Box>
                              <Typography variant="body2" color="textSecondary">
                                Age: {prediction.age} • BMI: {prediction.bmi} • Glucose: {prediction.avg_glucose_level} mg/dL
                              </Typography>
                              <Typography variant="caption" color="textSecondary">
                                {formatDate(prediction.created_at)}
                              </Typography>
                            </Box>
                          }
                        />
                        <IconButton
                          edge="end"
                          onClick={(e) => handleMenuOpen(e, prediction)}
                        >
                          <MoreVertIcon />
                        </IconButton>
                      </ListItem>
                      {index < recentPredictions.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Health Tips */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, backgroundColor: 'primary.light', color: 'primary.contrastText' }}>
            <Typography variant="h6" fontWeight="bold" sx={{ mb: 2 }}>
              💡 Health Tip of the Day
            </Typography>
            <Typography variant="body1">
              Regular physical activity for at least 150 minutes per week can significantly reduce your stroke risk.
              Even simple activities like brisk walking, swimming, or cycling can make a big difference in your cardiovascular health.
            </Typography>
          </Paper>
        </Grid>

        {/* Get Started Message for New Users */}
        {statistics?.total_predictions === 0 && (
          <Grid item xs={12}>
            <Card sx={{ backgroundColor: 'info.light', color: 'info.contrastText' }}>
              <CardContent sx={{ textAlign: 'center', p: 4 }}>
                <AssessmentIcon sx={{ fontSize: 60, mb: 2 }} />
                <Typography variant="h5" fontWeight="bold" sx={{ mb: 2 }}>
                  Ready to Get Started?
                </Typography>
                <Typography variant="body1" sx={{ mb: 3 }}>
                  Complete your first stroke risk assessment to get personalized insights
                  and recommendations for better health management.
                </Typography>
                <Button
                  variant="contained"
                  size="large"
                  onClick={() => navigate('/predict')}
                  sx={{
                    backgroundColor: 'white',
                    color: 'info.main',
                    '&:hover': {
                      backgroundColor: 'grey.100'
                    }
                  }}
                >
                  Start Your First Assessment
                </Button>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>

      {/* Action Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => handleViewPrediction(selectedPrediction?.id)}>
          <ListItemIcon>
            <ViewIcon fontSize="small" />
          </ListItemIcon>
          View Details
        </MenuItem>
        <MenuItem onClick={() => handleDeletePrediction(selectedPrediction?.id)}>
          <ListItemIcon>
            <DeleteIcon fontSize="small" />
          </ListItemIcon>
          Delete
        </MenuItem>
      </Menu>
    </Container>
  );
};

export default DashboardPage;
