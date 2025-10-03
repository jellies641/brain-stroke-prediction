import React, { useEffect, useState } from "react";
import {
  Container,
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Paper,
  Avatar,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  useTheme,
} from "@mui/material";
import {
  Psychology,
  History,
  TrendingUp,
  Person,
  Assessment,
  Warning,
  CheckCircle,
  Error,
  Add,
  Visibility,
  Delete,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { usePrediction } from "../contexts/PredictionContext";

const Dashboard = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const { user } = useAuth();
  const {
    predictions,
    getPredictionHistory,
    isLoading,
    calculateSummaryStats,
    getRiskColor,
    getRiskEmoji,
    deletePrediction,
  } = usePrediction();

  const [stats, setStats] = useState({
    total: 0,
    low_risk: 0,
    moderate_risk: 0,
    high_risk: 0,
    recent_predictions: 0,
  });

  useEffect(() => {
    // Only load prediction history if user is authenticated
    if (user) {
      getPredictionHistory();
    }
  }, [user]);

  useEffect(() => {
    // Calculate stats when predictions change
    const calculatedStats = calculateSummaryStats(predictions);
    setStats(calculatedStats);
  }, [predictions, calculateSummaryStats]);

  const handleDeletePrediction = async (predictionId) => {
    if (window.confirm("Are you sure you want to delete this prediction?")) {
      await deletePrediction(predictionId);
    }
  };

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return "Good morning";
    if (hour < 18) return "Good afternoon";
    return "Good evening";
  };

  const recentPredictions = predictions.slice(0, 5);

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Welcome Section */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom fontWeight="bold">
          {getGreeting()}, {user?.name?.split(" ")[0] || "there"}! 👋
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Welcome to your stroke risk assessment dashboard. Monitor your health
          journey and track your progress.
        </Typography>
      </Box>

      <Grid container spacing={4}>
        {/* Quick Actions */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h5" gutterBottom fontWeight="bold">
              Quick Actions
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Button
                  fullWidth
                  variant="contained"
                  size="large"
                  startIcon={<Psychology />}
                  onClick={() => navigate("/prediction")}
                  sx={{ py: 2 }}
                >
                  New Risk Assessment
                </Button>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Button
                  fullWidth
                  variant="outlined"
                  size="large"
                  startIcon={<History />}
                  onClick={() => navigate("/history")}
                  sx={{ py: 2 }}
                >
                  View All History
                </Button>
              </Grid>
            </Grid>
          </Paper>

          {/* Statistics Overview */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom fontWeight="bold">
              Your Statistics
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={6} md={3}>
                <Card
                  sx={{
                    textAlign: "center",
                    bgcolor: "primary.light",
                    color: "white",
                  }}
                >
                  <CardContent>
                    <Typography variant="h3" fontWeight="bold">
                      {stats.total}
                    </Typography>
                    <Typography variant="body2">Total Assessments</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={6} md={3}>
                <Card
                  sx={{
                    textAlign: "center",
                    bgcolor: "success.light",
                    color: "white",
                  }}
                >
                  <CardContent>
                    <Typography variant="h3" fontWeight="bold">
                      {stats.low_risk}
                    </Typography>
                    <Typography variant="body2">Low Risk</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={6} md={3}>
                <Card
                  sx={{
                    textAlign: "center",
                    bgcolor: "warning.light",
                    color: "white",
                  }}
                >
                  <CardContent>
                    <Typography variant="h3" fontWeight="bold">
                      {stats.moderate_risk}
                    </Typography>
                    <Typography variant="body2">Moderate Risk</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={6} md={3}>
                <Card
                  sx={{
                    textAlign: "center",
                    bgcolor: "error.light",
                    color: "white",
                  }}
                >
                  <CardContent>
                    <Typography variant="h3" fontWeight="bold">
                      {stats.high_risk}
                    </Typography>
                    <Typography variant="body2">High Risk</Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Profile Summary */}
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
              <Avatar
                sx={{
                  width: 60,
                  height: 60,
                  bgcolor: "primary.main",
                  mr: 2,
                  fontSize: "1.5rem",
                }}
              >
                {user?.name?.charAt(0)?.toUpperCase() || "U"}
              </Avatar>
              <Box>
                <Typography variant="h6" fontWeight="bold">
                  {user?.name || "User"}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {user?.email || "user@example.com"}
                </Typography>
              </Box>
            </Box>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<Person />}
              onClick={() => navigate("/profile")}
            >
              View Profile
            </Button>
          </Paper>

          {/* Health Tips */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom fontWeight="bold">
              💡 Health Tips
            </Typography>
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <CheckCircle color="success" fontSize="small" />
                </ListItemIcon>
                <ListItemText
                  primary="Regular Exercise"
                  secondary="30 minutes of moderate activity daily"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircle color="success" fontSize="small" />
                </ListItemIcon>
                <ListItemText
                  primary="Healthy Diet"
                  secondary="Low sodium, high fiber foods"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircle color="success" fontSize="small" />
                </ListItemIcon>
                <ListItemText
                  primary="Regular Check-ups"
                  secondary="Monitor blood pressure and cholesterol"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <Warning color="warning" fontSize="small" />
                </ListItemIcon>
                <ListItemText
                  primary="Quit Smoking"
                  secondary="Significantly reduces stroke risk"
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>

        {/* Recent Predictions */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                mb: 2,
              }}
            >
              <Typography variant="h5" fontWeight="bold">
                Recent Assessments
              </Typography>
              <Button
                variant="outlined"
                startIcon={<Add />}
                onClick={() => navigate("/prediction")}
              >
                New Assessment
              </Button>
            </Box>

            {isLoading ? (
              <LinearProgress />
            ) : recentPredictions.length === 0 ? (
              <Box sx={{ textAlign: "center", py: 4 }}>
                <Assessment
                  sx={{ fontSize: 64, color: "text.secondary", mb: 2 }}
                />
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  No assessments yet
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Take your first stroke risk assessment to get started.
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<Psychology />}
                  onClick={() => navigate("/prediction")}
                >
                  Start Assessment
                </Button>
              </Box>
            ) : (
              <Grid container spacing={2}>
                {recentPredictions.map((prediction, index) => (
                  <Grid item xs={12} md={6} lg={4} key={prediction.id || index}>
                    <Card sx={{ height: "100%" }}>
                      <CardContent>
                        <Box
                          sx={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "flex-start",
                            mb: 2,
                          }}
                        >
                          <Box>
                            <Typography variant="h6" gutterBottom>
                              {getRiskEmoji(prediction.risk_level)}{" "}
                              {prediction.risk_level || "Unknown"}
                            </Typography>
                            <Chip
                              label={`${Math.round((prediction.probability_score || 0) * 100)}% Risk`}
                              color={
                                getRiskColor(prediction.risk_level) ===
                                "#2e7d32"
                                  ? "success"
                                  : getRiskColor(prediction.risk_level) ===
                                      "#f57f17"
                                    ? "warning"
                                    : "error"
                              }
                              size="small"
                            />
                          </Box>
                          <Box>
                            <IconButton
                              size="small"
                              onClick={() => navigate(`/history`)}
                              title="View Details"
                            >
                              <Visibility fontSize="small" />
                            </IconButton>
                            <IconButton
                              size="small"
                              onClick={() =>
                                handleDeletePrediction(prediction.id)
                              }
                              title="Delete"
                            >
                              <Delete fontSize="small" />
                            </IconButton>
                          </Box>
                        </Box>

                        <Typography
                          variant="body2"
                          color="text.secondary"
                          gutterBottom
                        >
                          Age: {prediction.patient_summary?.age || "N/A"} • BMI:{" "}
                          {prediction.patient_summary?.bmi || "N/A"}
                        </Typography>

                        <Typography variant="caption" color="text.secondary">
                          {prediction.formatted_date ||
                            new Date(
                              prediction.created_at,
                            ).toLocaleDateString()}
                        </Typography>

                        <Box sx={{ mt: 2 }}>
                          <LinearProgress
                            variant="determinate"
                            value={(prediction.probability_score || 0) * 100}
                            color={
                              getRiskColor(prediction.risk_level) === "#2e7d32"
                                ? "success"
                                : getRiskColor(prediction.risk_level) ===
                                    "#f57f17"
                                  ? "warning"
                                  : "error"
                            }
                            sx={{ height: 6, borderRadius: 3 }}
                          />
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Medical Disclaimer */}
      <Paper
        sx={{
          mt: 4,
          p: 3,
          bgcolor: "warning.light",
          border: "1px solid",
          borderColor: "warning.main",
        }}
      >
        <Typography
          variant="h6"
          color="warning.dark"
          fontWeight="bold"
          gutterBottom
        >
          ⚠️ Important Medical Disclaimer
        </Typography>
        <Typography variant="body2" color="warning.dark">
          This dashboard and all assessments are for educational and
          informational purposes only. They should not be used as a substitute
          for professional medical advice, diagnosis, or treatment. Always
          consult qualified healthcare professionals for medical decisions and
          in case of emergency.
        </Typography>
      </Paper>
    </Container>
  );
};

export default Dashboard;
