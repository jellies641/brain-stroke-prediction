import React, { useState } from "react";
import {
  Container,
  Box,
  Typography,
  Paper,
  Grid,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Card,
  CardContent,
  Alert,
  AlertTitle,
  Chip,
  LinearProgress,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  CircularProgress,
} from "@mui/material";
import {
  Psychology,
  Assessment,
  CheckCircle,
  Warning,
  Error,
  TrendingUp,
  HealthAndSafety,
  LocalHospital,
  Recommend,
} from "@mui/icons-material";
import { usePrediction } from "../contexts/PredictionContext";
import { toast } from "react-toastify";

const Prediction = () => {
  const { makePrediction, isLoading } = usePrediction();
  const [result, setResult] = useState(null);
  const [formData, setFormData] = useState({
    name: "",
    age: "",
    gender: "",
    hypertension: false,
    heart_disease: false,
    ever_married: "",
    work_type: "",
    residence_type: "",
    avg_glucose_level: "",
    bmi: "",
    smoking_status: "",
    family_history_stroke: false,
    alcohol_consumption: "",
  });

  const [errors, setErrors] = useState({});

  const handleInputChange = (event) => {
    const { name, value, type, checked } = event.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));

    // Clear error when user starts typing
    if (errors[name]) {
      setErrors((prev) => ({
        ...prev,
        [name]: "",
      }));
    }
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.name.trim()) {
      newErrors.name = "Name is required";
    }

    if (!formData.age || formData.age < 1 || formData.age > 120) {
      newErrors.age = "Please enter a valid age (1-120)";
    }

    if (!formData.gender) {
      newErrors.gender = "Please select gender";
    }

    if (!formData.ever_married) {
      newErrors.ever_married = "Please select marital status";
    }

    if (!formData.work_type) {
      newErrors.work_type = "Please select work type";
    }

    if (!formData.residence_type) {
      newErrors.residence_type = "Please select residence type";
    }

    if (
      !formData.avg_glucose_level ||
      formData.avg_glucose_level < 50 ||
      formData.avg_glucose_level > 500
    ) {
      newErrors.avg_glucose_level =
        "Please enter valid glucose level (50-500 mg/dL)";
    }

    if (!formData.bmi || formData.bmi < 10 || formData.bmi > 60) {
      newErrors.bmi = "Please enter valid BMI (10-60)";
    }

    if (!formData.smoking_status) {
      newErrors.smoking_status = "Please select smoking status";
    }

    if (!formData.alcohol_consumption) {
      newErrors.alcohol_consumption = "Please select alcohol consumption";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!validateForm()) {
      toast.error("Please fix the form errors before submitting");
      return;
    }

    const predictionData = {
      ...formData,
      age: parseInt(formData.age),
      avg_glucose_level: parseFloat(formData.avg_glucose_level),
      bmi: parseFloat(formData.bmi),
      hypertension: formData.hypertension ? 1 : 0,
      heart_disease: formData.heart_disease ? 1 : 0,
      family_history_stroke: formData.family_history_stroke ? 1 : 0,
    };

    const response = await makePrediction(predictionData);

    if (response.success) {
      setResult(response.data);
      // Scroll to results
      setTimeout(() => {
        document.getElementById("results-section")?.scrollIntoView({
          behavior: "smooth",
        });
      }, 100);
    }
  };

  const handleReset = () => {
    setFormData({
      name: "",
      age: "",
      gender: "",
      hypertension: false,
      heart_disease: false,
      ever_married: "",
      work_type: "",
      residence_type: "",
      avg_glucose_level: "",
      bmi: "",
      smoking_status: "",
      family_history_stroke: false,
      alcohol_consumption: "",
    });
    setErrors({});
    setResult(null);
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel?.toUpperCase()) {
      case "LOW":
        return "success";
      case "MODERATE":
        return "warning";
      case "HIGH":
        return "error";
      default:
        return "info";
    }
  };

  const getRiskIcon = (riskLevel) => {
    switch (riskLevel?.toUpperCase()) {
      case "LOW":
        return <CheckCircle />;
      case "MODERATE":
        return <Warning />;
      case "HIGH":
        return <Error />;
      default:
        return <Assessment />;
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ textAlign: "center", mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom fontWeight="bold">
          🧠 Stroke Risk Assessment
        </Typography>
        <Typography
          variant="h6"
          color="text.secondary"
          sx={{ maxWidth: 600, mx: "auto" }}
        >
          Provide your health information to get a comprehensive stroke risk
          analysis powered by advanced machine learning algorithms.
        </Typography>
      </Box>

      <Grid container spacing={4}>
        {/* Assessment Form */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 4 }}>
            <Box sx={{ display: "flex", alignItems: "center", mb: 3 }}>
              <Psychology color="primary" sx={{ mr: 2, fontSize: 32 }} />
              <Typography variant="h5" fontWeight="bold">
                Health Assessment Form
              </Typography>
            </Box>

            <form onSubmit={handleSubmit}>
              <Grid container spacing={3}>
                {/* Personal Information */}
                <Grid item xs={12}>
                  <Typography variant="h6" color="primary" gutterBottom>
                    Personal Information
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                </Grid>

                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Full Name"
                    name="name"
                    value={formData.name}
                    onChange={handleInputChange}
                    error={!!errors.name}
                    helperText={errors.name}
                    required
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Age"
                    name="age"
                    type="number"
                    value={formData.age}
                    onChange={handleInputChange}
                    error={!!errors.age}
                    helperText={errors.age || "Enter your current age"}
                    required
                    inputProps={{ min: 1, max: 120 }}
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <FormControl fullWidth error={!!errors.gender} required>
                    <InputLabel>Gender</InputLabel>
                    <Select
                      name="gender"
                      value={formData.gender}
                      onChange={handleInputChange}
                      label="Gender"
                    >
                      <MenuItem value="Male">Male</MenuItem>
                      <MenuItem value="Female">Female</MenuItem>
                      <MenuItem value="Other">Other</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} md={6}>
                  <FormControl fullWidth error={!!errors.ever_married} required>
                    <InputLabel>Marital Status</InputLabel>
                    <Select
                      name="ever_married"
                      value={formData.ever_married}
                      onChange={handleInputChange}
                      label="Marital Status"
                    >
                      <MenuItem value="Yes">Married</MenuItem>
                      <MenuItem value="No">Never Married</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} md={6}>
                  <FormControl fullWidth error={!!errors.work_type} required>
                    <InputLabel>Work Type</InputLabel>
                    <Select
                      name="work_type"
                      value={formData.work_type}
                      onChange={handleInputChange}
                      label="Work Type"
                    >
                      <MenuItem value="Private">Private Company</MenuItem>
                      <MenuItem value="Self-employed">Self-employed</MenuItem>
                      <MenuItem value="Govt_job">Government Job</MenuItem>
                      <MenuItem value="children">Student/Child</MenuItem>
                      <MenuItem value="Never_worked">Never Worked</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} md={6}>
                  <FormControl
                    fullWidth
                    error={!!errors.residence_type}
                    required
                  >
                    <InputLabel>Residence Type</InputLabel>
                    <Select
                      name="residence_type"
                      value={formData.residence_type}
                      onChange={handleInputChange}
                      label="Residence Type"
                    >
                      <MenuItem value="Urban">Urban</MenuItem>
                      <MenuItem value="Rural">Rural</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                {/* Medical Information */}
                <Grid item xs={12} sx={{ mt: 2 }}>
                  <Typography variant="h6" color="primary" gutterBottom>
                    Medical Information
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                </Grid>

                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Average Glucose Level"
                    name="avg_glucose_level"
                    type="number"
                    value={formData.avg_glucose_level}
                    onChange={handleInputChange}
                    error={!!errors.avg_glucose_level}
                    helperText={
                      errors.avg_glucose_level || "Normal range: 70-140 mg/dL"
                    }
                    required
                    inputProps={{ min: 50, max: 500, step: 0.1 }}
                    InputProps={{
                      endAdornment: (
                        <Box
                          component="span"
                          sx={{
                            color: "text.secondary",
                            fontSize: "0.875rem",
                            ml: 1,
                          }}
                        >
                          mg/dL
                        </Box>
                      ),
                    }}
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="BMI (Body Mass Index)"
                    name="bmi"
                    type="number"
                    value={formData.bmi}
                    onChange={handleInputChange}
                    error={!!errors.bmi}
                    helperText={errors.bmi || "Normal range: 18.5-24.9"}
                    required
                    inputProps={{ min: 10, max: 60, step: 0.1 }}
                    InputProps={{
                      endAdornment: (
                        <Typography variant="body2" color="text.secondary">
                          kg/m²
                        </Typography>
                      ),
                    }}
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        name="hypertension"
                        checked={formData.hypertension}
                        onChange={handleInputChange}
                        color="primary"
                      />
                    }
                    label="Hypertension (High Blood Pressure)"
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        name="heart_disease"
                        checked={formData.heart_disease}
                        onChange={handleInputChange}
                        color="primary"
                      />
                    }
                    label="Heart Disease"
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        name="family_history_stroke"
                        checked={formData.family_history_stroke}
                        onChange={handleInputChange}
                        color="primary"
                      />
                    }
                    label="Family History of Stroke"
                  />
                </Grid>

                {/* Lifestyle Information */}
                <Grid item xs={12} sx={{ mt: 2 }}>
                  <Typography variant="h6" color="primary" gutterBottom>
                    Lifestyle Factors
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                </Grid>

                <Grid item xs={12} md={6}>
                  <FormControl
                    fullWidth
                    error={!!errors.smoking_status}
                    required
                  >
                    <InputLabel>Smoking Status</InputLabel>
                    <Select
                      name="smoking_status"
                      value={formData.smoking_status}
                      onChange={handleInputChange}
                      label="Smoking Status"
                    >
                      <MenuItem value="never smoked">Never Smoked</MenuItem>
                      <MenuItem value="formerly smoked">Former Smoker</MenuItem>
                      <MenuItem value="smokes">Current Smoker</MenuItem>
                      <MenuItem value="Unknown">Unknown</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} md={6}>
                  <FormControl
                    fullWidth
                    error={!!errors.alcohol_consumption}
                    required
                  >
                    <InputLabel>Alcohol Consumption</InputLabel>
                    <Select
                      name="alcohol_consumption"
                      value={formData.alcohol_consumption}
                      onChange={handleInputChange}
                      label="Alcohol Consumption"
                    >
                      <MenuItem value="Never">Never</MenuItem>
                      <MenuItem value="Occasionally">Occasionally</MenuItem>
                      <MenuItem value="Regularly">Regularly</MenuItem>
                      <MenuItem value="Heavy">Heavy</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                {/* Form Actions */}
                <Grid item xs={12} sx={{ mt: 3 }}>
                  <Box
                    sx={{ display: "flex", gap: 2, justifyContent: "center" }}
                  >
                    <Button
                      type="submit"
                      variant="contained"
                      size="large"
                      startIcon={
                        isLoading ? (
                          <CircularProgress size={20} />
                        ) : (
                          <Assessment />
                        )
                      }
                      disabled={isLoading}
                      sx={{ px: 4, py: 1.5 }}
                    >
                      {isLoading ? "Analyzing..." : "Analyze Risk"}
                    </Button>
                    <Button
                      variant="outlined"
                      size="large"
                      onClick={handleReset}
                      disabled={isLoading}
                    >
                      Reset Form
                    </Button>
                  </Box>
                </Grid>
              </Grid>
            </form>
          </Paper>
        </Grid>

        {/* Information Panel */}
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom fontWeight="bold">
              <HealthAndSafety
                color="primary"
                sx={{ mr: 1, verticalAlign: "middle" }}
              />
              Assessment Information
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              This assessment evaluates multiple risk factors including:
            </Typography>
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <CheckCircle color="success" fontSize="small" />
                </ListItemIcon>
                <ListItemText
                  primary="Demographics & Lifestyle"
                  secondary="Age, gender, work, residence"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircle color="success" fontSize="small" />
                </ListItemIcon>
                <ListItemText
                  primary="Medical History"
                  secondary="Heart disease, hypertension, family history"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircle color="success" fontSize="small" />
                </ListItemIcon>
                <ListItemText
                  primary="Health Metrics"
                  secondary="BMI, glucose levels, smoking habits"
                />
              </ListItem>
            </List>
          </Paper>

          <Alert severity="warning">
            <AlertTitle>Medical Disclaimer</AlertTitle>
            This tool provides educational information only and should not
            replace professional medical advice. Always consult healthcare
            professionals for medical decisions.
          </Alert>
        </Grid>
      </Grid>

      {/* Results Section */}
      {result && (
        <Box id="results-section" sx={{ mt: 6 }}>
          <Paper sx={{ p: 4 }}>
            <Typography
              variant="h4"
              gutterBottom
              fontWeight="bold"
              textAlign="center"
            >
              🎯 Assessment Results
            </Typography>

            <Grid container spacing={4}>
              {/* Risk Level Card */}
              <Grid item xs={12} md={4}>
                <Card sx={{ textAlign: "center", height: "100%" }}>
                  <CardContent sx={{ p: 3 }}>
                    <Box sx={{ mb: 2 }}>{getRiskIcon(result.risk_level)}</Box>
                    <Typography variant="h3" fontWeight="bold" gutterBottom>
                      {result.risk_level || "UNKNOWN"}
                    </Typography>
                    <Chip
                      label={`${Math.round((result.probability_score || 0) * 100)}% Risk`}
                      color={getRiskColor(result.risk_level)}
                      size="large"
                      sx={{ mb: 2 }}
                    />
                    <Typography variant="body2" color="text.secondary">
                      Confidence: {result.confidence || "Unknown"}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              {/* Risk Probability */}
              <Grid item xs={12} md={8}>
                <Card sx={{ height: "100%" }}>
                  <CardContent sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom fontWeight="bold">
                      <TrendingUp sx={{ mr: 1, verticalAlign: "middle" }} />
                      Risk Analysis
                    </Typography>
                    <Box sx={{ mb: 3 }}>
                      <Typography
                        variant="body2"
                        color="text.secondary"
                        gutterBottom
                      >
                        Stroke Probability:{" "}
                        {Math.round((result.probability_score || 0) * 100)}%
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={(result.probability_score || 0) * 100}
                        color={getRiskColor(result.risk_level)}
                        sx={{ height: 10, borderRadius: 5 }}
                      />
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      Model: {result.model_name || "Unknown"} (v
                      {result.model_version || "1.0"})
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              {/* Risk Factors */}
              <Grid item xs={12} md={6}>
                <Card sx={{ height: "100%" }}>
                  <CardContent sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom fontWeight="bold">
                      <Warning sx={{ mr: 1, verticalAlign: "middle" }} />
                      Risk Factors Identified
                    </Typography>
                    <List dense>
                      {result.risk_factors &&
                        result.risk_factors.map((factor, index) => (
                          <ListItem key={index} sx={{ px: 0 }}>
                            <ListItemIcon>
                              <Error color="error" fontSize="small" />
                            </ListItemIcon>
                            <ListItemText primary={factor} />
                          </ListItem>
                        ))}
                    </List>
                  </CardContent>
                </Card>
              </Grid>

              {/* Recommendations */}
              <Grid item xs={12} md={6}>
                <Card sx={{ height: "100%" }}>
                  <CardContent sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom fontWeight="bold">
                      <Recommend sx={{ mr: 1, verticalAlign: "middle" }} />
                      Recommendations
                    </Typography>
                    <List dense>
                      {result.recommendations &&
                        result.recommendations.map((recommendation, index) => (
                          <ListItem key={index} sx={{ px: 0 }}>
                            <ListItemIcon>
                              <LocalHospital color="primary" fontSize="small" />
                            </ListItemIcon>
                            <ListItemText
                              primary={recommendation}
                              primaryTypographyProps={{ variant: "body2" }}
                            />
                          </ListItem>
                        ))}
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Action Buttons */}
            <Box sx={{ textAlign: "center", mt: 4 }}>
              <Button
                variant="contained"
                size="large"
                onClick={handleReset}
                sx={{ mr: 2 }}
              >
                New Assessment
              </Button>
              <Button
                variant="outlined"
                size="large"
                onClick={() => window.print()}
              >
                Print Results
              </Button>
            </Box>
          </Paper>
        </Box>
      )}
    </Container>
  );
};

export default Prediction;
