import React, { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
  Alert,
  Card,
  CardContent,
  Stepper,
  Step,
  StepLabel,
  LinearProgress,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider
} from '@mui/material';
import {
  Person as PersonIcon,
  LocalHospital as MedicalIcon,
  Lifestyle as LifestyleIcon,
  Work as WorkIcon,
  Assessment as AssessmentIcon,
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { useFormik } from 'formik';
import * as Yup from 'yup';
import { useAuth } from '../contexts/AuthContext';
import { toast } from 'react-toastify';
import axios from 'axios';
import LoadingSpinner from '../components/Common/LoadingSpinner';

const PredictionPage = () => {
  const { user } = useAuth();
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showResult, setShowResult] = useState(false);

  const steps = [
    'Personal Information',
    'Medical History',
    'Lifestyle Factors',
    'Risk Assessment'
  ];

  // Form validation schema
  const validationSchema = Yup.object({
    name: Yup.string()
      .min(2, 'Name must be at least 2 characters')
      .required('Name is required'),
    age: Yup.number()
      .min(1, 'Age must be at least 1')
      .max(120, 'Age must be less than 120')
      .required('Age is required'),
    gender: Yup.string()
      .oneOf(['Male', 'Female', 'Other'], 'Please select a valid gender')
      .required('Gender is required'),
    ever_married: Yup.string()
      .oneOf(['Yes', 'No'], 'Please select marital status')
      .required('Marital status is required'),
    hypertension: Yup.boolean().required(),
    heart_disease: Yup.boolean().required(),
    avg_glucose_level: Yup.number()
      .min(50, 'Glucose level seems too low')
      .max(500, 'Glucose level seems too high')
      .required('Average glucose level is required'),
    bmi: Yup.number()
      .min(10, 'BMI seems too low')
      .max(60, 'BMI seems too high')
      .required('BMI is required'),
    smoking_status: Yup.string()
      .oneOf(['never smoked', 'formerly smoked', 'smokes', 'Unknown'], 'Please select smoking status')
      .required('Smoking status is required'),
    residence_type: Yup.string()
      .oneOf(['Urban', 'Rural'], 'Please select residence type')
      .required('Residence type is required'),
    work_type: Yup.string()
      .oneOf(['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'], 'Please select work type')
      .required('Work type is required'),
    family_history_stroke: Yup.boolean().required(),
    alcohol_consumption: Yup.string()
      .oneOf(['Never', 'Occasionally', 'Regularly', 'Heavy'], 'Please select alcohol consumption')
      .required('Alcohol consumption is required')
  });

  // Form handling with Formik
  const formik = useFormik({
    initialValues: {
      name: user?.name || '',
      age: user?.age || '',
      gender: user?.gender || '',
      ever_married: '',
      hypertension: false,
      heart_disease: false,
      avg_glucose_level: '',
      bmi: '',
      smoking_status: '',
      residence_type: '',
      work_type: '',
      family_history_stroke: false,
      alcohol_consumption: ''
    },
    validationSchema,
    onSubmit: async (values) => {
      setLoading(true);
      try {
        const response = await axios.post('/predict', values);
        setResult(response.data);
        setShowResult(true);
        toast.success('Risk assessment completed!');
        setActiveStep(3);
      } catch (error) {
        toast.error(error.response?.data?.error || 'Failed to complete assessment');
        console.error('Prediction error:', error);
      } finally {
        setLoading(false);
      }
    }
  });

  const handleNext = () => {
    const currentStepFields = getStepFields(activeStep);
    const hasErrors = currentStepFields.some(field =>
      formik.touched[field] && formik.errors[field]
    );

    if (!hasErrors && activeStep < steps.length - 1) {
      setActiveStep(activeStep + 1);
    }
  };

  const handleBack = () => {
    setActiveStep(activeStep - 1);
  };

  const getStepFields = (step) => {
    switch (step) {
      case 0:
        return ['name', 'age', 'gender', 'ever_married'];
      case 1:
        return ['hypertension', 'heart_disease', 'avg_glucose_level', 'bmi'];
      case 2:
        return ['smoking_status', 'residence_type', 'work_type', 'family_history_stroke', 'alcohol_consumption'];
      default:
        return [];
    }
  };

  const isStepComplete = (step) => {
    const fields = getStepFields(step);
    return fields.every(field => formik.values[field] !== '' && formik.values[field] !== undefined);
  };

  const getRiskLevelColor = (level) => {
    switch (level) {
      case 'LOW':
        return '#4CAF50';
      case 'MODERATE':
        return '#FF9800';
      case 'HIGH':
        return '#F44336';
      default:
        return '#757575';
    }
  };

  const getRiskIcon = (level) => {
    switch (level) {
      case 'LOW':
        return <CheckIcon />;
      case 'MODERATE':
        return <WarningIcon />;
      case 'HIGH':
        return <ErrorIcon />;
      default:
        return <InfoIcon />;
    }
  };

  const renderPersonalInfo = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom>
          Personal Information
        </Typography>
        <Typography variant="body2" color="textSecondary" sx={{ mb: 3 }}>
          Please provide your basic personal details for the assessment.
        </Typography>
      </Grid>

      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          id="name"
          name="name"
          label="Full Name"
          value={formik.values.name}
          onChange={formik.handleChange}
          onBlur={formik.handleBlur}
          error={formik.touched.name && Boolean(formik.errors.name)}
          helperText={formik.touched.name && formik.errors.name}
        />
      </Grid>

      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          id="age"
          name="age"
          label="Age"
          type="number"
          value={formik.values.age}
          onChange={formik.handleChange}
          onBlur={formik.handleBlur}
          error={formik.touched.age && Boolean(formik.errors.age)}
          helperText={formik.touched.age && formik.errors.age}
          inputProps={{ min: 1, max: 120 }}
        />
      </Grid>

      <Grid item xs={12} md={6}>
        <FormControl fullWidth error={formik.touched.gender && Boolean(formik.errors.gender)}>
          <InputLabel>Gender</InputLabel>
          <Select
            id="gender"
            name="gender"
            value={formik.values.gender}
            label="Gender"
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
          >
            <MenuItem value="Male">Male</MenuItem>
            <MenuItem value="Female">Female</MenuItem>
            <MenuItem value="Other">Other</MenuItem>
          </Select>
          {formik.touched.gender && formik.errors.gender && (
            <FormHelperText>{formik.errors.gender}</FormHelperText>
          )}
        </FormControl>
      </Grid>

      <Grid item xs={12} md={6}>
        <FormControl fullWidth error={formik.touched.ever_married && Boolean(formik.errors.ever_married)}>
          <InputLabel>Ever Married</InputLabel>
          <Select
            id="ever_married"
            name="ever_married"
            value={formik.values.ever_married}
            label="Ever Married"
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
          >
            <MenuItem value="Yes">Yes</MenuItem>
            <MenuItem value="No">No</MenuItem>
          </Select>
          {formik.touched.ever_married && formik.errors.ever_married && (
            <FormHelperText>{formik.errors.ever_married}</FormHelperText>
          )}
        </FormControl>
      </Grid>
    </Grid>
  );

  const renderMedicalHistory = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom>
          Medical History
        </Typography>
        <Typography variant="body2" color="textSecondary" sx={{ mb: 3 }}>
          Please provide information about your medical conditions and health metrics.
        </Typography>
      </Grid>

      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel>Do you have Hypertension?</InputLabel>
          <Select
            id="hypertension"
            name="hypertension"
            value={formik.values.hypertension}
            label="Do you have Hypertension?"
            onChange={formik.handleChange}
          >
            <MenuItem value={false}>No</MenuItem>
            <MenuItem value={true}>Yes</MenuItem>
          </Select>
        </FormControl>
      </Grid>

      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel>Do you have Heart Disease?</InputLabel>
          <Select
            id="heart_disease"
            name="heart_disease"
            value={formik.values.heart_disease}
            label="Do you have Heart Disease?"
            onChange={formik.handleChange}
          >
            <MenuItem value={false}>No</MenuItem>
            <MenuItem value={true}>Yes</MenuItem>
          </Select>
        </FormControl>
      </Grid>

      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          id="avg_glucose_level"
          name="avg_glucose_level"
          label="Average Glucose Level (mg/dL)"
          type="number"
          value={formik.values.avg_glucose_level}
          onChange={formik.handleChange}
          onBlur={formik.handleBlur}
          error={formik.touched.avg_glucose_level && Boolean(formik.errors.avg_glucose_level)}
          helperText={formik.touched.avg_glucose_level && formik.errors.avg_glucose_level || 'Normal range: 70-140 mg/dL'}
          inputProps={{ min: 50, max: 500, step: 0.1 }}
        />
      </Grid>

      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          id="bmi"
          name="bmi"
          label="Body Mass Index (BMI)"
          type="number"
          value={formik.values.bmi}
          onChange={formik.handleChange}
          onBlur={formik.handleBlur}
          error={formik.touched.bmi && Boolean(formik.errors.bmi)}
          helperText={formik.touched.bmi && formik.errors.bmi || 'BMI = weight(kg) / height(m)²'}
          inputProps={{ min: 10, max: 60, step: 0.1 }}
        />
      </Grid>
    </Grid>
  );

  const renderLifestyleFactors = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom>
          Lifestyle Factors
        </Typography>
        <Typography variant="body2" color="textSecondary" sx={{ mb: 3 }}>
          Information about your lifestyle and work environment.
        </Typography>
      </Grid>

      <Grid item xs={12} md={6}>
        <FormControl fullWidth error={formik.touched.smoking_status && Boolean(formik.errors.smoking_status)}>
          <InputLabel>Smoking Status</InputLabel>
          <Select
            id="smoking_status"
            name="smoking_status"
            value={formik.values.smoking_status}
            label="Smoking Status"
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
          >
            <MenuItem value="never smoked">Never Smoked</MenuItem>
            <MenuItem value="formerly smoked">Formerly Smoked</MenuItem>
            <MenuItem value="smokes">Currently Smokes</MenuItem>
            <MenuItem value="Unknown">Unknown</MenuItem>
          </Select>
          {formik.touched.smoking_status && formik.errors.smoking_status && (
            <FormHelperText>{formik.errors.smoking_status}</FormHelperText>
          )}
        </FormControl>
      </Grid>

      <Grid item xs={12} md={6}>
        <FormControl fullWidth error={formik.touched.residence_type && Boolean(formik.errors.residence_type)}>
          <InputLabel>Residence Type</InputLabel>
          <Select
            id="residence_type"
            name="residence_type"
            value={formik.values.residence_type}
            label="Residence Type"
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
          >
            <MenuItem value="Urban">Urban</MenuItem>
            <MenuItem value="Rural">Rural</MenuItem>
          </Select>
          {formik.touched.residence_type && formik.errors.residence_type && (
            <FormHelperText>{formik.errors.residence_type}</FormHelperText>
          )}
        </FormControl>
      </Grid>

      <Grid item xs={12} md={6}>
        <FormControl fullWidth error={formik.touched.work_type && Boolean(formik.errors.work_type)}>
          <InputLabel>Work Type</InputLabel>
          <Select
            id="work_type"
            name="work_type"
            value={formik.values.work_type}
            label="Work Type"
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
          >
            <MenuItem value="Private">Private Sector</MenuItem>
            <MenuItem value="Self-employed">Self-employed</MenuItem>
            <MenuItem value="Govt_job">Government Job</MenuItem>
            <MenuItem value="children">Student/Child</MenuItem>
            <MenuItem value="Never_worked">Never Worked</MenuItem>
          </Select>
          {formik.touched.work_type && formik.errors.work_type && (
            <FormHelperText>{formik.errors.work_type}</FormHelperText>
          )}
        </FormControl>
      </Grid>

      <Grid item xs={12} md={6}>
        <FormControl fullWidth error={formik.touched.alcohol_consumption && Boolean(formik.errors.alcohol_consumption)}>
          <InputLabel>Alcohol Consumption</InputLabel>
          <Select
            id="alcohol_consumption"
            name="alcohol_consumption"
            value={formik.values.alcohol_consumption}
            label="Alcohol Consumption"
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
          >
            <MenuItem value="Never">Never</MenuItem>
            <MenuItem value="Occasionally">Occasionally</MenuItem>
            <MenuItem value="Regularly">Regularly</MenuItem>
            <MenuItem value="Heavy">Heavy</MenuItem>
          </Select>
          {formik.touched.alcohol_consumption && formik.errors.alcohol_consumption && (
            <FormHelperText>{formik.errors.alcohol_consumption}</FormHelperText>
          )}
        </FormControl>
      </Grid>

      <Grid item xs={12}>
        <FormControl fullWidth>
          <InputLabel>Family History of Stroke</InputLabel>
          <Select
            id="family_history_stroke"
            name="family_history_stroke"
            value={formik.values.family_history_stroke}
            label="Family History of Stroke"
            onChange={formik.handleChange}
          >
            <MenuItem value={false}>No</MenuItem>
            <MenuItem value={true}>Yes</MenuItem>
          </Select>
        </FormControl>
      </Grid>
    </Grid>
  );

  const renderStepContent = (step) => {
    switch (step) {
      case 0:
        return renderPersonalInfo();
      case 1:
        return renderMedicalHistory();
      case 2:
        return renderLifestyleFactors();
      case 3:
        return result ? renderResults() : <LoadingSpinner message="Processing your assessment..." />;
      default:
        return null;
    }
  };

  const renderResults = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom>
          Risk Assessment Results
        </Typography>
      </Grid>

      <Grid item xs={12}>
        <Card
          sx={{
            border: `3px solid ${getRiskLevelColor(result.prediction.risk_level)}`,
            backgroundColor: `${getRiskLevelColor(result.prediction.risk_level)}10`
          }}
        >
          <CardContent sx={{ textAlign: 'center', p: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 2 }}>
              {React.cloneElement(getRiskIcon(result.prediction.risk_level), {
                sx: { fontSize: 48, color: getRiskLevelColor(result.prediction.risk_level), mr: 2 }
              })}
              <Typography variant="h3" fontWeight="bold" color={getRiskLevelColor(result.prediction.risk_level)}>
                {result.prediction.risk_level} RISK
              </Typography>
            </Box>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Probability Score: {(result.prediction.probability_score * 100).toFixed(1)}%
            </Typography>
            <Typography variant="body1" color="textSecondary">
              {result.prediction.recommendations.message}
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      {result.prediction.risk_factors.length > 0 && (
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Identified Risk Factors
              </Typography>
              <List>
                {result.prediction.risk_factors.map((factor, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <WarningIcon color="warning" />
                    </ListItemIcon>
                    <ListItemText primary={factor} />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      )}

      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Recommended Actions
            </Typography>
            <List>
              {result.prediction.recommendations.actions.map((action, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    <CheckIcon color="success" />
                  </ListItemIcon>
                  <ListItemText primary={action} />
                </ListItem>
              ))}
            </List>
            {result.prediction.recommendations.specific_actions && (
              <>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                  Specific Recommendations
                </Typography>
                <List>
                  {result.prediction.recommendations.specific_actions.map((action, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <InfoIcon color="info" />
                      </ListItemIcon>
                      <ListItemText primary={action} />
                    </ListItem>
                  ))}
                </List>
              </>
            )}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  if (loading && activeStep === 3) {
    return <LoadingSpinner fullScreen message="Analyzing your health data..." />;
  }

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Paper elevation={2} sx={{ p: 4 }}>
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Typography variant="h4" component="h1" fontWeight="bold" color="primary" sx={{ mb: 2 }}>
            🧠 Stroke Risk Assessment
          </Typography>
          <Typography variant="body1" color="textSecondary">
            Complete this comprehensive assessment to understand your stroke risk level
          </Typography>
        </Box>

        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label, index) => (
            <Step key={label} completed={isStepComplete(index)}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        <Box component="form" onSubmit={formik.handleSubmit}>
          {renderStepContent(activeStep)}

          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
            <Button
              onClick={handleBack}
              disabled={activeStep === 0}
              variant="outlined"
            >
              Back
            </Button>

            <Box>
              {activeStep === steps.length - 2 ? (
                <Button
                  type="submit"
                  variant="contained"
                  disabled={loading || !formik.isValid}
                  startIcon={<AssessmentIcon />}
                >
                  {loading ? 'Analyzing...' : 'Get Assessment'}
                </Button>
              ) : activeStep === steps.length - 1 ? (
                <Button
                  variant="contained"
                  onClick={() => window.location.reload()}
                >
                  New Assessment
                </Button>
              ) : (
                <Button
                  onClick={handleNext}
                  variant="contained"
                  disabled={!isStepComplete(activeStep)}
                >
                  Next
                </Button>
              )}
            </Box>
          </Box>
        </Box>
      </Paper>

      {/* Disclaimer Dialog */}
      <Dialog open={showResult} onClose={() => setShowResult(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          Medical Disclaimer
        </DialogTitle>
        <DialogContent>
          <Alert severity="warning" sx={{ mb: 2 }}>
            <Typography variant="body2">
              <strong>Important:</strong> This assessment is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.
            </Typography>
          </Alert>
          {result && (
            <Typography variant="body2" color="textSecondary">
              Your assessment has been saved to your history. You can view it anytime from your dashboard.
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowResult(false)} color="primary">
            I Understand
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default PredictionPage;
