import React, { useEffect, useState } from "react";
import {
  Container,
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from "@mui/material";
import {
  Visibility,
  Delete,
  FileDownload,
  Psychology,
  Warning,
  LocalHospital,
  Close,
} from "@mui/icons-material";
import { usePrediction } from "../contexts/PredictionContext";

const History = () => {
  const {
    predictions,
    getPredictionHistory,
    deletePrediction,
    isLoading,
    getRiskColor,
    getRiskEmoji,
  } = usePrediction();

  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [predictionToDelete, setPredictionToDelete] = useState(null);

  useEffect(() => {
    getPredictionHistory();
  }, []);

  const handleViewDetails = (prediction) => {
    setSelectedPrediction(prediction);
  };

  const handleCloseDetails = () => {
    setSelectedPrediction(null);
  };

  const handleDeleteClick = (prediction) => {
    setPredictionToDelete(prediction);
    setDeleteConfirmOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (predictionToDelete) {
      await deletePrediction(predictionToDelete.id);
      setDeleteConfirmOpen(false);
      setPredictionToDelete(null);
    }
  };

  const handleDeleteCancel = () => {
    setDeleteConfirmOpen(false);
    setPredictionToDelete(null);
  };

  const getRiskColorMui = (riskLevel) => {
    switch (riskLevel?.toUpperCase()) {
      case "LOW":
        return "success";
      case "MODERATE":
        return "warning";
      case "HIGH":
        return "error";
      default:
        return "default";
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom fontWeight="bold">
          📊 Assessment History
        </Typography>
        <Typography variant="h6" color="text.secondary">
          View and manage your stroke risk assessment history
        </Typography>
      </Box>

      {/* History Table */}
      <Paper sx={{ width: "100%", overflow: "hidden" }}>
        {isLoading ? (
          <LinearProgress />
        ) : predictions.length === 0 ? (
          <Box sx={{ textAlign: "center", py: 8 }}>
            <Psychology sx={{ fontSize: 64, color: "text.secondary", mb: 2 }} />
            <Typography variant="h5" color="text.secondary" gutterBottom>
              No assessments found
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
              You haven't completed any risk assessments yet.
            </Typography>
            <Button variant="contained" href="/prediction" sx={{ mt: 2 }}>
              Take Your First Assessment
            </Button>
          </Box>
        ) : (
          <TableContainer sx={{ maxHeight: 600 }}>
            <Table stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell>Date</TableCell>
                  <TableCell>Risk Level</TableCell>
                  <TableCell>Probability</TableCell>
                  <TableCell>Age</TableCell>
                  <TableCell>BMI</TableCell>
                  <TableCell>Glucose</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {predictions.map((prediction, index) => (
                  <TableRow hover key={prediction.id || index}>
                    <TableCell>
                      <Typography variant="body2">
                        {new Date(prediction.created_at || Date.now()).toLocaleDateString()}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {new Date(prediction.created_at || Date.now()).toLocaleTimeString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={`${getRiskEmoji(prediction.risk_level)} ${prediction.risk_level || "Unknown"}`}
                        color={getRiskColorMui(prediction.risk_level)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" fontWeight="bold">
                        {Math.round((prediction.probability_score || 0) * 100)}%
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={(prediction.probability_score || 0) * 100}
                        color={getRiskColorMui(prediction.risk_level)}
                        sx={{ mt: 0.5, height: 4, borderRadius: 2 }}
                      />
                    </TableCell>
                    <TableCell>
                      {prediction.patient_summary?.age || "N/A"}
                    </TableCell>
                    <TableCell>
                      {prediction.patient_summary?.bmi || "N/A"}
                    </TableCell>
                    <TableCell>
                      {prediction.patient_summary?.glucose_level || "N/A"}
                    </TableCell>
                    <TableCell>
                      <IconButton
                        size="small"
                        onClick={() => handleViewDetails(prediction)}
                        title="View Details"
                      >
                        <Visibility />
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={() => handleDeleteClick(prediction)}
                        title="Delete"
                        color="error"
                      >
                        <Delete />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Paper>

      {/* Details Dialog */}
      <Dialog
        open={!!selectedPrediction}
        onClose={handleCloseDetails}
        maxWidth="md"
        fullWidth
      >
        {selectedPrediction && (
          <>
            <DialogTitle sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <Typography variant="h5" fontWeight="bold">
                Assessment Details
              </Typography>
              <IconButton onClick={handleCloseDetails}>
                <Close />
              </IconButton>
            </DialogTitle>
            <DialogContent>
              <Grid container spacing={3}>
                {/* Risk Summary */}
                <Grid item xs={12} md={4}>
                  <Card>
                    <CardContent sx={{ textAlign: "center" }}>
                      <Typography variant="h4" fontWeight="bold" gutterBottom>
                        {getRiskEmoji(selectedPrediction.risk_level)}
                      </Typography>
                      <Typography variant="h5" fontWeight="bold" gutterBottom>
                        {selectedPrediction.risk_level || "Unknown"}
                      </Typography>
                      <Chip
                        label={`${Math.round((selectedPrediction.probability_score || 0) * 100)}% Risk`}
                        color={getRiskColorMui(selectedPrediction.risk_level)}
                      />
                    </CardContent>
                  </Card>
                </Grid>

                {/* Patient Info */}
                <Grid item xs={12} md={8}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom fontWeight="bold">
                        Patient Information
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">Age</Typography>
                          <Typography variant="body1">{selectedPrediction.patient_summary?.age || "N/A"}</Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">Gender</Typography>
                          <Typography variant="body1">{selectedPrediction.patient_summary?.gender || "N/A"}</Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">BMI</Typography>
                          <Typography variant="body1">{selectedPrediction.patient_summary?.bmi || "N/A"}</Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">Glucose Level</Typography>
                          <Typography variant="body1">{selectedPrediction.patient_summary?.glucose_level || "N/A"} mg/dL</Typography>
                        </Grid>
                      </Grid>
                    </CardContent>
                  </Card>
                </Grid>

                {/* Risk Factors */}
                <Grid item xs={12} md={6}>
                  <Card sx={{ height: "100%" }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom fontWeight="bold">
                        <Warning sx={{ mr: 1, verticalAlign: "middle" }} />
                        Risk Factors
                      </Typography>
                      <List dense>
                        {selectedPrediction.risk_factors && selectedPrediction.risk_factors.map((factor, index) => (
                          <ListItem key={index} sx={{ px: 0 }}>
                            <ListItemIcon>
                              <Warning color="error" fontSize="small" />
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
                    <CardContent>
                      <Typography variant="h6" gutterBottom fontWeight="bold">
                        <LocalHospital sx={{ mr: 1, verticalAlign: "middle" }} />
                        Recommendations
                      </Typography>
                      <List dense>
                        {selectedPrediction.recommendations && selectedPrediction.recommendations.map((rec, index) => (
                          <ListItem key={index} sx={{ px: 0 }}>
                            <ListItemIcon>
                              <LocalHospital color="primary" fontSize="small" />
                            </ListItemIcon>
                            <ListItemText primary={rec} />
                          </ListItem>
                        ))}
                      </List>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button
                variant="outlined"
                startIcon={<FileDownload />}
                onClick={() => window.print()}
              >
                Print Report
              </Button>
              <Button onClick={handleCloseDetails} variant="contained">
                Close
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteConfirmOpen} onClose={handleDeleteCancel}>
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this assessment? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteCancel}>Cancel</Button>
          <Button onClick={handleDeleteConfirm} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default History;
