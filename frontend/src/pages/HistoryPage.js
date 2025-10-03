import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Pagination,
  TextField,
  InputAdornment,
  FormControl,
  InputLabel,
  Select,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  MoreVert as MoreVertIcon,
  Visibility as ViewIcon,
  Delete as DeleteIcon,
  Download as DownloadIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  FileDownload as ExportIcon
} from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';
import axios from 'axios';
import { toast } from 'react-toastify';
import LoadingSpinner from '../components/Common/LoadingSpinner';

const HistoryPage = () => {
  const { user } = useAuth();

  const [loading, setLoading] = useState(true);
  const [predictions, setPredictions] = useState([]);
  const [pagination, setPagination] = useState({});
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [showDetails, setShowDetails] = useState(false);
  const [anchorEl, setAnchorEl] = useState(null);
  const [menuPrediction, setMenuPrediction] = useState(null);

  // Filters and search
  const [searchTerm, setSearchTerm] = useState('');
  const [riskFilter, setRiskFilter] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);

  useEffect(() => {
    fetchPredictions();
  }, [currentPage, searchTerm, riskFilter]);

  const fetchPredictions = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/history', {
        params: {
          page: currentPage,
          per_page: itemsPerPage,
          search: searchTerm,
          risk_level: riskFilter
        }
      });

      setPredictions(response.data.predictions);
      setPagination(response.data.pagination);
    } catch (error) {
      toast.error('Failed to load prediction history');
      console.error('History fetch error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleMenuOpen = (event, prediction) => {
    setAnchorEl(event.currentTarget);
    setMenuPrediction(prediction);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setMenuPrediction(null);
  };

  const handleViewDetails = async (predictionId) => {
    try {
      const response = await axios.get(`/history/${predictionId}`);
      setSelectedPrediction(response.data.prediction);
      setShowDetails(true);
    } catch (error) {
      toast.error('Failed to load prediction details');
    }
    handleMenuClose();
  };

  const handleDeletePrediction = async (predictionId) => {
    if (window.confirm('Are you sure you want to delete this prediction?')) {
      try {
        await axios.delete(`/history/${predictionId}`);
        toast.success('Prediction deleted successfully');
        fetchPredictions();
      } catch (error) {
        toast.error('Failed to delete prediction');
      }
    }
    handleMenuClose();
  };

  const handleExportData = async () => {
    try {
      const response = await axios.get('/history/export', {
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `stroke-predictions-${new Date().toISOString().split('T')[0]}.csv`);
      document.body.appendChild(link);
      link.click();
      link.remove();

      toast.success('Data exported successfully');
    } catch (error) {
      toast.error('Failed to export data');
    }
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'LOW': return 'success';
      case 'MODERATE': return 'warning';
      case 'HIGH': return 'error';
      default: return 'default';
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

  const handlePageChange = (event, value) => {
    setCurrentPage(value);
  };

  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value);
    setCurrentPage(1);
  };

  const handleRiskFilterChange = (event) => {
    setRiskFilter(event.target.value);
    setCurrentPage(1);
  };

  if (loading && predictions.length === 0) {
    return <LoadingSpinner fullScreen message="Loading your prediction history..." />;
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" component="h1" fontWeight="bold" sx={{ mb: 1 }}>
            📊 Prediction History
          </Typography>
          <Typography variant="body1" color="textSecondary">
            View and manage your stroke risk assessments
          </Typography>
        </Box>
        <Button
          variant="outlined"
          startIcon={<ExportIcon />}
          onClick={handleExportData}
          disabled={predictions.length === 0}
        >
          Export Data
        </Button>
      </Box>

      {/* Filters */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                placeholder="Search by name..."
                value={searchTerm}
                onChange={handleSearchChange}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon />
                    </InputAdornment>
                  )
                }}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel>Filter by Risk Level</InputLabel>
                <Select
                  value={riskFilter}
                  label="Filter by Risk Level"
                  onChange={handleRiskFilterChange}
                >
                  <MenuItem value="">All Levels</MenuItem>
                  <MenuItem value="LOW">Low Risk</MenuItem>
                  <MenuItem value="MODERATE">Moderate Risk</MenuItem>
                  <MenuItem value="HIGH">High Risk</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={2}>
              <Button
                fullWidth
                variant="text"
                onClick={() => {
                  setSearchTerm('');
                  setRiskFilter('');
                  setCurrentPage(1);
                }}
              >
                Clear Filters
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Results */}
      {predictions.length === 0 ? (
        <Card>
          <CardContent sx={{ textAlign: 'center', py: 8 }}>
            <Typography variant="h6" color="textSecondary" sx={{ mb: 2 }}>
              No predictions found
            </Typography>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 3 }}>
              {searchTerm || riskFilter
                ? 'Try adjusting your search filters'
                : 'Complete your first stroke risk assessment to see results here'
              }
            </Typography>
            {!searchTerm && !riskFilter && (
              <Button
                variant="contained"
                onClick={() => window.location.href = '/predict'}
              >
                Start Assessment
              </Button>
            )}
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Desktop Table View */}
          <Box sx={{ display: { xs: 'none', md: 'block' } }}>
            <TableContainer component={Paper} elevation={2}>
              <Table>
                <TableHead>
                  <TableRow sx={{ backgroundColor: 'grey.50' }}>
                    <TableCell>Name</TableCell>
                    <TableCell>Date</TableCell>
                    <TableCell>Age</TableCell>
                    <TableCell>BMI</TableCell>
                    <TableCell>Glucose</TableCell>
                    <TableCell>Risk Level</TableCell>
                    <TableCell>Score</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {predictions.map((prediction) => (
                    <TableRow key={prediction.id} hover>
                      <TableCell>
                        <Typography variant="subtitle2" fontWeight="medium">
                          {prediction.name}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {formatDate(prediction.created_at)}
                        </Typography>
                      </TableCell>
                      <TableCell>{prediction.age}</TableCell>
                      <TableCell>{prediction.bmi}</TableCell>
                      <TableCell>{prediction.avg_glucose_level} mg/dL</TableCell>
                      <TableCell>
                        <Chip
                          icon={getRiskIcon(prediction.risk_level)}
                          label={prediction.risk_level}
                          color={getRiskColor(prediction.risk_level)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" fontWeight="medium">
                          {(prediction.probability_score * 100).toFixed(1)}%
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <IconButton
                          onClick={(e) => handleMenuOpen(e, prediction)}
                          size="small"
                        >
                          <MoreVertIcon />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>

          {/* Mobile Card View */}
          <Box sx={{ display: { xs: 'block', md: 'none' } }}>
            <Grid container spacing={2}>
              {predictions.map((prediction) => (
                <Grid item xs={12} key={prediction.id}>
                  <Card>
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                        <Box>
                          <Typography variant="h6" fontWeight="bold">
                            {prediction.name}
                          </Typography>
                          <Typography variant="body2" color="textSecondary">
                            {formatDate(prediction.created_at)}
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Chip
                            icon={getRiskIcon(prediction.risk_level)}
                            label={prediction.risk_level}
                            color={getRiskColor(prediction.risk_level)}
                            size="small"
                          />
                          <IconButton
                            onClick={(e) => handleMenuOpen(e, prediction)}
                            size="small"
                          >
                            <MoreVertIcon />
                          </IconButton>
                        </Box>
                      </Box>
                      <Grid container spacing={2}>
                        <Grid item xs={3}>
                          <Typography variant="caption" color="textSecondary">Age</Typography>
                          <Typography variant="body2" fontWeight="medium">{prediction.age}</Typography>
                        </Grid>
                        <Grid item xs={3}>
                          <Typography variant="caption" color="textSecondary">BMI</Typography>
                          <Typography variant="body2" fontWeight="medium">{prediction.bmi}</Typography>
                        </Grid>
                        <Grid item xs={3}>
                          <Typography variant="caption" color="textSecondary">Glucose</Typography>
                          <Typography variant="body2" fontWeight="medium">{prediction.avg_glucose_level}</Typography>
                        </Grid>
                        <Grid item xs={3}>
                          <Typography variant="caption" color="textSecondary">Score</Typography>
                          <Typography variant="body2" fontWeight="medium">
                            {(prediction.probability_score * 100).toFixed(1)}%
                          </Typography>
                        </Grid>
                      </Grid>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>

          {/* Pagination */}
          {pagination.pages > 1 && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
              <Pagination
                count={pagination.pages}
                page={currentPage}
                onChange={handlePageChange}
                color="primary"
                size="large"
                showFirstButton
                showLastButton
              />
            </Box>
          )}
        </>
      )}

      {/* Action Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => handleViewDetails(menuPrediction?.id)}>
          <ListItemIcon>
            <ViewIcon fontSize="small" />
          </ListItemIcon>
          View Details
        </MenuItem>
        <MenuItem onClick={() => handleDeletePrediction(menuPrediction?.id)}>
          <ListItemIcon>
            <DeleteIcon fontSize="small" />
          </ListItemIcon>
          Delete
        </MenuItem>
      </Menu>

      {/* Details Dialog */}
      <Dialog
        open={showDetails}
        onClose={() => setShowDetails(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {selectedPrediction && getRiskIcon(selectedPrediction.risk_level)}
            <Box>
              <Typography variant="h6">
                Assessment Details: {selectedPrediction?.name}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {selectedPrediction && formatDate(selectedPrediction.created_at)}
              </Typography>
            </Box>
          </Box>
        </DialogTitle>
        <DialogContent>
          {selectedPrediction && (
            <>
              {/* Risk Level */}
              <Card sx={{ mb: 3, border: `2px solid`, borderColor: getRiskColor(selectedPrediction.risk_level) + '.main' }}>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Chip
                    icon={getRiskIcon(selectedPrediction.risk_level)}
                    label={`${selectedPrediction.risk_level} RISK`}
                    color={getRiskColor(selectedPrediction.risk_level)}
                    size="large"
                    sx={{ mb: 2, fontSize: '1.1rem', px: 2 }}
                  />
                  <Typography variant="h6">
                    Probability Score: {(selectedPrediction.probability_score * 100).toFixed(1)}%
                  </Typography>
                </CardContent>
              </Card>

              {/* Health Data */}
              <Typography variant="h6" gutterBottom>Health Information</Typography>
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={6} sm={3}>
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="caption" color="textSecondary">Age</Typography>
                    <Typography variant="h6">{selectedPrediction.age}</Typography>
                  </Paper>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="caption" color="textSecondary">Gender</Typography>
                    <Typography variant="h6">{selectedPrediction.gender}</Typography>
                  </Paper>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="caption" color="textSecondary">BMI</Typography>
                    <Typography variant="h6">{selectedPrediction.bmi}</Typography>
                  </Paper>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="caption" color="textSecondary">Glucose</Typography>
                    <Typography variant="h6">{selectedPrediction.avg_glucose_level}</Typography>
                  </Paper>
                </Grid>
              </Grid>

              {/* Risk Factors */}
              {selectedPrediction.risk_factors && selectedPrediction.risk_factors.length > 0 && (
                <>
                  <Typography variant="h6" gutterBottom>Identified Risk Factors</Typography>
                  <List dense>
                    {selectedPrediction.risk_factors.map((factor, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <WarningIcon color="warning" />
                        </ListItemIcon>
                        <ListItemText primary={factor} />
                      </ListItem>
                    ))}
                  </List>
                </>
              )}

              {/* Recommendations */}
              {selectedPrediction.recommendations && (
                <>
                  <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Recommendations</Typography>
                  <Alert severity="info" sx={{ mb: 2 }}>
                    {selectedPrediction.recommendations.message}
                  </Alert>
                  <List dense>
                    {selectedPrediction.recommendations.actions?.map((action, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <CheckCircleIcon color="success" />
                        </ListItemIcon>
                        <ListItemText primary={action} />
                      </ListItem>
                    ))}
                  </List>
                </>
              )}
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowDetails(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default HistoryPage;
