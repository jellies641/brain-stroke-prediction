import React, { createContext, useContext, useReducer } from "react";
import axios from "axios";
import { toast } from "react-toastify";

// Initial state
const initialState = {
  predictions: [],
  currentPrediction: null,
  isLoading: false,
  statistics: null,
};

// Action types
const actionTypes = {
  SET_LOADING: "SET_LOADING",
  SET_PREDICTIONS: "SET_PREDICTIONS",
  ADD_PREDICTION: "ADD_PREDICTION",
  SET_CURRENT_PREDICTION: "SET_CURRENT_PREDICTION",
  SET_STATISTICS: "SET_STATISTICS",
  CLEAR_PREDICTIONS: "CLEAR_PREDICTIONS",
  DELETE_PREDICTION: "DELETE_PREDICTION",
};

// Prediction reducer
const predictionReducer = (state, action) => {
  switch (action.type) {
    case actionTypes.SET_LOADING:
      return {
        ...state,
        isLoading: action.payload,
      };

    case actionTypes.SET_PREDICTIONS:
      return {
        ...state,
        predictions: action.payload,
        isLoading: false,
      };

    case actionTypes.ADD_PREDICTION:
      return {
        ...state,
        predictions: [action.payload, ...state.predictions],
        currentPrediction: action.payload,
        isLoading: false,
      };

    case actionTypes.SET_CURRENT_PREDICTION:
      return {
        ...state,
        currentPrediction: action.payload,
      };

    case actionTypes.SET_STATISTICS:
      return {
        ...state,
        statistics: action.payload,
      };

    case actionTypes.DELETE_PREDICTION:
      return {
        ...state,
        predictions: state.predictions.filter(
          (pred) => pred.id !== action.payload,
        ),
      };

    case actionTypes.CLEAR_PREDICTIONS:
      return {
        ...initialState,
      };

    default:
      return state;
  }
};

// Create context
const PredictionContext = createContext();

// Prediction provider component
export const PredictionProvider = ({ children }) => {
  const [state, dispatch] = useReducer(predictionReducer, initialState);

  // Make a new prediction
  const makePrediction = async (patientData) => {
    try {
      dispatch({ type: actionTypes.SET_LOADING, payload: true });

      const response = await axios.post("/api/predict", patientData);

      const prediction = {
        id: Date.now(), // Simple ID generation for demo
        ...response.data.prediction,
        patient_summary: response.data.patient_summary,
        created_at: new Date().toISOString(),
        timestamp: response.data.timestamp,
      };

      dispatch({
        type: actionTypes.ADD_PREDICTION,
        payload: prediction,
      });

      toast.success("Prediction completed successfully!");
      return { success: true, data: prediction };
    } catch (error) {
      dispatch({ type: actionTypes.SET_LOADING, payload: false });

      const message =
        error.response?.data?.message || "Prediction failed. Please try again.";
      toast.error(message);

      return { success: false, message };
    }
  };

  // Get prediction history
  const getPredictionHistory = async () => {
    try {
      dispatch({ type: actionTypes.SET_LOADING, payload: true });

      const response = await axios.get("/api/history");

      dispatch({
        type: actionTypes.SET_PREDICTIONS,
        payload: response.data.predictions || [],
      });

      return { success: true };
    } catch (error) {
      dispatch({ type: actionTypes.SET_LOADING, payload: false });

      // Don't show error toast for authentication errors
      if (error.response?.status === 401) {
        console.log("Authentication required for prediction history");
        return { success: false, message: "Authentication required" };
      }

      const message =
        error.response?.data?.message || "Failed to load prediction history.";
      toast.error(message);

      return { success: false, message };
    }
  };

  // Get prediction statistics
  const getStatistics = async () => {
    try {
      const response = await axios.get("/api/statistics");

      dispatch({
        type: actionTypes.SET_STATISTICS,
        payload: response.data.statistics,
      });

      return { success: true };
    } catch (error) {
      // Don't log errors for authentication issues
      if (error.response?.status === 401) {
        console.log("Authentication required for statistics");
        return { success: false, message: "Authentication required" };
      }

      const message =
        error.response?.data?.message || "Failed to load statistics.";
      console.error("Statistics error:", message);

      return { success: false, message };
    }
  };

  // Delete a prediction
  const deletePrediction = async (predictionId) => {
    try {
      await axios.delete(`/api/predictions/${predictionId}`);

      dispatch({
        type: actionTypes.DELETE_PREDICTION,
        payload: predictionId,
      });

      toast.success("Prediction deleted successfully!");
      return { success: true };
    } catch (error) {
      const message =
        error.response?.data?.message || "Failed to delete prediction.";
      toast.error(message);

      return { success: false, message };
    }
  };

  // Set current prediction for viewing details
  const setCurrentPrediction = (prediction) => {
    dispatch({
      type: actionTypes.SET_CURRENT_PREDICTION,
      payload: prediction,
    });
  };

  // Clear all predictions (e.g., on logout)
  const clearPredictions = () => {
    dispatch({ type: actionTypes.CLEAR_PREDICTIONS });
  };

  // Get risk level color
  const getRiskColor = (riskLevel) => {
    switch (riskLevel?.toUpperCase()) {
      case "LOW":
        return "#2e7d32"; // Green
      case "MODERATE":
        return "#f57f17"; // Orange
      case "HIGH":
        return "#d32f2f"; // Red
      default:
        return "#757575"; // Gray
    }
  };

  // Get risk level emoji
  const getRiskEmoji = (riskLevel) => {
    switch (riskLevel?.toUpperCase()) {
      case "LOW":
        return "🟢";
      case "MODERATE":
        return "🟡";
      case "HIGH":
        return "🔴";
      default:
        return "⚪";
    }
  };

  // Format prediction data for display
  const formatPrediction = (prediction) => {
    if (!prediction) return null;

    return {
      ...prediction,
      risk_color: getRiskColor(prediction.risk_level),
      risk_emoji: getRiskEmoji(prediction.risk_level),
      probability_percentage: Math.round(prediction.probability_score * 100),
      formatted_date: new Date(prediction.created_at).toLocaleDateString(),
      formatted_time: new Date(prediction.created_at).toLocaleTimeString(),
    };
  };

  // Calculate summary statistics
  const calculateSummaryStats = (predictions) => {
    if (!predictions || predictions.length === 0) {
      return {
        total: 0,
        low_risk: 0,
        moderate_risk: 0,
        high_risk: 0,
        average_age: 0,
        recent_predictions: 0,
      };
    }

    const stats = {
      total: predictions.length,
      low_risk: predictions.filter((p) => p.risk_level === "LOW").length,
      moderate_risk: predictions.filter((p) => p.risk_level === "MODERATE")
        .length,
      high_risk: predictions.filter((p) => p.risk_level === "HIGH").length,
      average_age:
        predictions.reduce((sum, p) => sum + (p.patient_summary?.age || 0), 0) /
        predictions.length,
      recent_predictions: predictions.filter((p) => {
        const predDate = new Date(p.created_at);
        const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
        return predDate > weekAgo;
      }).length,
    };

    return stats;
  };

  // Context value
  const value = {
    // State
    predictions: state.predictions,
    currentPrediction: state.currentPrediction,
    isLoading: state.isLoading,
    statistics: state.statistics,

    // Actions
    makePrediction,
    getPredictionHistory,
    getStatistics,
    deletePrediction,
    setCurrentPrediction,
    clearPredictions,

    // Helper functions
    getRiskColor,
    getRiskEmoji,
    formatPrediction,
    calculateSummaryStats,
  };

  return (
    <PredictionContext.Provider value={value}>
      {children}
    </PredictionContext.Provider>
  );
};

// Custom hook to use prediction context
export const usePrediction = () => {
  const context = useContext(PredictionContext);

  if (!context) {
    throw new Error("usePrediction must be used within a PredictionProvider");
  }

  return context;
};

export default PredictionContext;
