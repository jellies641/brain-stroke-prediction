import React from "react";
import { Box, Paper, Typography, Grid } from "@mui/material";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

const ModelPerformanceChart = () => {
  // Model comparison data
  const modelComparison = [
    { name: "Logistic Regression", accuracy: 99.5, selected: true },
    { name: "Random Forest", accuracy: 97.8, selected: false },
    { name: "SVM", accuracy: 96.3, selected: false },
    { name: "Gradient Boosting", accuracy: 98.1, selected: false },
  ];

  // Performance metrics breakdown
  const metricsData = [
    { metric: "Accuracy", value: 99.5 },
    { metric: "Precision", value: 98.7 },
    { metric: "Recall", value: 99.2 },
    { metric: "F1-Score", value: 98.9 },
  ];

  // Risk distribution data
  const riskDistribution = [
    { name: "Low Risk", value: 76, color: "#4caf50" },
    { name: "Moderate Risk", value: 19, color: "#ff9800" },
    { name: "High Risk", value: 5, color: "#f44336" },
  ];

  // Training progress simulation
  const trainingProgress = [
    { epoch: 1, accuracy: 88.5, loss: 0.32 },
    { epoch: 2, accuracy: 92.3, loss: 0.25 },
    { epoch: 3, accuracy: 95.7, loss: 0.18 },
    { epoch: 4, accuracy: 97.8, loss: 0.12 },
    { epoch: 5, accuracy: 98.9, loss: 0.08 },
    { epoch: 6, accuracy: 99.5, loss: 0.05 },
  ];

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <Paper sx={{ p: 2, border: "1px solid #ccc" }}>
          <Typography variant="body2">
            {`${label}: ${payload[0].value}%`}
          </Typography>
        </Paper>
      );
    }
    return null;
  };

  return (
    <Box sx={{ width: "100%" }}>
      <Grid container spacing={4}>
        {/* Model Comparison Chart */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: 350 }}>
            <Typography
              variant="h6"
              gutterBottom
              fontWeight="bold"
              color="primary.main"
            >
              Model Algorithm Comparison
            </Typography>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={modelComparison}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="name"
                  angle={-45}
                  textAnchor="end"
                  height={80}
                  fontSize={12}
                />
                <YAxis domain={[95, 100]} />
                <Tooltip content={<CustomTooltip />} />
                <Bar
                  dataKey="accuracy"
                  fill={(entry) => (entry.selected ? "#1976d2" : "#90caf9")}
                >
                  {modelComparison.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.selected ? "#1976d2" : "#90caf9"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Risk Distribution Pie Chart */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: 350 }}>
            <Typography
              variant="h6"
              gutterBottom
              fontWeight="bold"
              color="secondary.main"
            >
              Risk Category Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={riskDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {riskDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Performance Metrics */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: 350 }}>
            <Typography
              variant="h6"
              gutterBottom
              fontWeight="bold"
              color="success.main"
            >
              Model Performance Metrics
            </Typography>
            <Box sx={{ mt: 2 }}>
              {metricsData.map((metric, index) => (
                <Box key={index} sx={{ mb: 2 }}>
                  <Box
                    sx={{
                      display: "flex",
                      justifyContent: "space-between",
                      mb: 0.5,
                    }}
                  >
                    <Typography variant="body2" fontWeight="medium">
                      {metric.metric}
                    </Typography>
                    <Typography
                      variant="body2"
                      fontWeight="bold"
                      color="success.main"
                    >
                      {metric.value}%
                    </Typography>
                  </Box>
                  <Box
                    sx={{
                      width: "100%",
                      height: 8,
                      backgroundColor: "grey.200",
                      borderRadius: 4,
                      overflow: "hidden",
                    }}
                  >
                    <Box
                      sx={{
                        width: `${metric.value}%`,
                        height: "100%",
                        backgroundColor: "#4caf50",
                        borderRadius: 4,
                        transition: "width 1s ease-in-out",
                      }}
                    />
                  </Box>
                </Box>
              ))}
            </Box>
          </Paper>
        </Grid>

        {/* Training Progress */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: 350 }}>
            <Typography
              variant="h6"
              gutterBottom
              fontWeight="bold"
              color="info.main"
            >
              Training Progress Simulation
            </Typography>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={trainingProgress}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" />
                <YAxis yAxisId="left" domain={[85, 100]} />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  domain={[0.05, 0.35]}
                />
                <Tooltip />
                <Legend />
                <Bar
                  yAxisId="left"
                  dataKey="accuracy"
                  fill="#2196f3"
                  name="Accuracy (%)"
                />
                <Bar
                  yAxisId="right"
                  dataKey="loss"
                  fill="#ff5722"
                  name="Loss"
                />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ModelPerformanceChart;
