import React, { createContext, useContext, useReducer, useEffect } from "react";
import axios from "axios";
import { toast } from "react-toastify";

// Initial state
const initialState = {
  user: null,
  token: null,
  isLoading: false,
  isAuthenticated: false,
};

// Action types
const actionTypes = {
  AUTH_START: "AUTH_START",
  AUTH_SUCCESS: "AUTH_SUCCESS",
  AUTH_FAIL: "AUTH_FAIL",
  LOGOUT: "LOGOUT",
  SET_LOADING: "SET_LOADING",
  UPDATE_USER: "UPDATE_USER",
};

// Auth reducer
const authReducer = (state, action) => {
  switch (action.type) {
    case actionTypes.AUTH_START:
      return {
        ...state,
        isLoading: true,
      };

    case actionTypes.AUTH_SUCCESS:
      return {
        ...state,
        isLoading: false,
        isAuthenticated: true,
        user: action.payload.user,
        token: action.payload.token,
      };

    case actionTypes.AUTH_FAIL:
      return {
        ...state,
        isLoading: false,
        isAuthenticated: false,
        user: null,
        token: null,
      };

    case actionTypes.LOGOUT:
      return {
        ...initialState,
      };

    case actionTypes.SET_LOADING:
      return {
        ...state,
        isLoading: action.payload,
      };

    case actionTypes.UPDATE_USER:
      return {
        ...state,
        user: { ...state.user, ...action.payload },
      };

    default:
      return state;
  }
};

// Create context
const AuthContext = createContext();

// Auth provider component
export const AuthProvider = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState);

  // Configure axios defaults
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (token) {
      axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;
      // Validate token on app load
      validateToken(token);
    }
  }, []);

  // Set axios interceptors
  useEffect(() => {
    // Request interceptor
    const requestInterceptor = axios.interceptors.request.use(
      (config) => {
        if (state.token) {
          config.headers.Authorization = `Bearer ${state.token}`;
        }
        return config;
      },
      (error) => Promise.reject(error),
    );

    // Response interceptor
    const responseInterceptor = axios.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          logout();
          toast.error("Session expired. Please login again.");
        }
        return Promise.reject(error);
      },
    );

    return () => {
      axios.interceptors.request.eject(requestInterceptor);
      axios.interceptors.response.eject(responseInterceptor);
    };
  }, [state.token]);

  // Validate token
  const validateToken = async (token) => {
    try {
      dispatch({ type: actionTypes.SET_LOADING, payload: true });

      // Try API token validation
      const response = await axios.get("/api/auth/validate", {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (response.data.valid) {
        dispatch({
          type: actionTypes.AUTH_SUCCESS,
          payload: {
            user: response.data.user,
            token: token,
          },
        });
      } else {
        logout();
      }
    } catch (error) {
      console.error("Token validation failed:", error);
      logout();
    } finally {
      dispatch({ type: actionTypes.SET_LOADING, payload: false });
    }
  };

  // Login function
  const login = async (email, password) => {
    try {
      dispatch({ type: actionTypes.AUTH_START });

      // Try API login
      const response = await axios.post("/api/auth/login", {
        email,
        password,
      });

      const { user, access_token } = response.data;

      // Store token in localStorage
      localStorage.setItem("token", access_token);
      localStorage.setItem("user", JSON.stringify(user));

      // Set axios default header
      axios.defaults.headers.common["Authorization"] = `Bearer ${access_token}`;

      dispatch({
        type: actionTypes.AUTH_SUCCESS,
        payload: {
          user,
          token: access_token,
        },
      });

      toast.success(`Welcome back, ${user.name}!`);
      return { success: true };
    } catch (error) {
      const message = error.response?.data?.message || "Login failed";

      dispatch({ type: actionTypes.AUTH_FAIL });
      toast.error(message);

      return { success: false, message };
    }
  };

  // Signup function
  const signup = async (userData) => {
    try {
      dispatch({ type: actionTypes.AUTH_START });

      // Try API signup
      const response = await axios.post("/api/auth/signup", userData);

      const { user } = response.data;

      // Don't automatically login after signup - just show success
      dispatch({ type: actionTypes.AUTH_FAIL });

      toast.success(`Account created successfully! Please login to continue.`);
      return { success: true, requireLogin: true };
    } catch (error) {
      const message = error.response?.data?.message || "Signup failed";

      dispatch({ type: actionTypes.AUTH_FAIL });
      toast.error(message);

      return { success: false, message };
    }
  };

  // Google login function
  const googleLogin = async (tokenId) => {
    try {
      dispatch({ type: actionTypes.AUTH_START });

      const response = await axios.post("/api/auth/google", {
        tokenId,
      });

      const { user, access_token } = response.data;

      // Store token in localStorage
      localStorage.setItem("token", access_token);
      localStorage.setItem("user", JSON.stringify(user));

      // Set axios default header
      axios.defaults.headers.common["Authorization"] = `Bearer ${access_token}`;

      dispatch({
        type: actionTypes.AUTH_SUCCESS,
        payload: {
          user,
          token: access_token,
        },
      });

      toast.success(`Welcome, ${user.name}!`);
      return { success: true };
    } catch (error) {
      const message = error.response?.data?.message || "Google login failed";

      dispatch({ type: actionTypes.AUTH_FAIL });
      toast.error(message);

      return { success: false, message };
    }
  };

  // Logout function
  const logout = () => {
    // Remove from localStorage
    localStorage.removeItem("token");
    localStorage.removeItem("user");

    // Remove axios default header
    delete axios.defaults.headers.common["Authorization"];

    dispatch({ type: actionTypes.LOGOUT });
    toast.info("You have been logged out.");
  };

  // Update user profile
  const updateUser = async (updateData) => {
    try {
      dispatch({ type: actionTypes.SET_LOADING, payload: true });

      const response = await axios.put("/api/auth/profile", updateData);

      const updatedUser = response.data.user;

      // Update localStorage
      localStorage.setItem("user", JSON.stringify(updatedUser));

      dispatch({
        type: actionTypes.UPDATE_USER,
        payload: updatedUser,
      });

      toast.success("Profile updated successfully!");
      return { success: true };
    } catch (error) {
      const message = error.response?.data?.message || "Profile update failed";
      toast.error(message);
      return { success: false, message };
    } finally {
      dispatch({ type: actionTypes.SET_LOADING, payload: false });
    }
  };

  // Change password
  const changePassword = async (currentPassword, newPassword) => {
    try {
      dispatch({ type: actionTypes.SET_LOADING, payload: true });

      await axios.post("/api/auth/change-password", {
        currentPassword,
        newPassword,
      });

      toast.success("Password changed successfully!");
      return { success: true };
    } catch (error) {
      const message = error.response?.data?.message || "Password change failed";
      toast.error(message);
      return { success: false, message };
    } finally {
      dispatch({ type: actionTypes.SET_LOADING, payload: false });
    }
  };

  // Request password reset
  const requestPasswordReset = async (email) => {
    try {
      dispatch({ type: actionTypes.SET_LOADING, payload: true });

      await axios.post("/api/auth/forgot-password", { email });

      toast.success("Password reset email sent!");
      return { success: true };
    } catch (error) {
      const message =
        error.response?.data?.message || "Password reset request failed";
      toast.error(message);
      return { success: false, message };
    } finally {
      dispatch({ type: actionTypes.SET_LOADING, payload: false });
    }
  };

  // Context value
  const value = {
    // State
    user: state.user,
    token: state.token,
    isLoading: state.isLoading,
    isAuthenticated: state.isAuthenticated,

    // Actions
    login,
    signup,
    googleLogin,
    logout,
    updateUser,
    changePassword,
    requestPasswordReset,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// Custom hook to use auth context
export const useAuth = () => {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }

  return context;
};

export default AuthContext;
