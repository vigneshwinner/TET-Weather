/**
 * API Client Service
 * Handles all communication with the Flask backend
 */

import axios from 'axios';

// Configuration
const config = {
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000',
  timeout: 30000,
  retries: 3,
  retryDelay: 1000,
};

// Create axios instance
const api = axios.create({
  baseURL: config.baseURL,
  timeout: config.timeout,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Retry logic with exponential backoff
const retryRequest = async (fn, retries = config.retries) => {
  try {
    return await fn();
  } catch (error) {
    if (retries > 0 && isRetryable(error)) {
      await sleep(config.retryDelay * (config.retries - retries + 1));
      return retryRequest(fn, retries - 1);
    }
    throw error;
  }
};

const isRetryable = (error) => {
  if (!error.response) return true; // Network error
  const status = error.response.status;
  return status >= 500 || status === 429; // Server errors or rate limited
};

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Error handler
const handleError = (error) => {
  if (error.response) {
    // Server responded with error
    const { data, status } = error.response;
    return {
      error: true,
      message: data?.message || `Server error: ${status}`,
      status,
      data: data,
    };
  } else if (error.request) {
    // No response received
    return {
      error: true,
      message: 'Network error: Unable to reach the server',
      status: 0,
    };
  } else {
    // Request setup error
    return {
      error: true,
      message: error.message,
      status: 0,
    };
  }
};

// API Methods
export const apiClient = {
  /**
   * Health check
   */
  async health() {
    try {
      const response = await retryRequest(() => api.get('/health'));
      return response.data;
    } catch (error) {
      return handleError(error);
    }
  },

  /**
   * Get list of supported commodities
   */
  async getCommodities() {
    try {
      const response = await retryRequest(() => api.get('/commodities'));
      return response.data;
    } catch (error) {
      return handleError(error);
    }
  },

  /**
   * Get forecast for a commodity
   * @param {string} commodity - Ticker symbol
   * @param {string} week - Optional target week (YYYY-MM-DD)
   */
  async getForecast(commodity, week = null) {
    try {
      const params = { commodity };
      if (week) params.week = week;
      
      const response = await retryRequest(() => 
        api.get('/forecast', { params })
      );
      return response.data;
    } catch (error) {
      return handleError(error);
    }
  },

  /**
   * Get SSI time series
   * @param {string} commodity - Ticker symbol
   * @param {string} start - Start date (YYYY-MM-DD)
   * @param {string} end - End date (YYYY-MM-DD)
   */
  async getSSI(commodity, start = null, end = null) {
    try {
      const params = { commodity };
      if (start) params.start = start;
      if (end) params.end = end;
      
      const response = await retryRequest(() => 
        api.get('/ssi', { params })
      );
      return response.data;
    } catch (error) {
      return handleError(error);
    }
  },

  /**
   * Get backtest summary
   * @param {string} commodity - Ticker symbol
   */
  async getBacktestSummary(commodity) {
    try {
      const response = await retryRequest(() => 
        api.get('/backtest/summary', { params: { commodity } })
      );
      return response.data;
    } catch (error) {
      return handleError(error);
    }
  },

  /**
   * Run scenario simulation
   * @param {string} commodity - Ticker symbol
   * @param {object} scenario - Scenario parameters
   */
  async simulate(commodity, scenario) {
    try {
      const response = await retryRequest(() => 
        api.post('/simulate', { commodity, scenario })
      );
      return response.data;
    } catch (error) {
      return handleError(error);
    }
  },
};

export default apiClient;
