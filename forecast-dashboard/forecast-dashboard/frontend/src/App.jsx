import { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import ForecastPanel from './components/ForecastPanel';
import SSIPanel from './components/SSIPanel';
import PerformancePanel from './components/PerformancePanel';
import { apiClient } from './services/api';
import { RefreshCw, AlertCircle } from 'lucide-react';

function App() {
  // State
  const [commodities, setCommodities] = useState([]);
  const [selectedCommodity, setSelectedCommodity] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastRefresh, setLastRefresh] = useState(null);
  
  // Data state
  const [forecast, setForecast] = useState(null);
  const [ssiData, setSsiData] = useState(null);
  const [backtest, setBacktest] = useState(null);
  
  // Date range for SSI
  const [dateRange, setDateRange] = useState({
    start: getDefaultStartDate(),
    end: new Date().toISOString().split('T')[0],
  });

  // Load commodities on mount
  useEffect(() => {
    loadCommodities();
  }, []);

  // Load data when commodity changes
  useEffect(() => {
    if (selectedCommodity) {
      refreshData();
    }
  }, [selectedCommodity]);

  // Load data when date range changes
  useEffect(() => {
    if (selectedCommodity) {
      loadSSI();
    }
  }, [dateRange]);

  const loadCommodities = async () => {
    setLoading(true);
    const result = await apiClient.getCommodities();
    
    if (result.error) {
      setError(result.message);
    } else {
      setCommodities(result.commodities || []);
      if (result.commodities?.length > 0) {
        setSelectedCommodity(result.commodities[0]);
      }
    }
    setLoading(false);
  };

  const refreshData = useCallback(async () => {
    if (!selectedCommodity) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Load all data in parallel
      const [forecastResult, ssiResult, backtestResult] = await Promise.all([
        apiClient.getForecast(selectedCommodity),
        apiClient.getSSI(selectedCommodity, dateRange.start, dateRange.end),
        apiClient.getBacktestSummary(selectedCommodity),
      ]);

      if (forecastResult.error) {
        console.error('Forecast error:', forecastResult.message);
      } else {
        setForecast(forecastResult);
      }

      if (ssiResult.error) {
        console.error('SSI error:', ssiResult.message);
      } else {
        setSsiData(ssiResult);
      }

      if (backtestResult.error) {
        console.error('Backtest error:', backtestResult.message);
      } else {
        setBacktest(backtestResult);
      }

      setLastRefresh(new Date());
    } catch (err) {
      setError('Failed to load data. Please try again.');
    }
    
    setLoading(false);
  }, [selectedCommodity, dateRange]);

  const loadSSI = async () => {
    if (!selectedCommodity) return;
    
    const result = await apiClient.getSSI(
      selectedCommodity, 
      dateRange.start, 
      dateRange.end
    );
    
    if (!result.error) {
      setSsiData(result);
    }
  };

  const handleSimulate = async () => {
    if (!selectedCommodity) return;
    
    const result = await apiClient.simulate(selectedCommodity, {
      shock_magnitude: 1.5,
    });
    
    if (!result.error) {
      alert(`Simulation complete!\nBaseline return: ${result.baseline?.predicted_return}\nScenario return: ${result.scenario?.predicted_return}`);
    }
  };

  return (
    <div className="min-h-screen bg-surface-950">
      {/* Header */}
      <Header
        commodities={commodities}
        selectedCommodity={selectedCommodity}
        onCommodityChange={setSelectedCommodity}
      />

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Banner */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <span className="text-red-400">{error}</span>
          </div>
        )}

        {/* Controls */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <button
              onClick={refreshData}
              disabled={loading}
              className="btn btn-primary flex items-center gap-2"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh Data
            </button>
            
            {import.meta.env.VITE_ENABLE_SIMULATION === 'true' && (
              <button
                onClick={handleSimulate}
                disabled={loading || !selectedCommodity}
                className="btn btn-secondary"
              >
                Run Simulation
              </button>
            )}
          </div>
          
          {lastRefresh && (
            <span className="text-sm text-surface-200">
              Last updated: {lastRefresh.toLocaleTimeString()}
            </span>
          )}
        </div>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Forecast Panel - Full width on mobile, 1/3 on desktop */}
          <div className="lg:col-span-1">
            <ForecastPanel
              forecast={forecast}
              loading={loading}
              commodity={selectedCommodity}
            />
          </div>

          {/* Performance Panel - 2/3 width */}
          <div className="lg:col-span-2">
            <PerformancePanel
              backtest={backtest}
              loading={loading}
            />
          </div>

          {/* SSI Panel - Full width */}
          <div className="lg:col-span-3">
            <SSIPanel
              ssiData={ssiData}
              loading={loading}
              dateRange={dateRange}
              onDateRangeChange={setDateRange}
            />
          </div>
        </div>
      </main>
    </div>
  );
}

// Helper function
function getDefaultStartDate() {
  const date = new Date();
  date.setFullYear(date.getFullYear() - 1);
  return date.toISOString().split('T')[0];
}

export default App;
