import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Legend, AreaChart, Area
} from 'recharts';
import { 
  TrendingUp, TrendingDown, Minus, RefreshCw, Activity, 
  DollarSign, Percent, BarChart3, AlertTriangle
} from 'lucide-react';

// API base URL
const API_BASE = '/api';

// Fetch helper
async function fetchApi(endpoint) {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error(`API Error (${endpoint}):`, error);
    return null;
  }
}

// ============================================================================
// Components
// ============================================================================

function Header({ commodity, setCommodity, commodities, onRefresh }) {
  return (
    <header className="bg-surface-800 border-b border-slate-700 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-2xl font-bold text-white">TET-Weather</h1>
          <span className="text-slate-400">Forecast Dashboard</span>
        </div>
        
        <div className="flex items-center gap-4">
          <select
            value={commodity}
            onChange={(e) => setCommodity(e.target.value)}
            className="bg-surface-900 border border-slate-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-brand-500"
          >
            {commodities.map(c => (
              <option key={c} value={c}>{c.replace('_', ' ')}</option>
            ))}
          </select>
          
          <button
            onClick={onRefresh}
            className="flex items-center gap-2 bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 rounded-lg transition-colors"
          >
            <RefreshCw size={18} />
            Refresh
          </button>
        </div>
      </div>
    </header>
  );
}

function SignalCard({ forecast }) {
  if (!forecast) return <LoadingCard title="Current Signal" />;
  
  const signalColors = {
    'LONG': 'bg-emerald-500',
    'SHORT': 'bg-red-500',
    'FLAT': 'bg-slate-500'
  };
  
  const SignalIcon = forecast.signal === 'LONG' ? TrendingUp : 
                     forecast.signal === 'SHORT' ? TrendingDown : Minus;
  
  return (
    <div className="bg-surface-800 rounded-xl p-6 border border-slate-700">
      <h3 className="text-slate-400 text-sm font-medium mb-4">Current Signal</h3>
      
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`${signalColors[forecast.signal]} p-3 rounded-lg`}>
            <SignalIcon size={28} className="text-white" />
          </div>
          <div>
            <div className="text-3xl font-bold text-white">{forecast.signal}</div>
            <div className="text-slate-400 text-sm">Week of {forecast.date}</div>
          </div>
        </div>
        
        <div className="text-right">
          <div className="text-sm text-slate-400">Direction Prob</div>
          <div className="text-2xl font-semibold text-white">
            {(forecast.direction_probability * 100).toFixed(1)}%
          </div>
        </div>
      </div>
      
      <div className="mt-4 pt-4 border-t border-slate-700">
        <div className="flex justify-between text-sm">
          <span className="text-slate-400">Predicted Return</span>
          <span className={forecast.predicted_return > 0 ? 'text-emerald-400' : 'text-red-400'}>
            {(forecast.predicted_return * 100).toFixed(2)}%
          </span>
        </div>
        <div className="flex justify-between text-sm mt-2">
          <span className="text-slate-400">Confidence</span>
          <span className="text-white">{(forecast.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ title, value, format = 'number', icon: Icon, color = 'blue' }) {
  const colorClasses = {
    blue: 'bg-blue-500/20 text-blue-400',
    green: 'bg-emerald-500/20 text-emerald-400',
    red: 'bg-red-500/20 text-red-400',
    yellow: 'bg-yellow-500/20 text-yellow-400',
    purple: 'bg-purple-500/20 text-purple-400'
  };
  
  let displayValue = value;
  if (format === 'percent') displayValue = `${(value * 100).toFixed(1)}%`;
  else if (format === 'decimal') displayValue = value?.toFixed(2);
  
  return (
    <div className="bg-surface-800 rounded-xl p-5 border border-slate-700">
      <div className="flex items-center justify-between mb-3">
        <span className="text-slate-400 text-sm">{title}</span>
        {Icon && (
          <div className={`p-2 rounded-lg ${colorClasses[color]}`}>
            <Icon size={18} />
          </div>
        )}
      </div>
      <div className="text-2xl font-bold text-white">{displayValue ?? '-'}</div>
    </div>
  );
}

function PerformancePanel({ metrics, commodity }) {
  if (!metrics || !metrics[commodity]) {
    return <LoadingCard title="Performance Metrics" />;
  }
  
  const m = metrics[commodity];
  
  return (
    <div className="bg-surface-800 rounded-xl p-6 border border-slate-700">
      <h3 className="text-white font-semibold mb-4">Performance Metrics</h3>
      
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-surface-900 rounded-lg p-4">
          <div className="text-slate-400 text-sm">Total Return</div>
          <div className={`text-xl font-bold ${m.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {(m.total_return * 100).toFixed(1)}%
          </div>
        </div>
        
        <div className="bg-surface-900 rounded-lg p-4">
          <div className="text-slate-400 text-sm">Sharpe Ratio</div>
          <div className={`text-xl font-bold ${m.sharpe_ratio >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {m.sharpe_ratio?.toFixed(2)}
          </div>
        </div>
        
        <div className="bg-surface-900 rounded-lg p-4">
          <div className="text-slate-400 text-sm">Win Rate</div>
          <div className="text-xl font-bold text-white">
            {(m.win_rate * 100).toFixed(1)}%
          </div>
        </div>
        
        <div className="bg-surface-900 rounded-lg p-4">
          <div className="text-slate-400 text-sm">Hit Ratio</div>
          <div className="text-xl font-bold text-white">
            {(m.hit_ratio * 100).toFixed(1)}%
          </div>
        </div>
        
        <div className="bg-surface-900 rounded-lg p-4">
          <div className="text-slate-400 text-sm">Max Drawdown</div>
          <div className="text-xl font-bold text-red-400">
            -{(m.max_drawdown * 100).toFixed(1)}%
          </div>
        </div>
        
        <div className="bg-surface-900 rounded-lg p-4">
          <div className="text-slate-400 text-sm">Profit Factor</div>
          <div className="text-xl font-bold text-white">
            {m.profit_factor?.toFixed(2)}
          </div>
        </div>
      </div>
    </div>
  );
}

function EquityCurveChart({ curves, commodity }) {
  if (!curves || !curves[commodity]) {
    return <LoadingCard title="Equity Curve" />;
  }
  
  const data = curves[commodity].dates.map((date, i) => ({
    date: date,
    pnl: curves[commodity].cumulative_pnl[i] * 100
  }));
  
  return (
    <div className="bg-surface-800 rounded-xl p-6 border border-slate-700">
      <h3 className="text-white font-semibold mb-4">Cumulative PnL</h3>
      
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis 
            dataKey="date" 
            stroke="#64748b"
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => value.slice(5)}
          />
          <YAxis 
            stroke="#64748b"
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => `${value.toFixed(0)}%`}
          />
          <Tooltip
            contentStyle={{ 
              backgroundColor: '#1e293b', 
              border: '1px solid #475569',
              borderRadius: '8px'
            }}
            formatter={(value) => [`${value.toFixed(2)}%`, 'PnL']}
          />
          <Area 
            type="monotone" 
            dataKey="pnl" 
            stroke="#3b82f6" 
            fill="url(#pnlGradient)"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function AllCommoditiesChart({ curves }) {
  if (!curves || Object.keys(curves).length === 0) {
    return <LoadingCard title="All Commodities" />;
  }
  
  // Merge all commodities into single dataset
  const commodities = Object.keys(curves);
  const allDates = [...new Set(commodities.flatMap(c => curves[c].dates))].sort();
  
  const data = allDates.map(date => {
    const point = { date };
    commodities.forEach(c => {
      const idx = curves[c].dates.indexOf(date);
      if (idx !== -1) {
        point[c] = curves[c].cumulative_pnl[idx] * 100;
      }
    });
    return point;
  });
  
  const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
  
  return (
    <div className="bg-surface-800 rounded-xl p-6 border border-slate-700">
      <h3 className="text-white font-semibold mb-4">All Commodities Comparison</h3>
      
      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis 
            dataKey="date" 
            stroke="#64748b"
            tick={{ fontSize: 11 }}
            tickFormatter={(value) => value.slice(5)}
          />
          <YAxis 
            stroke="#64748b"
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => `${value.toFixed(0)}%`}
          />
          <Tooltip
            contentStyle={{ 
              backgroundColor: '#1e293b', 
              border: '1px solid #475569',
              borderRadius: '8px'
            }}
            formatter={(value) => value ? `${value.toFixed(2)}%` : '-'}
          />
          <Legend />
          {commodities.map((c, i) => (
            <Line 
              key={c}
              type="monotone" 
              dataKey={c} 
              stroke={colors[i % colors.length]}
              strokeWidth={2}
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function SignalHistoryChart({ signals }) {
  if (!signals || signals.length === 0) {
    return <LoadingCard title="Signal History" />;
  }
  
  const data = signals.slice(-26).map(s => ({
    date: s.date,
    position: s.position * 100,
    probability: (s.probability - 0.5) * 200 // Scale to +/- 100
  }));
  
  return (
    <div className="bg-surface-800 rounded-xl p-6 border border-slate-700">
      <h3 className="text-white font-semibold mb-4">Position History (Last 26 Weeks)</h3>
      
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis 
            dataKey="date" 
            stroke="#64748b"
            tick={{ fontSize: 10 }}
            tickFormatter={(value) => value.slice(5)}
          />
          <YAxis 
            stroke="#64748b"
            tick={{ fontSize: 12 }}
            domain={[-100, 100]}
            tickFormatter={(value) => `${value}%`}
          />
          <Tooltip
            contentStyle={{ 
              backgroundColor: '#1e293b', 
              border: '1px solid #475569',
              borderRadius: '8px'
            }}
            formatter={(value) => `${value.toFixed(1)}%`}
          />
          <Bar 
            dataKey="position" 
            fill="#3b82f6"
            name="Position"
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function ModelEvaluationTable({ evaluation }) {
  if (!evaluation) {
    return <LoadingCard title="Model Evaluation" />;
  }
  
  const commodities = Object.keys(evaluation);
  
  return (
    <div className="bg-surface-800 rounded-xl p-6 border border-slate-700">
      <h3 className="text-white font-semibold mb-4">Model Evaluation</h3>
      
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700">
              <th className="text-left py-3 px-2 text-slate-400">Commodity</th>
              <th className="text-right py-3 px-2 text-slate-400">MAE</th>
              <th className="text-right py-3 px-2 text-slate-400">R²</th>
              <th className="text-right py-3 px-2 text-slate-400">Accuracy</th>
              <th className="text-right py-3 px-2 text-slate-400">ROC-AUC</th>
              <th className="text-right py-3 px-2 text-slate-400">Brier</th>
            </tr>
          </thead>
          <tbody>
            {commodities.map(commodity => {
              const e = evaluation[commodity];
              return (
                <tr key={commodity} className="border-b border-slate-700/50 hover:bg-slate-700/20">
                  <td className="py-3 px-2 text-white font-medium">{commodity}</td>
                  <td className="py-3 px-2 text-right text-slate-300">{e.mae?.toFixed(4)}</td>
                  <td className={`py-3 px-2 text-right ${e.r2 >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {e.r2?.toFixed(4)}
                  </td>
                  <td className={`py-3 px-2 text-right ${e.accuracy > 0.5 ? 'text-emerald-400' : 'text-slate-300'}`}>
                    {(e.accuracy * 100).toFixed(1)}%
                  </td>
                  <td className={`py-3 px-2 text-right ${e.roc_auc > 0.5 ? 'text-emerald-400' : 'text-slate-300'}`}>
                    {e.roc_auc?.toFixed(4) ?? '-'}
                  </td>
                  <td className="py-3 px-2 text-right text-slate-300">
                    {e.brier_score?.toFixed(4) ?? '-'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function LoadingCard({ title }) {
  return (
    <div className="bg-surface-800 rounded-xl p-6 border border-slate-700 animate-pulse">
      <h3 className="text-slate-400 text-sm font-medium mb-4">{title}</h3>
      <div className="h-32 bg-slate-700 rounded-lg"></div>
    </div>
  );
}

// ============================================================================
// Main App
// ============================================================================

export default function App() {
  const [commodity, setCommodity] = useState('Henry_Hub');
  const [commodities, setCommodities] = useState(['Brent', 'Henry_Hub', 'Power', 'Copper', 'Corn']);
  const [forecast, setForecast] = useState(null);
  const [signals, setSignals] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [curves, setCurves] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);
  
  const loadData = async () => {
    setLoading(true);
    
    const [commoditiesData, forecastData, signalsData, metricsData, curvesData, evalData] = await Promise.all([
      fetchApi('/commodities'),
      fetchApi(`/forecast?commodity=${commodity}`),
      fetchApi(`/signals?commodity=${commodity}&limit=52`),
      fetchApi('/performance'),
      fetchApi('/equity_curve'),
      fetchApi('/evaluation')
    ]);
    
    if (commoditiesData?.commodities) setCommodities(commoditiesData.commodities);
    if (forecastData) setForecast(forecastData);
    if (signalsData?.signals) setSignals(signalsData.signals);
    if (metricsData?.metrics) setMetrics(metricsData.metrics);
    if (curvesData?.curves) setCurves(curvesData.curves);
    if (evalData?.evaluation) setEvaluation(evalData.evaluation);
    
    setLastUpdate(new Date());
    setLoading(false);
  };
  
  useEffect(() => {
    loadData();
  }, [commodity]);
  
  const handleRefresh = async () => {
    await fetchApi('/refresh');
    await loadData();
  };
  
  return (
    <div className="min-h-screen bg-surface-900">
      <Header 
        commodity={commodity}
        setCommodity={setCommodity}
        commodities={commodities}
        onRefresh={handleRefresh}
      />
      
      <main className="p-6">
        {/* Top Row: Signal + Performance */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <SignalCard forecast={forecast} />
          <div className="lg:col-span-2">
            <PerformancePanel metrics={metrics} commodity={commodity} />
          </div>
        </div>
        
        {/* Middle Row: Equity Curve + Signal History */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <EquityCurveChart curves={curves} commodity={commodity} />
          <SignalHistoryChart signals={signals} />
        </div>
        
        {/* All Commodities Comparison */}
        <div className="mb-6">
          <AllCommoditiesChart curves={curves} />
        </div>
        
        {/* Model Evaluation Table */}
        <div className="mb-6">
          <ModelEvaluationTable evaluation={evaluation} />
        </div>
        
        {/* Footer */}
        <footer className="text-center text-slate-500 text-sm mt-8">
          {lastUpdate && (
            <p>Last updated: {lastUpdate.toLocaleTimeString()}</p>
          )}
          <p className="mt-1">TET-Weather Forecast Dashboard v1.0</p>
        </footer>
      </main>
    </div>
  );
}
