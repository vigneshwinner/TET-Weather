import { useState } from 'react';
import Plot from 'react-plotly.js';
import { BarChart3, Calendar } from 'lucide-react';

function SSIPanel({ ssiData, loading, dateRange, onDateRangeChange }) {
  const [showComponents, setShowComponents] = useState(false);

  if (loading) {
    return (
      <div className="card animate-pulse">
        <div className="card-header">SSI Time Series</div>
        <div className="h-80 bg-surface-800 rounded"></div>
      </div>
    );
  }

  if (!ssiData || !ssiData.data?.length) {
    return (
      <div className="card">
        <div className="card-header">SSI Time Series</div>
        <p className="text-surface-200">No SSI data available</p>
      </div>
    );
  }

  // Prepare chart data
  const dates = ssiData.data.map(d => d.date);
  const ssiValues = ssiData.data.map(d => d.ssi_value);

  // Main SSI trace
  const traces = [
    {
      x: dates,
      y: ssiValues,
      type: 'scatter',
      mode: 'lines',
      name: 'SSI',
      line: {
        color: '#0ea5e9',
        width: 2,
      },
      fill: 'tozeroy',
      fillcolor: 'rgba(14, 165, 233, 0.1)',
    },
  ];

  // Add component traces if available and enabled
  if (showComponents && ssiData.data[0]?.components) {
    const componentColors = {
      momentum: '#10b981',
      mean_reversion: '#f59e0b',
      volatility: '#ef4444',
    };

    Object.keys(ssiData.data[0].components).forEach(component => {
      traces.push({
        x: dates,
        y: ssiData.data.map(d => d.components?.[component] || 0),
        type: 'scatter',
        mode: 'lines',
        name: component.charAt(0).toUpperCase() + component.slice(1),
        line: {
          color: componentColors[component] || '#6b7280',
          width: 1.5,
          dash: 'dot',
        },
      });
    });
  }

  const layout = {
    autosize: true,
    height: 320,
    margin: { l: 50, r: 30, t: 20, b: 50 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
      family: 'Inter, system-ui, sans-serif',
      color: '#94a3b8',
    },
    xaxis: {
      showgrid: false,
      zeroline: false,
      tickformat: '%b %Y',
      tickfont: { size: 11 },
    },
    yaxis: {
      showgrid: true,
      gridcolor: 'rgba(148, 163, 184, 0.1)',
      zeroline: true,
      zerolinecolor: 'rgba(148, 163, 184, 0.3)',
      tickfont: { size: 11 },
    },
    legend: {
      orientation: 'h',
      y: -0.2,
      x: 0.5,
      xanchor: 'center',
      font: { size: 11 },
    },
    hovermode: 'x unified',
    hoverlabel: {
      bgcolor: '#1e293b',
      bordercolor: '#334155',
      font: { color: '#f8fafc' },
    },
  };

  const config = {
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
    displaylogo: false,
    responsive: true,
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div className="card-header mb-0 flex items-center gap-2">
          <BarChart3 className="w-4 h-4" />
          Signal Strength Index (SSI)
        </div>
        
        <div className="flex items-center gap-4">
          {/* Component Toggle */}
          {ssiData.data[0]?.components && (
            <label className="flex items-center gap-2 text-sm text-surface-200 cursor-pointer">
              <input
                type="checkbox"
                checked={showComponents}
                onChange={(e) => setShowComponents(e.target.checked)}
                className="rounded border-surface-200/20 bg-surface-800 text-brand-500 focus:ring-brand-500"
              />
              Show Components
            </label>
          )}
          
          {/* Date Range Picker */}
          <div className="flex items-center gap-2">
            <Calendar className="w-4 h-4 text-surface-200" />
            <input
              type="date"
              value={dateRange.start}
              onChange={(e) => onDateRangeChange({ ...dateRange, start: e.target.value })}
              className="bg-surface-800 border border-surface-200/20 rounded px-2 py-1 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
            />
            <span className="text-surface-200">to</span>
            <input
              type="date"
              value={dateRange.end}
              onChange={(e) => onDateRangeChange({ ...dateRange, end: e.target.value })}
              className="bg-surface-800 border border-surface-200/20 rounded px-2 py-1 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
            />
          </div>
        </div>
      </div>

      {/* Chart */}
      <Plot
        data={traces}
        layout={layout}
        config={config}
        useResizeHandler={true}
        style={{ width: '100%' }}
      />

      {/* Stats */}
      <div className="mt-4 grid grid-cols-4 gap-4 text-center">
        <div className="p-3 bg-surface-800 rounded-lg">
          <div className="text-xs text-surface-200">Current</div>
          <div className={`font-mono font-medium ${ssiValues[ssiValues.length - 1] >= 0 ? 'positive' : 'negative'}`}>
            {ssiValues[ssiValues.length - 1]?.toFixed(3) || '-'}
          </div>
        </div>
        <div className="p-3 bg-surface-800 rounded-lg">
          <div className="text-xs text-surface-200">Mean</div>
          <div className="font-mono font-medium text-white">
            {(ssiValues.reduce((a, b) => a + b, 0) / ssiValues.length).toFixed(3)}
          </div>
        </div>
        <div className="p-3 bg-surface-800 rounded-lg">
          <div className="text-xs text-surface-200">Max</div>
          <div className="font-mono font-medium positive">
            {Math.max(...ssiValues).toFixed(3)}
          </div>
        </div>
        <div className="p-3 bg-surface-800 rounded-lg">
          <div className="text-xs text-surface-200">Min</div>
          <div className="font-mono font-medium negative">
            {Math.min(...ssiValues).toFixed(3)}
          </div>
        </div>
      </div>
    </div>
  );
}

export default SSIPanel;
