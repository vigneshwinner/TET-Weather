import Plot from 'react-plotly.js';
import { TrendingUp, Percent, AlertTriangle, Target, BarChart2 } from 'lucide-react';

function PerformancePanel({ backtest, loading }) {
  if (loading) {
    return (
      <div className="card animate-pulse">
        <div className="card-header">Performance</div>
        <div className="grid grid-cols-3 gap-4 mb-6">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-20 bg-surface-800 rounded"></div>
          ))}
        </div>
        <div className="h-48 bg-surface-800 rounded"></div>
      </div>
    );
  }

  if (!backtest) {
    return (
      <div className="card">
        <div className="card-header">Performance</div>
        <p className="text-surface-200">No backtest data available</p>
      </div>
    );
  }

  const { metrics, equity_curve } = backtest;

  // KPI data
  const kpis = [
    {
      label: 'Total Return',
      value: `${metrics.total_return >= 0 ? '+' : ''}${metrics.total_return?.toFixed(1)}%`,
      icon: TrendingUp,
      color: metrics.total_return >= 0 ? 'positive' : 'negative',
    },
    {
      label: 'Sharpe Ratio',
      value: metrics.sharpe_ratio?.toFixed(2),
      icon: Target,
      color: metrics.sharpe_ratio >= 1 ? 'positive' : metrics.sharpe_ratio >= 0.5 ? 'neutral' : 'negative',
    },
    {
      label: 'Max Drawdown',
      value: `${metrics.max_drawdown?.toFixed(1)}%`,
      icon: AlertTriangle,
      color: 'negative',
    },
    {
      label: 'Win Rate',
      value: `${metrics.win_rate?.toFixed(1)}%`,
      icon: Percent,
      color: metrics.win_rate >= 50 ? 'positive' : 'negative',
    },
    {
      label: 'Volatility',
      value: `${metrics.volatility?.toFixed(1)}%`,
      icon: BarChart2,
      color: 'neutral',
    },
    {
      label: 'Trades',
      value: metrics.num_trades,
      icon: BarChart2,
      color: 'neutral',
    },
  ];

  // Equity curve chart
  const equityDates = equity_curve?.map(d => d.date) || [];
  const equityValues = equity_curve?.map(d => d.value) || [];

  const chartData = [
    {
      x: equityDates,
      y: equityValues,
      type: 'scatter',
      mode: 'lines',
      name: 'Equity',
      line: {
        color: '#10b981',
        width: 2,
      },
      fill: 'tozeroy',
      fillcolor: 'rgba(16, 185, 129, 0.1)',
    },
  ];

  const layout = {
    autosize: true,
    height: 180,
    margin: { l: 50, r: 20, t: 10, b: 40 },
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
      tickfont: { size: 10 },
    },
    yaxis: {
      showgrid: true,
      gridcolor: 'rgba(148, 163, 184, 0.1)',
      zeroline: false,
      tickfont: { size: 10 },
      tickprefix: '$',
    },
    showlegend: false,
    hovermode: 'x',
    hoverlabel: {
      bgcolor: '#1e293b',
      bordercolor: '#334155',
      font: { color: '#f8fafc' },
    },
  };

  const config = {
    displayModeBar: false,
    responsive: true,
  };

  return (
    <div className="card">
      <div className="card-header flex items-center gap-2">
        <BarChart2 className="w-4 h-4" />
        Backtest Performance
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mb-6">
        {kpis.map((kpi, index) => (
          <div
            key={index}
            className="p-3 bg-surface-800 rounded-lg text-center"
          >
            <div className="flex items-center justify-center mb-2">
              <kpi.icon className={`w-4 h-4 ${kpi.color}`} />
            </div>
            <div className={`text-lg font-bold font-mono ${kpi.color}`}>
              {kpi.value}
            </div>
            <div className="text-xs text-surface-200 mt-1">
              {kpi.label}
            </div>
          </div>
        ))}
      </div>

      {/* Equity Curve */}
      <div>
        <h4 className="text-sm text-surface-200 mb-2">Equity Curve</h4>
        {equity_curve?.length > 0 ? (
          <Plot
            data={chartData}
            layout={layout}
            config={config}
            useResizeHandler={true}
            style={{ width: '100%' }}
          />
        ) : (
          <div className="h-48 flex items-center justify-center text-surface-200">
            No equity data available
          </div>
        )}
      </div>

      {/* Last Updated */}
      {backtest.last_updated && (
        <div className="mt-4 text-xs text-surface-200/60 text-right">
          Last updated: {new Date(backtest.last_updated).toLocaleString()}
        </div>
      )}
    </div>
  );
}

export default PerformancePanel;
