import { TrendingUp, TrendingDown, Target, Calendar } from 'lucide-react';

function ForecastPanel({ forecast, loading, commodity }) {
  if (loading) {
    return (
      <div className="card animate-pulse">
        <div className="card-header">Forecast</div>
        <div className="space-y-4">
          <div className="h-16 bg-surface-800 rounded"></div>
          <div className="h-16 bg-surface-800 rounded"></div>
          <div className="h-16 bg-surface-800 rounded"></div>
        </div>
      </div>
    );
  }

  if (!forecast) {
    return (
      <div className="card">
        <div className="card-header">Forecast</div>
        <p className="text-surface-200">No forecast data available</p>
      </div>
    );
  }

  const isPositive = forecast.direction_probability > 0.5;
  const directionText = isPositive ? 'Bullish' : 'Bearish';
  const confidenceLevel = getConfidenceLevel(forecast.confidence);

  return (
    <div className="card">
      <div className="card-header flex items-center gap-2">
        <Target className="w-4 h-4" />
        Next Week Forecast
      </div>

      {/* Direction Indicator */}
      <div className="mb-6">
        <div className={`flex items-center gap-3 ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
          {isPositive ? (
            <TrendingUp className="w-8 h-8" />
          ) : (
            <TrendingDown className="w-8 h-8" />
          )}
          <div>
            <div className="text-2xl font-bold">{directionText}</div>
            <div className="text-sm opacity-75">
              {(forecast.direction_probability * 100).toFixed(1)}% probability
            </div>
          </div>
        </div>
      </div>

      {/* Predicted Return */}
      <div className="mb-6 p-4 bg-surface-800 rounded-lg">
        <div className="text-sm text-surface-200 mb-1">Predicted Return</div>
        <div className={`kpi-value ${forecast.predicted_return >= 0 ? 'positive' : 'negative'}`}>
          {forecast.predicted_return >= 0 ? '+' : ''}{(forecast.predicted_return * 100).toFixed(2)}%
        </div>
      </div>

      {/* Confidence Gauge */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-surface-200">Confidence</span>
          <span className={`text-sm font-medium ${getConfidenceColor(confidenceLevel)}`}>
            {confidenceLevel}
          </span>
        </div>
        <div className="h-2 bg-surface-800 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-500 ${getConfidenceBarColor(confidenceLevel)}`}
            style={{ width: `${forecast.confidence * 100}%` }}
          />
        </div>
      </div>

      {/* Meta Info */}
      <div className="flex items-center gap-2 text-xs text-surface-200">
        <Calendar className="w-3 h-3" />
        <span>Week of {forecast.week}</span>
      </div>
      <div className="mt-1 text-xs text-surface-200/60">
        Model: {forecast.model_version}
      </div>
    </div>
  );
}

function getConfidenceLevel(confidence) {
  if (confidence >= 0.7) return 'High';
  if (confidence >= 0.4) return 'Medium';
  return 'Low';
}

function getConfidenceColor(level) {
  switch (level) {
    case 'High': return 'text-emerald-400';
    case 'Medium': return 'text-amber-400';
    default: return 'text-red-400';
  }
}

function getConfidenceBarColor(level) {
  switch (level) {
    case 'High': return 'bg-emerald-500';
    case 'Medium': return 'bg-amber-500';
    default: return 'bg-red-500';
  }
}

export default ForecastPanel;
