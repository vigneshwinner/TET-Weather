import { ChevronDown, Activity } from 'lucide-react';

function Header({ commodities, selectedCommodity, onCommodityChange }) {
  return (
    <header className="bg-surface-900 border-b border-surface-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo / Title */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-brand-600 rounded-lg flex items-center justify-center">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-white">Forecast Dashboard</h1>
              <p className="text-xs text-surface-200">Energy & Commodities</p>
            </div>
          </div>

          {/* Commodity Selector */}
          <div className="flex items-center gap-4">
            <label className="text-sm text-surface-200">Commodity:</label>
            <div className="relative">
              <select
                value={selectedCommodity || ''}
                onChange={(e) => onCommodityChange(e.target.value)}
                className="appearance-none bg-surface-800 border border-surface-200/20 rounded-lg px-4 py-2 pr-10 text-white font-medium focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent cursor-pointer min-w-[140px]"
              >
                {commodities.map((commodity) => (
                  <option key={commodity} value={commodity}>
                    {getCommodityName(commodity)}
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-surface-200 pointer-events-none" />
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

// Helper to get display names for commodities
function getCommodityName(ticker) {
  const names = {
    'NG': 'Natural Gas',
    'CL': 'Crude Oil',
    'HO': 'Heating Oil',
    'RB': 'RBOB Gasoline',
    'ERCOT': 'ERCOT Power',
  };
  return names[ticker] || ticker;
}

export default Header;
