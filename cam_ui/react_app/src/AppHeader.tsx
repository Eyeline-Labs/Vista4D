import React from 'react';
import './AppHeader.css';

interface AppHeaderProps {
  title: string;
  logoImage?: string;
  showTooltip?: boolean;
  tooltipContent?: string;
  badge?: 'alpha' | 'beta';
}

const AppHeader: React.FC<AppHeaderProps> = ({
  title,
  logoImage,
  showTooltip = false,
  tooltipContent,
  badge
}) => {
  const [tooltipVisible, setTooltipVisible] = React.useState(false);

  return (
    <header className="app-header">
      <div
        className="app-header-content"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          width: '100%'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <h1 className="app-title">{title}</h1>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {showTooltip && tooltipContent && (
            <div
              style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}
              onMouseEnter={() => setTooltipVisible(true)}
              onMouseLeave={() => setTooltipVisible(false)}
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#ffffff"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{ cursor: 'pointer' }}
              >
                <circle cx="12" cy="12" r="10" />
                <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                <circle cx="12" cy="17" r="0.5" fill="#ffffff" />
              </svg>
              {tooltipVisible && (
                <div
                  style={{
                    position: 'absolute',
                    top: '30px',
                    right: '0',
                    backgroundColor: '#1a1a1a',
                    color: '#fff',
                    padding: '0.75rem',
                    borderRadius: '6px',
                    fontSize: '0.85rem',
                    width: '300px',
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
                    border: '1px solid #333',
                    zIndex: 1000,
                    lineHeight: '1.4'
                  }}
                >
                  {tooltipContent}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default AppHeader;
