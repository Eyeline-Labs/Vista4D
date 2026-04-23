import React from 'react';
import './AppBackground.css';

interface AppBackgroundProps {
  children: React.ReactNode;
}

const AppBackground: React.FC<AppBackgroundProps> = ({ children }) => {
  return <div className="app-background">{children}</div>;
};

export default AppBackground;
