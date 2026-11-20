// web/src/theme/ThemeContext.tsx
import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { THEME_PRESETS, ThemePreset, DEFAULT_PRESET_ID } from '../../../src/theme-presets';

interface ThemeState {
  presetId: string;
  mode: 'light' | 'dark' | 'system';
  customTokens: Record<string, string>;
  setPreset: (id: string) => void;
  setMode: (mode: 'light' | 'dark' | 'system') => void;
  setCustomToken: (key: string, value: string) => void;
  resetCustomTokens: () => void;
  availablePresets: readonly ThemePreset[];
  currentPreset: ThemePreset;
}

const ThemeContext = createContext<ThemeState | null>(null);

const STORAGE_KEY = 'audiotools-theme';

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [presetId, setPresetId] = useState(DEFAULT_PRESET_ID);
  const [mode, setMode] = useState<'light' | 'dark' | 'system'>('system');
  const [customTokens, setCustomTokens] = useState<Record<string, string>>({});
  const [hydrated, setHydrated] = useState(false);

  // Load from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        setPresetId(parsed.presetId ?? DEFAULT_PRESET_ID);
        setMode(parsed.mode ?? 'system');
        setCustomTokens(parsed.customTokens ?? {});
      }
    } catch { /* ignore */ }
    setHydrated(true);
  }, []);

  // Persist changes. A blocked or missing store must not take Settings down.
  useEffect(() => {
    if (!hydrated) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ presetId, mode, customTokens }));
    } catch { /* ignore */ }
  }, [presetId, mode, customTokens, hydrated]);

  // Apply to document.documentElement
  useEffect(() => {
    if (!hydrated) return;
    const root = document.documentElement;
    const preset = THEME_PRESETS.find(p => p.id === presetId) ?? THEME_PRESETS[0];
    
    root.setAttribute('data-accent', preset.accentId);
    root.setAttribute('data-theme', mode === 'system' 
      ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
      : mode);
    
    // Apply custom tokens
    Object.entries(customTokens).forEach(([key, value]) => {
      root.style.setProperty(key, value);
    });
  }, [presetId, mode, customTokens, hydrated]);

  const currentPreset = THEME_PRESETS.find(p => p.id === presetId) ?? THEME_PRESETS[0];

  const setPreset = (id: string) => {
    setPresetId(id);
    if (id === 'custom') {
      // Initialize custom tokens from current preset
      const preset = THEME_PRESETS.find(p => p.id === presetId);
      if (preset && preset.accentId !== 'custom') {
        // Copy current computed values to custom slots
        const computed = getComputedStyle(document.documentElement);
        const newCustom: Record<string, string> = {};
        [
          '--color-accent-primary', '--color-accent-primary-hover',
          '--color-accent-secondary', '--color-accent-secondary-hover',
          '--color-accent-tertiary', '--color-accent-tertiary-hover',
          '--color-bg-deepest', '--color-bg-deep', '--color-bg-raised', '--color-bg-highlight',
          '--color-text-primary', '--color-text-secondary', '--color-text-muted'
        ].forEach(key => {
          const val = computed.getPropertyValue(key).trim();
          if (val) newCustom[key.replace('--', '--custom-')] = val;
        });
        setCustomTokens(newCustom);
      }
    }
  };

  const setCustomToken = (key: string, value: string) => {
    setCustomTokens(prev => ({ ...prev, [key]: value }));
  };

  const resetCustomTokens = () => setCustomTokens({});

  return (
    <ThemeContext.Provider value={{
      presetId, mode, customTokens,
      setPreset, setMode, setCustomToken, resetCustomTokens,
      availablePresets: THEME_PRESETS,
      currentPreset
    }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used within ThemeProvider');
  return ctx;
}