import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { User } from '../types';

interface AuthState {
  isAuthenticated: boolean;
  isLoading: boolean;
  user: User | null;
  language: 'FR' | 'EN';
  login: (user: User, accessToken: string, refreshToken: string) => void;
  logout: () => void;
  setLanguage: (lang: 'FR' | 'EN') => void;
  checkAuth: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      isAuthenticated: false,
      isLoading: false,
      user: null,
      language: 'FR',
      login: (user, accessToken, refreshToken) => {
        if (typeof window !== 'undefined') {
          localStorage.setItem('access_token', accessToken);
          localStorage.setItem('refresh_token', refreshToken);
        }
        set({ isAuthenticated: true, user });
      },
      logout: () => {
        if (typeof window !== 'undefined') {
          localStorage.removeItem('access_token');
          localStorage.removeItem('refresh_token');
        }
        set({ isAuthenticated: false, user: null });
      },
      setLanguage: (lang) => set({ language: lang }),
      checkAuth: () => {
        if (typeof window !== 'undefined') {
          const token = localStorage.getItem('access_token');
          if (token) {
            // we assume if there's a token, they are authenticated until api fails
            // user object can be restored from persist or needs fetching. 
            // In this implementation, persist will handle user object restoration automatically.
            set({ isAuthenticated: true });
          } else {
            set({ isAuthenticated: false, user: null });
          }
        }
      }
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ 
        language: state.language,
        user: state.user,
        isAuthenticated: state.isAuthenticated 
      }),
    }
  )
);
