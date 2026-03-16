"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Eye, EyeOff, Loader2 } from 'lucide-react';
import api from '@/lib/api';
import { useAuthStore } from '@/store/authStore';

export default function LoginPage() {
  const router = useRouter();
  const login = useAuthStore((state) => state.login);
  
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const response = await api.post('/auth/login', {
        email,
        password,
      });

      const { user, access_token, refresh_token } = response.data;
      login(user, access_token, refresh_token);
      router.push('/dashboard');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Une erreur est survenue lors de la connexion.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0A1628] flex items-center justify-center p-4">
      <div className="w-full max-w-md bg-[#0F1F3D] rounded-xl shadow-2xl p-8 border border-gray-800">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Analyse IA</h1>
          <p className="text-gray-400">Connectez-vous à votre compte</p>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/50 text-red-500 rounded-lg p-3 mb-6 text-sm">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Email
            </label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-[#0A1628] border border-gray-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-[#2563EB] focus:border-transparent transition-all"
              placeholder="votre@email.com"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Mot de passe
            </label>
            <div className="relative">
              <input
                type={showPassword ? 'text' : 'password'}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-[#0A1628] border border-gray-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-[#2563EB] focus:border-transparent transition-all pr-12"
                placeholder="••••••••"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white transition-colors p-1"
              >
                {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
                className="w-4 h-4 rounded border-gray-700 bg-[#0A1628] text-[#2563EB] focus:ring-[#2563EB] focus:ring-offset-[#0F1F3D]"
              />
              <span className="text-sm text-gray-300">Se souvenir de moi</span>
            </label>
            <a href="#" className="text-sm text-[#2563EB] hover:text-blue-400 transition-colors">
              Mot de passe oublié ?
            </a>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-[#2563EB] hover:bg-blue-600 text-white font-medium py-3 rounded-lg transition-all flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <Loader2 className="animate-spin" size={20} />
                <span>Connexion...</span>
              </>
            ) : (
              <span>Se connecter</span>
            )}
          </button>
        </form>

        <p className="mt-8 text-center text-sm text-gray-400">
          Pas encore de compte ?{' '}
          <button
            onClick={() => router.push('/register')}
            className="text-[#2563EB] hover:text-blue-400 transition-colors font-medium"
          >
            S'inscrire
          </button>
        </p>
      </div>
    </div>
  );
}
