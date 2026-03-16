"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Eye, EyeOff, Loader2 } from 'lucide-react';
import api from '@/lib/api';
import { useAuthStore } from '@/store/authStore';

export default function RegisterPage() {
  const router = useRouter();
  const setLanguage = useAuthStore((state) => state.setLanguage);
  const currentLanguage = useAuthStore((state) => state.language);
  
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Password strength logic
  const getPasswordStrength = (pass: string) => {
    let score = 0;
    if (!pass) return 0;
    if (pass.length >= 8) score += 1;
    if (/[A-Z]/.test(pass)) score += 1;
    if (/[a-z]/.test(pass)) score += 1;
    if (/[0-9]/.test(pass)) score += 1;
    if (/[^A-Za-z0-9]/.test(pass)) score += 1;
    return score;
  };

  const strength = getPasswordStrength(password);
  
  const getStrengthColor = () => {
    if (strength === 0) return 'bg-gray-700';
    if (strength <= 2) return 'bg-red-500';
    if (strength <= 4) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const getStrengthLabel = () => {
    if (strength === 0) return '';
    if (strength <= 2) return 'Faible';
    if (strength <= 4) return 'Moyen';
    return 'Fort';
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    // Basic Validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setError('Veuillez entrer une adresse email valide.');
      setIsLoading(false);
      return;
    }

    if (password !== confirmPassword) {
      setError('Les mots de passe ne correspondent pas.');
      setIsLoading(false);
      return;
    }

    if (strength < 3) {
      setError('Le mot de passe est trop faible. Veuillez ajouter des majuscules, chiffres ou symboles.');
      setIsLoading(false);
      return;
    }

    try {
      await api.post('/auth/register', {
        full_name: fullName,
        email,
        password,
      });

      // Redirect after 800ms
      setTimeout(() => {
        router.push('/dashboard');
      }, 800);
      
    } catch (err: any) {
      setError(err.response?.data?.detail || "Une erreur est survenue lors de l'inscription.");
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0A1628] flex items-center justify-center p-4">
      {/* Language Toggle */}
      <div className="absolute top-4 right-4 flex space-x-2 bg-[#0F1F3D] rounded-lg p-1 border border-gray-800">
        <button
          onClick={() => setLanguage('FR')}
          className={`px-3 py-1 rounded-md text-sm transition-all ${currentLanguage === 'FR' ? 'bg-[#2563EB] text-white' : 'text-gray-400 hover:text-white'}`}
        >
          FR
        </button>
        <button
          onClick={() => setLanguage('EN')}
          className={`px-3 py-1 rounded-md text-sm transition-all ${currentLanguage === 'EN' ? 'bg-[#2563EB] text-white' : 'text-gray-400 hover:text-white'}`}
        >
          EN
        </button>
      </div>

      <div className="w-full max-w-md bg-[#0F1F3D] rounded-xl shadow-2xl p-8 border border-gray-800">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Inscription</h1>
          <p className="text-gray-400">Créez votre compte Analyse IA</p>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/50 text-red-500 rounded-lg p-3 mb-6 text-sm">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Nom complet
            </label>
            <input
              type="text"
              required
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              className="w-full bg-[#0A1628] border border-gray-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-[#2563EB] focus:border-transparent transition-all"
              placeholder="Jean Dupont"
            />
          </div>

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
            
            {/* Password Strength Meter */}
            {password && (
              <div className="mt-2 space-y-1">
                <div className="flex space-x-1 h-1.5">
                  {[1, 2, 3, 4, 5].map((level) => (
                    <div
                      key={level}
                      className={`flex-1 rounded-full ${
                        strength >= level ? getStrengthColor() : 'bg-gray-700'
                      } transition-colors duration-300`}
                    />
                  ))}
                </div>
                <div className="text-xs text-right text-gray-400">
                  {getStrengthLabel()}
                </div>
              </div>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Confirmer le mot de passe
            </label>
            <div className="relative">
              <input
                type={showConfirmPassword ? 'text' : 'password'}
                required
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="w-full bg-[#0A1628] border border-gray-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-[#2563EB] focus:border-transparent transition-all pr-12"
                placeholder="••••••••"
              />
              <button
                type="button"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white transition-colors p-1"
              >
                {showConfirmPassword ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-[#2563EB] hover:bg-blue-600 text-white font-medium py-3 rounded-lg transition-all flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <Loader2 className="animate-spin" size={20} />
                <span>Inscription en cours...</span>
              </>
            ) : (
              <span>S'inscrire</span>
            )}
          </button>
        </form>

        <p className="mt-8 text-center text-sm text-gray-400">
          Déjà un compte ?{' '}
          <button
            onClick={() => router.push('/login')}
            className="text-[#2563EB] hover:text-blue-400 transition-colors font-medium"
          >
            Se connecter
          </button>
        </p>
      </div>
    </div>
  );
}
