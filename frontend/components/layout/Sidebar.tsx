import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Brain, MessageSquare, Upload, History, Shield, Settings, LogOut, Globe, Menu, X } from 'lucide-react';
import { useAuthStore } from '@/store/authStore';
import { useChatStore } from '@/store/chatStore';
import { logout } from '@/lib/auth';
import { t } from '@/lib/i18n';

interface SidebarProps {
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
  onLogout: () => void;
}

export default function Sidebar({ isOpen, setIsOpen, onLogout }: SidebarProps) {
  const pathname = usePathname();
  const { user } = useAuthStore();
  const { language: lang, setLanguage } = useChatStore();

  const navItems = [
    { href: '/dashboard', icon: MessageSquare, label: t(lang, 'dashboard') },
    { href: '/upload', icon: Upload, label: t(lang, 'upload') },
    { href: '/history', icon: History, label: t(lang, 'history') },
    { href: '/rgpd', icon: Shield, label: t(lang, 'rgpd') },
    { href: '/settings', icon: Settings, label: t(lang, 'settings') || 'Settings' },
  ];

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 md:hidden" 
          onClick={() => setIsOpen(false)}
        />
      )}

      <aside className={`
        fixed inset-y-0 left-0 z-50 w-[250px] bg-[rgba(15,31,61,0.8)] backdrop-blur-md border-r border-blue-500/10 flex flex-col transition-transform duration-300 ease-in-out md:relative md:translate-x-0
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        {/* Header */}
        <div className="h-14 flex items-center justify-between px-4 border-b border-blue-500/10 flex-shrink-0">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center flex-shrink-0">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-white tracking-wide">Analyse IA</span>
          </div>
          <button 
            className="md:hidden text-gray-400 hover:text-white"
            onClick={() => setIsOpen(false)}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 py-6 px-3 space-y-1.5 overflow-y-auto">
          {navItems.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setIsOpen(false)}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
                  active
                    ? 'bg-blue-600/15 text-white border-l-4 border-blue-500 pl-3'
                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                }`}
              >
                <item.icon className="w-5 h-5 flex-shrink-0" />
                {item.label}
              </Link>
            );
          })}
        </nav>

        {/* User Profile Footer */}
        <div className="p-4 border-t border-blue-500/10 bg-black/20 space-y-3">
          <div className="flex items-center gap-3 px-2">
            <div className="w-9 h-9 rounded-full bg-blue-600/30 border border-blue-500/50 flex items-center justify-center flex-shrink-0 text-white font-semibold">
              {(user?.email || 'U')[0].toUpperCase()}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-white truncate">
                {user?.full_name || 'Utilisateur'}
              </p>
              <p className="text-xs text-blue-300/60 truncate">
                {user?.email || 'user@example.com'}
              </p>
            </div>
          </div>
          
          <div className="pt-2 flex flex-col gap-1">
            <button
              onClick={() => setLanguage(lang === 'fr' ? 'en' : 'fr')}
              className="flex items-center gap-3 px-2 py-2 w-full rounded-md text-sm text-gray-400 hover:text-white hover:bg-white/5 transition-all"
            >
              <Globe className="w-4 h-4" />
              {lang === 'fr' ? 'Passer en Anglais' : 'Switch to French'}
            </button>
            <button
              onClick={onLogout}
              className="flex items-center gap-3 px-2 py-2 w-full rounded-md text-sm text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-all"
            >
              <LogOut className="w-4 h-4" />
              {t(lang, 'logout') || 'Déconnexion'}
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}
