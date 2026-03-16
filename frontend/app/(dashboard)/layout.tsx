'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Loader2, Menu } from 'lucide-react'
import { useAuthStore } from '@/store/authStore'
import { useChatStore } from '@/store/chatStore'
import { logout, isAuthenticated } from '@/lib/auth'
import Sidebar from '@/components/layout/Sidebar'

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const { language: lang } = useChatStore()
  const [isChecking, setIsChecking] = useState(true)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)

  useEffect(() => {
    const checkAuth = () => {
      if (!isAuthenticated()) {
        router.replace('/login')
        return
      }
      setIsChecking(false)
    }
    checkAuth()
  }, [router])

  const handleLogout = async () => {
    await logout()
    router.push('/login')
  }

  // Show spinner while auth is being verified — prevents flash of unauthenticated content
  if (isChecking) {
    return (
      <div className="h-screen flex items-center justify-center bg-[#0A1628]">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />
          <span className="text-xs text-blue-300/40">
            {lang === 'fr' ? 'Vérification...' : 'Checking auth...'}
          </span>
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen flex overflow-hidden bg-[#0A1628]">
      {/* Sidebar Component */}
      <Sidebar 
        isOpen={isSidebarOpen} 
        setIsOpen={setIsSidebarOpen} 
        onLogout={handleLogout} 
      />

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden">
        {/* Mobile Header (Hidden on Desktop) */}
        <header className="md:hidden h-14 flex items-center justify-between px-4 border-b border-blue-500/10 bg-[#0A1628] flex-shrink-0">
          <span className="font-bold text-white tracking-wide">Analyse IA</span>
          <button 
            onClick={() => setIsSidebarOpen(true)}
            className="text-gray-400 hover:text-white p-1"
          >
            <Menu className="w-6 h-6" />
          </button>
        </header>

        {/* Desktop Status Header (Hidden on Mobile) */}
        <header className="hidden md:flex h-14 items-center justify-between px-6 border-b border-blue-500/10 bg-[#0A1628] flex-shrink-0">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-blue-300/50 font-mono">
              {lang === 'fr' ? 'Système opérationnel' : 'System operational'}
            </span>
          </div>
          <span className="text-xs text-blue-300/30 font-mono">mistral-nemo · Ollama</span>
        </header>

        {/* Page Content */}
        <div className="flex-1 overflow-x-hidden overflow-y-auto">
          {children}
        </div>
      </main>
    </div>
  )
}
