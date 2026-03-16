import { ChatMessage } from '@/types';
import { User, Brain } from 'lucide-react';
import { useAuthStore } from '@/store/authStore';

interface MessageBubbleProps {
  message: ChatMessage;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const { user } = useAuthStore();

  return (
    <div className={`w-full flex ${isUser ? 'justify-end' : 'justify-start'} mb-6 group`}>
      <div className={`flex max-w-[85%] md:max-w-[75%] gap-4 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        
        {/* Avatar */}
        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1
          ${isUser 
            ? 'bg-blue-600/20 border border-blue-500/30 text-blue-400' 
            : 'bg-[#0F1F3D] border border-blue-500/20 text-white'}`}
        >
          {isUser ? (
             <span className="text-xs font-bold">
               {user?.full_name ? user.full_name[0].toUpperCase() : 'U'}
             </span>
          ) : (
             <Brain className="w-4 h-4" />
          )}
        </div>

        {/* Message Content */}
        <div className="flex flex-col gap-1 min-w-0">
          {/* Name & Time */}
          <div className={`flex items-center gap-2 text-xs text-blue-300/40 px-1
            ${isUser ? 'justify-end' : 'justify-start'}`}>
            <span>{isUser ? (user?.full_name || 'Vous') : 'Analyse IA'}</span>
            <span>•</span>
            <span>
              {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          </div>
          
          {/* Bubble wrapper */}
          <div className={`relative px-5 py-3.5 rounded-2xl text-[15px] leading-relaxed break-words shadow-sm
            ${isUser 
              ? 'bg-gradient-to-br from-blue-600 to-blue-700 text-white rounded-tr-sm' 
              : 'bg-[#0F1F3D] border border-blue-500/10 flex flex-col gap-3 text-gray-200 rounded-tl-sm'}`}
          >
            {/* Thinking / Streaming Animation */}
            {message.isStreaming && !message.content ? (
              <div className="flex items-center gap-1.5 h-6">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            ) : (
              // Text Content
              <div className="whitespace-pre-wrap">{message.content}</div>
            )}
            
            {/* If there's an analysis result attached (Handled via Dashboard wrapper for now, or can be mounted here) */}
          </div>
        </div>
      </div>
    </div>
  );
}
