import { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, Loader2 } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (content: string) => void;
  isProcessing: boolean;
  lang: 'fr' | 'en';
}

export default function ChatInput({ onSendMessage, isProcessing, lang }: ChatInputProps) {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isFrench = lang === 'fr';

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isProcessing) return;
    
    onSendMessage(input.trim());
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'; // Reset height
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="p-4 bg-[#0A1628] border-t border-blue-500/10">
      <div className="max-w-4xl mx-auto relative flex items-end gap-2 bg-[#0F1F3D] border border-blue-500/20 rounded-2xl p-2 shadow-sm focus-within:ring-1 focus-within:ring-blue-500 focus-within:border-blue-500 transition-all">
        
        {/* Attachment Button */}
        <button 
          type="button"
          disabled={isProcessing}
          className="p-3 text-gray-400 hover:text-white hover:bg-white/5 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
          title={isFrench ? "Joindre un fichier" : "Attach a file"}
        >
          <Paperclip className="w-5 h-5" />
        </button>

        {/* Text Area */}
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isProcessing}
          placeholder={isFrench ? "Posez une question sur vos données..." : "Ask a question about your data..."}
          className="w-full bg-transparent text-white placeholder-blue-300/30 resize-none outline-none py-3 max-h-[200px] overflow-y-auto text-[15px] leading-relaxed disabled:opacity-50 disabled:cursor-not-allowed"
          rows={1}
        />

        {/* Send Button */}
        <button
          onClick={handleSubmit}
          disabled={!input.trim() || isProcessing}
          className={`p-3 rounded-xl flex items-center justify-center transition-all flex-shrink-0
            ${input.trim() && !isProcessing
              ? 'bg-blue-600 text-white hover:bg-blue-500 shadow-sm'
              : 'bg-white/5 text-gray-500 cursor-not-allowed'
            }`}
        >
          {isProcessing ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Send className="w-5 h-5" />
          )}
        </button>
      </div>
      <div className="text-center mt-2 text-xs text-blue-300/30">
        {isFrench 
          ? "L'IA peut faire des erreurs. Vérifiez les informations importantes." 
          : "AI can make mistakes. Verify important information."}
      </div>
    </div>
  );
}
