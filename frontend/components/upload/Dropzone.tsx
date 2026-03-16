import { useState, useCallback } from 'react';
import { UploadCloud, FileType, AlertCircle } from 'lucide-react';
import { useChatStore } from '@/store/chatStore';

interface DropzoneProps {
  onFileSelect: (file: File) => void;
  accept: string; // e.g., ".csv" or ".pdf"
  maxSizeMB: number;
  label: string;
}

export default function Dropzone({ onFileSelect, accept, maxSizeMB, label }: DropzoneProps) {
  const [isDragActive, setIsDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { language: lang } = useChatStore();
  const isFrench = lang === 'fr';

  const validateAndSelectFile = (file: File) => {
    setError(null);
    
    // Check file type
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (fileExtension !== accept.toLowerCase()) {
      setError(isFrench 
        ? `Format invalide. Seul le format ${accept} est accepté.` 
        : `Invalid format. Only ${accept} is accepted.`);
      return;
    }

    // Check file size limit
    if (file.size > maxSizeMB * 1024 * 1024) {
      setError(isFrench
        ? `Le fichier dépasse la taille maximale de ${maxSizeMB} MB.`
        : `File exceeds the maximum size of ${maxSizeMB} MB.`);
      return;
    }

    onFileSelect(file);
  };

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSelectFile(e.dataTransfer.files[0]);
    }
  }, [accept, maxSizeMB, lang]);

  const onFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSelectFile(e.target.files[0]);
    }
    // Reset value so the same file can be selected again
    e.target.value = '';
  };

  return (
    <div className="flex flex-col w-full">
      <h3 className="text-white font-medium mb-3">{label}</h3>
      
      <label
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={`relative flex flex-col items-center justify-center w-full h-48 border-2 border-dashed rounded-xl transition-all cursor-pointer bg-[#0F1F3D] overflow-hidden
          ${isDragActive 
            ? 'border-blue-400 bg-blue-600/10 scale-[1.02]' 
            : 'border-blue-500/30 hover:bg-blue-600/5 hover:border-blue-400/60'
          }`}
      >
        <div className="flex flex-col items-center justify-center pt-5 pb-6 px-4 text-center">
          <UploadCloud className={`w-10 h-10 mb-3 ${isDragActive ? 'text-blue-400 animate-bounce' : 'text-blue-500'}`} />
          <p className="mb-2 text-sm text-gray-300">
            <span className="font-semibold">{isFrench ? 'Cliquez pour uploader' : 'Click to upload'}</span> {isFrench ? 'ou glissez-déposez' : 'or drag and drop'}
          </p>
          <p className="text-xs text-blue-300/60 font-mono">
            {accept.toUpperCase()} ({isFrench ? 'Max' : 'Max'} {maxSizeMB} MB)
          </p>
        </div>
        <input 
          type="file" 
          className="hidden" 
          accept={accept}
          onChange={onFileInputChange} 
        />
      </label>

      {error && (
        <div className="mt-3 flex items-center gap-2 text-sm text-red-400 bg-red-500/10 py-2 px-3 rounded-lg border border-red-500/20">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
