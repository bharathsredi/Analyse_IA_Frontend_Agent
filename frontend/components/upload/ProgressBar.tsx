interface ProgressBarProps {
  progress: number;
  filename: string;
}

export default function ProgressBar({ progress, filename }: ProgressBarProps) {
  return (
    <div className="w-full bg-[#0F1F3D] border border-blue-500/20 rounded-xl p-4 shadow-sm">
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm font-medium text-white truncate max-w-[80%]">{filename}</span>
        <span className="text-sm font-semibold text-blue-400">{Math.round(progress)}%</span>
      </div>
      <div className="w-full bg-[#0A1628] rounded-full h-2.5 overflow-hidden">
        <div 
          className="bg-gradient-to-r from-blue-600 to-blue-400 h-2.5 rounded-full transition-all duration-300 ease-out" 
          style={{ width: `${progress}%` }} 
        />
      </div>
    </div>
  );
}
