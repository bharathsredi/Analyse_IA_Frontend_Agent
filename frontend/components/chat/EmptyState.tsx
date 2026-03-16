import { Brain } from 'lucide-react';

interface EmptyStateProps {
  onSelectPrompt: (prompt: string) => void;
  lang: 'fr' | 'en';
}

export default function EmptyState({ onSelectPrompt, lang }: EmptyStateProps) {
  const isFrench = lang === 'fr';

  const prompts = [
    {
      title: isFrench ? 'Analyse des données' : 'Data Analysis',
      text: isFrench ? 'Quelles sont les valeurs manquantes ou atypiques dans ce jeu de données ?' : 'What are the missing or outlier values in this dataset?'
    },
    {
      title: isFrench ? 'Performance' : 'Performance',
      text: isFrench ? 'Évalue les performances historiques des ventes ou du pipeline.' : 'Evaluate the historical performance of sales or the pipeline.'
    },
    {
      title: isFrench ? 'Prédiction' : 'Prediction',
      text: isFrench ? 'Y a-t-il des tendances saisonnières qui se dégagent ?' : 'Are there any seasonal trends emerging?'
    },
    {
      title: isFrench ? 'Automatisation' : 'Automation',
      text: isFrench ? 'Suggère un modèle de machine learning pour ces données.' : 'Suggest a machine learning model for this data.'
    }
  ];

  return (
    <div className="flex flex-col items-center justify-center h-full px-4 text-center max-w-3xl mx-auto">
      <div className="w-16 h-16 bg-blue-600/20 text-blue-500 rounded-2xl flex items-center justify-center mb-6 border border-blue-500/30 shadow-lg shadow-blue-500/10">
        <Brain className="w-8 h-8" />
      </div>
      
      <h2 className="text-2xl md:text-3xl font-bold text-white mb-3">
        {isFrench ? 'Comment puis-je vous aider aujourd\'hui ?' : 'How can I help you today?'}
      </h2>
      <p className="text-blue-300/60 mb-10 max-w-lg">
        {isFrench 
          ? 'Posez-moi une question sur vos données, ou sélectionnez l\'une de nos suggestions ci-dessous pour démarrer rapidement.' 
          : 'Ask me a question about your data, or select one of our suggestions below to get started quickly.'}
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
        {prompts.map((prompt, i) => (
          <button
            key={i}
            onClick={() => onSelectPrompt(prompt.text)}
            className="flex flex-col text-left p-4 rounded-xl border border-blue-500/10 bg-[#0F1F3D] hover:bg-blue-600/10 hover:border-blue-500/30 transition-all group"
          >
            <span className="font-semibold text-white mb-1 group-hover:text-blue-400 transition-colors">
              {prompt.title}
            </span>
            <span className="text-sm text-blue-300/60 leading-relaxed">
              {prompt.text}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
