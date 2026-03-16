export interface User {
  id: string;
  email: string;
  full_name: string;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
}

export interface LoginRequest {
  email: string;
  password?: string;
}

export interface RegisterRequest {
  full_name: string;
  email: string;
  password?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  language?: 'FR' | 'EN' | 'fr' | 'en';
  timestamp: Date;
  analysis_result?: AnalysisResult;
  analysis?: AnalysisResult;
  isStreaming?: boolean;
}

export interface UploadedFile {
  file_id: string;
  filename: string;
  original_name?: string;
  size?: number;
  size_bytes: number;
  type: string;
  upload_date?: string;
  uploaded_at: string;
}

export interface BackendResult {
  task_id: string;
  status: 'PENDING' | 'SUCCESS' | 'FAILURE';
  result?: AnalysisResult;
  answer?: string;
}

export interface AnalysisResult {
  model?: string;
  best_model?: string;
  rows?: number;
  columns?: number;
  top_features?: TopFeature[];
  anomalies?: BackendAnomalies;
  metrics?: Record<string, number>;
  eda_insights?: any;
}

export interface TopFeature {
  name?: string;
  feature?: string;
  value?: number;
  importance?: number;
  importance_percentage: number;
}

export interface Anomaly {
  id: string;
  description: string;
  severity: 'low' | 'medium' | 'high';
}

export interface BackendAnomalies {
  high: number;
  medium: number;
  low: number;
  total: number;
  percentage: number;
}

export interface HistoryItem {
  id: string;
  type: string;
  query: string;
  response: string;
  response_preview: string;
  timestamp: string;
  created_at: string;
}

export interface RgpdConsent {
  analytics: boolean;
  marketing?: boolean;
  storage: boolean;
  ai_processing: boolean;
  [key: string]: boolean | undefined;
}

export interface AuditEntry {
  id: string;
  action: string;
  timestamp: string;
  details?: string;
}
