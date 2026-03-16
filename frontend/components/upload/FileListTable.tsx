import { UploadedFile } from '@/types';
import { Card, Title, Table, TableHead, TableRow, TableHeaderCell, TableBody, TableCell, Text } from '@tremor/react';
import { FileText, FileSpreadsheet, Trash2, Loader2 } from 'lucide-react';

interface FileListTableProps {
  files: UploadedFile[];
  onDelete: (fileId: string) => void;
  isDeletingId: string | null;
  lang: 'fr' | 'en';
}

export default function FileListTable({ files, onDelete, isDeletingId, lang }: FileListTableProps) {
  const isFrench = lang === 'fr';

  const formatSize = (bytes: number) => {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };
  
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString(isFrench ? 'fr-FR' : 'en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getFileIcon = (type: string) => {
    if (!type) return <FileText className="w-5 h-5 text-red-400" />;
    if (type.includes('csv') || type.includes('excel')) {
      return <FileSpreadsheet className="w-5 h-5 text-emerald-400" />;
    }
    return <FileText className="w-5 h-5 text-red-400" />;
  };

  return (
    <Card className="w-full bg-[#0F1F3D] border-[#243B6E]">
      <Title className="text-white mb-6">{isFrench ? 'Fichiers Uploadés' : 'Uploaded Files'}</Title>
      
      {files.length === 0 ? (
        <div className="py-8 text-center border border-dashed border-blue-500/20 rounded-lg bg-[#0A1628]">
          <Text className="text-blue-300/50">
            {isFrench ? 'Aucun fichier uploadé pour le moment.' : 'No files uploaded yet.'}
          </Text>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <Table className="mt-2">
            <TableHead>
              <TableRow className="border-b border-blue-500/10">
                <TableHeaderCell className="text-blue-300">{isFrench ? 'Type' : 'Type'}</TableHeaderCell>
                <TableHeaderCell className="text-blue-300">{isFrench ? 'Nom du fichier' : 'Filename'}</TableHeaderCell>
                <TableHeaderCell className="text-blue-300">{isFrench ? 'Taille' : 'Size'}</TableHeaderCell>
                <TableHeaderCell className="text-blue-300">{isFrench ? "Date d'upload" : 'Upload Date'}</TableHeaderCell>
                <TableHeaderCell className="text-right text-blue-300">{isFrench ? 'Actions' : 'Actions'}</TableHeaderCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {files.map((file) => (
                <TableRow key={file.file_id} className="border-b border-blue-500/5 hover:bg-blue-600/5 transition-colors">
                  <TableCell>
                    {getFileIcon(file.type || '')}
                  </TableCell>
                  <TableCell className="font-medium text-white max-w-[200px] truncate">
                    {file.original_name || file.filename}
                  </TableCell>
                  <TableCell className="text-gray-400 tabular-nums">
                    {formatSize(file.size_bytes || file.size || 0)}
                  </TableCell>
                  <TableCell className="text-gray-400 tabular-nums">
                    {formatDate(file.uploaded_at || file.upload_date || new Date().toISOString())}
                  </TableCell>
                  <TableCell className="text-right">
                    <button
                      onClick={() => onDelete(file.file_id)}
                      disabled={isDeletingId === file.file_id}
                      className="text-gray-500 hover:text-red-400 transition-colors p-2 rounded-lg hover:bg-red-500/10 disabled:opacity-50"
                      title={isFrench ? "Supprimer le fichier" : "Delete file"}
                    >
                      {isDeletingId === file.file_id ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Trash2 className="w-4 h-4" />
                      )}
                    </button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </Card>
  );
}
