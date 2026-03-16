import { Card, Title, Text, Table, TableHead, TableRow, TableHeaderCell, TableBody, TableCell, Badge } from '@tremor/react';
import { BackendAnomalies } from '@/types';

interface AnomalyTableProps {
  anomalies: BackendAnomalies;
}

export default function AnomalyTable({ anomalies }: AnomalyTableProps) {
  const data = [
    { level: 'High Risk', count: anomalies.high, color: 'red' },
    { level: 'Medium Risk', count: anomalies.medium, color: 'orange' },
    { level: 'Low Risk', count: anomalies.low, color: 'yellow' },
  ];

  return (
    <Card className="w-full bg-[#0F1F3D] border-[#243B6E]">
      <Title className="text-white">Détection d'Anomalies</Title>
      <Text className="text-blue-300/60 mb-4">
        {anomalies.total} anomalies détectées au total ({anomalies.percentage.toFixed(2)}%)
      </Text>

      <Table className="mt-5">
        <TableHead>
          <TableRow className="border-b border-blue-500/10">
            <TableHeaderCell className="text-blue-300">Niveau de Risque</TableHeaderCell>
            <TableHeaderCell className="text-right text-blue-300">Nombre d'Anomalies</TableHeaderCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((item) => (
            <TableRow key={item.level} className="border-b border-blue-500/5 hover:bg-blue-600/5 transition-colors">
              <TableCell>
                <Badge color={item.color as any}>{item.level}</Badge>
              </TableCell>
              <TableCell className="text-right text-white font-medium">
                {item.count}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Card>
  );
}
