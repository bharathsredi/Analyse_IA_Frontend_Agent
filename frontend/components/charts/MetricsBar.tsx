import { BarList, Card, Title, Text, Bold, Flex } from '@tremor/react';

interface MetricsBarProps {
  metrics: Record<string, number>;
}

export default function MetricsBar({ metrics }: MetricsBarProps) {
  const data = Object.entries(metrics).map(([name, value]) => ({
    name,
    value,
  }));

  return (
    <Card className="max-w-md bg-[#0F1F3D] border-[#243B6E]">
      <Title className="text-white">Aperçu des Métriques</Title>
      <Text className="text-blue-300/60 mb-4">Valeurs générales de l'analyse</Text>
      <Flex className="mt-4">
        <Text className="text-blue-300">Métrique</Text>
        <Text className="text-blue-300"><Bold>Valeur</Bold></Text>
      </Flex>
      <BarList data={data} className="mt-2" color="blue" />
    </Card>
  );
}
