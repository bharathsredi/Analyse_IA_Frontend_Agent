import { TopFeature } from '@/types';
import { Card, Title, BarChart, Text } from '@tremor/react';

interface ShapChartProps {
  features: TopFeature[];
}

export default function ShapChart({ features }: ShapChartProps) {
  const chartData = features.map(f => ({
    name: f.name || f.feature || 'Unknown',
    "Importance": f.importance_percentage || f.importance || f.value || 0
  }));

  return (
    <Card className="w-full bg-[#0F1F3D] border-[#243B6E]">
      <Title className="text-white">Impact des Features (SHAP)</Title>
      <Text className="text-blue-300/60 mb-6">Variables les plus discriminantes du modèle</Text>
      
      <BarChart
        className="mt-6 h-72"
        data={chartData}
        index="name"
        categories={["Importance"]}
        colors={["blue"]}
        yAxisWidth={48}
        showAnimation={true}
        valueFormatter={(number) => `${number.toFixed(2)}%`}
      />
    </Card>
  );
}
