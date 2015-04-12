using System;

namespace Smart.NET.NeuralNetwork
{
    public class Layer
    {
        private readonly NeuralNetwork m_neuralNetwork;
        private readonly Action<Layer> m_activator;

        public readonly Int32 LayerId;
        public readonly Int32 NodeCount;
        public readonly Layer PreviousLayer;

        public Double[] PreviousToLayerWeights;
        public Double[] Biases;
        public Double[] Values;

        private Layer(NeuralNetwork neuralNetwork, Action<Layer> activator = null, Layer previousLayer = null)
        {
            LayerId = Environment.TickCount;
            m_neuralNetwork = neuralNetwork;
            m_activator = activator;
            PreviousLayer = previousLayer;
        }

        public Layer(NeuralNetwork neuralNetwork, int nodeCount)
            : this(neuralNetwork)
        {
            NodeCount = nodeCount;
            Values = new Double[nodeCount];
        }

        public Layer(NeuralNetwork neuralNetwork, Double[] values)
            : this(neuralNetwork)
        {
            NodeCount = values.Length;
            Values = values;
        }

        public Layer(NeuralNetwork neuralNetwork, int nodeCount, Action<Layer> activator, Layer previousLayer)
            : this(neuralNetwork, activator, previousLayer)
        {
            NodeCount = nodeCount;
            Values = new Double[nodeCount];
        }

        public Layer(NeuralNetwork neuralNetwork, Double[] values, Action<Layer> activator, Layer previousLayer)
            : this(neuralNetwork, activator, previousLayer)
        {
            NodeCount = values.Length;
            Values = values;
        }

        public void Activate()
        {
            m_activator(this);
        }
    }
}
