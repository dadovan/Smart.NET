using System;

namespace Smart.NET.NeuralNetwork
{
    public static class SoftmaxLayerActivator
    {
        public static void Activate(Layer layer)
        {
            var max = layer.Values[0];
            for (var i = 0; i < layer.NodeCount; i++)
                if (layer.Values[i] > max)
                    max = layer.Values[i];
            var scale = 0.0d;
            for (var i = 0; i < layer.NodeCount; i++)
                scale += Math.Exp(layer.Values[i] - max);
            for (var i = 0; i < layer.NodeCount; i++)
                layer.Values[i] = Math.Exp(layer.Values[i] - max) / scale;
        }
    }
}
