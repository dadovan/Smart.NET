using Microsoft.VisualStudio.TestTools.UnitTesting;
using Smart.NET.Utils;
using System;
using System.Threading.Tasks;

namespace Smart.NET.NeuralNetwork
{
    public class NeuralNetwork
    {
        public Layer[] Layers;

        public NeuralNetwork()
        {
        }

        public void ComputeForwardOutput()
        {
            for (var layerIndex = 0; layerIndex < Layers.Length - 1; layerIndex++)
            {
                ComputeLayerForwardValues(Layers[layerIndex], Layers[layerIndex + 1]);
                Layers[layerIndex + 1].Activate();
            }
        }

        private void ComputeLayerForwardValues(Layer layer, Layer forwardLayer)
        {
            Parallel.For(0, forwardLayer.NodeCount, (forwardIndex) =>
            {
                var value = forwardLayer.Biases[forwardIndex];
                for (var layerIndex = 0; layerIndex < layer.NodeCount; layerIndex++)
                {
                    value += (layer.Values[layerIndex] * forwardLayer.PreviousToLayerWeights[(layerIndex * forwardLayer.NodeCount) + forwardIndex]);
                }
                forwardLayer.Values[forwardIndex] = value;
            });
        }
    }
}

namespace Smart.NET.NeuralNetwork.Test
{
    [TestClass]
    public class NeuralNetworkTests
    {
        [TestMethod]
        public void BasicFeedForwardTest()
        {
            var nn = new NeuralNetwork();

            var random = new Random(19741104);
            var iLayer = new Layer(nn, new[] { 0.46246d, 0.56387d, 0.98518d });

            var hLayer = new Layer(nn, 4, TanhLayerActivator.Activate, iLayer);
            hLayer.PreviousToLayerWeights = new[] { 0.57426d, 0.56688d, 0.30634d, 0.62644d, 0.07241d, 0.35078d, 0.62603d, 0.12419d, 0.52969d, 0.78555d, 0.44469d, 0.17940d };
            hLayer.Biases = new[] { 0.84816d, 0.00589d, 0.41893d, 0.31132d };

            var oLayer = new Layer(nn, 2, SoftmaxLayerActivator.Activate, hLayer);
            oLayer.PreviousToLayerWeights = new[] { 0.185883618512137d, 0.262706959742451d, 0.382791653919402d, 0.648913248744287d, 0.149259824375277d, 0.0459774290425598d, 0.771017016270672d, 0.413325088291115d };
            oLayer.Biases = new[] { 0.226960628399141d, 0.0423739743616311d };

            nn.Layers = new[] { iLayer, hLayer, oLayer };

            nn.ComputeForwardOutput();

            hLayer.Values.AssertIsApproximately(new[] { 0.932393086142774d, 0.845384790862073d, 0.874453762631829d, 0.689914067784969d });
            oLayer.Values.AssertIsApproximately(new[] { 0.55603223859186d, 0.44396776140814d });
        }
    }
}