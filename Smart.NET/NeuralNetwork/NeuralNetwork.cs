using Microsoft.VisualStudio.TestTools.UnitTesting;
using Smart.NET.Utils;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace Smart.NET.NeuralNetwork
{
    public class NeuralNetwork
    {
        public Layer[] Layers;

        public NeuralNetwork()
        {
        }

        public static NeuralNetwork Load(String filename)
        {
            var nn = new NeuralNetwork();
            using (var br = new BinaryReader(File.Open(filename, FileMode.Open, FileAccess.Read, FileShare.None)))
            {
                Assert.AreEqual(1, br.ReadInt16()); // File format version
                var layers = new Layer[br.ReadInt32()];
                layers[0] = new Layer(nn, br.ReadInt32());
                for (var layerIndex = 1; layerIndex < layers.Length; layerIndex++)
                {
                    var nodeCount = br.ReadInt32();
                    Action<Layer> activationFunction;
                    switch (br.ReadInt32())
                    {
                        case 0:
                            activationFunction = SoftmaxLayerActivator.Activate;
                            break;
                        case 1:
                            activationFunction = TanhLayerActivator.Activate;
                            break;
                        default:
                            throw new InvalidDataException("Unknown activation function");
                    }
                    var layer = new Layer(nn, nodeCount, activationFunction, layers[layerIndex - 1]);
                    layer.Biases = new Double[br.ReadInt32()];
                    Assert.AreEqual(nodeCount, layer.Biases.Length);
                    for (var i = 0; i < layer.Biases.Length; i++)
                        layer.Biases[i] = br.ReadDouble();
                    layer.PreviousToLayerWeights = new Double[br.ReadInt32()];
                    Assert.AreEqual(nodeCount * layers[layerIndex - 1].NodeCount, layer.PreviousToLayerWeights.Length);
                    for (var i = 0; i < layer.PreviousToLayerWeights.Length; i++)
                        layer.PreviousToLayerWeights[i] = br.ReadDouble();
                    layers[layerIndex] = layer;
                }
                nn.Layers = layers;
            }
            return nn;
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

        public void ValidateWithConfusionMatrix(List<Tuple<Double[], Double[]>> testData)
        {
            // Compute the confusion matrix
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            var iLayer = Layers[0];
            var oLayer = Layers[Layers.Length - 1];
            var dCount = oLayer.NodeCount;
            var confusionMatrix = new Int32[dCount, dCount];
            foreach (var tuple in testData)
            {
                iLayer.Values = tuple.Item1;
                ComputeForwardOutput();

                var dValues = tuple.Item2;

                var actual = oLayer.Values.Select((v, i) => new { Value = v, Index = i }).OrderByDescending(v => v.Value).First().Index;
                var expected = tuple.Item2.Select((v, i) => new { Value = v, Index = i }).First(v => v.Value == 1d).Index;

                confusionMatrix[expected, actual]++;
            }
            stopwatch.Stop();

            // Render the matrix
            Console.WriteLine("Total ms: {0}", stopwatch.ElapsedMilliseconds);
            Console.Write("\t");
            for (var i = 0; i < dCount; i++)
            {
                Console.Write("{0}\t", i);
            }
            Console.WriteLine("Recall");
            for (var i = 0; i < dCount; i++)
            {
                Console.Write("{0}\t", i);
                var correct = 0;
                var total = 0;
                for (var j = 0; j < dCount; j++)
                {
                    if (j == i)
                        correct += confusionMatrix[i, j];
                    total += confusionMatrix[i, j];
                    Console.Write("{0}\t", confusionMatrix[i, j]);
                }
                var recall = (Double)correct / (Double)total;
                Console.WriteLine(recall.ToString("F3"));
            }
            var precision = new Double[dCount];
            for (var i = 0; i < dCount; i++)
            {
                var correct = 0;
                var total = 0;
                for (var j = 0; j < dCount; j++)
                {
                    if (j == i)
                        correct += confusionMatrix[j, i];
                    total += confusionMatrix[j, i];
                }
                precision[i] = (Double)correct / (Double)total;
            }
            Console.Write("Prec\t");
            for (var i = 0; i < dCount; i++)
            {
                Console.Write(precision[i].ToString("F3") + "\t");
            }
            Console.WriteLine();
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