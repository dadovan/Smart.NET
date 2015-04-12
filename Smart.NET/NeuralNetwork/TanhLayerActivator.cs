using Microsoft.VisualStudio.TestTools.UnitTesting;
using Smart.NET.Utils;
using System;
using System.Threading.Tasks;

namespace Smart.NET.NeuralNetwork
{
    public static class TanhLayerActivator
    {
        public static void Activate(Layer layer)
        {
            Parallel.For(0, layer.NodeCount, (layerIndex) =>
            {
                layer.Values[layerIndex] = Math.Tanh(layer.Values[layerIndex]);
            });
        }
    }
}

namespace Smart.NET.NeuralNetwork.Test
{
    [TestClass]
    public class TanhLayerActivatorTests
    {
        [TestMethod]
        public void GeneralTest()
        {
            var layer1 = new Layer(null, new[] { 0.46246d, 0.56387d, 0.98518d });
            var layer2 = new Layer(null, new[] { 0.84816d, 0.00589d, 0.41893d, 0.31132d }, TanhLayerActivator.Activate, layer1);
            layer2.PreviousToLayerWeights = new[] { 0.57426d, 0.56688d, 0.30634d, 0.62644d, 0.07241d, 0.35078d, 0.62603d, 0.12419d, 0.52969d, 0.78555d, 0.44469d, 0.17940d };
            layer2.Biases = new[] { 0.84816d, 0.00589d, 0.41893d, 0.31132d };

            layer2.Activate();

            layer2.Values.AssertIsApproximately(new[] { 0.690106988761489d, 0.00588993188878884d, 0.396028631872285d, 0.30163747397554d });
        }
    }
}