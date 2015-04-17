using Smart.NET.NeuralNetwork;
using Smart.NET.Utils;

namespace Smart.NET_CLI
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var nn = NeuralNetwork.Load(@"..\..\DataFiles\MNIST.model");
            var testData = MNIST.LoadMNISTData(@"..\..\DataFiles\test_10k_28x28.bin");
            nn.ValidateWithConfusionMatrix(testData);
        }
    }
}
