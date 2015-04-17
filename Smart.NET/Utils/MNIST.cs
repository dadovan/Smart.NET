using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;

namespace Smart.NET.Utils
{
    public class MNIST
    {
        public static List<Tuple<Double[], Double[]>> LoadMNISTData(String filename)
        {
            var data = new List<Tuple<Double[], Double[]>>();
            using (var br = new BinaryReader(File.OpenRead(filename)))
            {
                var iValueCount = br.ReadInt32();
                var dValueCount = br.ReadInt32();
                Assert.AreEqual(784, iValueCount);
                Assert.AreEqual(10, dValueCount);
                var count = br.ReadInt32();
                Console.WriteLine("Loading {0} data rows from MNIST data file {1}", count, filename);
                data = new List<Tuple<Double[], Double[]>>(count);
                for (var row = 0; row < count; row++)
                {
                    var iValues = new Double[iValueCount];
                    var dValues = new Double[dValueCount];
                    for (var i = 0; i < iValues.Length; i++)
                        iValues[i] = br.ReadDouble();
                    for (var i = 0; i < dValues.Length; i++)
                        dValues[i] = br.ReadDouble();
                    data.Add(new Tuple<Double[], Double[]>(iValues, dValues));
                }
            }
            return data;
        }
    }
}
