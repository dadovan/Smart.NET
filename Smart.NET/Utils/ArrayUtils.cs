using System;

namespace Smart.NET.Utils
{
    public class ArrayUtils
    {
        public static Double[] NewRandomArray(Random random, Int32 length)
        {
            var array = new Double[length];
            for (var i = 0; i < length; i++)
                array[i] = random.NextDouble();
            return array;
        }
    }
}
