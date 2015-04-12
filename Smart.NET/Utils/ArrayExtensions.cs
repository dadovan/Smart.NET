using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace Smart.NET.Utils
{
    public static class ArrayExtensions
    {
        public static void AssertIsApproximately(this Double[] array1, Double[] array2, Double delta = 0.00000001d)
        {
            Assert.IsNotNull(array2);
            Assert.AreEqual(array1.Length, array2.Length);
            for (var i = 0; i < array1.Length; i++)
            {
                if (delta == 0d)
                    Assert.AreEqual(array1[i], array2[i], "Item at index {0} is different (max delta = 0)", i);
                else
                    Assert.AreEqual(array1[i], array2[i], delta, "Item at index {0} is different (max delta = {1})", i, delta);
            }
        }
    }
}
