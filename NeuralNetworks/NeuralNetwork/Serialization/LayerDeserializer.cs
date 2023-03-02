using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Layers;
using System;

namespace NeuralNetwork.Serialization
{
    internal static class LayerDeserializer
    {
        public static ILayer Deserialize(ISerializedLayer serializedLayer, int batchSize)
        {
            switch (serializedLayer.Type)
            {
                case LayerType.Standard:
                    var standardSerialized = serializedLayer as SerializedStandardLayer;
                    return DeserializeStandardLayer(standardSerialized, batchSize);
                //TODO: complete depending on what layers are implemented
                default:
                    throw new InvalidOperationException("Unknown layer type to deserialize");
            }
        }
        // This method can be modified if necessary
        private static ILayer DeserializeStandardLayer(SerializedStandardLayer standardSerialized, int batchSize)
        {
            var weights = Matrix<double>.Build.DenseOfRowArrays(standardSerialized.Weights);
            var bias = Matrix<double>.Build.DenseOfColumnArrays(new double[][] { standardSerialized.Bias });
            var activator = ActivatorFactory.Build(standardSerialized.ActivatorType);
            return new StandardLayer(weights, bias, batchSize, activator);
        }
    }
}