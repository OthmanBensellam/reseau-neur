using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Layers;
using System;

namespace NeuralNetwork.Serialization
{
    internal class LayerSerializer : ILayerSerializer
    {

        //TODO: modify depending on the implemented layers
        public ISerializedLayer Serialize(ILayer layer)
        {
            switch (layer)
            {
                case StandardLayer standardLayer:
                    return SerializeStandardLayer(standardLayer);                
                default:
                    throw new InvalidOperationException("Unknown layer type: " + layer.GetType());
            }
        }

        private ISerializedLayer SerializeStandardLayer(StandardLayer standardLayer)
        {
            throw new NotImplementedException();
        }
    }
}