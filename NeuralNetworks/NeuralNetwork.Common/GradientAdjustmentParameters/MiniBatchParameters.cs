namespace NeuralNetwork.Common.GradientAdjustmentParameters
{
    /// <summary>
    /// Parameters for adjusting the gradient update using the Momentum technique.
    /// </summary>
    /// <seealso cref="NeuralNetwork.Common.GradientAdjustmentParameters.IGradientAdjustmentParameters" />
    public class MiniBatchParameters : IGradientAdjustmentParameters
    {
        public GradientAdjustmentType Type => GradientAdjustmentType.MiniBatch;

        public double LearningRate { get; set; }
        public int M { get; set; }

        public MiniBatchParameters()
        {
        }
    }
}