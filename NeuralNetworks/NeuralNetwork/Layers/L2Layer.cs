using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using System;
using System.Text.RegularExpressions;

namespace NeuralNetwork.Layers
{
	internal class L2Layer : ILayer
	{

		public int InputSize { get; }
		public int LayerSize { get; }
		public int BatchSize { get; set; }
		//public IGradientAdjustmentParameters Params_ { get; set; }
		public Matrix<double> Activation { get; set; }

		public Matrix<double> WeightedError { get; set; }//=> throw new NotImplementedException();
														 //velocity doit probablement pas rester ici 
														 //velocity doit etre initialisée a 0 <- prendre en consideration, quelle est la dim -> enlever du constructeur

		//the things above are the same as those of the StandardLayer majority (or all) of wich should be deleted 
		//the things under are specific to the L2Layer

		//the hyper parameter k
		public double k_ { get; set; }
		//underlaying Layer:
		public StandardLayer uLayer_ { get; set; }



		public L2Layer(double k,StandardLayer uLayer)
		{
			k_ = k;
			uLayer_ = uLayer; //verify if this part works.
		}
		public void Propagate(Matrix<double> input)
		{
			//invoke forward propagation on the underlying layer
			uLayer_.Propagate(input);
		}

		public void BackPropagate(Matrix<double> upstreamWeightedErrors)
		{
			//invoke backprop and compute weight gradients on the undelying layer(done implicitly)
			uLayer_.BackPropagate(upstreamWeightedErrors);
			//Multiply weights of the underlying layer by the penalty coefficient k_
			uLayer_.Weights_ *= k_;
			//add the result to the weight gradient of the underlying layer
			uLayer_.GradWeight_ += uLayer_.Weights_;
		}

		public void UpdateParameters()
		{
			//invoke parameter update on the underlying layer
			//(the weight gradient are already updated in back prop ??)
			uLayer_.UpdateParameters();
		}

		public bool Equals(ILayer other)
		{
			throw new NotImplementedException();
		}
	}
}
