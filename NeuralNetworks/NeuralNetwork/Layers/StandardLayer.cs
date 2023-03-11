using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using System;
using System.Text.RegularExpressions;

namespace NeuralNetwork.Layers
{
    internal class StandardLayer : ILayer
    {

        public int InputSize { get; }
        public int LayerSize { get; }
        public IActivator Activator { get; }
        public int BatchSize { get; set; }
        //a enlever:
        public Matrix<double> B_ { get; set; }

        //ajout poids initiaux:

        public Matrix<double> Weights_ { get; set; }
        public Matrix<double> Bias_ { get; set; }
        //ajout Grad
        public Matrix<double> GradWeight_ { get; set; }
        public Matrix<double> GradBias_ { get; set; }
        //valeurs bi
        public Matrix<double> Alphas_ { get; set; }
        public Matrix<double> Output_ { get; set; }
        public IGradientAdjustmentParameters Params_ { get; set; }
        public Matrix<double> Activation { get; set; }

        public Matrix<double> WeightedError { get; set; }//=> throw new NotImplementedException();
        //velocity doit probablement pas rester ici 
        //velocity doit etre initialisée a 0 <- prendre en consideration, quelle est la dim -> enlever du constructeur
        public Matrix<double> VelocityB_ { get; set; }
        public Matrix<double> VelocityW_ { get; set; }

        //int i_;



        public StandardLayer(Matrix<double> initialWeights, Matrix<double> initialBias, int batchSize, IActivator activator,IGradientAdjustmentParameters Params)
        {
            Params_ = Params;
            LayerSize = initialWeights.ColumnCount;
            InputSize = initialWeights.RowCount;
            BatchSize = batchSize;
            Activator = activator ?? throw new ArgumentNullException(nameof(activator));
            Bias_ = initialBias;
            Weights_ = initialWeights;
            Activation = Matrix<double>.Build.Dense(LayerSize, BatchSize);
            //enlever la vilocity d'ici apres 
            //w input layersize
            //b layer, 1
            VelocityW_ = Matrix<double>.Build.Dense(InputSize, LayerSize, 0.0);
            VelocityB_ = Matrix<double>.Build.Dense(LayerSize, 1,0.0);
            //verify if need to be initialised (size etc)
            //GradWeight_ = Matrix<double>.Build.Dense(LayerSize, InputSize); ;
            //GradBias_ = Matrix<double> ;
            //penser a trouver une architecture pr enlever la velocité d'ici
            //VelocityB_ = VelocityB;
            //i_ = 0;
            //VelocityW_ = VelocityW;

        }
        public void Propagate(Matrix<double> input)
        {
            Alphas_ = input;
            //Ligne en dessous a mettre en cas de probleme de size (batch heterogene)
            //Activation = Matrix<double>.Build.Dense(input.RowCount, input.ColumnCount);
            //ligne probablement inutile : son but et d'eviter la modification de weight par .Transpose
            //Matrix<double> WeightsCopy = Matrix<double>.Build.Dense(Weights_.RowCount, Weights_.ColumnCount);
            Output_ = Weights_.Transpose() * input + Bias_;
            Output_.Map(Activator.Apply, Activation);
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            B_ = Output_.Map(Activator.ApplyDerivative);
            B_ = B_.PointwiseMultiply(upstreamWeightedErrors);
            WeightedError = Weights_ * B_;
            GradWeight_ = Alphas_ * B_.Transpose() / BatchSize;
            /*
            MathNet.Numerics.LinearAlgebra.Vector<double> v = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(BatchSize, 1.0);
            Matrix<double> VectorAsMatriceC = Matrix<double>.Build.Dense(BatchSize, 1);
            VectorAsMatriceC.InsertColumn(0, v);
            */
            //Matrix<double> M.Dense(3, 4, 1.0);
            //
            //GradBias_ = (Bj * VectorAsMatriceC) / (double)BatchSize;
            GradBias_ = (B_ * Matrix<double>.Build.Dense(BatchSize, BatchSize, 1.0)) / BatchSize;
        }

        public void UpdateParameters()
        {
            if (Params_.Type == GradientAdjustmentType.FixedLearningRate) 
            {
                Weights_ -= (((FixedLearningRateParameters)Params_).LearningRate) * GradWeight_;
                Bias_ -= (((FixedLearningRateParameters)Params_).LearningRate) * GradBias_;
            }
            else if (Params_.Type == GradientAdjustmentType.Momentum)
            {
                VelocityW_ = (((MomentumParameters)Params_).Momentum) * VelocityW_ - (((MomentumParameters)Params_).LearningRate) * GradWeight_;
                Weights_ += VelocityW_;
                VelocityB_ = (((MomentumParameters)Params_).Momentum) * VelocityB_ - (((MomentumParameters)Params_).LearningRate) * GradBias_;
                Bias_ += VelocityB_;
            }
           
                    
            //(((FixedLearningRateParameters)Params_).LearningRate)*
        }

        public bool Equals(ILayer other)
        {
            throw new NotImplementedException();
        }
    }
}
