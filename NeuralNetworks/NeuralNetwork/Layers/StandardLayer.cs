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

            //verify if need to be initialised (size etc)
            //GradWeight_ = Matrix<double>.Build.Dense(LayerSize, InputSize); ;
            //GradBias_ = Matrix<double> ;
            //penser a trouver une architecture pr enlever la velocité d'ici
            //VelocityB_ = VelocityB;
            //VelocityW_ = VelocityW;

        }
        public void Propagate(Matrix<double> input)
        {
            Alphas_ = input;
            //Ligne en dessous a mettre en cas de probleme de size (batch heterogene)
            //Activation = Matrix<double>.Build.Dense(input.RowCount, input.ColumnCount);
            //ligne probablement inutile : son but et d'eviter la modification de weight par .Transpose
            Matrix<double> WeightsCopy = Matrix<double>.Build.Dense(Weights_.RowCount, Weights_.ColumnCount);
            Output_ = WeightsCopy.Transpose() * input + Bias_;
            Output_.Map(Activator.Apply, Activation);
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            Matrix<double> B = (Output_.Map(Activator.ApplyDerivative)).PointwiseMultiply(upstreamWeightedErrors);
            WeightedError = Weights_ * B;
            GradWeight_ = (Alphas_ * B.Transpose()) / (double)BatchSize;

            //creating a vector of BS ones
            MathNet.Numerics.LinearAlgebra.Vector<double> v = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(BatchSize, 1.0);
            Matrix<double> VectorAsMatriceC = Matrix<double>.Build.Dense(BatchSize, 1);
            VectorAsMatriceC.InsertColumn(0, v);

            GradBias_ = (B * VectorAsMatriceC) / (double)BatchSize;
        }

        public void UpdateParameters()
        {
            Weights_ -= (((FixedLearningRateParameters)Params_).LearningRate)* GradWeight_;
            Bias_ -= (((FixedLearningRateParameters)Params_).LearningRate)* GradBias_;
        }

        public bool Equals(ILayer other)
        {
            throw new NotImplementedException();
        }
    }
}


//a priori le seul endroit qu'on aura a modifier est l'update parameter
//UP question a poser
/*
else if (Type_ == GradientAdjustmentType.Momentum)
{
    while (i + NBatch < C_)
    {
        //calcul de velocity peut etre ecrit d'une merilleure maniére:
        VelocityB_.SetSubMatrix(0, VelocityB_.RowCount, i, i + NBatch, ((MomentumParameters)Params_).Momentum * VelocityB_.SubMatrix(0, VelocityB_.RowCount, i, i + NBatch) - ((MomentumParameters)Params_).LearningRate * GradBias_.SubMatrix(0, GradBias_.RowCount, i, i + NBatch));
        VelocityW_.SetSubMatrix(0, VelocityW_.RowCount, i, i + NBatch, ((MomentumParameters)Params_).Momentum * VelocityW_.SubMatrix(0, VelocityW_.RowCount, i, i + NBatch) - ((MomentumParameters)Params_).LearningRate * GradWeight_.SubMatrix(0, GradWeight_.RowCount, i, i + NBatch));
        Weights_.SetSubMatrix(0, Weights_.RowCount, i, i + NBatch, Weights_.SubMatrix(0, Weights_.RowCount, i, i + NBatch) + VelocityW_.SubMatrix(0, VelocityW_.RowCount, i, i + NBatch));
        Bias_.SetSubMatrix(0, Bias_.RowCount, i, i + NBatch, Bias_.SubMatrix(0, Bias_.RowCount, i, i + NBatch) + VelocityB_.SubMatrix(0, VelocityB_.RowCount, i, i + NBatch));
        i += NBatch;
    }
    if (i < C_ - 1)
    {
        NBatch = C_ - 1 - i;
        //calcul de velocity peut etre ecrit d'une merilleure maniére:
        VelocityB_.SetSubMatrix(0, VelocityB_.RowCount, i, i + NBatch, ((MomentumParameters)Params_).Momentum * VelocityB_.SubMatrix(0, VelocityB_.RowCount, i, i + NBatch) - ((MomentumParameters)Params_).LearningRate * GradBias_.SubMatrix(0, GradBias_.RowCount, i, i + NBatch));
        VelocityW_.SetSubMatrix(0, VelocityW_.RowCount, i, i + NBatch, ((MomentumParameters)Params_).Momentum * VelocityW_.SubMatrix(0, VelocityW_.RowCount, i, i + NBatch) - ((MomentumParameters)Params_).LearningRate * GradWeight_.SubMatrix(0, GradWeight_.RowCount, i, i + NBatch));
        Weights_.SetSubMatrix(0, Weights_.RowCount, i, i + NBatch, Weights_.SubMatrix(0, Weights_.RowCount, i, i + NBatch) + VelocityW_.SubMatrix(0, VelocityW_.RowCount, i, i + NBatch) );
        Bias_.SetSubMatrix(0, Bias_.RowCount, i, i + NBatch, Bias_.SubMatrix(0, Bias_.RowCount, i, i + NBatch) + VelocityB_.SubMatrix(0, VelocityB_.RowCount, i, i + NBatch));
    }
}*/
