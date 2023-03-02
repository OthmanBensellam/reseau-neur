using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using System;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Numerics;

namespace NeuralNetwork.Layers
{
    internal class StandardLayer : ILayer
    {
        public int InputSize { get; }
        public int LayerSize { get; }
        public IActivator Activator { get; }
        public int BatchSize { get; set; }

        public Matrix<double> Activation { get; set; }
        public Matrix<double> WeightedError { get; set; }

        //ajout Grad
        public Matrix<double> GradWeight_ { get; set; }
        public Matrix<double> GradBias_ { get; set; }

        //ajout poids initiaux:

        public Matrix<double> Weights_ { get; set; }
        public Matrix<double> Bias_ { get; set; }

        //type
        public GradientAdjustmentType Type_  { get; set; }
        public IGradientAdjustmentParameters Params_ { get; set; }// modifier pour un truc plus generique permettant d'integrer les autres types 
        // velocity doit probablement pas rester ici 
        //velocity doit etre initialisée a 0 <- prendre en consideration, quelle est la dim -> enlever du constructeur
        public Matrix<double> VelocityB_ { get; set; }
        public Matrix<double> VelocityW_ { get; set; }

        public StandardLayer(Matrix<double> initialWeights, Matrix<double> initialBias, Matrix<double> VelocityB, Matrix<double> VelocityW, int batchSize, IActivator activator)
        {
            LayerSize = initialWeights.ColumnCount;
            InputSize = initialWeights.RowCount;
            BatchSize = batchSize;
            Activator = activator; //?? throw new ArgumentNullException(nameof(activator));
            Bias_ = initialBias;
            Weights_ = initialWeights;
            //penser a trouver une architecture pr enlever la velocité d'ici
            VelocityB_ = VelocityB;
            VelocityW_ = VelocityW;
            Activation = Matrix<double>.Build.Dense(LayerSize, InputSize);
            //verify if need to be initialised (size etc)
            GradWeight_ = Matrix<double>.Build.Dense(LayerSize, InputSize); ;
            GradBias_ = Matrix<double>.Build.Dense(LayerSize, InputSize); ;

        }

        public void Propagate(Matrix<double> input )
        {
            int C_ = input.ColumnCount;
            int NBatch = (int)(C_ / BatchSize);
            int i = 0;
            while(i+NBatch<C_)
            {
                //definir la matrice a l'ext pr optim
                Matrix<double> M = Weights_.SubMatrix(0, Weights_.RowCount, i, i + NBatch).Transpose() * input.SubMatrix(0, input.RowCount, i, i+NBatch) + Bias_.SubMatrix(0, Bias_.RowCount, i, i+NBatch);
                Activation.SetSubMatrix(0, Bias_.RowCount, i, i+NBatch, M.Map(Activator.Apply));
                i += NBatch;
            }
            if(i < C_-1)
            {
                NBatch = C_ - 1 - i;
                Matrix<double> M = Weights_.SubMatrix(0, Weights_.RowCount, i, i + NBatch).Transpose() + Bias_.SubMatrix(0, Bias_.RowCount, i, i + NBatch);
                Activation.SetSubMatrix(0, Bias_.RowCount, i, i + NBatch, M.Map(Activator.Apply));
            }
        }


        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            int C_ = Activation.ColumnCount;
            int NBatch = (int)(C_ / BatchSize);
            int i = 0;
            MathNet.Numerics.LinearAlgebra.Vector<double> v = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(NBatch, 1.0);
            while (i + NBatch < C_)
            {
                //def la matrice B a l'ext pr optim
                Matrix<double> B = Activation.SubMatrix(0, Activation.RowCount, i, i+NBatch).Map(Activator.ApplyDerivative) * upstreamWeightedErrors.SubMatrix(0, upstreamWeightedErrors.RowCount, i, i + NBatch).Transpose();
                WeightedError.SetSubMatrix(0, WeightedError.RowCount, i, i + NBatch, (Weights_.SubMatrix(0, Weights_.RowCount, i, i + NBatch) * B).Transpose());
                GradWeight_.SetSubMatrix(0, GradWeight_.RowCount, i, i + NBatch, (Activation.SubMatrix(0, Activation.RowCount, i, i + NBatch) * B.Transpose()) / (double)BatchSize);
                GradBias_.SetSubMatrix(0, GradBias_.RowCount, i, i + NBatch, B * (double)BatchSize);
                i += NBatch;
            }
            if (i < C_ - 1)
            {
                NBatch = C_ - 1 - i;
                v = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(NBatch, 1.0);
                Matrix<double> B = Activation.SubMatrix(0, Activation.RowCount, i, i + NBatch).Map(Activator.ApplyDerivative) * upstreamWeightedErrors.SubMatrix(0, upstreamWeightedErrors.RowCount, i, i + NBatch).Transpose();
                WeightedError.SetSubMatrix(0, WeightedError.RowCount, i, i + NBatch, (Weights_.SubMatrix(0, Weights_.RowCount, i, i + NBatch) * B).Transpose());
                GradWeight_.SetSubMatrix(0, GradWeight_.RowCount, i, i + NBatch, (Activation.SubMatrix(0, Activation.RowCount, i, i + NBatch) * B.Transpose()) / (double)BatchSize);
                GradBias_.SetSubMatrix(0, GradBias_.RowCount, i, i + NBatch, B * (double)BatchSize);
            }
        }
        //C:\Users\othman\Downloads\dl-pricer\ensimag-dl-pricing\Examples\BooleanFunctionTester\xor-trained.json
        public void UpdateParameters()
        {//sc2
         // sah re reflechir je suis perdu un peu 
            int C_ = Activation.ColumnCount;
            int NBatch = (int)(C_ / BatchSize);
            int i = 0;
            if (Type_ == GradientAdjustmentType.FixedLearningRate)
            {
                while (i + NBatch < C_)
                {
                    Weights_.SetSubMatrix(0, Weights_.RowCount, i, i + NBatch, Weights_.SubMatrix(0, Weights_.RowCount, i, i + NBatch) - ((FixedLearningRateParameters)Params_).LearningRate * GradWeight_.SubMatrix(0, GradWeight_.RowCount, i, i + NBatch));
                    Bias_.SetSubMatrix(0, Bias_.RowCount, i, i + NBatch, Bias_.SubMatrix(0, Bias_.RowCount, i, i + NBatch) - ((FixedLearningRateParameters)Params_).LearningRate * GradBias_.SubMatrix(0, GradBias_.RowCount, i, i + NBatch));
                    i += NBatch;
                }
                if (i < C_ - 1)
                {
                    NBatch = C_ - 1 - i;
                    Weights_.SetSubMatrix(0, Weights_.RowCount, i, i + NBatch, Weights_.SubMatrix(0, Weights_.RowCount, i, i + NBatch) - ((FixedLearningRateParameters)Params_).LearningRate * GradWeight_.SubMatrix(0, GradWeight_.RowCount, i, i + NBatch));
                    Bias_.SetSubMatrix(0, Bias_.RowCount, i, i + NBatch, Bias_.SubMatrix(0, Bias_.RowCount, i, i + NBatch) - ((FixedLearningRateParameters)Params_).LearningRate * GradBias_.SubMatrix(0, GradBias_.RowCount, i, i + NBatch));
                }
            }
            //a priori le seul endroit qu'on aura a modifier est l'update parameter
            //UP question a poser 
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
            }
        }

        public bool Equals(ILayer other)
        {
            throw new NotImplementedException();
        }
    }
}