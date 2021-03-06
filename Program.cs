﻿using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/// <summary>
/// Overly verbose implementation of a neural network 
/// so I can understand what it's doing before I condense
/// it down to matrix shenanigans
/// </summary>
namespace NN
{
    // Signatures for plugin function types
    // Kinda a hacky C# version of typeof, I guess
    using ErrorFunc = Func<float[], float[], float>;
    using OptimizeFunc = Func<Sample[], float>;
    using ActivatorFunc = Func<float, float>;

    /// <summary>
    /// Interface for an activation function of a node
    /// </summary>
    interface IActivationFunction
    {
        float F(float x);
        float FPrime(float x); 
    }

    class IdentityActivation : IActivationFunction
    {
        public float F(float x)
        {
            return x;
        }

        public float FPrime(float x)
        {
            return 1.0f;
        }
    }

    class LogisticActivation : IActivationFunction
    {
        public float F(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }

        public float FPrime(float x)
        {
            return F(x) * (1.0f - F(x));
        }
    }

    class TanHActivation : IActivationFunction
    {
        public float F(float x)
        {
            return 2.0f / (1.0f + (float)Math.Exp(-2.0f * x)) - 1.0f;
        }

        public float FPrime(float x)
        {
            float e = F(x);
            return 1.0f - e * e;
        }
    }

    class SoftPlusActivation : IActivationFunction
    {
        public float F(float x)
        {
            return (float)Math.Log(1.0f + Math.Exp(x));
        }

        public float FPrime(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }
    }

    /// <summary>
    /// Variation of the hyperbolic tangent activation function
    /// (LeCun, et al. 1998 "Efficient BackProp")
    /// 
    /// Note that this requires a few extra steps:
    /// * Each activator output should be centered around 0
    /// * The output nodes should be [-1, 1]
    /// </summary>
    class LeCunTanHActivation : IActivationFunction
    {
        public float F(float x)
        {
            return 1.7159f * (float)Math.Tanh(2.0f / 3.0f * x);
        }

        public float FPrime(float x)
        {
            float e = (float)(Math.Exp(-2.0f * x / 3.0f) + Math.Exp(2.0f * x / 3.0f));
            return 4.5757f / (e * e);
        }
    }

    /// <summary>
    /// Exponential linear unit activator - with a fixed alpha (for now)
    /// (Clevert, et al. 2015 "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)")
    /// </summary>
    class ELUActivation : IActivationFunction
    {
        const float ALPHA = 0.03f;

        public float F(float x)
        {
            if (x < 0)
            {
                return ALPHA * ((float)Math.Exp(x) - 1.0f);
            }

            return x;
        }

        public float FPrime(float x)
        {
            if (x < 0)
            {
                return F(x) + ALPHA;
            }

            return 1;
        }
    }

    class Utility
    {
        private static Random random = new System.Random();
        
        public static void Shuffle<T>(IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = random.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        /// <summary>
        /// Generate a random number with gaussian (normal) distribution. 
        /// Fast (less trig) technique for this is via http://www.design.caltech.edu/erik/Misc/Gaussian.html
        /// </summary>
        public static float GaussianRandom()
        {
            double x1, x2, w, y1, y2;

            do
            {
                x1 = 2.0 * random.NextDouble() - 1.0;
                x2 = 2.0 * random.NextDouble() - 1.0;
                w = x1 * x1 + x2 * x2;
            } while (w >= 1.0);

            w = Math.Sqrt((-2.0 * Math.Log(w)) / w);

            y1 = (float)(x1 * w);
            y2 = (float)(x2 * w);

            return (float)y1;
        }

        public static Matrix<double> Softmax(Matrix<double> x)
        {
            double sum = 0;

            x.Map(u => Math.Exp(u));

            foreach (var val in x.Enumerate())
            {
                sum += val;
            }

            return x / sum;
        }

        /// <summary>
        /// Euclidean distance between two vectors
        /// </summary>
        /// <param name="predicted"></param>
        /// <param name="actual"></param>
        /// <returns></returns>
        public static float EuclideanDistance(float[] predicted, float[] actual)
        {
            if (predicted.Length != actual.Length)
            {
                throw new Exception("Vector length mismatch");
            }

            float sum = 0;
            for (int i = 0; i < predicted.Length; i++)
            {
                sum += (predicted[i] - actual[i]) * (predicted[i] - actual[i]);
            }

            return (float)Math.Sqrt(sum);
        }

        /// <summary>
        /// MSD of two vectors
        /// 
        /// In short: sum[i=0->n]((predicted_i - actual_i) / n)
        /// </summary>
        /// <param name="predicted"></param>
        /// <param name="actual"></param>
        /// <returns></returns>
        public static float MeanSignedDeviation(float[] predicted, float[] actual)
        {
            if (predicted.Length != actual.Length)
            {
                throw new Exception("Vector length mismatch");
            }

            float sum = 0;
            float n = predicted.Length;

            for (int i = 0; i < n; i++)
            {
                sum += (predicted[i] - actual[i]) / n;
            }

            return sum;
        }

        /// <summary>
        /// MSE of two vectors
        /// 
        /// In short: (1/n)*sum[i=0->n]((predicted_i - actual_i)^2)
        /// </summary>
        /// <param name="predicted"></param>
        /// <param name="actual"></param>
        /// <returns></returns>
        public static float MeanSquaredError(float[] predicted, float[] actual)
        {
            if (predicted.Length != actual.Length)
            {
                throw new Exception("Vector length mismatch");
            }

            float sum = 0;
            float n = predicted.Length;

            for (int i = 0; i < n; i++)
            {
                sum += (predicted[i] - actual[i]) * (predicted[i] - actual[i]);
            }

            return sum / n;
        }

        /// <summary>
        /// MAE of two vectors
        /// 
        /// In short: (1/n) * sum[i=0->n](abs(predicted_i - actual_i))
        /// </summary>
        /// <param name="predicted"></param>
        /// <param name="actual"></param>
        /// <returns></returns>
        public static float MeanAbsoluteError(float[] predicted, float[] actual)
        {
            if (predicted.Length != actual.Length)
            {
                throw new Exception("Vector length mismatch");
            }

            float sum = 0;
            float n = predicted.Length;

            for (int i = 0; i < n; i++)
            {
                sum += Math.Abs(predicted[i] - actual[i]);
            }

            return sum / n;
        }

        /// <summary>
        /// SSE of two vectors
        /// 
        /// In short: sum[i=0->n]((predicted_i - actual_i)^2)
        /// </summary>
        /// <param name="predicted"></param>
        /// <param name="actual"></param>
        /// <returns></returns>
        public static float SumSquaredError(float[] predicted, float[] actual)
        {
            if (predicted.Length != actual.Length)
            {
                throw new Exception("Vector length mismatch");
            }

            float sum = 0;
            float n = predicted.Length;

            for (int i = 0; i < n; i++)
            {
                sum += (predicted[i] - actual[i]) * (predicted[i] - actual[i]);
            }

            return sum;
        }

        /// <summary>
        /// Simple difference, using only the first index of both vectors
        /// 
        /// predicted - actual, obv
        /// </summary>
        /// <param name="predicted"></param>
        /// <param name="actual"></param>
        /// <returns></returns>
        public static float SimpleDifferenceError(float[] predicted, float[] actual)
        {
            return predicted[0] - actual[0];
        }

        public static string VecToString(float[] vec)
        {
            return '[' + string.Join(",", vec) + ']';
        }
    }

    class Node
    {
        /// <summary>
        /// Weights between *previous* nodes and this node
        /// </summary>
        public float[] weights;

        /// <summary>
        /// Tracking of weight deltas between previous nodes,
        /// used for factoring in momentum
        /// </summary>
        public float[] previousWeightDelta;
        
        /// <summary>
        /// Weighted sum of previous node outputs + bias
        /// </summary>
        public float input;

        /// <summary>
        /// Activation function output of this node
        /// </summary>
        public float output;

        /// <summary>
        /// Bias at this node
        /// </summary>
        public float bias;

        /// <summary>
        /// Tracking of previous bias delta,
        /// used for factoring in momentum
        /// </summary>
        public float previousBiasDelta;

        /// <summary>
        /// Tracked during training. 
        /// (predicted_class - actual_class) * sigmoidDeriv(input)
        /// </summary>
        public float delta;

        /// <summary>
        /// Create a new NN node
        /// </summary>
        /// <param name="prevNodes">Number of nodes in the previous layer</param>
        public Node(int prevNodes)
        {
            bias = Utility.GaussianRandom();
            output = 0;
            delta = 0;
            input = 0;

            if (prevNodes > 0)
            {
                weights = new float[prevNodes];
                previousWeightDelta = new float[prevNodes];

                for (int n = 0; n < prevNodes; n++)
                {
                    weights[n] = Utility.GaussianRandom();
                    previousWeightDelta[n] = 0;
                }
            }
        }
    }
    
    class Sample
    {
        public float[] attr;
        public string classification;
    }

    class NeuralNetwork
    {
        public float trainingRate;
        public float momentum;
        public int epoch;
        public float errorThreshold;

        /// <summary>
        /// If there is only one output node, samples either
        /// match the positive class label (1) or don't (0)
        /// </summary>
        public string positiveClass;
        public IActivationFunction activator;

        public int minibatchSize;

        private Node[] inputLayer;
        private Node[][] layers;

        private List<string> classifications;

        public NeuralNetwork(int inputNodes, int hiddenNodes = 0, int outputNodes = 1)
        {
            BuildNetwork(inputNodes, hiddenNodes, outputNodes);
        }
        
        public void BuildNetwork(int inputNodes, int hiddenNodes = 0, int outputNodes = 1)
        {
            // Use Weka's 'a' setting if not specified
            if (hiddenNodes < 1)
            {
                hiddenNodes = (inputNodes + outputNodes) / 2;
            }
            
            // Setup input layer
            inputLayer = new Node[inputNodes];
            for (int i = 0; i < inputNodes; i++)
            {
                inputLayer[i] = new Node(0);
            }
            
            // Setup hidden layer(s) - just one for now
            layers = new Node[2][];
            layers[0] = new Node[hiddenNodes];
            for (int i = 0; i < hiddenNodes; i++)
            {
                layers[0][i] = new Node(inputNodes);
            }

            // Setup output layer
            layers[1] = new Node[outputNodes];
            for (int i = 0; i < outputNodes; i++)
            {
                layers[1][i] = new Node(hiddenNodes);
            }
        }

        /// <summary>
        /// Record the nominal classifications for later transforming into a binary vec
        /// </summary>
        public void RecordClassifications(Sample[] samples)
        {
            classifications = new List<string>();
            
            foreach (var sample in samples)
            {
                if (!classifications.Contains(sample.classification))
                {
                    classifications.Add(sample.classification);
                }
            }

            // Ensure that classifications are always in the same order
            classifications.Sort();
        }

        public Sample[] GetTrainingSamples()
        {
            var samples = new Sample[4];

            samples[0] = new Sample()
            {
                attr = new float[3] { 0, 0, 1f },
                classification = "Zero"
            };

            samples[1] = new Sample()
            {
                attr = new float[3] { 0, 1f, 1f },
                classification = "One"
            };

            samples[2] = new Sample()
            {
                attr = new float[3] { 1f, 0, 1f },
                classification = "One"
            };

            samples[3] = new Sample()
            {
                attr = new float[3] { 1f, 1f, 1f },
                classification = "Zero"
            };

            return samples;
        }

        public Sample[] GetEasierTrainingSamples()
        {
            var samples = new Sample[4];

            samples[0] = new Sample()
            {
                attr = new float[3] { 0, 0, 1f },
                classification = "Zero"
            };

            samples[1] = new Sample()
            {
                attr = new float[3] { 1f, 1f, 1f },
                classification = "One"
            };

            samples[2] = new Sample()
            {
                attr = new float[3] { 1f, 0, 1f },
                classification = "One"
            };

            samples[3] = new Sample()
            {
                attr = new float[3] { 0, 1f, 1f },
                classification = "Zero"
            };

            return samples;
        }
        
        /// <summary>
        /// Returns the class hypothesis after forward 
        /// propagating the sample through the network
        /// </summary>
        /// <param name="sample"></param>
        /// <returns>hypthesis</returns>
        public float[] FeedForward(Sample sample)
        {
            var previousLayer = inputLayer;

            // Feed sample as the input layer's outputs
            for (int i = 0; i < inputLayer.Length; i++)
            {
                inputLayer[i].output = sample.attr[i];
            }
            
            foreach (var layer in layers)
            {
                foreach (var node in layer)
                {
                    // For all nodes in the previous layer
                    //      current node's input += edge weight * previous node's output
                    // Also include the bias as an additional special node,
                    // which has a static output but no edge weight
                    node.input = 0;
                    for (int j = 0; j < previousLayer.Length; j++)
                    {
                        node.input += node.weights[j] * previousLayer[j].output; // + node.bias;
                    }
                    
                    node.input += node.bias;

                    // Output is the activation function of the input
                    node.output = activator.F(node.input);
                }

                previousLayer = layer;
            }

            return GetOutput();
        }

        private float[] GetOutput()
        {
            var outputLayer = layers.Last();

            float[] output = new float[outputLayer.Length];
            
            for (int i = 0; i < outputLayer.Length; i++)
            {
                output[i] = outputLayer[i].output;
            }
            
            return output;
        }

        private float BackPropagate(Sample sample)
        {
            float weightDelta = 0;
            float biasDelta = 0;
            float error = 0;

            // Back-propagate the error to update weights/biases
            Node[] forwardLayer = null;

            for (int i = layers.Length - 1; i >= 0; i--)
            {
                var layer = layers[i];

                // If we're in the output layer, set the error delta 
                // to the input hypothesis error term (typically hypothesis - actual)
                if (forwardLayer == null)
                {
                    // foreach (var node in layer.nodes)
                    for (int j = 0; j < layer.Length; j++)
                    {
                        Node node = layer[j];

                        // Convert the classification to either 1 (output node is for the same class)
                        // or 0 (output node is for a different class). If there's only one output node,
                        // our actual is 1 iff the sample's classification matches positiveClass (0 otherwise)

                        // We also use values [0.1, 0.9] to try to quicken optimization 
                        // (LeCun, et al. 1998 "Efficient BackProp")
                        float actual = 0.1f; // -0.9f;
                        if (layer.Length < 2)
                        {
                            if (sample.classification == positiveClass)
                            {
                                actual = 0.9f;
                            }
                        }
                        else if (sample.classification == classifications[j])
                        {
                            actual = 0.9f;
                        }
                        
                        // Reported overall error will be (1/2)*sum((actual - hypothesis)^2)
                        error += (float)Math.Pow(actual - node.output, 2);

                        // Set the delta to the derivative of the error (hypothesis - actual)
                        //  * derivative of the aggregate inputs
                        node.delta = (node.output - actual) * activator.FPrime(node.input);
                    }
                }
                else
                {
                    // If we're in a hidden layer, aggregate the edge weight * delta of 
                    // every node connected to this node from the forward layer
                    for (int j = 0; j < layer.Length; j++)
                    {
                        var node = layer[j];

                        float sum = 0;
                        foreach (var forwardNode in forwardLayer)
                        {
                            sum += forwardNode.delta * activator.FPrime(node.input) * forwardNode.weights[j];
                        }

                        node.delta = sum;
                    }
                }

                forwardLayer = layer;
            }

            // Update weights and biases based on backprop deltas
            var prevLayer = inputLayer;
            foreach (var layer in layers)
            {
                foreach (var node in layer)
                {
                    // Update weight between previous nodes and this node
                    for (int j = 0; j < prevLayer.Length; j++)
                    {
                        weightDelta = trainingRate * node.delta * prevLayer[j].output;

                        // Factor in momentum to the weight to reduce oscillation
                        weightDelta += momentum * node.previousWeightDelta[j];
                        node.previousWeightDelta[j] = weightDelta;

                        node.weights[j] -= weightDelta;
                    }

                    // Update bias for this node 
                    biasDelta = trainingRate * node.delta;

                    // Factor in momentum to the bias
                    biasDelta += momentum * node.previousBiasDelta;
                    node.previousBiasDelta = biasDelta;

                    node.bias -= biasDelta;
                }
                
                prevLayer = layer;
            }

            // Matrix version of this sorta kinda:
            // error[layer] = ElementWiseMultiply( // .* in Matlab
            //                  weights[layer].transpose * error[layer + 1], 
            //                  SigmoidDerivative(input values at [layer + 1])
            //                      OR
            //                  ElementWiseMultiply(activations[layer], (1 - activations[layer]))
            //              )

            return error / 2;
        }

        /// <summary>
        /// Perform training with the given samples using SGD
        /// (back propagating an error per distinct sample trained with)
        /// </summary>
        /// <param name="samples"></param>
        /// <returns>Mean SSE error over the samples</returns>
        public float StochasticGradientDescent(Sample[] samples)
        {
            float totalError = 0;
            
            Utility.Shuffle(samples);

            foreach (var sample in samples)
            {
                // Feed forward through the network 
                FeedForward(sample);
                
                // Backprop the error through the network
                float error = BackPropagate(sample);
                totalError += error;
            }
            
            return totalError / samples.Length;
        }
        
        public float[] Train(
            Sample[] samples,
            OptimizeFunc optimizeFunc
        ) {
            // Error recorded for each iteration
            List<float> errorList = new List<float>();
            
            RecordClassifications(samples);

            // Iterate over the epoch, running the desired optimization algorithm for each iteration
            float error = 1.0f;
            int iteration;
            for (iteration = 0; iteration < epoch && error > errorThreshold; iteration++)
            {
                error = optimizeFunc(samples);
                errorList.Add(error);
                Console.WriteLine("Iteration " + iteration + " Error: " + error);
            }

            if (iteration == epoch)
            {
                Console.WriteLine(
                    "Gave up after " + iteration + " iterations. Error: " + errorList.Last().ToString("0.00000")
                );
            }
            else
            {
                Console.WriteLine(
                    "Hit threshold of " + errorList.Last().ToString("0.00000") + " at " + iteration + "th iteration"
                );
            }

            return errorList.ToArray();
        }

        /// <summary>
        /// Get the predicted class index given a hypothesis vector (output node values)
        /// </summary>
        /// <param name="hypothesis"></param>
        /// <returns></returns>
        private int GetPredictedClassIndex(float[] hypothesis, out float certainty)
        {
            int bestIndex = 0;
            for (int i = 0; i < hypothesis.Length; i++)
            {
                if (hypothesis[i] > hypothesis[bestIndex])
                {
                    bestIndex = i;
                }
            }

            certainty = hypothesis[bestIndex];
            return bestIndex;
        }

        /// <summary>
        /// Display a Weka-style confusion matrix (and other statistics)
        /// for a binary classification (1 output node)
        /// </summary>
        public void Test(Sample[] samples)
        {
            RecordClassifications(samples);

            int n = 2;
            float accuracy = 0;
            var confusion = new float[n, n];

            // Certainty of class per sample
            var certainty = new float[samples.Length];
            
            for (int y = 0; y < n; y++)
            {
                for (int x = 0; x < n; x++)
                {
                    confusion[x, y] = 0;
                }
            }

            // Console.WriteLine("Sample  Class Certainty");
            for (int i = 0; i < samples.Length; i++)
            {
                var hypothesis = FeedForward(samples[i]);

                // Note these are inverted (0 = positive class) just
                // so that we can display the positive class first in the 
                // confusion matrix
                var predicted = Convert.ToInt32(hypothesis[0] < 0.5f);
                var actual = Convert.ToInt32(samples[i].classification != positiveClass);
                
                if (predicted == actual)
                {
                    accuracy++;
                }

                confusion[predicted, actual]++;
                certainty[i] = hypothesis[0];
                
                // Report certainty for this sample
                // Console.WriteLine(
                //     i.ToString().PadLeft(6) + "  " + 
                //     certainty[i].ToString("0.0000").PadLeft(4)
                // );
            }

            // Confusion matrix header of codes per class (weka-style)
            int padding = samples.Length.ToString().Length + 1;
            
            Console.WriteLine("T".PadLeft(padding) + "F" .PadLeft(padding) + "  <-- classified as");

            for (int y = 0; y < n; y++)
            {
                for (int x = 0; x < n; x++)
                {
                    Console.Write(confusion[x, y].ToString().PadLeft(padding));
                }

                // End of line class name
                Console.WriteLine(" | " + (y == 0 ? "T" : "F"));
            }

            Console.WriteLine("Accuracy: " + (accuracy / samples.Length * 100.0f).ToString("0.0000") + "%");
        }

        /// <summary>
        /// Run the given test samples through the previously trained NN
        /// and dumps a Weka-style confusion matrix and some statistical results
        /// </summary>
        /// <param name="samples"></param>
        public void TestMulticlass(Sample[] samples)
        {
            RecordClassifications(samples);

            int n = classifications.Count;
            float accuracy = 0;

            // Confusion matrix of values
            var confusion = new float[n, n];

            // Certainty of class per sample
            var certainty = new float[samples.Length];

            for (int y = 0; y < n; y++)
            {
                for (int x = 0; x < n; x++)
                {
                    confusion[x, y] = 0;
                }
            }

            // Console.WriteLine("Sample  Class Certainty");
            for (int i = 0; i < samples.Length; i++)
            {
                var hypothesis = FeedForward(samples[i]);
                var predicted = GetPredictedClassIndex(hypothesis, out certainty[i]);
                var actual = classifications.FindIndex(x => x == samples[i].classification);

                if (predicted == actual)
                {
                    accuracy++;
                }

                confusion[predicted, actual]++;

                // Report certainty for this sample
                // Console.WriteLine(
                //     i.ToString().PadLeft(6) + "  " + 
                //     certainty[i].ToString("0.0000").PadLeft(4)
                // );
            }
         
            // Confusion matrix header of codes per class (weka-style)
            string alpha = "abcdefghijklmnopqrstuvwxyz";
            int padding = samples.Length.ToString().Length + 1;

            for (int x = 0; x < n; x++)
            {
                Console.Write(alpha[x].ToString().PadLeft(padding));
            }
            Console.WriteLine("   <-- classified as");

            for (int y = 0; y < n; y++)
            {
                for (int x = 0; x < n; x++)
                {
                    Console.Write(confusion[x, y].ToString().PadLeft(padding));
                }

                // End of line class name
                Console.WriteLine(" | " + alpha[y] + " = " + classifications[y]);
            }

            Console.WriteLine("Accuracy: " + (accuracy / samples.Length * 100.0f).ToString("0.0000") + "%");
        }
        
    }

    class Program
    {
        /// <summary>
        /// Apply normalization to the sample set, 
        /// since I hate doing it in Excel by hand every time.
        /// </summary>
        /// <param name="samples"></param>
        static void NormalizeSamples(Sample[] samples)
        {
            var n = samples[0].attr.Length;
            var min = new float[n];
            var max = new float[n];

            // Grab range of each attribute
            foreach (var sample in samples)
            {
                for (int i = 0; i < n; i++)
                {
                    min[i] = Math.Min(min[i], sample.attr[i]);
                    max[i] = Math.Max(max[i], sample.attr[i]);
                }
            }

            // Normalize each attribute to [-0.5, 0.5]
            foreach (var sample in samples)
            {
                for (int i = 0; i < n; i++)
                {
                    if (max[i] != min[i])
                    {
                        sample.attr[i] = (sample.attr[i] - min[i]) / (max[i] - min[i]) - 0.5f;
                    }
                }
            }
        }

        static Sample[] LoadSamplesFromCSV(string filename)
        {
            List<Sample> samples = new List<Sample>();

            using (FileStream fs = File.OpenRead(filename))
            {
                using (StreamReader reader = new StreamReader(fs))
                {
                    // Skip header line
                    reader.ReadLine();
                    while (!reader.EndOfStream)
                    {
                        string line = reader.ReadLine();
                        string[] values = line.Split(',');

                        // Last value is class #
                        var sample = new Sample()
                        {
                            classification = values.Last(),
                            attr = new float[values.Length - 1]
                        };

                        // Rest are attributes 
                        for (int i = 0; i < values.Length - 1; i++)
                        {
                            sample.attr[i] = float.Parse(values[i]);
                        }

                        samples.Add(sample);
                    }

                }
            }

            return samples.ToArray();
        }

        static void WriteIterationLog(NeuralNetwork network, float[] error)
        {
            using (FileStream fs = File.Create(@"iterations-error.csv"))
            {
                using (StreamWriter writer = new StreamWriter(fs))
                {
                    // Header 
                    writer.WriteLine
                        ("epoch=" + network.epoch +
                        " threshold=" + network.errorThreshold +
                        " rate=" + network.trainingRate +
                        " momentum=" + network.momentum +
                        " minibatch=" + network.minibatchSize
                    );
                    
                    for (int i = 0; i < error.Length; i++)
                    {
                        writer.WriteLine(error[i]);
                    }
                }
            }
        }

        static void WriteIterationGroup(NeuralNetwork network, float[][] error)
        {
            int maxIterations = 0;
            for (int run = 0; run < error.Length; run++)
            {
                maxIterations = Math.Max(maxIterations, error[run].Length);
            }

            using (FileStream fs = File.Create(@"iteration-group.csv"))
            {
                using (StreamWriter writer = new StreamWriter(fs))
                {
                    // Header
                    writer.WriteLine
                        ("epoch=" + network.epoch + 
                        " threshold=" + network.errorThreshold + 
                        " rate=" + network.trainingRate +
                        " momentum=" + network.momentum +
                        " minibatch=" + network.minibatchSize 
                    );


                    // Column headers
                    for (int run = 0; run < error.Length; run++)
                    {
                        writer.Write("R" + run + ",");
                    }

                    writer.Write("\n");

                    // Column data
                    for (int iter = 0; iter < maxIterations; iter++)
                    {
                        for (int run = 0; run < error.Length; run++)
                        {
                            if (error[run].Length <= iter)
                            {
                                writer.Write(",");
                            }
                            else
                            {
                                writer.Write(error[run][iter] + ",");
                            }
                        }

                        writer.Write("\n");
                    }
                }
            }
        }
        
        static void IrisBinaryTest()
        {
            int runs = 5;

            float[][] error = new float[runs][];

            NeuralNetwork network = null;
            for (int i = 0; i < runs; i++)
            {
                // Need to reinitialize network & samples, otherwise
                // we have old network settings
                network = new NeuralNetwork(4, 0, 1)
                {
                    trainingRate = 0.3f,
                    momentum = 0.2f,
                    epoch = 1000,
                    errorThreshold = 0.01f,
                    activator = new TanHActivation()
                };

                // var samples = network.GetEasierTrainingSamples();
                // var samples = network.GetTrainingSamples();
                // var samples = LoadSamplesFromCSV("iris-two-class-normalized.csv");
                var samples = LoadSamplesFromCSV("iris-normalized-mean-zero.csv");

                error[i] = network.Train(
                    samples,
                    network.StochasticGradientDescent
                );

                network.Test(samples);
            }

            WriteIterationGroup(network, error);
        }

        static void IrisMulticlassTest()
        {
            int runs = 5;

            float[][] error = new float[runs][];

            NeuralNetwork network = null;
            for (int i = 0; i < runs; i++)
            {
                network = new NeuralNetwork(4, 0, 3)
                {
                    trainingRate = 0.3f,
                    momentum = 0.2f,
                    epoch = 5000,
                    errorThreshold = 0.01f,
                    activator = new TanHActivation(),
                    minibatchSize = 3,
                    positiveClass = "Baseline"
                };
                
                var samples = LoadSamplesFromCSV("iris-normalized-mean-zero.csv");

                error[i] = network.Train(
                    samples,
                    network.StochasticGradientDescent
                );
                
                network.TestMulticlass(samples);
            }

            WriteIterationGroup(network, error);
        }

        static void RealDataTest()
        {
            // 10 vec3 per sample, 300 nodes
            // Limit: 199 vec3 (including 0)
            var network = new NeuralNetwork(597, 0, 1)
            {
                trainingRate = 0.3f,
                momentum = 0.2f,
                epoch = 500,
                errorThreshold = 0.01f,
                activator = new LogisticActivation(),
                positiveClass = "Baseline"
            };

            var samples = LoadSamplesFromCSV("samples-various-0.01s.csv");
            NormalizeSamples(samples);

            var errors = network.Train(samples, network.StochasticGradientDescent);

            network.Test(samples);

            WriteIterationLog(network, errors);
        }

        static void Vec3Test()
        {
            var network = new Vector3NeuralNetwork(100)
            {
                trainingRate = 1.0f,
                momentum = 0.5f,
                epoch = 500,
                errorThreshold = 0.001f,
                positiveClass = "Spiral"
            };

            var trainingSamples = Vector3NeuralNetwork.LoadSamplesFromCSV("vec3-train.csv");
            var testSamples = Vector3NeuralNetwork.LoadSamplesFromCSV("vec3-test.csv");
            
            var errors = network.Train(trainingSamples);
            network.Test(testSamples);
        }

        static void Main(string[] args)
        {
            Vec3Test();

            Console.ReadLine();
        }
    }
}
