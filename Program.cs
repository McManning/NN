using MathNet.Numerics.LinearAlgebra;
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
        public Dictionary<Node, float> weights;

        /// <summary>
        /// Tracking of weight deltas between previous nodes,
        /// used for factoring in momentum
        /// </summary>
        public Dictionary<Node, float> previousWeightDelta;
        
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

        public Node()
        {
            weights = new Dictionary<Node, float>();
            previousWeightDelta = new Dictionary<Node, float>();

            bias = Utility.GaussianRandom();
            output = 0;
            delta = 0;
            input = 0;
        }

        /// <summary>
        /// Add link to a node in the previous layer
        /// </summary>
        /// <param name="node"></param>
        public void AddPrevious(Node node) {
            weights[node] = Utility.GaussianRandom();
            previousWeightDelta[node] = 0;
        }

        public string WeightsToString()
        {
            return string.Join(",", weights.Values);
        }
    }

    class Layer
    {
        public Node[] nodes;

        public int Count
        {
            get
            {
                return nodes.Length;
            }
        }

        public float[] Output
        {
            get
            {
                float[] output = new float[nodes.Length];
                for (int i = 0; i < nodes.Length; i++)
                {
                    output[i] = nodes[i].output;
                }

                return output;
            }
        }

        public Layer(int nodeCount)
        {
            nodes = new Node[nodeCount];
            for (int i = 0; i < nodeCount; i++)
            {
                nodes[i] = new Node();
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

        private List<Layer> layers;
        private Layer inputLayer;

        private List<string> classifications;

        public NeuralNetwork(int inputNodes, int hiddenNodes = 0, int outputNodes = 1)
        {
            layers = new List<Layer>();

            BuildNetwork(inputNodes, hiddenNodes, outputNodes);
        }

        public void Print()
        {
            Console.WriteLine("==== Input Layer ====");

            for (int n = 0; n < inputLayer.nodes.Length; n++)
            {
                Console.WriteLine("Node" + n +
                    " <output=" + inputLayer.nodes[n].output +
                    ", bias=" + inputLayer.nodes[n].bias +
                    ">"
                );
            }

            for (int i = 0; i < layers.Count; i++)
            {
                Console.WriteLine("==== Layer ".PadLeft(i * 10 + 10, ' ') + i + " ====");
                
                for (int n = 0; n < layers[i].nodes.Length; n++)
                {
                    Console.WriteLine("Node".PadLeft(i * 10 + 10, ' ') + n + 
                        " <output=" + layers[i].nodes[n].output +
                        ", bias=" + layers[i].nodes[n].bias +
                        ", weights=" + layers[i].nodes[n].WeightsToString() + 
                        ">"
                    );
                }
            }
        }

        public void BuildNetwork(int inputNodes, int hiddenNodes = 0, int outputNodes = 1)
        {
            // Use Weka's 'a' setting if not specified
            if (hiddenNodes < 1)
            {
                hiddenNodes = (inputNodes + outputNodes) / 2;
            }

            var input = new Layer(inputNodes);
            var hidden = new Layer(hiddenNodes);
            var output = new Layer(outputNodes);

            Console.WriteLine(
                "Building network of " + 
                input.nodes.Length + " inputs, " + 
                hidden.nodes.Length + " hidden, and " + 
                output.nodes.Length + " outputs"
            );

            inputLayer = input;

            layers.Add(hidden);
            layers.Add(output);

            // Add connections between nodes
            foreach (var node in hidden.nodes)
            {
                foreach (var prev in input.nodes)
                {
                    node.AddPrevious(prev);
                }
            }

            foreach (var node in output.nodes)
            {
                foreach (var prev in hidden.nodes)
                {
                    node.AddPrevious(prev);
                }
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
            Layer previousLayer = inputLayer;

            // Feed sample as the input layer's outputs
            for (int i = 0; i < sample.attr.Length; i++)
            {
                inputLayer.nodes[i].output = sample.attr[i];
            }
            
            foreach (var layer in layers)
            {
                foreach (var node in layer.nodes)
                {
                    // For all nodes in the previous layer
                    //      current node's input += edge weight * previous node's output
                    // Also include the bias as an additional special node,
                    // which has a static output but no edge weight
                    node.input = 0;
                    foreach (var prev in node.weights)
                    {
                        node.input += prev.Value * prev.Key.output; // + node.bias;
                    }
                    
                    node.input += node.bias;

                    // Output is the activation function of the input
                    node.output = activator.F(node.input);
                }

                previousLayer = layer;
            }

            return layers.Last().Output;
        }

        private float BackPropagate(Sample sample)
        {
            float weightDelta = 0;
            float biasDelta = 0;
            float error = 0;

            // Back-propagate the error to update weights/biases
            Layer forwardLayer = null;
            foreach (var layer in layers.Reverse<Layer>())
            {
                // If we're in the output layer, set the error delta 
                // to the input hypothesis error term (typically hypothesis - actual)
                if (forwardLayer == null)
                {
                    // foreach (var node in layer.nodes)
                    for (int i = 0; i < layer.nodes.Length; i++)
                    {
                        Node node = layer.nodes[i];

                        // Convert the classification to either 1 (output node is for the same class)
                        // or 0 (output node is for a different class). If there's only one output node,
                        // our actual is 1 iff the sample's classification matches positiveClass (0 otherwise)

                        // We also use values [0.1, 0.9] to try to quicken optimization 
                        // (LeCun, et al. 1998 "Efficient BackProp")
                        float actual = 0.1f; // -0.9f;
                        if (layer.nodes.Length < 2)
                        {
                            if (sample.classification == positiveClass)
                            {
                                actual = 0.9f;
                            }
                        }
                        else if (sample.classification == classifications[i])
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
                    foreach (var node in layer.nodes)
                    {
                        float sum = 0;
                        foreach (var forwardNode in forwardLayer.nodes)
                        {
                            sum += forwardNode.delta * activator.FPrime(node.input) * forwardNode.weights[node];
                        }

                        node.delta = sum;
                    }
                }

                forwardLayer = layer;
            }

            // Update weights and biases based on backprop deltas
            Layer prevLayer = inputLayer;
            foreach (var layer in layers)
            {
                foreach (var node in layer.nodes)
                {
                    // Update weight between previous nodes and this node
                    foreach (var prevNode in prevLayer.nodes)
                    {
                        weightDelta = trainingRate * node.delta * prevNode.output;

                        // Factor in momentum to the weight to reduce oscillation
                        weightDelta += momentum * node.previousWeightDelta[prevNode];
                        node.previousWeightDelta[prevNode] = weightDelta;

                        node.weights[prevNode] -= weightDelta;
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
        
        /* public void FastNN(Sample[] samples)
         {
             Matrix<double> w1 = Matrix<double>.Build.Random(3, 3);
             Matrix<double> w2 = Matrix<double>.Build.Random(3, 2);

             // 10 iter
             // 3 features
             // 2 classes
             for (int iter = 0; iter < 10; iter++)
             {
                 Console.WriteLine("Iteration " + iter);

                 Utility.Shuffle(samples);

                 // step, subset of 'minibatch' for bigger data
                 grad = MinibatchGradient();

                 // minibatch_grad
                 List<Matrix<double>> xs = new List<Matrix<double>>();
                 List<Matrix<double>> hs = new List<Matrix<double>>();
                 List<Matrix<double>> errors = new List<Matrix<double>>();

                 for (int i = 0; i < samples.Length; i++) {
                     // Type copy
                     var attr = new double[samples[i].attr.Length];
                     for (int xx = 0; xx < attr.Length; xx++)
                     {
                         attr[xx] = samples[i].attr[xx];
                     }

                     var x = Vector<double>.Build.DenseOfArray(attr).ToColumnMatrix();

                     // Forward prop
                     var h = x * w1;

                     // ReLU f(x) = max(0, x)
                     h.Map(u => Math.Max(0, u));

                     // Hidden layer to output
                     var class_pred = Softmax(h * w2);

                     // [0 0] matrix of 2 class problem
                     // var y_true = Vector<double>.Build.Dense(2, 0).ToColumnMatrix();
                     // There's some probability distribution stuff, basically it looks to
                     // set anything with not the same class as 0, 1 o/w. We'll just use
                     // the raw classes

                     var classes = new double[2] { 0, 1 }; // TODO: Clearly not correct
                     var class_true = Vector<double>.Build.DenseOfArray(classes).ToColumnMatrix();

                     var error = class_true - class_pred;

                     xs.Add(x);
                     hs.Add(h);
                     errors.Add(error);
                 }

                 // Backward prop
                 var dw2 = hs.
             }

             // shuffle shite
             // ... oh I wasn't shuffling... hmm.
             Utility.Shuffle(samples);

             // Backward prop

         }*/

    }

    class Program
    {
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

        static void WriteIterationLog(float[] error, int iterations)
        {
            using (FileStream fs = File.Create(@"iterations-" + iterations + ".csv"))
            {
                using (StreamWriter writer = new StreamWriter(fs))
                {
                    writer.WriteLine("Iteration,MSE");
                    for (int i = 0; i < iterations; i++)
                    {
                        writer.WriteLine(i + "," + error[i]);
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
        
        static void Main(string[] args)
        {
            int runs = 5;

            float[][] error = new float[runs][];

            NeuralNetwork network = null;
            for (int i = 0; i < runs; i++)
            {
                // Need to reinitialize network & samples, otherwise
                // we have old network settings
                network = new NeuralNetwork(597, 20, 1)
                {
                    trainingRate = 0.5f,
                    momentum = 0, // 0.2f,
                    epoch = 1000,
                    errorThreshold = 0.001f,
                    activator = new TanHActivation(),
                    minibatchSize = 3,
                    positiveClass = "Baseline"
                };

                // var samples = network.GetEasierTrainingSamples();
                // var samples = network.GetTrainingSamples();
                // var samples = LoadSamplesFromCSV("iris-two-class-normalized.csv");
                // var samples = LoadSamplesFromCSV("iris-normalized-mean-zero.csv");
                var samples = LoadSamplesFromCSV("samples-various-0.01s.csv");

                error[i] = network.Train(
                    samples, 
                    network.StochasticGradientDescent
                );

                network.Test(samples);
                // network.TestMulticlass(samples);
            }

            WriteIterationGroup(network, error);

            Console.ReadLine();
        }
    }
}
