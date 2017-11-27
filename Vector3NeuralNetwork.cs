using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN
{
    /// <summary>
    /// Placeholder Vec3 implementation - to be swapped with Unity's implementation 
    /// </summary>
    class Vector3
    {
        public float x, y, z;

        public static Vector3 zero
        {
            get
            {
                return new Vector3(0, 0, 0);
            }
        }

        public static Vector3 one
        {
            get
            {
                return new Vector3(1f, 1f, 1f);
            }
        }

        public float magnitude
        {
            get
            {
                // Or sqrt(dot(this, this)), whichever
                return (float)Math.Sqrt(x * x + y * y + z * z);
            }
        }

        /// <summary>
        /// Faster distance calculation for something like:
        /// offset = other.position - transform.position
        /// sqrLen = offset.sqrMagnitude
        /// if sqrLen lt closeDistance * closeDistance
        ///     do a thing
        /// </summary>
        public float sqrMagnitude
        {
            get
            {
                // Or dot product of self, whatev
                return x * x + y * y + z * z;
            }
        }

        public Vector3 normalized
        {
            get
            {
                var mag = magnitude;
                if (mag > 0)
                {
                    return this / mag;
                }

                // Zero vec, return self
                return new Vector3(x, y, z);
            }
        }
        
        public Vector3(float x, float y, float z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public static Vector3 operator +(Vector3 a, Vector3 b)
        {
            return new Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
        }

        public static Vector3 operator -(Vector3 a, Vector3 b)
        {
            return new Vector3(a.x - b.x, a.y - b.y, a.z - b.z);
        }

        public static Vector3 operator *(Vector3 a, float d)
        {
            return new Vector3(a.x * d, a.y * d, a.z * d);
        }

        public static Vector3 operator /(Vector3 a, float d)
        {
            return new Vector3(a.x / d, a.y / d, a.z / d);
        }

        public static float Dot(Vector3 lhs, Vector3 rhs)
        {
            return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z + rhs.z;
        } 
        
        public static float Distance(Vector3 a, Vector3 b)
        {
            return (a - b).magnitude;
        }
    }

    /// <summary>
    /// I'm not creative in naming this.
    /// 
    /// This is a variation of the classic Neural Network that is tailored specifically
    /// for operating on Vector3 position data and classifying Vec3 position lists.
    /// 
    /// In short:
    ///     * Everything is treated internally as a Vec3 direction
    ///         That means that when we are adjusting nodes during Backprop, 
    ///         the node can be moved anywhere in full 3 dimensions
    ///     * During FeedForward, input vec3 points are weighted in importance,
    ///         along with some bias vec3 for each node.
    ///         The output of FeedForward for a positive match ends up being > 1
    ///         (this was discovered through trial and error)
    ///     * During BackProp, the error is the square magnitude of the distance
    ///         between the (single) output node and either Vec3.one for training
    ///         a positive or Vec3.zero for training against a negative. 
    /// </summary>
    class Vector3NeuralNetwork
    {
        /// <summary>
        /// Node that represents a point in 3D space
        /// </summary>
        public class Node
        {
            /// <summary>
            /// Weights between *previous* nodes and this node
            /// </summary>
            public Vector3[] weights;

            /// <summary>
            /// Tracking of weight deltas between previous nodes,
            /// used for factoring in momentum
            /// </summary>
            public Vector3[] previousWeightDelta;

            /// <summary>
            /// Weighted sum of previous node outputs + bias
            /// </summary>
            public Vector3 input;

            /// <summary>
            /// Activation function output of this node
            /// </summary>
            public Vector3 output;

            /// <summary>
            /// Bias at this node
            /// </summary>
            public Vector3 bias;

            /// <summary>
            /// Tracking of previous bias delta,
            /// used for factoring in momentum
            /// </summary>
            public Vector3 previousBiasDelta;

            /// <summary>
            /// Tracked during training. 
            /// (predicted_class - actual_class) * sigmoidDeriv(input)
            /// </summary>
            public Vector3 delta;

            /// <summary>
            /// Create a new NN node
            /// </summary>
            /// <param name="prevNodes">Number of nodes in the previous layer</param>
            public Node(int prevNodes)
            {
                bias = new Vector3(
                    Utility.GaussianRandom(),
                    Utility.GaussianRandom(),
                    Utility.GaussianRandom()
                );

                previousBiasDelta = Vector3.zero;
                output = Vector3.zero;
                delta = Vector3.zero;
                input = Vector3.zero;

                if (prevNodes > 0)
                {
                    weights = new Vector3[prevNodes];
                    previousWeightDelta = new Vector3[prevNodes];

                    for (int n = 0; n < prevNodes; n++)
                    {
                        weights[n] = new Vector3(
                            Utility.GaussianRandom(),
                            Utility.GaussianRandom(),
                            Utility.GaussianRandom()
                        );
                        previousWeightDelta[n] = Vector3.zero;
                    }
                }
            }
        }

        /// <summary>
        /// Sample containing a series of Vector3's associated with some class
        /// </summary>
        public class Sample
        {
            public Vector3[] attr;
            public string classification;
        }

        public float trainingRate;
        public float momentum;
        public int epoch;
        public float errorThreshold;

        /// <summary>
        /// If there is only one output node, samples either
        /// match the positive class label (1) or don't (0)
        /// </summary>
        public string positiveClass;
        
        private Node[] inputLayer;
        private Node[][] layers;

        public Vector3NeuralNetwork(int inputNodes, int hiddenNodes = 0)
        {
            BuildNetwork(inputNodes, hiddenNodes);
        }

        public void BuildNetwork(int inputNodes, int hiddenNodes = 0)
        {
            // Use Weka's 'a' setting if not specified
            if (hiddenNodes < 1)
            {
                hiddenNodes = (inputNodes + 1) / 2;
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
            layers[1] = new Node[1];
            layers[1][0] = new Node(hiddenNodes);
        }

        /// <summary>
        /// Logistic sigmoid function for activation. 
        /// Seems to be the most efficient (for now) for Vec3
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Vector3 Activation(Vector3 input)
        {
            return new Vector3(
                1.0f / (1.0f + (float)Math.Exp(-input.x)),
                1.0f / (1.0f + (float)Math.Exp(-input.y)),
                1.0f / (1.0f + (float)Math.Exp(-input.z))
            );
        }

        /// <summary>
        /// Derivative of the logistic sigmoid function
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Vector3 ActivationDerivative(Vector3 input)
        {
            var a = Activation(input);
            return new Vector3(
                a.x * (1.0f - a.x),
                a.y * (1.0f - a.y),
                a.z * (1.0f - a.z)
            );
        }

        /// <summary>
        /// Returns the hypothesis Vec3. This vector has a square magnitude > 1.0 
        /// for a predicted match against the positive class
        /// </summary>
        /// <param name="sample"></param>
        /// <returns>hypthesis</returns>
        public Vector3 FeedForward(Sample sample)
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
                    node.input = Vector3.zero;
                    for (int j = 0; j < previousLayer.Length; j++)
                    {
                        // Piecewise multiply weight against previous output
                        node.input.x += node.weights[j].x * previousLayer[j].output.x;
                        node.input.y += node.weights[j].y * previousLayer[j].output.y;
                        node.input.z += node.weights[j].z * previousLayer[j].output.z;
                    }

                    node.input += node.bias;

                    // Output is the activation function of the input
                    node.output = Activation(node.input);
                }

                previousLayer = layer;
            }

            // Return output of last layer's node
            return layers[layers.Length - 1][0].output;
        }

        private float BackPropagate(Sample sample)
        {
            Vector3 weightDelta = Vector3.zero;
            Vector3 biasDelta = Vector3.zero;
            float error = 0;
            
            Node[] forwardLayer = null;
            for (int i = layers.Length - 1; i >= 0; i--)
            {
                var layer = layers[i];
                
                if (forwardLayer == null) // Output layer
                {
                    for (int j = 0; j < layer.Length; j++)
                    {
                        Node node = layer[j];

                        // We want the final node to target vec3.one for a positive result.
                        // This is basically a 3-dimensional convergence to 3 positive results
                        // at once through the same network pass.
                        var actual = sample.classification == positiveClass 
                            ? Vector3.one 
                            : Vector3.zero;
                        
                        // Reported overall error will be (1/2)*sum((actual - hypothesis)^2)
                        error += (actual - node.output).sqrMagnitude;

                        // Set the delta to the derivative of the error (hypothesis - actual)
                        //  * derivative of the aggregate inputs
                        var deriv = ActivationDerivative(node.input);
                        var offset = node.output - actual;
                        node.delta.x = deriv.x * offset.x;
                        node.delta.y = deriv.y * offset.y;
                        node.delta.z = deriv.z * offset.z;
                    }
                }
                else
                {
                    // If we're in a hidden layer, aggregate the edge weight * delta of 
                    // every node connected to this node from the forward layer
                    for (int j = 0; j < layer.Length; j++)
                    {
                        var node = layer[j];
                        var sum = Vector3.zero;
                        var deriv = ActivationDerivative(node.input);
                        
                        foreach (var forwardNode in forwardLayer)
                        {
                            sum.x += forwardNode.delta.x * forwardNode.weights[j].x * deriv.x;
                            sum.y += forwardNode.delta.y * forwardNode.weights[j].y * deriv.y;
                            sum.z += forwardNode.delta.z * forwardNode.weights[j].z * deriv.z;
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
                        weightDelta.x = trainingRate * node.delta.x * prevLayer[j].output.x;
                        weightDelta.y = trainingRate * node.delta.y * prevLayer[j].output.y;
                        weightDelta.z = trainingRate * node.delta.z * prevLayer[j].output.z;

                        // Factor in momentum to the weight to reduce oscillation
                        weightDelta += node.previousWeightDelta[j] * momentum;
                        node.previousWeightDelta[j] = weightDelta;

                        node.weights[j] -= weightDelta;
                    }

                    // Update bias for this node 
                    biasDelta = node.delta * trainingRate;

                    // Factor in momentum to the bias
                    biasDelta += node.previousBiasDelta * momentum;
                    node.previousBiasDelta = biasDelta;

                    node.bias -= biasDelta;
                }

                prevLayer = layer;
            }
            
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

        public float[] Train(Sample[] samples)
        {
            // Error recorded for each iteration
            List<float> errorList = new List<float>();
            
            // Iterate over the epoch, running the desired optimization algorithm for each iteration
            float error = 1.0f;
            int iteration;
            for (iteration = 0; iteration < epoch && error > errorThreshold; iteration++)
            {
                error = StochasticGradientDescent(samples);
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

        public void Test(Sample[] samples)
        {
            int n = 2;
            float accuracy = 0;
            var confusion = new float[n, n];

            var positiveThreshold = 1.0f;

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
                // confusion matrix. In reality, hypothesis.sqrMagnitude should be > positiveThreshold^2 for a positive match
                var predicted = Convert.ToInt32(hypothesis.sqrMagnitude < positiveThreshold * positiveThreshold);
                var actual = Convert.ToInt32(samples[i].classification != positiveClass);

                if (predicted == actual)
                {
                    accuracy++;
                }

                confusion[predicted, actual]++;
                certainty[i] = (Vector3.zero - hypothesis).sqrMagnitude;

                // Report certainty for this sample
                Console.WriteLine(
                     i.ToString().PadLeft(6) + "  " + 
                     certainty[i].ToString("0.0000").PadLeft(4)
                );
            }

            // Confusion matrix header of codes per class (weka-style)
            int padding = samples.Length.ToString().Length + 1;

            Console.WriteLine("T".PadLeft(padding) + "F".PadLeft(padding) + "  <-- classified as");

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

        public static Sample[] LoadSamplesFromCSV(string filename)
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
                            attr = new Vector3[(values.Length - 1) / 3]
                        };

                        // Rest are vec3 attributes
                        for (int i = 0; i < sample.attr.Length; i++)
                        {
                            sample.attr[i] = new Vector3(
                                float.Parse(values[i * 3]),
                                float.Parse(values[i * 3 + 1]),
                                float.Parse(values[i * 3 + 2])
                            );
                        }

                        samples.Add(sample);
                    }

                }
            }

            return samples.ToArray();
        }

    }
}
