using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CsvHelper;

namespace MachineLearning {
    public class DecisionTree {
        public class Node {
            #region vars / constructor
            public Dictionary<string, Node> pointers { get; set; }
            public List<Dictionary<string, string>> dataset { get; set; }
            public string clss { get; set; }
            public string confidence { get; set; }
            public string attribute { get; set; }
            public Node() {
                pointers = new Dictionary<string, Node>();
                clss = "";
                confidence = "";
            }
            public Node(string _clss, string conf) {
                pointers = new Dictionary<string, Node>();
                clss = _clss;
                confidence = conf;
            }
            #endregion
            #region toString()
            public override string ToString() {
                return $"class={clss}, confidence={confidence}, attr={attribute}, data count={dataset.Count}, pointers count ={pointers.Count} ";
            }
            #endregion
        }

        #region vars /constructor
        static int eCount = 0;
        int logLvl = 0;
        public int minRecordsToSplitNode;
        public double accuracyWanted = 0.950;
        static public double totalEntropy = 0;
        public static bool firstCall = true;
        public static List<string> classList = new List<string>();
        Node root;
        public List<Dictionary<string, string>> regressionRecords = new List<Dictionary<string, string>>();
        public List<Dictionary<string, string>> desiredRecords = new List<Dictionary<string, string>>();
        public DecisionTree(int minRecordsToSplitNode = 10, double accuracyWanted = 0.950) {
            root = new Node();
            this.minRecordsToSplitNode = minRecordsToSplitNode;
            this.accuracyWanted = accuracyWanted;
        }
        public DecisionTree(List<Dictionary<string, string>> data) { buildDecisionTree(data); }
        #endregion
        // Build Tree
        #region recursiveEvaluate
        public void recursiveEvaluate(Node dta) {
            evaluateNode(dta);
            if (dta.pointers.Count == 0) return;
            foreach (var k in dta.pointers.Keys)
                recursiveEvaluate(dta.pointers[k]);
        }
        #endregion
        #region buildDecisionTree
        public void buildDecisionTree(List<Dictionary<string, string>> data) {
            root = new Node();
            root.dataset = data;
            Console.WriteLine("Starting to train...");
            Console.WriteLine($"{root.dataset.Count} training records");
            var before = DateTime.Now;
            recursiveEvaluate(root);
            var after = DateTime.Now;
            Console.WriteLine($"Finished training. {eCount} nodes created in {Math.Round((after - before).TotalSeconds, 2)} s.");
        }
        #endregion
        #region evaluateNode
        private int evaluateNode(Node node) {
            eCount++;
            if (node.dataset.Count <= 1) return 0; // No data
            if (node.dataset[0].Keys.Count <= 1) return 0; // No attributes to select
            int atrributesLeft = 0;
            var highestGainKey = "";
            var highestGainValue = double.MinValue;
            // Find Attribute with highest gain
            foreach (var k in node.dataset[0].Keys) {
                var gn = gain(k, node.dataset);
                if (k != "class" && gn > highestGainValue) {
                    highestGainValue = gn;
                    highestGainKey = k;
                }
            }
            // Find All possible values of Attribute Selected
            var possibleValues = new Dictionary<string, List<Dictionary<string, string>>>();
            foreach (var record in node.dataset) {
                if (record.ContainsKey(highestGainKey))
                    if (!possibleValues.ContainsKey(record[highestGainKey]))
                        possibleValues.Add(record[highestGainKey], new List<Dictionary<string, string>> { record });
                    else possibleValues[record[highestGainKey]].Add(record);
            }
            // Set current node Prediction
            var d = likelyClass(node.dataset);
            node.clss = d.Split(":")[0];
            node.confidence = (double.Parse(d.Split(":")[1]) / (double)node.dataset.Count).ToString();
            node.attribute = highestGainKey;

            // Dont Overfit / make so much sub branches
            if (node.dataset.Count < minRecordsToSplitNode) return 0;

            // Create Children Nodes from all possible values
            foreach (var k in possibleValues.Keys) {
                var newNode = new Node();
                newNode.dataset = possibleValues[k];
                foreach (var record in newNode.dataset)
                    record.Remove(highestGainKey);
                var prediction = likelyClass(newNode.dataset);
                newNode.clss = prediction.Split(":")[0];
                newNode.confidence = (double.Parse(prediction.Split(":")[1]) / (double)newNode.dataset.Count).ToString();
                atrributesLeft = newNode.dataset[0].Keys.Count;
                if (double.Parse(newNode.confidence) > double.Parse(node.confidence) || newNode.dataset.Count > minRecordsToSplitNode)
                    node.pointers.Add($"{highestGainKey}={k}", newNode);
            }
            if (logLvl > 2) Console.WriteLine($"Splitting Attribute = {highestGainKey}");
            return atrributesLeft;
        }
        #endregion
        #region likelyClass
        private string likelyClass(List<Dictionary<string, string>> dataset) {
            var classCount = new Dictionary<string, int>();
            foreach (var record in dataset) {
                if (classCount.ContainsKey(record["class"])) classCount[record["class"]]++;
                else classCount.Add(record["class"], 1);
            }
            var highestClassKey = "";
            var highestClassCount = 0;
            foreach (var k in classCount.Keys) {
                if (classCount[k] > highestClassCount) {
                    highestClassKey = k;
                    highestClassCount = classCount[k];
                }
            }
            return $"{highestClassKey}:{highestClassCount}";
        }
        #endregion

        // Calculations for Entropy & InfoGain 
        #region entropy
        public static double entropy(List<Dictionary<string, string>> data) {
            Dictionary<string, int> classes = new Dictionary<string, int>();
            foreach (var record in data) {
                if (classes.ContainsKey(record["class"]))
                    classes[record["class"]]++;
                else classes[record["class"]] = 1;
            }
            double entropy = 0;
            foreach (var k in classes.Keys) {
                if (firstCall) classList.Add(k);
                double p = classes[k] / (double)data.Count;
                entropy += (-1) * p * Math.Log2(p);
            }
            if (firstCall) { Console.WriteLine("Possible Classes"); foreach (var c in classList) Console.WriteLine(c); firstCall = false; }
            totalEntropy = entropy;
            return entropy;
        }
        #endregion
        #region countOfClassPer
        public static int countOfClassPer(string key, string val, string clss, List<Dictionary<string, string>> data) {
            int count = 0;
            foreach (var record in data)
                if (record.ContainsKey(key) && record[key] == val && record["class"] == clss)
                    count++;
            return count;
        }
        #endregion
        #region entropyForAttr
        public static double entropyForAttr(string attr, List<Dictionary<string, string>> data) {
            Dictionary<string, int> values = new Dictionary<string, int>();
            foreach (var record in data) {
                if (record.ContainsKey(attr))
                    if (values.ContainsKey(record[attr]))
                        values[record[attr]]++;
                    else values[record[attr]] = 1;
            }
            double entropy = 0;
            foreach (var k in values.Keys) {
                double weight = values[k] / (double)data.Count;
                foreach (var c in classList) {
                    if (values[k] != 0) {
                        double p = countOfClassPer(attr, k, c, data) / (double)values[k];
                        if (p != 0) entropy += (-1) * weight * p * Math.Log2(p);
                    }
                }
            }
            return entropy;
        }
        #endregion
        #region gain
        public static double gain(string attr, List<Dictionary<string, string>> data) {
            return entropy(data) - entropyForAttr(attr, data);
        }
        #endregion

        // Predict & Test
        #region makePrediction
        public bool makePrediction(Dictionary<string, string> record, string answer = null, int index = 0) {
            var node = root;
            var nodeConfidence = double.Parse(node.confidence);
            while (nodeConfidence < accuracyWanted && node.pointers.Count != 0) {
                var key = $"{node.attribute}={record[node.attribute]}";
                if (node.pointers.ContainsKey(key)) {
                    node = node.pointers[key];
                    nodeConfidence = double.Parse(node.confidence);
                }
                else break;
            }
            if (logLvl >= 4) Console.WriteLine($"#{index} Predicted class : {node.clss}, Confidence : {node.confidence}");
            if (answer != null && node.clss == answer) return true;
            else if (logLvl >= 2) Console.WriteLine($"Incorrect prediction for #{index}. Correct Answer = {answer ?? ""}");
            return false;
        }
        #endregion 
        #region batchTest
        public void regressionTest() {
            batchTest(regressionRecords, "Regression Test");
        }
        public void progressTest() {
            batchTest(desiredRecords, "Desired Test");
        }
        public void batchTest(List<Dictionary<string, string>> testData, string testName = "Untitled") {
            if (logLvl >= 4) Console.WriteLine($"Starting test '{testName}'.");
            int correct = 0;
            int index = 0;
            int failed = 0;
            foreach (var record in testData) {
                try {
                    index++;
                    if (logLvl > 4) {
                        string output = "";
                        foreach (var k in record.Keys)
                            output += $"{k}:{record[k]}, ";
                        Console.WriteLine($"Test Record =  {output}");
                    }
                    if (makePrediction(record, record["class"], index)) {
                        if (testName != "Regression Test") regressionRecords.Add(record);
                        correct++;
                    }
                    else if (testName != "Progress Test") desiredRecords.Add(record);
                }
                catch { failed++; }
            }
            double acc = Math.Round(correct / (double)(testData.Count - failed), 2) * 100;
            Console.WriteLine($"{acc}% Accuracy for test '{testName}'. ({correct}/{testData.Count})");
        }
        #endregion

        // Print Tree Data
        #region printTop
        public string printTop() {
            string output = "";
            string rootNode = $"\n**\nroot node : \n {root.ToString()} \n**";
            output += rootNode;
            Console.WriteLine(rootNode);
            foreach (var k in root.pointers.Keys) {
                string childNode = $"\n**\n{k} node : \n {root.pointers[k].ToString()} \n**";
                Console.WriteLine(childNode);
                output += childNode;
            }
            return output;
        }
        #endregion
        #region printNodeRecursive
        void printNodeRecursive(Node nde) {
            Console.WriteLine(nde.ToString());
            if (nde.pointers.Count == 0) return;
            foreach (var k in nde.pointers.Keys)
                printNodeRecursive(nde.pointers[k]);
        }
        #endregion
        #region printTree
        public void printTree() {
            printNodeRecursive(root);
        }
        #endregion
    }
    static class DataUtils {
        // converts csv file to List<Dictionary<string,string>>
        #region convertCSV
        public static List<Dictionary<string, string>> convertCSV(string filePath, string columnToPredict, List<string> removeTheseColumns = null) {

            var records = new List<Dictionary<string, string>>();
            using (var reader = new StreamReader(filePath))
            using (var csv = new CsvReader(reader, System.Globalization.CultureInfo.InvariantCulture)) {
                var r = csv.GetRecords<dynamic>();
                foreach (var l in r) {
                    var dict = new Dictionary<string, string>();
                    foreach (var property in (IDictionary<String, Object>)l)
                        dict.Add(property.Key == columnToPredict ? "class" : property.Key, (string)property.Value ?? "");
                    if (removeTheseColumns != null) {
                        foreach (var k in removeTheseColumns)
                            if (dict.ContainsKey(k)) dict.Remove(k);
                    }
                    if (!string.IsNullOrEmpty(dict["class"]))
                        records.Add(dict);
                }
            }
            return records;
        }
        #endregion
    }
    static class Program {
        static void Main(string[] args) {
            // Example Usage:
	    // var tree = new DecisionTree(minRecordsToSplitNode: 6, accuracyWanted: 0.92);
            // tree.buildDecisionTree(trainingData);
            // tree.batchTest(testData, "Unseen Samples");
            // tree.regressionTest();
            // tree.progressTest();
            // tree.printTop();
        }
    }
}
