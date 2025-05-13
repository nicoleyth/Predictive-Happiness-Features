"""
Author: Nicole Huang & Aiden Kim
Date: 3/26/25 (from lab 6) with some changes made 4/30/25
Description: Stores Partition and Example class. 
Implemented best feature functions by information gain and 
classification accuracy within the Partition class. 
"""

################################################################################
# IMPORTS
################################################################################

import math

################################################################################
# CLASSES
################################################################################

class Example:

    def __init__(self, features, label):
        """Helper class (like a struct) that stores info about each example."""
        # dictionary. key=feature name: value=feature value for this example
        self.features = features
        self.label = label

    def change_label(self, new_val):
        self.label = new_val

class Partition:

    def __init__(self, data, F):
        """Store information about a dataset"""
        # list of Examples
        self.data = data
        self.n = len(self.data)

        # dictionary. key=feature name: value=list of possible values
        self.F = F
    
    def calc_label_prob(self, label):
        """
        This function calculates the probability for the given label in parameter. 
        """
        count = sum(1 for example in self.data if example.label == label)
        return count / self.n
    
    def calc_entropy(self):
        """
        This function calculates the entropy of the data. 
        """
        labels = set(example.label for example in self.data)
        
        entropy = 0
        for label in labels:
            prob = self.calc_label_prob(label)
            if prob > 0:  # avoid math calculation error
                entropy -= prob * math.log2(prob)
        return entropy
    
    def group_by_feature(self, feat_name):
        """
        Sort the data by extracting data only with specific feature.
        """
        feat_data = {}

        #sort examples (ex) by feature values
        for ex in self.data:
            feat_val = ex.features[feat_name]
            if feat_val not in feat_data:
                feat_data[feat_val] = []
            feat_data[feat_val].append(ex)
        return feat_data
    
    def group_by_cont_feature(self, feat_name, threshold):
        """
        Groups the data based on a continuous feature given threshold.
        """
        below_thresh = []
        above_thresh = []
        
        #compares the data values to threshhold and add ex to corresponding list
        for ex in self.data:
            feat_val = ex.features[feat_name]
            if feat_val <= threshold:
                below_thresh.append(ex)
            else:
                above_thresh.append(ex)
        
        return below_thresh, above_thresh
    
    def find_continuous_thresholds(self, feat_name):
        """
        Calculates threshholds for continuous threshold for given feature. 
        """
        # sort by continuous feature val
        sorted_examples = sorted(self.data, key=lambda ex: ex.features[feat_name])
        
        thresholds = []
        
        # threshhold whenever more than one of the same feature val
        for i in range(1, len(sorted_examples)):
            prev_example = sorted_examples[i-1]
            curr_example = sorted_examples[i]
            
            prev_value = prev_example.features[feat_name]
            curr_value = curr_example.features[feat_name]
            
            prev_label = prev_example.label
            curr_label = curr_example.label
            
            # check if the feature value is the same as previous
            if prev_value == curr_value:
                thresholds.append(prev_value)
                sorted_examples[i].change_label = 2
                sorted_examples[i-1].change_label = 2
                

        #check whenever feature label changes
        low_range = sorted_examples[0].features[feat_name]
        high_range = 0
        for i in range(1, len(sorted_examples)):
            prev_example = sorted_examples[i-1]
            curr_example = sorted_examples[i]
            
            prev_value = prev_example.features[feat_name]
            curr_value = curr_example.features[feat_name]
            prev_label = prev_example.label
            curr_label = curr_example.label
            # check if label changes
            if prev_label != curr_label:
                high_range = curr_label
                # Create a threshold at the midpoint between the two values
                threshold = (low_range + high_range) / 2
                thresholds.append(threshold)
                low_range = high_range
        
        return thresholds

    # def find_continuous_thresholds(self, feat_name):
    #     """
    #     Calculates thresholds for a continuous feature by looking at points where the label changes.
    #     """
    #     # Sort examples by the feature value
    #     sorted_examples = sorted(self.data, key=lambda ex: ex.features[feat_name])

    #     # Early exit if there's not enough data
    #     if len(sorted_examples) < 2:
    #         return []

    #     thresholds = []

    #     # Loop through consecutive pairs to find threshold where label changes
    #     for i in range(1, len(sorted_examples)):
    #         prev_value = sorted_examples[i - 1].features[feat_name]
    #         curr_value = sorted_examples[i].features[feat_name]

    #         prev_label = sorted_examples[i - 1].label
    #         curr_label = sorted_examples[i].label

    #         # Only consider a threshold if feature values differ AND labels change
    #         if prev_value != curr_value and prev_label != curr_label:
    #             threshold = (prev_value + curr_value) / 2
    #             thresholds.append(threshold)

    #     return thresholds    
    
    def calc_feat_entropy(self, feat_name, is_cont):
        """
        Computes the individual entropy for a specific given feat_name.
        """
        if is_cont:
            #find threshold for continuous features
            thresholds = self.find_continuous_thresholds(feat_name)
            best_entropy = float('inf')

            #calculates entropy after sorting continuous features
            for threshold in thresholds:
                below, above = self.group_by_cont_feature(feat_name, threshold)
                partition_below = Partition(below, self.F)
                partition_above = Partition(above, self.F)
                entropy = ((len(below) / self.n) * partition_below.calc_entropy() +
                           (len(above) / self.n) * partition_above.calc_entropy()
                )
                #compares and stores best entropy
                if entropy < best_entropy:
                    best_entropy = entropy

        else: #if data is categorical/discrete
            feat_data = self.group_by_feature(feat_name)
            best_entropy = 0
            #calculate weighted entropy per feature
            for value, subset in feat_data.items():
                subset_size = len(subset)
                subset_partition = Partition(subset, self.F)
                weighted_entropy = (subset_size / self.n) * subset_partition.calc_entropy()
                best_entropy += weighted_entropy
        return best_entropy
    
    def calc_gain(self, feature, is_cont=False):
        """
        This function calculates the info gain for specific feature.
        """
        # calc total entropy
        tot_entropy = self.calc_entropy()
        
        # calc entropy based on the feature
        feat_entropy = self.calc_feat_entropy(feature, is_cont)
        # print(f"{feature} entropy: {feat_entropy}")
        
        # calculate information gain
        gain = tot_entropy - feat_entropy
        return gain
    
    def best_feature(self):
        """
        This function identifies the best feature, associated with the max gain.
        """
        attribute_gains = self.calc_gain_by_feature()
        
        # identifying feature with max gains
        best_feature = max(attribute_gains, key=attribute_gains.get)
        return best_feature
    
    def calc_gain_by_feature(self):
        """
        Calculate the gain value per feature. Return dictionary of gain values.
        """
        attribute_gains = {}
        
        # calculate info gain for per feature
        # print()
        # print("Info Gain: ")
        for feature in self.F:
            is_continuous = isinstance(self.data[0].features[feature], (int, float))
            gain = self.calc_gain(feature, is_continuous)
            attribute_gains[feature] = gain
        #     print("{:<15}{:<10} ".format(feature, float(f'{gain:.6f}')))
        # print()

        return attribute_gains

    
    def calc_classify_acc(self, feat_name):
        """
        This function calculates the classification accuracy for the given feature.
        """
        feat_data = self.group_by_feature(feat_name)
        corr_pred = 0
        tot_pred = 0
        
        # predict the majority label and calculate accuracy
        for value, examples in feat_data.items():
            # majority label for this feature value
            label_counts = {}
            for example in examples:
                if example.label not in label_counts:
                    label_counts[example.label] = 1
                else:
                    label_counts[example.label] += 1
            majority_label = max(label_counts, key=label_counts.get)
            
            # correct predictions calculation
            corr_pred += sum(1 for example in examples if example.label == majority_label)
            tot_pred += len(examples)
        
        accuracy = corr_pred / tot_pred
        return accuracy
    
    def best_feature_by_accuracy(self):
        """
        This function identifies the best feature based on classification accuracy.
        """
        attribute_acc = {}
        
        # calculate accuracy per feature
        print("\nClassification Accuracy: ")
        for feature in self.F:
            accuracy = self.calc_classify_acc(feature)
            attribute_acc[feature] = accuracy
            print("{:<15}{:<10}".format(feature, float(f'{accuracy:.6f}')))
        print()
        
        # identify feature with max accuracy
        best_feature = max(attribute_acc, key=attribute_acc.get)
        return best_feature