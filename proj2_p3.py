# Project 2 - Part III
# Tolu Adeleye (tadel002)
# This program implements both Forward Selection and Backward Elimination
#
# Small dataset:
#   Forward:  best subset = [5,3], accuracy = .920
#   Backward: best subset = [2, 4, 5, 7, 10], accuracy = ...
#
# Large dataset:
#   Forward:  best subset = [27, 1], accuracy = 0.955
#   Backward: best subset = [27], accuracy = 0.847
#
# Titanic dataset:
#   Forward:  best subset = [2], accuracy = .780
#   Backward: best subset = [2], accuracy = .780

import math

def load_dataset(path):
    # reading the dataset - label is first, then all the features
    data = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                # labels are in scientific notation for some reason (0.0e+00 or 1.0e+00)
                label = int(float(parts[0]))
            except ValueError:
                # probably a header row or something messed up, just skip it
                continue
            feats = [float(x) for x in parts[1:]]
            data.append((label, feats))
    return data


def normalize_dataset(data):
    # doing z-score normalization so everything's on the same scale
    if not data:
        return data

    num_features = len(data[0][1])
    total_instances = float(len(data))

    # first pass: get the means
    means = [0.0] * num_features
    for label, feats in data:
        for idx, val in enumerate(feats):
            means[idx] += val
    for idx in range(num_features):
        means[idx] /= total_instances

    # second pass: calculate standard deviations
    variances = [0.0] * num_features
    for label, feats in data:
        for idx, val in enumerate(feats):
            diff = val - means[idx]
            variances[idx] += diff * diff
    
    stds = [0.0] * num_features
    for idx in range(num_features):
        if variances[idx] == 0.0:
            stds[idx] = 1.0  # avoid division by zero
        else:
            stds[idx] = math.sqrt(variances[idx] / total_instances)

    # now normalize everything
    normalized = []
    for label, feats in data:
        normalized_feats = []
        for idx, val in enumerate(feats):
            normalized_feats.append((val - means[idx]) / stds[idx])
        normalized.append((label, normalized_feats))
    return normalized

def euclidean_distance(vec_a, vec_b, which_features):
    sum_squared = 0.0
    for idx in which_features:
        diff = vec_a[idx] - vec_b[idx]
        sum_squared += diff * diff
    return math.sqrt(sum_squared)

def nn_predict(training_data, test_features, which_features):
    # simple 1-nearest neighbor - find closest training point
    closest_distance = None
    closest_label = None
    for label, feats in training_data:
        dist = euclidean_distance(feats, test_features, which_features)
        if closest_distance is None or dist < closest_distance:
            closest_distance = dist
            closest_label = label
    return closest_label

def leave_one_out_accuracy(data, feature_subset):
    # using leave-one-out cross validation
    # feature_subset has 1-based indices like [1, 3, 5]
    total = len(data)
    if total == 0:
        return 0.0

    # edge case: if no features selected, just use majority class
    if not feature_subset:
        label_counts = {}
        for label, feats in data:
            label_counts[label] = label_counts.get(label, 0) + 1
        majority = max(label_counts, key=label_counts.get)
        return label_counts[majority] / float(total)

    # convert to 0-based indexing for actual array access
    feature_indices = [f - 1 for f in feature_subset]

    num_correct = 0
    for i in range(total):
        test_label, test_feats = data[i]
        # train on everything except instance i
        train_data = data[:i] + data[i + 1:]
        prediction = nn_predict(train_data, test_feats, feature_indices)
        if prediction == test_label:
            num_correct += 1
    return num_correct / float(total)

def forward_selection(data, total_features, verbose=True):
    selected_features = []
    best_features_so_far = []
    best_accuracy_so_far = leave_one_out_accuracy(data, selected_features)

    if verbose:
        print('Running nearest neighbor with no features (default rate), '
              'using "leave-one-out" evaluation, I get an accuracy of {:.3f}'.format(best_accuracy_so_far))
        print("Beginning search.")

    # gradually add features one at a time
    for level in range(1, total_features + 1):
        best_new_feature = None
        best_accuracy_this_level = -1.0

        # try adding each unused feature
        for feature_num in range(1, total_features + 1):
            if feature_num in selected_features:
                continue
            candidate = selected_features + [feature_num]
            accuracy = leave_one_out_accuracy(data, candidate)
            if verbose:
                print("Using feature(s) {} accuracy is {:.3f}".format(candidate, accuracy))
            if accuracy > best_accuracy_this_level:
                best_accuracy_this_level = accuracy
                best_new_feature = feature_num

        selected_features.append(best_new_feature)
        if verbose:
            print("Feature set {} was best, accuracy is {:.3f}".format(selected_features, best_accuracy_this_level))

        if best_accuracy_this_level < best_accuracy_so_far:
            if verbose:
                print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        else:
            best_accuracy_so_far = best_accuracy_this_level
            best_features_so_far = list(selected_features)

    if verbose:
        print("Finished search!! The best feature subset is {}, "
              "which has an accuracy of {:.3f}".format(best_features_so_far, best_accuracy_so_far))

    return best_features_so_far, best_accuracy_so_far

def backward_elimination(data, total_features, verbose=True):
    # start with all features and remove one at a time
    selected_features = list(range(1, total_features + 1))
    best_features_so_far = list(selected_features)
    best_accuracy_so_far = leave_one_out_accuracy(data, selected_features)

    if verbose:
        print('Running nearest neighbor with all {} features, using "leave-one-out" evaluation, '
              'I get an accuracy of {:.3f}'.format(total_features, best_accuracy_so_far))
        print("Beginning search.")

    for level in range(total_features, 0, -1):
        worst_feature = None
        best_accuracy_this_level = -1.0

        # try removing each feature
        for feature_num in selected_features:
            candidate = [x for x in selected_features if x != feature_num]
            accuracy = leave_one_out_accuracy(data, candidate)
            if verbose:
                print("Using feature(s) {} accuracy is {:.3f}".format(candidate, accuracy))
            if accuracy > best_accuracy_this_level:
                best_accuracy_this_level = accuracy
                worst_feature = feature_num

        selected_features.remove(worst_feature)
        if verbose:
            print("Feature set {} was best, accuracy is {:.3f}".format(selected_features, best_accuracy_this_level))

        if best_accuracy_this_level < best_accuracy_so_far:
            if verbose:
                print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        else:
            best_accuracy_so_far = best_accuracy_this_level
            best_features_so_far = list(selected_features)

    if verbose:
        print("Finished search!! The best feature subset is {}, "
              "which has an accuracy of {:.3f}".format(best_features_so_far, best_accuracy_so_far))

    return best_features_so_far, best_accuracy_so_far

def main():
    print("Welcome to Tolu Adeleye's Feature Selection Algorithm.")
    filename = input("Type in the name of the file to test: ").strip()

    data = load_dataset(filename)
    if not data:
        print("Could not load any data from file:", filename)
        return

    data = normalize_dataset(data)
    total_features = len(data[0][1])
    print("This dataset has {} instances and {} features.".format(len(data), total_features))

    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    print("3) Tolu's Special Algorithm")
    choice = input().strip()

    if choice == "1":
        forward_selection(data, total_features, verbose=True)
    elif choice == "2":
        backward_elimination(data, total_features, verbose=True)
    else:
        print("Special algorithm is not implemented for this project. (I focused on getting Forward and Backward working well.)")

if __name__ == "__main__":
    main()
