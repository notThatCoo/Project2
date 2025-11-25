Tolu Adeleye, tadel002

Project 2 – Part I
CS 170: Introduction to AI
11/24/2025

import random

#evaluation fucntion
def evaluate(features):
    return random.uniform(0, 100)


def forward_selection(num_features):
    print()
    # initial state is the empty feature set
    open_list = [[]]

    best_state = []
    best_score = evaluate(best_state)
    print(f'Using no features and "random" evaluation, I get an accuracy of {best_score:.1f}%')
    print("Beginning search.")

    # this loop adds one feature per level
    for level in range(1, num_features + 1):
        next_states = []
        best_state_this_level = None
        best_score_this_level = -1.0

        # this loop generates neighbors by adding one unused feature
        for state in open_list:
            for f in range(1, num_features + 1):
                if f not in state:
                    new_state = state + [f]
                    acc = evaluate(new_state)
                    print(f"Using feature(s) {sorted(new_state)} accuracy is {acc:.1f}%")

                    next_states.append(new_state)

                    if acc > best_score_this_level:
                        best_score_this_level = acc
                        best_state_this_level = new_state

        # this saves only the best state for the next level
        open_list = [best_state_this_level]
        print(f"Feature set {sorted(best_state_this_level)} was best, accuracy is {best_score_this_level:.1f}%")

        # this conditional checks if accuracy improved
        if best_score_this_level < best_score:
            print("(Warning, Accuracy has decreased!)")
        else:
            best_score = best_score_this_level
            best_state = sorted(best_state_this_level)

    print(f"Finished search!! The best feature subset is {best_state}, "
          f"which has an accuracy of {best_score:.1f}%")


def backward_elimination(num_features):
    print()
    current = list(range(1, num_features + 1))
    base = evaluate(current)
    print(f"Using all features {current} accuracy is {base:.1f}%")
    print("Beginning search.")

    best_overall_acc = base
    best_overall_set = current.copy()

    # each level, remove one feature
    for level in range(num_features, 0, -1):
        best_remove = None
        best_acc = -1.0

        for f in current:
            trial = [x for x in current if x != f]
            acc = evaluate(trial)
            print(f"Using feature(s) {sorted(trial)} accuracy is {acc:.1f}%")

            if acc > best_acc:
                best_acc = acc
                best_remove = f

        current.remove(best_remove)
        print(f"Feature set {sorted(current)} was best, accuracy is {best_acc:.1f}%")

        if best_acc < best_overall_acc:
            print("(Warning, Accuracy has decreased!)")
        else:
            best_overall_acc = best_acc
            best_overall_set = sorted(current)

    print(f"Finished search!! The best feature subset is {best_overall_set}, "
          f"which has an accuracy of {best_overall_acc:.1f}%")


def main():
    print("Welcome to Tolu Adeleye's Feature Selection Algorithm.")
    while True:
        try:
            n = int(input("Please enter total number of features: ").strip())
            if n > 0:
                break
        except:
            pass
        print("Please enter a valid positive integer.")

    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    print("3) Tolu's Special Algorithm")
    choice = input().strip()

    if choice == "1":
        forward_selection(n)
    elif choice == "2":
        backward_elimination(n)
    elif choice == "3":
        print("You chose option 3, but it is not implemented yet.")
    else:
        print("\nUnrecognized choice. Exiting.")


if __name__ == "__main__":
    main()
