import sys
import collections

if __name__ == "__main__":
    if 0 == int(sys.argv[1]):
        print("Discrete A3C experiment")
        import A3C_experiment
    else:
        print("Continuous A3C experiment")
        import A3C_experiment_continuous
