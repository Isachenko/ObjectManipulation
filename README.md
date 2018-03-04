# ObjectManipulation
Small object manipulation project with uArm and ML.

To run this experiment the requeriments are:
-
-
-
-


To run it you have to open with pyhton main.py and add the next 4 arguments

-$1: Discrete or continuous
    - 0: Discrete experiment
    - 1: Continuous experiment

-$2: Type of experiment
    - 1: 'distance'
        Small cube and the reward is proportional to the speed of the cube
    - 2: 'left'
        The reward is greater if the arm go left
    - 3: 'big'
        Big cube and the reward is proportional to the speed of the cube
    - 4: 'random'
        There are two cubes, one of them is place in a random place and reward is inverse proportional to the distance.

-$3: Number of workers in the experiment

-$4: Value coefficient in the loss expression. # Just works in discrete so far

To run in peregrine run.
    - sbatch run_A3C.sh [$arguments]