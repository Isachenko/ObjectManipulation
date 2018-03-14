# ObjectManipulation
Small object manipulation project with uArm and ML.

To run this experiment the requeriments are:


To run it you have to open with pyhton main.py and add the next 4 arguments:

* $1: Discrete or continuous
    - 0: Discrete experiment
    - 1: Continuous experiment

* $2: Type of experiment
    - 0: Speed, agent gets reward for cube speed.
    - 1: Distance, agent gets reward for minimazing distance between 2 cubes.
    - 2: Left, agent gets reward for moving arm to the left.
    - 3: Big, the same as 0, but cube is bigger.
    - 4: Random, the same as 0(speed), but with random placement of cube.

* $3: Number of workers in the experiment

* $4: Value coefficient in the loss expression. # Just works in discrete so far

To run in peregrine run.
    - sbatch run_A3C.sh [$arguments]