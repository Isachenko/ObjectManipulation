import v_rep_environment
import agent
import v_rep_experiment

import tensorflow as tf

if __name__ == "__main__":
    print(tf.__version__)
    env = v_rep_environment.VRepEnvironment()
    agent = agent.Agent()
    experiment1 = v_rep_experiment.VRepExperiment(env, agent)
    experiment1.run()

