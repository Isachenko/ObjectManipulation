import v_rep_environment
import agent
import v_rep_experiment
import imageio

if __name__ == "__main__":
    #imageio.plugins.ffmpeg.download()
    env = v_rep_environment.VRepEnvironment()
    agent = agent.Agent()
    experiment1 = v_rep_experiment.VRepExperiment(env, agent)
    experiment1.run()

