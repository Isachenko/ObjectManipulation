import vrep
import time

class VRepExperiment():
    def __init__(self, env, agent, epochs_number=3, max_t=500):
        self.env = env
        self.agent = agent
        self.epochs_number = epochs_number
        self.max_t = max_t

    def run(self):
        self.env.connect_to_vrep()
        for epoch in range(self.epochs_number):
            self.env.reset()
            self.env.start() #initialize firs state
            self.agent.set_observation(self.env.get_state(), self.env.get_reward())

            for t in range(0, self.max_t):
                action = self.agent.get_action()
                self.env.make_action(action)
                self.agent.set_observation(self.env.get_state(), self.env.get_reward())
                #time.sleep(0.1)

        self.env.disconnect_from_vrep()


