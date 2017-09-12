class Agent:

    def __init__(self):
        self.state = []
        self.reward = 0

    def get_action(self):
        #all RL should go in this section
        action = [1,2,3,4,5]
        return action

    def set_observation(self, state, reward):
        self.state = state
        self.state = reward
        print("new state: ", state.__dict__)
        print("new reward: ", reward)


