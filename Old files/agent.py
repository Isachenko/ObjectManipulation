import tensorflow as tf

#Cacla
# 1: Initialize θ0 (belowVt(s)=V(s,θt)), ψ0, s0.
# 2: for t ∈ {0,1,2,...} do
# 3: Chooseat ∼π(st,ψt)
# 4: Perform at , observe rt +1 and st +1
# 5: δt =rt+1+γVt(st+1)−Vt(st)
# 6: θt+1=θt+αt(st)δt∇θVt(st)
# 7: ifδt >0then
# 8: ψt+1 =ψt +βt(st)(at −Ac(st,ψt))∇ψAc(st,ψt)
# 9: if st +1 is terminal then
# 10:
# Reinitialize st+1

class Agent:

    def __init__(self):
        self.state = []
        self.prev_state = []
        self.reward = 0
        self.prev_state
        self.gamma = 1
        self.alpha = 0.2
        self.betta = 0.2
        print("Agent created, tf version: ", tf.__version__)

    #Ac(st, ψt)
    def Actor(self):
        pass


    # return a[t] ∼π(st,ψt)
    def policy(self):
        return [1,2,3,4,5]

    #V(s, θt)
    def V(self, state):
        return 1



    def get_action(self):
        #all RL should go in this section
        return self.policy()

    def set_observation(self, state, reward):
        self.prev_state = state
        self.state = state
        self.state = reward

    def update_params(self):
        delta = self.reward + self.gamma*self.V(self.state) - self.V(self.prev_state)
        self.update_tetta(delta)
        if delta > 0:
            self.update_psy()

    def update_tetta(self, delta):
        pass

    def update_psy(self):
        pass



