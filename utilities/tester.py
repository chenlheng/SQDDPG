import numpy as np
import torch
from utilities.util import *
import time


class PGTester(object):

    def __init__(self, env, behaviour_net, args):
        self.env = env
        self.behaviour_net = behaviour_net.cuda().eval() if args.cuda else behaviour_net.eval()
        self.args = args
        self.cuda_ = self.args.cuda and torch.cuda.is_available()

    def action_logits(self, state, last_action, info):
        return self.behaviour_net.policy(state, last_action, info=info)

    def run_step(self, state, last_action, info={}):
        state = cuda_wrapper(prep_obs(state).contiguous().view(1, self.args.agent_num, self.args.obs_size), cuda=self.cuda_)
        action_out = self.action_logits(state, last_action, info)
        action = select_action(self.args, action_out, status='test')
        _, actual = translate_action(self.args, action, self.env)
        next_state, reward, done, _ = self.env.step(actual)
        disp = 'The rewards of agents are:'
        for r in reward:
            disp += ' '+str(r)[:7]
        print (disp+'.')
        return next_state, action, done

    def run_game(self, episodes, render):
        action = cuda_wrapper(torch.zeros((1, self.args.agent_num, self.args.action_dim)), cuda=self.cuda_)
        action[:, 0, :] += 1
        if self.args.model_name == 'coma':
            info = {}
            info['get_episode'] = True
            self.behaviour_net.init_hidden(batch_size=1)
            self.behaviour_net.add_hidden()
        for ep in range(episodes):
            print ('The episode {} starts!'.format(ep))
            state = self.env.reset()
            while True:
                if render:
                    self.env.render()
                state, action, done = self.run_step(state, action, info=info)
                time.sleep(0.1)
                if np.all(done):
                    print ('The episode {} is finished!'.format(ep))
                    break



class QTester(PGTester):

    def __init__(self, env, behaviour_net, args):
        super(QTester, self).__init__(env, behaviour_net, args)

    def action_logits(self, state, last_action):
        return self.behaviour_net.value(state, last_action)
