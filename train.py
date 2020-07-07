import numpy as np
from utilities.trainer import *
import torch
# from arguments import *
import os
from utilities.util import *
from utilities.logger import Logger
import argparse
import pickle

parser = argparse.ArgumentParser(description='Test rl agent.')
parser.add_argument('--save-path', type=str, nargs='?', default='/home/lhchen/nas/svrl/res/simple_spread_3/sqddpg/',
                    help='Please input the directory of saving model.')
parser.add_argument('--no', type=str, default='test')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--scenario', type=str, default='simple_spread_3')

# hyper-params
parser.add_argument('--policy_lrate', type=float, default=1e-4)
parser.add_argument('--value_lrate', type=float, default=1e-3)
parser.add_argument('--train_episodes_num', type=int, default=180000)
argv = parser.parse_args()

if argv.scenario == 'simple_spread_3':
    from args.simple_spread_3_sqddpg import *
elif argv.scenario == 'simple_spread_6':
    from args.simple_spread_6_sqddpg import *
else:
    raise NotImplementedError
args = args._replace(
    policy_lrate=argv.policy_lrate,
    train_episodes_num=argv.train_episodes_num,
    value_lrate=argv.value_lrate
)

torch.manual_seed(argv.seed)
np.random.seed(argv.seed)

if argv.save_path[-1] is '/':
    save_path = argv.save_path + '%s/' % argv.no
else:
    save_path = argv.save_path + '/' + '%s/' % argv.no
if not os.path.exists(save_path):
    os.mkdir(save_path)

# create save folders
if 'model_save' not in os.listdir(save_path):
    os.mkdir(save_path + 'model_save')
if 'tensorboard' not in os.listdir(save_path):
    os.mkdir(save_path + 'tensorboard')
if log_name not in os.listdir(save_path + 'model_save/'):
    os.mkdir(save_path + 'model_save/' + log_name)
if log_name not in os.listdir(save_path + 'tensorboard/'):
    os.mkdir(save_path + 'tensorboard/' + log_name)
else:
    path = save_path + 'tensorboard/' + log_name
    for f in os.listdir(path):
        file_path = os.path.join(path, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

logger = Logger(save_path + 'tensorboard/' + log_name)

model = Model[model_name]

strategy = Strategy[model_name]

print('{}\n'.format(args))

if strategy == 'pg':
    train = PGTrainer(args, model, env(), logger, args.online)
elif strategy == 'q':
    raise NotImplementedError('This needs to be implemented.')
else:
    raise RuntimeError('Please input the correct strategy, e.g. pg or q.')

stat = dict()
episode_rewards = []
final_ep_rewards = []

for i in range(args.train_episodes_num):
    # print('%i train_episodes' % i)
    ep_rew = train.run(stat)
    episode_rewards.append(ep_rew)
    # print(i, episodes_rewards[-1])
    train.logging(stat)
    if i % args.save_model_freq == args.save_model_freq - 1:
        train.print_info(stat)
        torch.save({'model_state_dict': train.behaviour_net.state_dict()},
                   save_path + 'model_save/' + log_name + '/model.pt')
        # print('The model is saved!\n')
        with open(save_path + 'model_save/' + log_name + '/log.txt', 'w+') as file:
            file.write(str(args) + '\n')
            file.write(str(i))

    if i > 0 and i % 1000 == 0:
        final_ep_rewards.append(np.mean(episode_rewards[-1000:]))
        print(i, final_ep_rewards[-1])

rew_file_name = save_path + 'exp_rewards.pkl'
with open(rew_file_name, 'wb') as fp:
    pickle.dump(final_ep_rewards, fp)
