from agent_dir.agent import Agent
from torch.autograd import Variable
import scipy.misc
import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F


def count_frequency(my_list):
    freq = {}
    for items in my_list:
        freq[items] = my_list.count(items)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN, self).__init__(env)

        if args.test_dqn:
            # you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################
        self.env = env
        self.args = args
        self.model_t = TargetModel().cuda()
        self.log_file = 'train_log_dqn.txt'
        self.action_list = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
        self.loss = nn.MSELoss().cuda()
        self.optimizer = torch.optim.Adam(self.model_t.parameters(), lr=3e-4)

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        reward_sum = 0
        episode_number = 0
        prev_x = None
        x_cache, x_next_cache, reward_cache, pred_cache, info_cache = list(), list(), list(), list(), list()
        begin_obv = self.env.reset()
        return begin_obv, prev_x, x_cache, x_next_cache, reward_cache, pred_cache, info_cache, reward_sum, episode_number

    def train(self, n_epochs=1000000):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        for epoch in range(n_epochs):
            with open(self.log_file, "a") as f:
                f.write(f'\nepoch: {epoch}')
            reward_cache, pred_cache, prob_cache = self.sample_action(eps=5)
            with open(self.log_file, "a") as f:
                f.write('\n')
            for _ in range(10):
                self.update_model(reward_cache, pred_cache, prob_cache, n_step=50)
            del reward_cache, pred_cache, prob_cache
            if epoch % 100 == 0:
                save_model(self.model_t, address=f'bestmodel/dqn_model_{epoch}.pkl', epoch=epoch)

    def make_action(self, observation, test=True):
        ##################
        # YOUR CODE HERE #
        ##################
        train_dataset = Dataset_sample(observation, self.action_list)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
        self.model_t.eval()
        for i, (s, a) in enumerate(dataloader, 0):
            s = s.cuda()
            a = a.cuda()
            prob = self.model_t(s, a)
            predicted = select_action(prob.data)
        if test:
            return int(predicted.cpu().data)
        else:
            return int(predicted.cpu().data), prob

    def sample_action(self, eps=5):
        count_freq = []
        cur_x, prev_x, _, _, reward_cache, pred_cache, prob_cache, reward_sum, episode_number = self.init_game_setting()
        for batch in range(eps):
            done = False
            while not done:
                x = cur_x - prev_x if prev_x is not None else np.zeros(cur_x.shape)
                prev_x = cur_x
                int_pred, prob_pred = self.make_action(x, test=False)
                cur_x, reward, done, info = self.env.step(int_pred)
                reward_cache.append(reward)
                pred_cache.append(int_pred)
                prob_cache.append(prob_pred)
                count_freq.append(int_pred)
                reward_sum += reward
                # if reward > 0:
                #     print(('ep %d: game running, reward: %f !!!!!!' % (episode_number, reward)))
                #     with open(self.log_file, "a") as f:
                #         f.write(('\nep %d: game running, reward: %f !!!!!!' % (episode_number, reward)))

            episode_number += 1
            print(('ep %d: game finished, reward: %f' % (episode_number, reward_sum)))
            with open(self.log_file, "a") as f:
                f.write(('\nep %d: game finished, reward: %f' % (episode_number, reward_sum)))
            count_frequency(count_freq)
            cur_x, prev_x, _, _, _, _, _, reward_sum, episode_number = self.init_game_setting()
            count_freq = []
        return reward_cache, pred_cache, prob_cache

    def update_model(self, reward_cache, pred_cache, prob_cache, n_step=10):
        train_dataset = Dataset_update(reward_cache, pred_cache, prob_cache, n_step=n_step)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
        self.model_t.train()
        for i, (y_pred, y_true) in enumerate(dataloader, 0):
            y_true = y_true.cuda()
            self.optimizer.zero_grad()
            loss = self.loss(y_pred, y_true)
            loss.backward(retain_graph=True)
            self.optimizer.step()
        # # del reward_cache, pred_cache, prob_cache
        print(f'loss: {loss.item()}')
        with open(self.log_file, "a") as f:
            f.write(f'| loss: {loss.item()}')


def select_action(prob_list):
    softmax_prob = F.softmax(prob_list.view(1, -1), dim=1)
    prob_dist = torch.distributions.Categorical(softmax_prob)
    predicted = prob_dist.sample()
    return predicted


class TargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        cc = 8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )
        self.encoder.add_module('layer_1', _Residual_Block(8, 16, scale=1.0))
        self.encoder.add_module('avg_1', nn.AvgPool2d(4))
        self.encoder.add_module('layer_3', _Residual_Block(16, 4, scale=1.0))
        self.encoder.add_module('avg_3', nn.AvgPool2d(8))
        self.fc_situation = nn.Linear(4, 8)
        self.fc_action = nn.Linear(4, 8)
        self.fc_final = nn.Linear(16, 1)

        weight_init(self.modules())

    def forward(self, x, action):
        x = x.view(x.size(0), 1, 80, 80)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_situation(x)
        y = self.fc_action(action)
        x = torch.cat([x, y], dim=1)
        x = self.fc_final(x)
        return x


def weight_init(modules):
    for m in modules:
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)


class _Residual_Block(nn.Module):
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(_Residual_Block, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output, identity_data)))
        return output


class Dataset_sample(Dataset):
    def __init__(self, observation, pred):
        self.data = [self.prepro(observation)]
        self.pred = torch.Tensor(pred)

    def __getitem__(self, index):
        return self.data[0], self.pred[index]

    def __len__(self):
        return len(self.pred)

    def prepro(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        result = I.astype(np.float).ravel()
        return torch.Tensor(result)


class Dataset_update(Dataset):
    def __init__(self, reward_cache, pred_cache, prob_cache, n_step=5):
        self.reward = reward_cache
        self.pred = pred_cache
        self.prob = prob_cache
        self.n_step = n_step
        self.n = len(reward_cache)
        self.action_list = torch.Tensor([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]).cuda()

    def __getitem__(self, index):
        mask = self.action_list[self.pred[index]].view(-1, 1)
        x = self.prob[index]
        if index < self.n - self.n_step:
            r = torch.sum(torch.Tensor(self.reward[index:index + self.n_step]))
            y_prob = self.prob[index + self.n_step]
            y_pred = select_action(y_prob)
            y = r + y_prob[y_pred]
            return x * mask, y * mask
        else:
            return x * mask, 0 * mask

    def __len__(self):
        return len(self.pred)

    def prepro(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        result = I.astype(np.float).ravel()
        return torch.Tensor(result)


def save_model(model, address='model.pkl', epoch=None, iters=0):
    # if not os.path.exists(address):
    #     os.makedirs(address)
    torch.save({'modelr_state_dict': model.state_dict(),
                'epoch': epoch,
                'iters': iters,
                }, address)
