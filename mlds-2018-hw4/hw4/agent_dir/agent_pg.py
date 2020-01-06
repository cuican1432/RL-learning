from agent_dir.agent import Agent
from torch.autograd import Variable
import scipy.misc
import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F


def save_model(model, address='model.pkl', epoch=None, iters=0):
    # if not os.path.exists(address):
    #     os.makedirs(address)
    torch.save({'modelr_state_dict': model.state_dict(),
                'epoch': epoch,
                'iters': iters,
                }, address)


def prepro(o, image_size=[80, 80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = o.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    resized = np.expand_dims(resized.astype(np.float32), axis=2)
    return np.moveaxis(resized, [0, 1, 2, 3], [2, 3, 0, 1])


def prepro_prefer(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    result = I.astype(np.float).ravel()
    return torch.Tensor(result)


def CountFrequency(my_list):
    freq = {}
    for items in my_list:
        freq[items] = my_list.count(items)

    for key, value in freq.items():
        print("% d : % d" % (key, value))
        with open('train_log.txt', "a") as f:
            f.write("\n% d : % d" % (key, value))


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG, self).__init__(env)

        if args.test_pg:
            # you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################
        self.env = env
        self.args = args
        self.model = RLModel()
        self.model = self.model.cuda()
        self.loss = nn.MSELoss().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.log_file = 'train_log.txt'

    def init_game_setting(self):
        """

        Testing function will call this function at the beginning of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        begin_obv = self.env.reset()
        prev_x = None  # used in computing the difference frame
        xs, hs, dlogps, drs = [], [], [], []
        running_reward = None
        reward_sum = 0
        episode_number = 0
        return begin_obv, prev_x, xs, hs, dlogps, drs, running_reward, reward_sum, episode_number

    def train(self, n_epochs=1000000):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        for epoch in range(n_epochs):
            xs, hs, preds, drs = self.sample_action(eps=1)
            epr = self.advantage_funct(np.vstack(drs))
            self.update_model(xs, epr, preds, hs)
            del xs, hs, preds, drs
            if epoch % 100 == 0:
                save_model(self.model, address=f'bestmodel/model_{epoch}.pkl', epoch=epoch)

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        train_dataset = Dataset_sample(observation)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        for i, x_test in enumerate(dataloader, 0):
            x_test = x_test.cuda()
            prob = self.model(x_test)
            # _, predicted = torch.max(prob, 1)
            softmax_prob = F.softmax(prob, dim=1)
            prob_dist = torch.distributions.Categorical(softmax_prob)
            predicted = prob_dist.sample()
        if test:
            return int(predicted.cpu().data)
        else:
            return int(predicted.cpu().data), prob, predicted, torch.Tensor([0])  # replace x_test with None

    def sample_action(self, eps=5):
        count_freq = []
        cur_x, prev_x, xs, hs, dlogps, drs, running_reward, reward_sum, episode_number = self.init_game_setting()
        for batch in range(eps):
            done = False
            while not done:
                x = cur_x - prev_x if prev_x is not None else np.zeros(cur_x.shape)
                prev_x = cur_x
                pred, h, predicted, x = self.make_action(x, test=False)
                xs.append(x)
                hs.append(h)
                dlogps.append(predicted)
                count_freq.append(pred)
                cur_x, reward, done, info = self.env.step(pred)
                drs.append(reward)
                reward_sum += reward
                if reward == 1:  # Pong has either +1 or -1 reward exactly when game ends.
                    print(('ep %d: game running, reward: %f' % (episode_number, reward)) + (
                        '' if reward == -1 else ' !!!!!!!!'))
                    with open(self.log_file, "a") as f:
                        f.write(('\nep %d: game running, reward: %f' % (episode_number, reward)) + (
                            '' if reward == -1 else ' !!!!!!!!'))

            episode_number += 1
            print(('ep %d: game finished, reward: %f' % (episode_number, reward_sum)))
            with open(self.log_file, "a") as f:
                f.write(('\nep %d: game finished, reward: %f' % (episode_number, reward_sum)))

            CountFrequency(count_freq)
            cur_x, prev_x, _, _, _, _, _, reward_sum, _ = self.init_game_setting()
            count_freq = []
        return xs, hs, dlogps, drs

    def update_model(self, x, reward, pred, pred_prob):
        print('update model')
        train_dataset = Dataset_update(x, reward, pred, pred_prob)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        for i, (data, pred_prob, target, reward) in enumerate(dataloader, 0):
            self.optimizer.zero_grad()
            rl_loss = self.compute_loss(pred_prob, target, reward)
            rl_loss.backward()
            self.optimizer.step()
        print(f'loss: {rl_loss.item()}')
        with open(self.log_file, "a") as f:
            f.write(f'\nloss: {rl_loss.item()}')

    def advantage_funct(self, r, gamma=0.9):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0:
                # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r

    def compute_loss(self, y_pred, y_true, reward):
        loss = nn.CrossEntropyLoss().cuda()
        n = reward.shape[0]
        loss_sum = 0
        for i in range(n):
            loss_sum += loss(y_pred[i], y_true[i]) * reward[i] * (-1)
        return loss_sum


class Dataset_update(Dataset):
    def __init__(self, matrix, loss, pred, pred_prob):
        self.data = matrix
        self.loss = loss
        self.pred = pred
        self.pred_prob = pred_prob

    def __getitem__(self, index):
        data = self.data[index]
        label = self.pred[index]
        reward = torch.Tensor(self.loss[index]).cuda()
        pred_prob = self.pred_prob[index]
        return data, pred_prob, label, reward

    def __len__(self):
        return len(self.data)


class Dataset_sample(Dataset):
    def __init__(self, observation):
        self.data = [prepro_prefer(observation)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class RLModel(nn.Module):
    def __init__(self, output_size=6):
        super().__init__()

        cc = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(1, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )
        self.encoder.add_module('layer_1', _Residual_Block(32, 32, scale=1.0))
        self.encoder.add_module('avg_1', nn.AvgPool2d(2))
        self.encoder.add_module('layer_2', _Residual_Block(32, 64, scale=1.0))
        self.encoder.add_module('avg_2', nn.AvgPool2d(2))
        self.encoder.add_module('layer_3', _Residual_Block(64, 64, scale=1.0))
        self.encoder.add_module('avg_3', nn.AvgPool2d(2))
        self.encoder.add_module('layer_4', _Residual_Block(64, 16, scale=1.0))
        self.encoder.add_module('avg_4', nn.AvgPool2d(2))
        self.encoder.add_module('layer_5', _Residual_Block(16, 6, scale=1.0))
        self.encoder.add_module('avg_5', nn.AvgPool2d(2))

        self.fc = nn.Linear(6, output_size)
        weight_init(self.modules())

    def forward(self, x):
        x = x.view(x.size(0), 1, 80, 80)
        x = self.encoder(x)
        x = x.view(1, -1)
        x = self.fc(x)
        return x


def weight_init(modules):
    for m in modules:
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            # if m.bias:
            #     torch.nn.init.xavier_uniform_(m.bias)


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
