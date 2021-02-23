import torch
import numpy as np
from torch import nn
from codes.common_cmk import funcs
from codes.MemeryNet.lib import descriptor
import pickle

class EntNet(nn.Module):

    # descriptor
    s = descriptor.s_AttriAccess()
    G = descriptor.G_AttriAccess()
    H = descriptor.H_AttriAccess()
    new_H = descriptor.new_H_AttriAccess()
    q = descriptor.q_AttriAccess()
    p = descriptor.p_AttriAccess()
    u = descriptor.u_AttriAccess()
    ans_vector = descriptor.ans_vector_AttriAccess()
    ans = descriptor.ans_AttriAccess()

    def __init__(self, W, input_size, H_size, X_size, Y_size, Z_size, R_size, K_size, device):

        super(EntNet, self).__init__()
        self.record_allowed = False
        self.H_size = H_size
        self.device = device
        # dynamic memory
        self.H = nn.init.normal_(torch.empty(H_size, dtype=torch.float, device=self.device), mean=0.0, std=0.1)
        self.W = W.clone().detach()

        # embedding parameters
        self.params = nn.ParameterDict({
            'F': nn.Parameter(nn.init.normal_(torch.empty(input_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),

            # shared parameters
            'X': nn.Parameter(nn.init.normal_(torch.empty(X_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),
            'Y': nn.Parameter(nn.init.normal_(torch.empty(Y_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),
            'Z': nn.Parameter(nn.init.normal_(torch.empty(Z_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),

            # answer parameters
            'R': nn.Parameter(nn.init.normal_(torch.empty(R_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),
            'K': nn.Parameter(nn.init.normal_(torch.empty(K_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1))
        })

        # dropout
        self.dropout = nn.Dropout(p=0.3)

        # init records setting
        self.reset_record_status()

    def reset_record_status(self):
        self.story_index = 0
        self.state_path = {}
        self.matrics = {}
        self.params_matrics = {}
        self.grads_matrics = {}
        self.record_allowed = False  # allow the descriptor record each of state from model

    def init_record_status_for_new_story(self):
        if self.record_allowed:
            self.story_index += 1
            self.matrics[self.story_index] = {}
            self.state_path[self.story_index] = []

    def record_params(self):

        # record the learnable parameter value
        if self.story_index not in self.params_matrics: self.params_matrics[self.story_index] = {}
        for key, param in self.params.items(): # nn.ParameterDict()
            param_arr = param.data.detach().cpu().unsqueeze(dim=0).numpy()
            if key in self.params_matrics[self.story_index]:
                self.params_matrics[self.story_index][key] = np.concatenate((self.params_matrics[self.story_index][key], param_arr), axis=0)
            else:
                self.params_matrics[self.story_index][key] = param_arr

        # record the gradient
        if self.story_index not in self.grads_matrics: self.grads_matrics[self.story_index] = {}
        for key, param in self.params.items():
            grad_arr = param.grad.detach().cpu().unsqueeze(dim=0).numpy()
            if key in self.grads_matrics[self.story_index]:
                self.grads_matrics[self.story_index][key] = np.concatenate((self.grads_matrics[self.story_index][key], grad_arr), axis=0)
            else:
                self.grads_matrics[self.story_index][key] = grad_arr

        # # register the grads hook
        # self.params_grads[self.story_index] = {}
        # modules = self.named_modules()
        # for name, module in modules:
        #     module.register_backward_hook(self.hook_fn_backward)
        #     self.params_grads[self.story_index][str(module)] = {}

    # def hook_fn_backward(self, module, grad_inputs, grad_outputs):
    #     for grad_output in grad_outputs:
    #         self.params_grads[self.story_index] = grad_output.detach().cpu().numpy()
    #     for grad_input in grad_inputs:
    #         self.params_grads[self.story_index] = grad_input.detach().cpu().numpy()

    def snapshot(self, path, matrix_name, params_name, grad_name, state_path_name, episode):
        matrix_full_path = path + '/' + matrix_name.format(episode)
        params_full_path = path + '/' + params_name.format(episode)
        grad_full_path = path + '/' + grad_name.format(episode)
        matrics_path_full_path = path + '/' + state_path_name.format(episode)
        # save matrics
        with open(matrix_full_path, 'wb') as f:
            pickle.dump(self.matrics, f, pickle.HIGHEST_PROTOCOL)
        with open(params_full_path, 'wb') as f:
            pickle.dump(self.params_matrics, f, pickle.HIGHEST_PROTOCOL)
        with open(grad_full_path, 'wb') as f:
            pickle.dump(self.grads_matrics, f, pickle.HIGHEST_PROTOCOL)
        with open(matrics_path_full_path, 'wb') as f:
            pickle.dump(self.state_path, f, pickle.HIGHEST_PROTOCOL)
        # init records setting
        self.reset_record_status()

    def run_model(self, dataset, criterion, optimizer, device, mode="train"):
        """
        :param dataset: collections.namedtuple('DataSet', ["E_s", 'Q', "ans_vector", "ans", "new_story", "end_story", 'stories', 'q'])
        :param criterion: criterion
        :param optimizer: optimizer
        :param mode: string: train/test
        :return: detached_loss, predict_ans
        """
        if mode == "Train":
            self.train()
        elif mode == "Test":
            self.eval()
        self.forward(dataset.E_s, new_story=dataset.new_story)
        predict = self.answer(dataset.Q)
        loss = criterion(predict, torch.tensor([dataset.ans], device=device))
        if mode == "Train":
            if dataset.end_story:
                loss.backward()
                if self.record_allowed: self.record_params()
                optimizer.step()
                optimizer.zero_grad()
            else:
                loss.backward(retain_graph=True)
                if self.record_allowed: self.record_params()
        # detach the loss and predicted vector
        detached_loss = loss.detach().cpu().item()
        predict_ans = torch.argmax(predict.detach().cpu()).item() # get the ans value in integer
        return detached_loss, predict_ans

    def forward(self, E_s, new_story=True):
        """
        k = sentence length
        m = memory size
        :param E_s: [ torch.tensor = facts in word embeddings ]
        :return: ans_vector (n,1)
        """
        self.prepare_memory(new_story)
        for E in E_s:
            # E = torch.tensor(data=E, requires_grad=True, dtype=torch.float)   # (64*k)
            self.s = torch.mul(self.params['F'], E).sum(dim=1).unsqueeze(1)  # (64*1)
            self.G = nn.Softmax(dim=1)((torch.mm(self.s.t(), self.H) + torch.mm(self.s.t(), self.W)))  # (1*m)
            self.new_H = nn.Sigmoid()(torch.mm(self.dropout(self.params['X']), self.H) +
                                 torch.mm(self.dropout(self.params['Y']), self.W) +
                                 torch.mm(self.dropout(self.params['Z']), self.s))  # (64*m)
            self.H = funcs.unitVector_2d(self.H + torch.mul(self.G, self.new_H), dim=0)  # (64*m)

    def answer(self, Q):
        """
        :param Q: torch.tensor = Questions in word embeddings (n,PAD_MAX_LENGTH)
        :return: ans_vector (n,1)
        """
        # answer the question
        Q.requires_grad_()
        self.q = torch.mul(self.params['F'], Q).sum(dim=1).unsqueeze(1)  # (64*1)
        self.p = nn.Softmax(dim=1)(torch.mm(self.q.t(), self.H))  # (1*m)
        self.u = torch.mul(self.p, self.H).sum(dim=1).unsqueeze(1)  # (64*1)
        self.unit_params('R', dim=1)
        self.ans_vector = torch.mm(self.params['R'], nn.Sigmoid()(self.q + torch.mm(self.params['K'], self.u)))  # (k,1)
        self.ans = nn.LogSoftmax(dim=1)(self.ans_vector.t())
        return self.ans

    def prepare_memory(self, new_story):
        if new_story:
            self.init_record_status_for_new_story()
            if self.record_allowed: self.state_path[self.story_index].append('params')  # beginning at each start of story-question pair
            self.H = nn.init.normal_(self.H).detach()
        else:
            if self.record_allowed: self.state_path[self.story_index].append('params')  # beginning at each start of story-question pair
            # self.H = self.H.detach()

    def unit_params(self, name, dim):
        magnitude = self.params[name].data.detach().pow(2).sum(dim=dim).sqrt().unsqueeze(dim=dim)
        self.params[name].data = self.params[name] / magnitude