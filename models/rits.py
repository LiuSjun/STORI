import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math


# class FeatureRegression(nn.Module):   #多元回归估计，自己根据需要定制全连接层
#     def __init__(self, input_size):
#         super(FeatureRegression, self).__init__()
#         self.build(input_size)
#
#     def build(self, input_size):
#         self.W = Parameter(torch.Tensor(input_size, input_size))
#         self.b = Parameter(torch.Tensor(input_size))
#
#         m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
#         self.register_buffer('m', m)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.W.size(0))
#         self.W.data.uniform_(-stdv, stdv)
#         if self.b is not None:
#             self.b.data.uniform_(-stdv, stdv)
#
#     def forward(self, x):
#         z_h = F.linear(x, self.W * Variable(self.m), self.b)
#         return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

# SELECT_SIZE可能有问题

class RITS_H(nn.Module):
    def __init__(self, rnn_hid_size, SEQ_LEN, SELECT_SIZE):
        super(RITS_H, self).__init__()
        self.rnn_hid_size = rnn_hid_size
        self.SEQ_LEN = SEQ_LEN
        self.SELECT_SIZE = SELECT_SIZE
        self.build()

    def build(self):
        self.rnn_cell0 = nn.LSTMCell(self.SELECT_SIZE * 2, self.rnn_hid_size)
        self.rnn_cell1 = nn.LSTMCell(self.SELECT_SIZE * 2, self.rnn_hid_size)
        self.temp_decay_h = TemporalDecay(input_size = self.SELECT_SIZE, output_size = self.rnn_hid_size, diag = False)
        self.temp_decay_h0 = TemporalDecay(input_size=self.SELECT_SIZE, output_size=self.rnn_hid_size, diag=False)
        self.hist_reg0 = nn.Linear(self.rnn_hid_size, self.SELECT_SIZE)
        # self.feat_reg0 = FeatureRegression(self.SELECT_SIZE * 2)  #只需输入两维
        # self.weight_combine = nn.Linear(self.SELECT_SIZE * 2, self.SELECT_SIZE)

    # def forward(self, data, f_data, direct):
    def forward(self, data, direct):

        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        bsize = values.size()[0]

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']
        evals = evals[:, :, 2]
        eval_masks = eval_masks[:, :, 2]
        evals = torch.reshape(evals, (bsize, self.SEQ_LEN, 1))
        eval_masks = torch.reshape(eval_masks, (bsize, self.SEQ_LEN, 1))

        h0 = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c0 = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        h1 = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c1 = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        if torch.cuda.is_available():
            h0, c0 = h0.cuda(), c0.cuda()
            h1, c1 = h1.cuda(), c1.cuda()

        x_loss0 = 0.0
        loss1 = 0.0
        imputations0 = []

        for t in range(self.SEQ_LEN):

            x_y = values[:, t, 2]
            m_y = masks[:, t, 2]
            d_y = deltas[:, t, 2]

            x_y = torch.reshape(x_y, (bsize, 1))
            m_y = torch.reshape(m_y, (bsize, 1))
            d_y = torch.reshape(d_y, (bsize, 1))

            gamma_h = self.temp_decay_h(d_y)

            # h0 = h0 * gamma_h

            x_h = self.hist_reg0(h0)
            x_c = m_y * x_y + (1 - m_y) * x_h
            x_loss0 = x_loss0 + torch.sum(torch.abs(x_y - x_h) * m_y) / (torch.sum(m_y) + 1e-5)
            loss1 += torch.sum(torch.abs(x_y - x_h) * m_y) / (torch.sum(m_y) + 1e-5)

            inputs1 = values[:, t, 0:2]
            h1, c1 = self.rnn_cell1(inputs1, (h1, c1))

            h1 = h1 * gamma_h

            inputs = torch.cat([x_c, m_y], dim=1)
            h0, c0 = self.rnn_cell0(inputs, (h1, c1))

            imputations0.append(x_c.unsqueeze(dim = 1))

        imputations0 = torch.cat(imputations0, dim = 1)

        # return {'loss': x_loss0, 'imputations': imputations0,\
        #         'evals': evals, 'eval_masks': eval_masks, 'imputations_z': imputationsz,\
        #         'loss1': loss1, 'loss2': loss2, 'loss3': loss3}
        return {'loss': x_loss0, 'imputations': imputations0,\
                'evals': evals, 'eval_masks': eval_masks,\
                'loss1': loss1}

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class Model(nn.Module):
    def __init__(self, rnn_hid_size, INPUT_SIZE, SEQ_LEN, SELECT_SIZE):
        super(Model, self).__init__()
        self.RITS_H = RITS_H(rnn_hid_size, SEQ_LEN, SELECT_SIZE)

    def forward(self, data, direct):
        out2 = self.RITS_H(data, direct)

        loss2 = out2['loss']

        # return {'loss': loss1+loss2, 'imputations': out2['imputations'],\
        #         'evals': out2['evals'], 'eval_masks': out2['eval_masks'], 'imputations_z': out2['imputations_z'],\
        #         'loss1': out2['loss1'], 'loss2': out2['loss2'], 'loss3': out2['loss3'], 'loss_F': loss1}
        return {'loss': loss2, 'imputations': out2['imputations'],\
                'evals': out2['evals'], 'eval_masks': out2['eval_masks'], 'loss1': out2['loss1']}


    def run_on_batch(self, data, optimizer, epoch = None):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret