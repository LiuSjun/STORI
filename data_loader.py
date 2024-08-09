import os
import ujson as json
import numpy as np
from scipy.signal import savgol_coeffs
from scipy.ndimage import convolve1d
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class MyTrainSet(Dataset):
    def __init__(self):
        super(MyTrainSet, self).__init__()
        self.content = open('D:/studydata/OSC/data/CDL_sub/DLdata/SAR_NDVI_train.json').readlines()

        indices = np.arange(len(self.content))

        val_indices = np.random.choice(indices, len(self.content) // 10)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        # print(idx)
        rec = json.loads(self.content[idx])

        # 训练和测试分开
        rec['is_train'] = 1

        # 数据集划分训练和测试（4：1）
        # if idx in self.val_indices:
        #     rec['is_train'] = 0
        # else:
        #     rec['is_train'] = 1

        return rec


class MyTestSet(Dataset):
    def __init__(self):
        super(MyTestSet, self).__init__()
        # self.content = open('D:/studydata/OSC/data/CDL_sub/DLdata/simu_gap/SAR_NDVI_testM_16.json').readlines()
        self.content = open('D:/studydata/OSC/data/CDL_sub/DLdata/simu_gap/SAR_NDVI_testM_16.json').readlines()

        indices = np.arange(len(self.content))

        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])

        rec['is_train'] = 0
        # if idx in self.val_indices:
        #     rec['is_train'] = 0
        # else:
        #     rec['is_train'] = 1
        return rec


def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))
    scaler = MinMaxScaler(feature_range=[-1, 1])

    def to_tensor_dict(recs):

        coeffs_long_trend = savgol_coeffs(19, 2)
        values0 = np.array(list(map(lambda r: list(map(lambda x: x['values'], r)), recs)))
        sar1 = values0[:, :, 0]
        sar2 = values0[:, :, 1]
        bsize = values0.shape[0]
        for k in range(bsize):
            onesar1 = sar1[k:k+1, :].T
            onesar2 = sar2[k:k+1, :].T
            onesar1 = scaler.fit_transform(onesar1)
            onesar2 = scaler.fit_transform(onesar2)
            onesar1 = onesar1[:, 0]
            onesar2 = onesar2[:, 0]
            onesar10 = convolve1d(onesar1, coeffs_long_trend, mode="wrap")
            onesar20 = convolve1d(onesar2, coeffs_long_trend, mode="wrap")
            values0[k, :, 0] = onesar10
            values0[k, :, 1] = onesar20
        values = torch.FloatTensor(values0)

        # values = torch.FloatTensor(
        #     list(map(lambda r: list(map(lambda x: x['values'], r)), recs)))

        masks = torch.FloatTensor(
            list(map(lambda r: list(map(lambda x: x['masks'], r)), recs)))
        deltas = torch.FloatTensor(
            list(map(lambda r: list(map(lambda x: x['deltas'], r)), recs)))
        forwards = torch.FloatTensor(
            list(map(lambda r: list(map(lambda x: x['forwards'], r)), recs)))

        evals = torch.FloatTensor(
            list(map(lambda r: list(map(lambda x: x['evals'], r)), recs)))
        eval_masks = torch.FloatTensor(
            list(map(lambda r: list(map(lambda x: x['eval_masks'], r)), recs)))

        # print('values:{}'.format(values.size()))
        # print('!!')
        # print('masks:{}'.format(masks.size()))
        # print('deltas:{}'.format(deltas.size()))
        # print('forwards:{}'.format(forwards.size()))
        # print('evals:{}'.format(evals.size()))
        # print('eval_masks:{}'.format(eval_masks.size()))

        # return {
        #     'values': values.permute(0, 2, 1),
        #     'forwards': forwards.permute(0, 2, 1),
        #     'masks': masks.permute(0, 2, 1),
        #     'deltas': deltas.permute(0, 2, 1),
        #     'evals': evals.permute(0, 2, 1),
        #     'eval_masks': eval_masks.permute(0, 2, 1)
        # }

        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals,
                'eval_masks': eval_masks}

    ret_dict = {
        'forward': to_tensor_dict(forward),
        'backward': to_tensor_dict(backward)
    }

    ret_dict['labels'] = torch.FloatTensor(
        list(map(lambda x: x['label'], recs)))
    ret_dict['is_train'] = torch.FloatTensor(
        list(map(lambda x: x['is_train'], recs)))

    # print('values:{}'.format(ret_dict['forward']['values'].size()))
    # print('!!')
    # print('masks:{}'.format(masks.size()))
    # print('deltas:{}'.format(deltas.size()))
    # print('forwards:{}'.format(forwards.size()))
    # print('evals:{}'.format(evals.size()))
    # print('eval_masks:{}'.format(eval_masks.size()))

    return ret_dict


def get_train_loader(batch_size=64, shuffle=True):
    data_set = MyTrainSet()
    data_iter = DataLoader(dataset=data_set, \
                           batch_size=batch_size, \
                           num_workers=8, \
                           shuffle=shuffle, \
                           pin_memory=True, \
                           collate_fn=collate_fn
                           )
    return data_iter


def get_test_loader(batch_size=64, shuffle=False):
    data_set = MyTestSet()
    data_iter = DataLoader(dataset=data_set, \
                           batch_size=batch_size, \
                           num_workers=8, \
                           shuffle=shuffle, \
                           pin_memory=True, \
                           collate_fn=collate_fn
                           )

    return data_iter

# if __name__ == '__main__':
#     data_set = MyTrainSet()
#     data_iter = DataLoader(dataset=data_set, \
#                            batch_size=128, \
#                            num_workers=8, \
#                            shuffle=True, \
#                            pin_memory=True, \
#                            collate_fn=collate_fn
#                            )

