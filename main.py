import torch
import torch.optim as optim
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
import models
from support.early_stopping import EarlyStopping
import argparse
import data_loader
from math import sqrt
from sklearn import metrics
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=250)  # 350
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--model', type=str, default='rits')
parser.add_argument('--hid_size', type=int, default=96)
parser.add_argument('--SEQ_LEN', type=int, default=46)
parser.add_argument('--INPUT_SIZE', type=int, default=3)
parser.add_argument('--SELECT_SIZE', type=int, default=1)
# parser.add_argument('--impute_weight', type=float, default=0.3)
# parser.add_argument('--label_weight', type=float, default=1.0)
args = parser.parse_args(args=[])

#定义保存输出文件
def savePreprocessedData(path, data):
    with open(path +".npy", 'bw') as outfile:
        np.save(outfile, data)


def run_train():
    model = getattr(models, args.model).Model(args.hid_size, args.INPUT_SIZE, args.SEQ_LEN, args.SELECT_SIZE)
    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    # Early Stopping
    # initialize the early_stopping object
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 50
    early_stopping = EarlyStopping(savepath='D:/studydata/OSC/data/CDL_sub/result/RITS/CDL_0512.pt',
                                   patience=patience, verbose=True, useralystop=False, delta=0.01)

    # 训练过程

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0008)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    # data_iter = data_loader.get_loader(batch_size=args.batch_size)
    data_iter = data_loader.get_train_loader(batch_size=args.batch_size)

    # epoch_loss = np.zeros((args.epochs))

    tb = SummaryWriter('D:/studydata/OSC/data/CDL_sub/result/RITS/log')
    # tensorboard --logdir=./result/log
    # tensorboard --logdir=D:/studydata/OSC/data/CIA_sub/result/BRITS/log

    for epoch in range(args.epochs):
        model.train()

        run_loss = 0.0
        rmse = 0.0
        # f_b_mae = 0.0

        run_loss1 = 0.0

        for idx, data in tqdm(enumerate(data_iter)):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer, epoch)

            run_loss += ret['loss'].item()
            run_loss1 += ret['loss1'].item()

            eval_masks = ret['eval_masks'].data.cpu().numpy()

            # series_f = ret['imputations_f'].data.cpu().numpy()
            # series_b = ret['imputations_b'].data.cpu().numpy()
            # series_f = series_f[np.where(eval_masks == 1)]
            # series_b = series_b[np.where(eval_masks == 1)]
            # f_b_mae += np.abs(series_f - series_b).mean()

            if epoch+1 % 100 == 0:
                eval_ = ret['evals'].data.cpu().numpy()
                imputation = ret['imputations'].data.cpu().numpy()
                eval_ = eval_[np.where(eval_masks == 1)]
                imputation = imputation[np.where(eval_masks == 1)]
                rmse += sqrt(metrics.mean_squared_error(eval_, imputation))


            # print('\r Progress epoch {}, {:.2f}%, average loss {}'.format(
            #     epoch, (idx + 1) * 100.0 / len(data_iter),
            #     run_loss / (idx + 1.0)))

        print("Epochs: ", epoch, " Loss: ", run_loss / (idx + 1.0))
        # epoch_loss[epoch] = run_loss / (idx + 1.0)
        # savePreprocessedData("D:/studydata/OSC/code/DL_test/BRITS-test/V1/result/04/model_trainLoss", epoch_loss)

        # 训练测试分开
        # test_data_iter = data_loader.get_test_loader(
        #     batch_size=args.batch_size)
        # valid_loss = evaluate(model, test_data_iter, train_flag = 1)

        # valid_loss = rmse / (idx + 1.0)
        valid_loss = run_loss / (idx + 1.0)
        if epoch + 1 % 100 == 0:
            print("Epochs: ", epoch, " Auc metrics: ", rmse / (idx + 1.0))

        tb.add_scalar('Total Loss', valid_loss, epoch)
        tb.add_scalar('Hndvi-pre Loss', run_loss1 / (idx + 1.0), epoch)
        # tb.add_scalar('Forward-backward MAE', f_b_mae / (idx + 1.0), epoch)


        # 数据集划分训练和测试（4：1）
        # valid_loss = evaluate(model, data_iter, train_flag=1)

        # early stop
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    scheduler.step()
    tb.close()

def evaluate_model():
    model = getattr(models, args.model).Model(args.hid_size, args.INPUT_SIZE, args.SEQ_LEN, args.SELECT_SIZE)
    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    savepath='D:/studydata/OSC/data/CDL_sub/result/RITS/CDL_0512.pt'
    # savepath = 'D:/studydata/OSC/data/CDL_sub/result/BRITS_MODSAR/CDL_0506.pt'

    #有真实值时，带评价的模型预测测试
    test(model,savepath)

    #只把模型作为预测
    # predict(model,savepath)

def test(model, savepath):

    model.load_state_dict(torch.load(savepath))

    test_data_iter = data_loader.get_test_loader(
        batch_size=args.batch_size)

    train_flag = 0

    model.eval()

    evals = []
    imputations = []

    save_impute = []
    save_evals = []

    for idx, data in tqdm(enumerate(test_data_iter)):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        # save the imputation results which is used to test the improvement of traditional methods with imputed values
        # save_impute.append(ret['imputations'].data.cpu().numpy())
        # save_label.append(ret['labels'].data.cpu().numpy())

        # pred = ret['predictions'].data.cpu().numpy()
        # label = ret['labels'].data.cpu().numpy()
        # is_train = ret['is_train'].data.cpu().numpy()

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        # imputation_z = ret['imputations_z'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

        # save the imputation results which is used to test the improvement of traditional methods with imputed values
        # save_impute.append(imputation[np.where(eval_masks == 1)])
        # save_evals.append(eval_[np.where(eval_masks == 1)])
        save_impute.append(imputation)
        save_evals.append(eval_)
        # save_label.append(ret['labels'].data.cpu().numpy())

        # save_impute_z.append(imputation_z[np.where(eval_masks == 1)])


    evals = np.asarray(evals)
    imputations = np.asarray(imputations)


    print('MAE', np.abs(evals - imputations).mean())
    print('MRE', np.abs(evals - imputations).sum() / np.abs(evals).sum())
    print('RMSE',sqrt(metrics.mean_squared_error(evals,imputations)))
    print('R', pearsonr(evals,imputations))


    save_impute = np.concatenate(save_impute, axis=0)
    save_evals = np.concatenate(save_evals, axis=0)


    # save_impute_z = np.concatenate(save_impute_z, axis=0)

    if train_flag == 0:
        # np.save('D:/studydata/OSC/data/CDL_sub/simu_gap/result/BRITS/{}_CDL_simugap_predicteddata16'.format(args.model), save_impute)
        # np.save('D:/studydata/OSC/data/CDL_sub/simu_gap/result/BRITS/{}_CDL_simugap_truedata'.format(args.model), save_evals)

        # np.save('D:/studydata/OSC/data/CIA_sub/discuss_rits/result/{}_CIA_test_predicteddata16'.format(args.model), save_impute)
        np.save('D:/studydata/OSC/data/CDL_sub/discuss_rits/result/{}_CDL_test_predicteddata16'.format(args.model),save_impute)
        # np.save('D:/studydata/OSC/data/CDL_sub/result/RITS/{}_CDL_validation_predicteddata'.format(args.model),save_impute)


        # np.save('./result/04/{}_predicteddata_fromsar'.format(args.model), save_impute_z)
        # np.save('./result/{}_label'.format(args.model), save_label)

    # valid_loss = sqrt(metrics.mean_squared_error(evals,imputations))

def predict(model, savepath):

    model.load_state_dict(torch.load(savepath))

    test_data_iter = data_loader.get_test_loader(
        batch_size=args.batch_size)

    model.eval()

    save_impute = []

    for idx, data in enumerate(test_data_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        # save the imputation results which is used to test the improvement of traditional methods with imputed values
        # save_impute.append(ret['imputations'].data.cpu().numpy())
        # save_label.append(ret['labels'].data.cpu().numpy())

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        # save the imputation results which is used to test the improvement of traditional methods with imputed values
        save_impute.append(imputation[np.where(eval_masks == 1)])


    save_impute = np.concatenate(save_impute, axis=0)
    # save_label = np.concatenate(save_label, axis=0)

    np.save('D:/studydata/OSC/data/CIA_sub/simu_gap/result/BRITS/{}_CIA_test_datafilled'.format(args.model), save_impute)


if __name__ == '__main__':
    #模型训练，训练时评价指标
    run_train()

    # evaluate the best model应用训练好的模型，进行预测和保存
    # evaluate_model()

