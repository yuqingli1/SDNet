import torch
import numpy as np
import random

import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpus}"if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True


seed = 20
setup_seed(seed)

epochs = 100

batch_size = 256

learning_rate = 5e-5
unet_lr = 6e-4
betas = (0.5, 0.9)

# min_loss
min_loss = 1

def cal_ACC_npy(predict, truth):
    vy_ = predict - np.mean(predict, axis=1).reshape((predict.shape[0], 1))
    vy = truth - np.mean(truth, axis=1).reshape((truth.shape[0], 1))
    cc = np.sum(vy_ * vy, axis=-1) / (
            np.sqrt(np.sum(vy_ ** 2, axis=-1)) * np.sqrt(np.sum(vy ** 2, axis=-1)) + 1e-8)
    average_cc = np.mean(cc)
    return average_cc

def cal_ACC_tensor(predict, truth):
    vy_ = predict - torch.mean(predict, dim=1).unsqueeze(1)
    vy = truth - torch.mean(truth, dim=1).unsqueeze(1)
    cc = torch.sum(vy_ * vy, dim=-1) / (
            torch.sqrt(torch.sum(vy_ ** 2, dim=-1)) * torch.sqrt(torch.sum(vy ** 2, dim=-1)) + 1e-8)
    average_cc = torch.mean(cc)
    return average_cc

def caL_rrmse_tensor(predict, truth):
    rrmse = torch.sqrt(torch.sum(torch.square(predict - truth), dim=1))
    rrmse = rrmse / torch.sqrt(torch.sum(torch.square(truth), dim=1))
    return torch.mean(rrmse, dim=0).item()

def caL_rrmse_npy(predict, truth):
    rrmse = np.sqrt(np.sum(np.square(predict - truth), axis=1))
    rrmse = rrmse / np.sqrt(np.sum(np.square(truth), axis=1))
    return np.mean(rrmse, axis=0)


def cal_SNR(predict, truth):
    if isinstance(predict, torch.Tensor) and isinstance(truth, torch.Tensor):
        PS = torch.sum(torch.square(truth), dim=1)  # power of signal
        PN = torch.sum(torch.square((predict - truth)), dim=1)  # power of noise
        ratio = PS / PN
        SNR = 10 * torch.log10(ratio)
        SNR = torch.mean(SNR, dim=0).item()
    else:
        PS = np.sum(np.square(truth), axis=1)  # power of signal
        PN = np.sum(np.square((predict - truth)), axis=1)  # power of noise
        ratio = PS / PN
        SNR = 10 * np.log10(ratio)
        SNR = np.mean(SNR, axis=0)
    return SNR

