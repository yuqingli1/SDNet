import numpy as np
import gc
import torch.optim.lr_scheduler
from config import *
from datetime import datetime
from dataloader import *
from network import *
from torch.utils.tensorboard import SummaryWriter

# load Unet
Unet = ResUnet(1)
Unet.to(device)

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
writer_path = "./log/"+TIMESTAMP

denoise_net = SCNN()
denoise_net = denoise_net.to(device)


loss_l2 = torch.nn.MSELoss()
loss_bce = torch.nn.BCELoss()


optimizer = torch.optim.Adam([{'params': denoise_net.parameters(), 'lr': learning_rate, 'betas': betas},
                              {'params': Unet.parameters(), 'lr': unet_lr, 'betas': betas}])

# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

total_train_step = 0
total_val_step = 0


val_frequency = 1

for epoch in range(epochs):

    total_train_loss_per_epoch = 0
    average_train_loss_per_epoch = 0
    train_step_num = 0

    # train
    Unet.train()
    denoise_net.train()
    for batch_idx, data in enumerate(train_dataloader):
        train_step_num += 1

        x, y, location = data
        x = x.float().to(device)
        y = y.float().to(device)
        location = location.float().to(device)

        mask = Unet(x)
        outputs = denoise_net(x)

        loss_mse = loss_l2(outputs, y)  # 降噪损失

        outputs = mask.ge(0.5) * outputs + mask.le(0.5) * x

        loss_seg = loss_bce(mask, location)

        loss = loss_mse + loss_seg

        total_train_loss_per_epoch = total_train_loss_per_epoch + loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

    average_train_loss_per_epoch = total_train_loss_per_epoch / train_step_num
    if epoch % val_frequency == 0:
        print("epoch:", epoch)
        print('train loss: ', average_train_loss_per_epoch)
        with SummaryWriter(writer_path) as writer:
            writer.add_scalar("train_loss", average_train_loss_per_epoch, global_step=epoch)

    # test
    val_step_num = 0
    total_val_loss_per_epoch = 0
    average_val_loss_per_epoch = 0

    total_val_Unet_accuracy = 0
    sum_cc = 0
    sum_rrmse = 0

    Unet.eval()
    denoise_net.eval()

    with torch.no_grad():
        if epoch % val_frequency == 0:
            for batch_idx, data in enumerate(val_dataloader):
                val_step_num += 1

                x, y, location = data
                x = x.float().to(device)
                y = y.float().to(device)
                location = location.float().to(device)

                mask = Unet(x)
                outputs = denoise_net(x)

                outputs = mask.ge(0.5) * outputs + mask.le(0.5) * x

                loss_mse = loss_l2(outputs, y)
                loss_seg = loss_bce(mask, location)

                loss = loss_mse + loss_seg

                total_val_loss_per_epoch += loss_mse.item()

                # 计算ACC
                cc = cal_ACC_tensor(outputs.detach(), y.detach())
                sum_cc += cc
                # 计算RRMSE
                rrmse = caL_rrmse_tensor(outputs.detach(), y.detach())
                sum_rrmse += rrmse

            average_val_loss_per_epoch = total_val_loss_per_epoch/val_step_num
            print('val loss: ', average_val_loss_per_epoch)

            acc = sum_cc.item() / val_step_num
            rrmse = sum_rrmse / val_step_num

            print('[val Average CC: ', acc, '], [val_RRMSE:', rrmse, ']')

            with SummaryWriter(writer_path) as writer:
                writer.add_scalar("val_loss", average_val_loss_per_epoch, global_step=epoch)
                writer.add_scalar("val_ACC", acc, global_step=epoch)
                writer.add_scalar("val_RRMSE", rrmse, global_step=epoch)

            if average_val_loss_per_epoch < min_loss:
                print('save model ！')
                torch.save({'Unet':Unet.state_dict(), 'denoise_net':denoise_net.state_dict()},
                           'checkpoint/'+denoise_net.__class__.__name__+""+Unet.__class__.__name__+'_Motion.pkl')
                min_loss = average_val_loss_per_epoch

    gc.collect()
    torch.cuda.empty_cache()



