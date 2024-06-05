import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from vgg16 import sep_vgg16, weights_init, gt_loss

from utils.callbacks import LossHistory
from utils.dataloader import VGGDataset, VGG_dataset_collate
from utils.utils import (seed_everything, show_config, worker_init_fn, get_lr)

from tqdm import tqdm


if __name__ == "__main__":
    pretrained = False

    # model_path = 'E:/project/logs/best_epoch_weights.pth'
    model_path = ''
    save_dir = 'logs/'
    train_annotation_path = 'train.txt'
    val_annotation_path = 'val.txt'

    num_classes = 6

    input_shape = [150, 150]
    Epochs = 60
    batch_size = 4
    # 保存间隔时间
    save_period = 5

    # 模型的最大学习率
    Init_lr = 1e-4
    # 模型的最小学习率
    Min_lr = Init_lr * 0.01

    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0

    # 在训练时进行评估
    eval_flag = True
    # 评估间隔时间
    eval_period = 5

    Cuda = True
    train_gpu = [0, ]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in train_gpu)
    ngpus_per_node = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))

    seed = 11
    seed_everything(seed)

    model = sep_vgg16(num_classes, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    # 记录Loss
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    print(model)
    model_train = model.train()

    if Cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(
        model_path=model_path, input_shape=input_shape,
        Epochs=Epochs, batch_size=batch_size,
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
        save_period=save_period, save_dir=save_dir, num_train=num_train, num_val=num_val
    )

    # 自适应调整学习率
    nbs = 16
    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    train_dataset = VGGDataset(train_lines, input_shape, train=True)
    val_dataset = VGGDataset(val_lines, input_shape, train=False)

    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, drop_last=True,
                     collate_fn=VGG_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, drop_last=True,
                         collate_fn=VGG_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))


    # 开始模型训练
    for epoch in range(0, Epochs):
        total_loss = 0
        val_loss = 0
        print('Start Train')
        with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epochs}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_step:
                    break
                images, labels = batch[0], batch[1]
                with torch.no_grad():
                    if Cuda:
                        images = images.cuda()

                optimizer.zero_grad()

                losses = gt_loss(model, images, labels)
                losses.backward()
                optimizer.step()

                total_loss += losses.item()

                pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

        print('Finish Train')
        print('Start Validation')
        with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epochs}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen_val):
                if iteration >= epoch_step_val:
                    break
                images, labels = batch[0], batch[1]
                with torch.no_grad():
                    if Cuda:
                        images = images.cuda()

                    optimizer.zero_grad()
                    val_total = gt_loss(model, images, labels)
                    val_loss += val_total.item()

                    pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                    pbar.update(1)

        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epochs))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        # 保存权值
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epochs:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
                epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

    loss_history.writer.close()
