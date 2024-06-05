from PIL import Image
import torch
from vgg16 import sep_vgg16
import numpy as np
from functools import partial
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os

from torch.utils.data import DataLoader
from utils.dataloader import VGGDataset, VGG_dataset_collate
from tqdm import tqdm
from utils.utils import (seed_everything, show_config, worker_init_fn, get_lr)

def preprocess_image(image_path, input_shape=(150, 150)):
    image = Image.open(image_path)

    # 获得图像的高宽与目标高宽
    image_w, image_h = image.size
    h, w = input_shape

    scale = min(w / image_w, h / image_h)
    new_w = int(image_w * scale)
    new_h = int(image_h * scale)
    dx = (w - new_w) // 2
    dy = (h - new_h) // 2

    image = image.resize((new_w, new_h), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    data = np.array(new_image, np.float32)
    data = np.transpose(data, (2, 0, 1))

    return torch.tensor(data, dtype=torch.float).unsqueeze(0)


def predict(model, image_tensor, device):
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
    return torch.argmax(output, dim=1).item()


if __name__ == "__main__":
    model = sep_vgg16(6)


    # model_path = 'E:/project/logs/loss_2024_05_28_10_42_25/ep060-loss1.747-val_loss1.628.pth'
    model_path = 'E:/project/logs/loss_2024_05_28_14_49_01/best_epoch_weights.pth'
    test_annotation_path = 'test.txt'
    output_predictions_path = 'predictions.txt'

    with open(test_annotation_path, encoding='utf-8') as f:
        test_lines = f.readlines()
    num_test = len(test_lines)

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

    model = model.to(device)
    seed = 11
    seed_everything(seed)
    test_dataset = VGGDataset(test_lines, (150, 150), train=False)
    gen = DataLoader(test_dataset, shuffle=True, batch_size=1, pin_memory=True, drop_last=True,
                     collate_fn=VGG_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

    print('Start predicting...')
    true_labels = []
    predictions = []
    count = 0
    for iteration, batch in enumerate(gen):
        print(count)
        count += 1
        # for line in tqdm(test_lines):
        #     line = line.strip()
        #     parts = line.split()
        image = batch[0]
        true_label = int(batch[1][0])
        prediction = predict(model, image, device)
        true_labels.append(true_label)
        predictions.append(prediction)

    with open(output_predictions_path, 'w') as f:
        for true_label, prediction in zip(true_labels, predictions):
            f.write(f"True label: {true_label}, Prediction: {prediction}\n")

    accuracy = accuracy_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions, average='macro')
    precision = precision_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')

    print(f'准确率: {accuracy:.4f}')
    print(f'召回率: {recall:.4f}')
    print(f'精确率: {precision:.4f}')
    print(f'F1分数: {f1:.4f}')
