import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import random


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        # 7x7的平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # 特征提取层，定义在下面
        x = self.features(x)
        # todo:使用CAM的时候需要把下面两行注释去掉
        # x.register_hook(self.save_gradient)
        # self.feature_maps = x
        x = self.avgpool(x)
        # 高维的特征表示转换为一维向量，以便输入到模型的分类器
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def save_gradient(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.feature_maps


# 特征提取，数字n代表卷n个通道为3的3x3的卷积核；'M'代表最大化池化，2x2，步长为2，图像尺寸减半
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            # 最大化池化，2x2，步长为2
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 输出通道为n的3x3的卷积核
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def sep_vgg16(num_classes, pretrained=False):
    model = VGG(make_layers(cfg), num_classes)
    if pretrained:
        model_path = 'E:/project/logs/best_epoch_weights.pth'
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

    # # 此处提取前30个层，事实上18个就足够
    # features = list(model.features)[:30]
    #
    # features = nn.Sequential(*features, model.avgpool, model.classifier)
    # return features
    return model


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def gt_loss(model, images, labels):
    output = model.forward(images)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, torch.tensor(list(map(int, labels))).to('cuda:0'))
    return loss


def compute_cam(model, input_tensor, target_class):
    model.eval()
    output = model(input_tensor)

    # Backward pass to get gradients
    model.zero_grad()
    target = output[0][target_class]
    target.backward()

    # Get the gradients and the feature maps
    gradients = model.get_activations_gradient()
    feature_maps = model.get_activations(input_tensor).squeeze().cpu().data.numpy()

    # Global average pooling to get weights
    weights = np.mean(gradients.squeeze().cpu().data.numpy(), axis=(1, 2))

    # Compute CAM
    cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * feature_maps[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam, torch.argmax(output, dim=1).item()


def visualize_cam(cam, img_path, true_label, predicted_label):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_img = heatmap + np.float32(img) / 255
    cam_img = cam_img / np.max(cam_img)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display original image
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title("Original Image")

    # Display CAM image
    ax[1].imshow(cam_img)
    ax[1].axis('off')
    ax[1].set_title(f"CAM\nTrue: {true_label}, Pred: {predicted_label}")
    plt.show()


if __name__ == "__main__":
    # Load model
    num_classes = 6
    model = sep_vgg16(num_classes, pretrained=True)

    # Load and preprocess image
    train_annotation_path = 'train.txt'
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()

    for _ in range(10):
        i = random.randint(0, 3000)
        img_path = train_lines[i].split()[0]
        preprocess = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(img_path)
        input_tensor = preprocess(img).unsqueeze(0)

        # Compute CAM
        target_class = int(train_lines[i].split()[1])  # Change to the class you want to visualize
        cam, output = compute_cam(model, input_tensor, target_class)

        # Visualize CAM
        visualize_cam(cam, img_path, target_class, output)
