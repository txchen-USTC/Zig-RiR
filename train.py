import argparse
import torch.backends.cudnn as cudnn
import setproctitle
from dataset import Data
from models.Zig_RiR import ZRiR
import torch.nn as nn
from torch.utils.data import DataLoader
import torch




class CrossEntropyLoss(nn.Module):
    def __init__(self, weights=None, ignore_index=255):
        super(CrossEntropyLoss, self).__init__()
        if weights is not None:
            weights = torch.from_numpy(np.array(weights)).float().cuda()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weights)

    def forward(self, prediction, label):
        loss = self.ce_loss(prediction, label)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            temp_prob = torch.unsqueeze(temp_prob, 1)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class loss(nn.Module):
    def __init__(self, model, args2):
        super(loss, self).__init__()
        self.model = model
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(args2.nclass)
    def forward(self, input, label, train):
        output = self.model(input)
        if train:
            loss = self.dice_loss(output, label.long()) + self.ce_loss(output, label.long())
            return loss
        else:
            return output


def get_model(args2):
    model = ZRiR(channels=[64, 128, 256, 512], num_classes=args2.nclass, img_size=args2.crop_size[0], in_chans=3)
    model = loss(model, args2)
    model = model.cuda()
    return model


def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, warmup_iter=None, power=0.9):
    if warmup_iter is not None and cur_iters < warmup_iter:
        lr = base_lr * cur_iters / (warmup_iter + 1e-8)
    elif warmup_iter is not None:
        lr = base_lr * ((1-float(cur_iters - warmup_iter) / (max_iters - warmup_iter))**(power))
    else:
        lr = base_lr * ((1 - float(cur_iters / max_iters)) ** (power))
    optimizer.param_groups[0]['lr'] = lr


def train():
    args2 = parse_args()
    model = get_model(args2)
    data_train = Data(train=True, dataset=args2.dataset, crop_szie=args2.crop_size)
    dataloader_train = DataLoader(
        data_train,
        batch_size=args2.train_batchsize,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        sampler=None)
    optimizer = torch.optim.AdamW([{'params':
                                        filter(lambda p: p.requires_grad,
                                               model.parameters()),
                                    'lr': args2.lr}],
                                  lr=args2.lr,
                                  betas=(0.9, 0.999),
                                  eps=1e-08,
                                  weight_decay=0.0001,
                                  )

    for epoch in range(args2.end_epoch):
        model.train()
        setproctitle.setproctitle("Zig_RiR:" + str(epoch) + "/" + "{}".format(args2.end_epoch))
        for i, sample in enumerate(dataloader_train):
            image, label = sample['image'], sample['label']
            image, label = image.cuda(), label.cuda()
            label = label.long().squeeze(1)
            losses = model(image, label, True)
            loss = losses.mean()
            lenth_iter = len(dataloader_train)
            adjust_learning_rate(optimizer,
                                args2.lr,
                                args2.end_epoch * lenth_iter,
                                i + epoch * lenth_iter,
                                args2.warm_epochs * lenth_iter
                                )
            model.zero_grad()
            loss.backward()
            optimizer.step()


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--end_epoch", type=int, default=3)
    parser.add_argument("--warm_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--train_batchsize", type=int, default=1)
    parser.add_argument("--crop_size", type=int, nargs='+', default=[256, 256], help='H, W')
    parser.add_argument("--nclass", type=int, default=2)
    args2 = parser.parse_args()
    return args2



if __name__ == '__main__':
    cudnn.benchmark = True
    cudnn.enabled = True
    train()



