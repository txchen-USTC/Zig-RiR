import torch
import numpy as np
import torch.nn.functional as F
from train2d import get_model
import argparse
from torch.utils.data import DataLoader
from dataset2d import Data



class Evaluator:
    def __init__(self, cuda=True):
        self.cuda = cuda

        self.MAE = list()
        self.Recall = list()
        self.Precision = list()
        self.Accuracy = list()
        self.Dice = list()
        self.IoU = list()


    def evaluate(self, pred, gt):

        pred_binary = (pred >= 0.5).float().cuda()
        pred_binary_inverse = (pred_binary == 0).float().cuda()
        gt_binary = (gt >= 0.5).float().cuda()
        gt_binary_inverse = (gt_binary == 0).float().cuda()
        MAE = torch.abs(pred_binary - gt_binary).mean().cuda()
        TP = pred_binary.mul(gt_binary).sum().cuda()
        FP = pred_binary.mul(gt_binary_inverse).sum().cuda()
        TN = pred_binary_inverse.mul(gt_binary_inverse).sum().cuda()
        FN = pred_binary_inverse.mul(gt_binary).sum().cuda()
        if TP.item() == 0:
            TP = torch.Tensor([1]).cuda()
        Recall = TP / (TP + FN)
        Precision = TP / (TP + FP)
        Dice = 2 * Precision * Recall / (Precision + Recall)
        Accuracy = (TP + TN) / (TP + FP + FN + TN)
        IoU = TP / (TP + FP + FN)

        return MAE.data.cpu().numpy().squeeze(), Recall.data.cpu().numpy().squeeze(), Precision.data.cpu().numpy().squeeze(), Accuracy.data.cpu().numpy().squeeze(), Dice.data.cpu().numpy().squeeze(), IoU.data.cpu().numpy().squeeze()


    def update(self, pred, gt):
        mae, recall, precision, accuracy, dice, ioU = self.evaluate(pred, gt)
        self.MAE.append(mae)
        self.Recall.append(recall)
        self.Precision.append(precision)
        self.Accuracy.append(accuracy)
        self.Dice.append(dice)
        self.IoU.append(ioU)

    def show(self ,flag = True):
        if flag == True:
            print("MAE:", "%.2f" % (np.mean(self.MAE ) *100) ,"  Recall:", "%.2f" % (np.mean(self.Recall ) *100), "  Pre:", "%.2f" % (np.mean(self.Precision ) *100),
                  "  Acc:", "%.2f" % (np.mean(self.Accuracy ) *100) ,"  Dice:", "%.2f" % (np.mean(self.Dice ) *100)
                  ,"  IoU:" , "%.2f" % (np.mean(self.IoU ) *100))
            print('\n')
        return np.mean(self.MAE ) *100 ,np.mean(self.Recall ) *100 ,np.mean(self.Precision ) *100 ,np.mean \
            (self.Accuracy ) *100 ,np.mean(self.Dice ) *100 ,np.mean(self.IoU) *100


def Eval(dataloader_test, model, args2):

    model.eval()
    if args2.dataset in ['ISIC16', 'ISIC18']:
        evaluator = Evaluator()

    if args2.dataset in ['acdc']:
        evaluator_RV =  Evaluator()
        evaluator_Myo = Evaluator()
        evaluator_LV =  Evaluator()


    if args2.dataset in ['synapse']:
        evaluator_A = Evaluator()
        evaluator_G = Evaluator()
        evaluator_LK = Evaluator()
        evaluator_RK = Evaluator()
        evaluator_L = Evaluator()
        evaluator_P = Evaluator()
        evaluator_Sp = Evaluator()
        evaluator_St = Evaluator()

    with torch.no_grad():
        for i, sample in enumerate(dataloader_test):
            image, label = sample['image'], sample['label']
            image, label = image.cuda(), label.cuda()
            label = label.long().squeeze(1)
            logit = model(image, label, False)


            if args2.dataset in ['ISIC16', 'ISIC18']:


                predictions = torch.argmax(logit, dim=1)
                predictions = F.one_hot(predictions.long(), num_classes=args2.nclass)
                new_labels = F.one_hot(label.long(), num_classes=args2.nclass)
                evaluator.update(predictions[0, :, :, 1], new_labels[0, :, :, 1].float())


            if args2.dataset in ['synapse']:
                predictions = torch.argmax(logit, dim=1)
                pred = F.one_hot(predictions.long(), num_classes=args2.nclass)
                new_labels = F.one_hot(label.long(), num_classes=args2.nclass)

                evaluator_A.update(pred[0, :, :, 1], new_labels[0, :, :, 1].float())
                evaluator_G.update(pred[0, :, :, 2], new_labels[0, :, :, 2].float())
                evaluator_LK.update(pred[0, :, :, 3], new_labels[0, :, :, 3].float())
                evaluator_RK.update(pred[0, :, :, 4], new_labels[0, :, :, 4].float())
                evaluator_L.update(pred[0, :, :, 5], new_labels[0, :, :, 5].float())
                evaluator_P.update(pred[0, :, :, 6], new_labels[0, :, :, 6].float())
                evaluator_Sp.update(pred[0, :, :, 7], new_labels[0, :, :, 7].float())
                evaluator_St.update(pred[0, :, :, 8], new_labels[0, :, :, 8].float())


            if args2.dataset in ['acdc']:
                predictions = torch.argmax(logit, dim=1)
                pred = F.one_hot(predictions.long(), num_classes=args2.nclass)
                new_labels = F.one_hot(label.long(), num_classes=args2.nclass)
                evaluator_RV.update(pred[0, :, :, 1], new_labels[0, :, :, 1].float())
                evaluator_Myo.update(pred[0, :, :, 2], new_labels[0, :, :, 2].float())
                evaluator_LV.update(pred[0, :, :, 3], new_labels[0, :, :, 3].float())



    if args2.dataset in ['ISIC16', 'ISIC18']:
        MAE, Rec, Pre, Acc, Dice, IoU = evaluator.show(False)
        print("MAE: ", "%.2f" % MAE, "  Recall: ", "%.2f" % Rec, " Pre: ", "%.2f" % Pre,
              " Acc: ", "%.2f" % Acc, " Dice: ", "%.2f" % Dice, " IoU: ", "%.2f" % IoU)


    if args2.dataset in ['acdc']:
        MAE_RV, Recall_RV, Pre_RV, Acc_RV, Dice_RV, IoU_RV = evaluator_RV.show(False)
        MAE_Myo, Recall_Myo, Pre_Myo, Acc_Myo, Dice_Myo, IoU_Myo = evaluator_Myo.show(False)
        MAE_LV, Recall_LV, Pre_LV, Acc_LV, Dice_LV, IoU_LV = evaluator_LV.show(False)

        MAE = (MAE_RV + MAE_Myo + MAE_LV) / 3
        Rec = (Recall_RV + Recall_Myo + Recall_LV) / 3
        Pre = (Pre_RV + Pre_Myo + Pre_LV) / 3
        Acc = (Acc_RV + Acc_Myo + Acc_LV) / 3
        Dice = (Dice_RV + Dice_Myo + Dice_LV) / 3
        IoU = (IoU_RV + IoU_Myo + IoU_LV) / 3
        print("MAE: ", "%.2f" % MAE, "  Recall: ", "%.2f" % Rec, " Pre: ", "%.2f" % Pre,
              " Acc: ", "%.2f" % Acc, " Dice: ", "%.2f" % Dice, " IoU: ", "%.2f" % IoU)



    if args2.dataset in ['synapse']:
        MAE_A, Recall_A, Pre_A, Acc_A, Dice_A, IoU_A = evaluator_A.show(False)
        MAE_G, Recall_G, Pre_G, Acc_G, Dice_G, IoU_G = evaluator_G.show(False)
        MAE_LK, Recall_LK, Pre_LK, Acc_LK, Dice_LK, IoU_LK = evaluator_LK.show(False)
        MAE_RK, Recall_RK, Pre_RK, Acc_RK, Dice_RK, IoU_RK = evaluator_RK.show(False)
        MAE_L, Recall_L, Pre_L, Acc_L, Dice_L, IoU_L = evaluator_L.show(False)
        MAE_P, Recall_P, Pre_P, Acc_P, Dice_P, IoU_P = evaluator_P.show(False)
        MAE_Sp, Recall_Sp, Pre_Sp, Acc_Sp, Dice_Sp, IoU_Sp = evaluator_Sp.show(False)
        MAE_St, Recall_St, Pre_St, Acc_St, Dice_St, IoU_St = evaluator_St.show(False)

        MAE = (MAE_A + MAE_G + MAE_LK + MAE_RK + MAE_L + MAE_P + MAE_Sp + MAE_St) / 8
        Rec = (Recall_A + Recall_G + Recall_LK + Recall_RK + Recall_L + Recall_P + Recall_Sp + Recall_St) / 8
        Pre = (Pre_A + Pre_G + Pre_LK + Pre_RK + Pre_L + Pre_P + Pre_Sp + Pre_St) / 8
        Acc = (Acc_A + Acc_G + Acc_LK + Acc_RK + Acc_L + Acc_P + Acc_Sp + Acc_St) / 8
        Dice = (Dice_A + Dice_G + Dice_LK + Dice_RK + Dice_L + Dice_P + Dice_Sp + Dice_St) / 8
        IoU = (IoU_A + IoU_G + IoU_LK + IoU_RK + IoU_L + IoU_P + IoU_Sp + IoU_St) / 8


        print("MAE: ", "%.2f" % MAE, "  Recall: ", "%.2f" % Rec, " Pre: ", "%.2f" % Pre,
              " Acc: ", "%.2f" % Acc, " Dice: ", "%.2f" % Dice, " IoU: ", "%.2f" % IoU)


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--crop_size", type=int, nargs='+', default=[256, 256], help='H, W')
    parser.add_argument("--nclass", type=int, default=2)
    args2 = parser.parse_args()

    return args2


def test():
    args2 = parse_args()
    model = get_model(args2)
    pre_dict = torch.load('./checkpoint.pkl', map_location='cpu')
    model.load_state_dict(pre_dict)

    data_test = Data(train=False, dataset=args2.dataset, crop_szie=args2.crop_size)
    dataloader_test = DataLoader(
        data_test,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=None)
    Eval(dataloader_test, model, args2)


if __name__ == '__main__':
    test()