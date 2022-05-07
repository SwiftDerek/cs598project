

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument("--process_raw", action='store_true', help="If specified, additionally save trajectories without normalized features")
parser.add_argument("--save_intermediate", action="store_true", help="If specified, save off intermediate tables used to construct final patient table")
pargs = parser.parse_args()



def train_eval(p_dict, phase='train'):



    for i,data in enumerate(tqdm(data_loader)):
        if args.use_visit:
            if args.gpu:
                variable(x.cuda()) for x in data ]loss_output[0]

        function.compute_metric(output, labels, time, classification_loss_output, classification_metric_dict, phase)


        if phase == 'train':
            optimizer.zero_grad()
            loss_gradient.backward()
            optimizer.step()




    print('\nEpoch: {:d} \t Phase: {:s} \n'.format(epoch, phase))
    metric = function.print_metric('classification', classification_metric_dict, phase)
    if args.phase != 'train':
        print 'metric = ', metric
        print
        print
        return


        print('valid: metric: {:3.4f}\t epoch: {:d}\n'.format(metric, epoch))
        print('\t\t\t valid: best_metric: {:3.4f}\t epoch: {:d}\n'.format(p_dict['best_metric'][0], p_dict['best_metric'][1]))  
    else:
        print('train: metric: {:3.4f}\t epoch: {:d}\n'.format(metric, epoch))



def main():
    p_dict = dict() 
    p_dict['args'] = args
    args.split_nn = args.split_num + args.split_nor * 3
    args.vocab_size = args.split_nn * 145 + 1
    print 'vocab_size', args.vocab_size




    train_dataset  = dataloader.DataSet(
                patient_train, 
                patient_time_record_dict,
                patient_label_dict,
                patient_master_dict, 
                args=args,
                phase='train')
    train_loader = DataLoader(
                dataset=train_dataset, 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=8, 
                pin_memory=True)
    val_dataset  = dataloader.DataSet(
                patient_valid, 
                patient_time_record_dict,
                patient_label_dict,
                patient_master_dict, 
                args=args,
                phase='val')
    val_loader = DataLoader(
                dataset=val_dataset, 
                batch_size=args.batch_size,
                shuffle=False, 
                num_workers=8, 
                pin_memory=True)

    p_dict['train_loader'] = train_loader
    p_dict['val_loader'] = val_loader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(1473, 64)
        self.fc2 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# initialize the NN
model = Net()
print(model)

    cudnn.benchmark = True
    net = lstm.LSTM(args)
    if args.gpu:
        net = net.cuda()
        p_dict['loss'] = loss.Loss().cuda()
    else:
        p_dict['loss'] = loss.Loss()



    p_dict['epoch'] = 0
    p_dict['best_metric'] = [0, 0]

