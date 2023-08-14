import os
import datetime
import numpy as np
from tqdm import tqdm
import pickle

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
#from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import Model, Model_no_transformer, Model_no_Ablang
from data import MyDataset
from util import *
from torchvision.ops.focal_loss import sigmoid_focal_loss

class Runner:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Run on device: %s' % self.device)
        self.criterion = sigmoid_focal_loss
        self.model = Model()
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            print(f'Loaded model from {model_path}')
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-5)
#        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=100)


    def train(self, train_loader):
        train_loss = 0
        for batch in train_loader:
            inputs, masks, labels = batch['x'].to(self.device), batch['mask'].to(self.device), batch['y'].to(self.device)
            token = batch['token'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs, masks, token)
            loss = self.criterion(outputs, labels, reduction='mean')
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        return train_loss


    def val(self, val_loader):
        self.model.eval()
        val_loss = 0
        outputs_list, labels_list = np.array([]), np.array([])
        with torch.no_grad():
            for batch in val_loader:
                inputs, masks, labels = batch['x'].to(self.device), batch['mask'].to(self.device), batch['y'].to(self.device)
                token = batch['token'].to(self.device)
                outputs = self.model(inputs, masks, token)
                loss = self.criterion(outputs, labels, reduction='mean')
                val_loss += loss.item()
                outputs_list = np.append(outputs_list, outputs.cpu().numpy()[:,1])
                labels_list = np.append(labels_list, labels.cpu().numpy()[:,1])
        val_loss /= len(val_loader)
#        self.scheduler.step(val_loss)
        return val_loss, outputs_list, labels_list
        

def train_eval(train_file, save_dir, feature_dir):
    dataset = MyDataset(train_file, feature_dir)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])        
    print('Train dataset', len(train_dataset))
    print('Val dataset', len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    best_loss, best_idx = 1, -1
    train_loss_list, val_loss_list, auc_list = [], [], []
    total_epoch, patience = 300, 20

    runner = Runner()
    with tqdm(total=total_epoch) as pbar:
        for epoch in range(total_epoch):
            train_loss = runner.train(train_loader)
            val_loss, outputs, labels = runner.val(val_loader)
            auc = compute_auc(labels, outputs)

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            auc_list.append(auc)

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                best_idx = epoch
                torch.save(runner.model.state_dict(), save_dir)
                counter = 0
            else:
                counter += 1
                if counter > patience:
                    break

            pbar.set_postfix(train_loss=train_loss, val_loss=val_loss)
            pbar.update()
            if epoch % 5 == 4:
                draw_loss(train_loss_list, val_loss_list)
                draw_auc_epoch(auc_list)
    print(f"Best model is from epoch {best_idx}: {round(auc,3)}")

def predict(test_file, model_path, feature_dir):
    test_dataset = MyDataset(test_file, feature_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    runner = Runner(model_path=model_path)
    val_loss, outputs, labels = runner.val(test_loader)
    return outputs


def Bagging(train_file, test_file, feature_dir, k):
    num_models = 5
    for idx in range(num_models):
        train_eval(train_file, f'tmp/best_model_{k}_{idx}.pth', feature_dir)
    if type(test_file) != list:
        test_file_list = [test_file]
    else:
        test_file_list = test_file
        
    res_all = []
    for test in test_file_list:
        print(test)
        res = []
        for idx in range(num_models):
            result = predict(test, f'tmp/best_model_{k}_{idx}.pth', feature_dir)
            print(result.shape)
            res.append(result)
        res = np.mean(res, axis=0)
        res_all.append(res)
    if type(test_file) != list:
        return res
    else:
        return res_all


def run_LICTOR():
    str_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    feature_dir = 'dataset/LICTOR_fasta/protein_dict.pkl'
    for i in range(1, 11):
        train_file = f'dataset/LICTOR_fasta/10fold/train_{i}.fasta'
        test_file = f'dataset/LICTOR_fasta/10fold/test_{i}.fasta'
        result = Bagging(train_file, test_file, feature_dir, i)
        os.makedirs(f'result/LICTOR/WSK_result_{str_time}', exist_ok=True)
        with open(f'result/LICTOR/WSK_result_{str_time}/{i}.pkl', 'wb') as fw:
            pickle.dump(result, fw)
        


def run_fasta(name):
    str_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    feature_dir = f'dataset/{name}/protein_dict.pkl'
    for i in range(1, 11):
        train_file = f'dataset/{name}/10fold/train_{i}.fasta'
        test_file = f'dataset/{name}/10fold/test_{i}.fasta'
        result = Bagging(train_file, test_file, feature_dir, i)
        os.makedirs(f'result/{name}/WSK_result_{str_time}', exist_ok=True)
        with open(f'result/{name}/WSK_result_{str_time}/{i}.pkl', 'wb') as fw:
            pickle.dump(result, fw)

def run_LICTOR_clinical():
    str_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    feature_dir = 'dataset/LICTOR_clinical'    
    train_file = f'dataset/LICTOR_clinical/train.txt'
    test_file = f'dataset/LICTOR_clinical/test.fasta'
    result = Bagging(train_file, test_file, feature_dir, 0)
    os.makedirs(f'result/LICTOR_clinical/WSK_result_{str_time}', exist_ok=True)
    with open(f'result/LICTOR_clinical/WSK_result_{str_time}/0.pkl', 'wb') as fw:
        pickle.dump(result, fw)

def run_AL_base():
    str_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    feature_dir = 'dataset/AL-base'
    for i in range(1, 11):
        train_file1 = f'dataset/AL-base/kappa/train_{i}.fasta'
        train_file2 = f'dataset/AL-base/lambda/train_{i}.fasta'
        test_file1 = f'dataset/AL-base/kappa/test_{i}.fasta'
        test_file2 = f'dataset/AL-base/lambda/test_{i}.fasta'
        result = Bagging([train_file1, train_file2], [test_file1, test_file2], feature_dir, 0)
        os.makedirs(f'result/AL-base/both_{str_time}', exist_ok=True)
        with open(f'result/AL-base/both_{str_time}/{i}.pkl', 'wb') as fw:
            pickle.dump(result, fw)
    '''
    for i in range(1, 11):
        train_file = f'dataset/AL-base/kappa/train_{i}.fasta'
        test_file1 = f'dataset/AL-base/kappa/test_{i}.fasta'
        test_file2 = f'dataset/AL-base/lambda/test_{i}.fasta'
        result = Bagging(train_file, [test_file1, test_file2], feature_dir, 0)
        with open(f'result/AL-base/kappa/{i}.pkl', 'wb') as fw:
            pickle.dump(result, fw)
    for i in range(1, 11):
        train_file = f'dataset/AL-base/lambda/train_{i}.fasta'
        test_file1 = f'dataset/AL-base/kappa/test_{i}.fasta'
        test_file2 = f'dataset/AL-base/lambda/test_{i}.fasta'
        result = Bagging(train_file, [test_file1, test_file2], feature_dir, 0)
        with open(f'result/AL-base/lambda/{i}.pkl', 'wb') as fw:
            pickle.dump(result, fw)
    '''
    
def run_ensemble():
    feature_dir = 'dataset/Combine'    
    train_file = f'dataset/Combine/train.fasta'
    test_file = f'dataset/Combine/test.fasta'
    '''
    for idx in range(5):
        train_eval(train_file, f'tmp/best_model_0_{idx}.pth', feature_dir)
    '''
    res = []
    for idx in range(5):
        result = predict(test_file, f'tmp/best_model_0_{idx}.pth', feature_dir)
        res.append(result)
        result = np.mean(res, axis=0)
        with open(f'result/ensemble/{idx}.pkl', 'wb') as fw:
            pickle.dump(result, fw)

if __name__ == '__main__':
#    run_LICTOR()
#    run_fasta('RFAmy')
    run_fasta('VLAmy')
#    run_fasta('Kappa')
#    run_AL_base()
#    run_ensemble()