import pickle
import numpy as np
from pyteomics import fasta
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, filename, feature_dir):
        self._amino_acids = ['-', 'G','A','S','P','V','T','C','L','I','N','D','Q','K','E','M','H','F','R','Y','W']
        self.AA_token = {aa:idx for idx, aa in enumerate(self._amino_acids)}
        if type(filename) == list:
            file_type = filename[0].split('.')[-1]
        else:
            file_type = filename.split('.')[-1]
        self.protein_dict= pickle.load(open(feature_dir, 'rb'))
        if file_type == 'fasta':
            feat_list, y_list = self.load_fasta(filename)
        else:
            feat_list, y_list = self.load_data(filename)
        self.data = []
        for feat, y in zip(feat_list, y_list):
            x = feat['protein'][0]
            seq = feat['protein'][1]
            token = np.array([self.AA_token[seq[i]] if i < len(seq) else 0 for i in range(128)])
#            mask = np.concatenate([np.zeros(len(x)), np.ones(128 - len(x))])
            x = np.pad(x, ((0, 128 - len(x)), (0, 0)), 'constant', constant_values=(0))
            mask = (np.count_nonzero(x, axis = 1)) == 0
            self.data.append({
                'x': x,
                'token': token,
                'mask': mask,
                'y': np.array([0, 1], dtype=np.float32) if y == 1 else np.array([1, 0], dtype=np.float32)
            })
    
    def load_data(self, filename):
        inputs, outputs = [], []
        with open(filename, 'r') as fr:
            lines = fr.readlines()
            cnt = 0
            for idx, l in enumerate(lines):
                line = l.strip()
                if idx % 6 == 0:
                    name = line.split(' ')[0]
                    pos = name.rfind('IG')
                    output = 1 if (line.split(' ')[-1] == 'tox') else 0
                    outputs.append(output)
                if idx % 6 == 1:
                    seq = line.replace('-', '')
                    protein = self.protein_dict[seq]
                # end of one input
                if idx % 6 == 5:
                    cnt += 1
                    inputs.append({
                        'protein': protein,
                    })
        return inputs, outputs

    def load_fasta(self, filenames):
        inputs, outputs = [], []
        if type(filenames) != list:
            filenames = [filenames]
        for filename in filenames:
            dataset = fasta.read(filename)
            cnt = 0
            for x in dataset:
                name, seq = x.description, x.sequence
                output = name[-1] == '1'
                cnt += 1
                if name not in self.protein_dict:
                    print(f'{name} not in protein dict')
                inputs.append({
                    'protein': self.protein_dict[name],
                })
                outputs.append(output)
        return inputs, outputs


    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    

def test_LICTOR(name, feature_name):
    i = 1
    train_file = f'dataset/{name}/10fold/train_{i}.txt'
    test_file = f'dataset/{name}/10fold/test_{i}.txt'
    train = MyDataset(train_file, feature_name)
    loader = DataLoader(train, 32, shuffle=True)
    for batch in loader:
        print(batch['x'].shape, batch['mask'].shape, batch['protein_id'].shape, batch['y'].shape)
        break

def test_fasta(name, feature_name):
    i = 1
    train_file = f'dataset/{name}/10fold/train_{i}.fasta'
    test_file = f'dataset/{name}/10fold/test_{i}.fasta'
    train = MyDataset(train_file, feature_name)
    loader = DataLoader(train, 32, shuffle=True)
    for batch in loader:
        print(batch['x'].shape, batch['mask'].shape, batch['y'].shape)
        print(batch['token'].shape)
        break

if __name__ == '__main__':
#    test_LICTOR('LICTOR', 'dataset/LICTOR/dump_features')
    test_fasta('LICTOR_fasta', 'dataset/LICTOR_fasta/protein_dict.pkl')
#    test_fasta('VLAmy', 'dataset/VLAmy')