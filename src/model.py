import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from data import MyDataset
from torchvision.ops.focal_loss import sigmoid_focal_loss

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        dim_in = 768
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_in, 
            nhead=4, 
            dim_feedforward=512, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.linear2 = nn.Linear(dim_in, 2)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, mask, token):
        x = self.transformer_encoder(x, src_key_padding_mask = mask)
        x = x[:, 0, :]
        x = x.squeeze(1)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

class Model_no_transformer(nn.Module):
    def __init__(self):
        super().__init__()
        dim_in = 768
        self.linear2 = nn.Linear(dim_in, 2)
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AvgPool1d(dim_in)
    
    def forward(self, x, mask, token):
        x = torch.mean(x, dim=1)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
    
class Model_no_Ablang(nn.Module):
    def __init__(self):
        super().__init__()
        dim_in = 768
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_in, 
            nhead=4, 
            dim_feedforward=512, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.linear2 = nn.Linear(dim_in, 2)
        self.softmax = nn.Softmax(dim=1)
        self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=dim_in)

    
    def forward(self, x, mask, token):
        x = self.embedding(token)
        x = self.transformer_encoder(x, src_key_padding_mask = mask)
        x = x[:, 0, :]
        x = x.squeeze(1)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
        
def test_LICTOR(name, feature_dir):
    i = 1
    model = Model()
    train_file = f'dataset/{name}/10fold/train_{i}.txt'
    test_file = f'dataset/{name}/10fold/test_{i}.txt'
    dataset = MyDataset(train_file, feature_dir)
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])        
    print('Train dataset', len(train_dataset))
    print('Val dataset', len(val_dataset))

    loader = DataLoader(train_dataset, 32, shuffle=True)
    criterion = sigmoid_focal_loss
    for batch in loader:
        x, mask, y_label = batch['x'], batch['mask'], batch['y']
        y_pred = model(x, mask)
        loss = criterion(y_pred, y_label, reduction='mean')
        print(loss)
        break

def test_fasta(name, feature_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i = 1
#    model = Model()
    model = Model_no_transformer().to(device)
    train_file = f'dataset/{name}/10fold/train_{i}.fasta'
    test_file = f'dataset/{name}/10fold/test_{i}.fasta'
    dataset = MyDataset(train_file, feature_dir)
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])        
    print('Train dataset', len(train_dataset))
    print('Val dataset', len(val_dataset))

    loader = DataLoader(train_dataset, 32, shuffle=True)
    criterion = sigmoid_focal_loss
    for batch in loader:
        x, mask, y_label = batch['x'].to(device), batch['mask'].to(device), batch['y'].to(device)
        seq = batch['token'].to(device)
        y_pred = model(x, mask, seq)
        loss = criterion(y_pred, y_label, reduction='mean')
        print(loss)

if __name__ == '__main__':
#    test_LICTOR('LICTOR', 'dataset/LICTOR/dump_features')
    test_fasta('LICTOR_fasta', 'dataset/LICTOR_fasta/protein_dict.pkl')
#    test_fasta('VLAmy', 'dataset/VLAmy')