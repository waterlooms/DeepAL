import ablang
import pickle
from abnumber import Chain
import torch
import torch.nn as nn
from pyteomics import fasta

def load_fasta(filename, protein_dict):
    encoder = ablang.pretrained("light") # working with light chains
    encoder.freeze()
    dataset = fasta.read(filename, 'r')
    protein_list, name_list, V_list, J_list = [], [], [], []
    alignment = False
    for x in dataset:
        seq = x.sequence
        seq_name = x.description
        if alignment:
            try:
                chain = Chain(seq, scheme='imgt', assign_germline = True)
                aligned = ''
                for i in range(1, 128):
                    name = f'L{i}'
                    try:
                        aa = chain[name]
                    except:
                        aa = '-'
                    aligned += aa
                if 'X' in aligned or 'Z' in aligned:
                    print('X or Z in sequence')
                    continue
                name_list.append(seq_name)
                protein_list.append(aligned)
            except Exception as e:
                print(e)
                print('Error!', seq_name, seq)
        else:
            name_list.append(seq_name)
            protein_list.append(seq)
    print("Computing features...")
    protein_features = encoder(protein_list, mode = 'rescoding')
    for name, aligned, feature in zip(name_list, protein_list, protein_features):
        protein_dict[name] = [feature, aligned]
    return protein_dict

def dump_features(dataset_name):
    protein_dict = {}
    if dataset_name == 'LICTOR_fasta':
        protein_dict = load_fasta('dataset/LICTOR_fasta/10fold/train_1.fasta', protein_dict)
        protein_dict = load_fasta('dataset/LICTOR_fasta/10fold/test_1.fasta', protein_dict)
        pickle.dump(protein_dict,open('dataset/LICTOR_fasta/protein_dict.pkl', 'wb'))
    if dataset_name == 'VLAmy':
        protein_dict = load_fasta('dataset/VLAmy/10fold/train_1.fasta', protein_dict)
        protein_dict = load_fasta('dataset/VLAmy/10fold/test_1.fasta', protein_dict)
        pickle.dump(protein_dict,open('dataset/VLAmy/protein_dict.pkl', 'wb'))
    if dataset_name == 'LICTOR_clinical':
        protein_dict = load_fasta('dataset/LICTOR_clinical/test.fasta', protein_dict)
        pickle.dump(protein_dict,open('dataset/LICTOR_clinical/protein_dict.pkl', 'wb'))
    

'''
def test_feature():
    protein_list = ['PEP', 'TIDE']
    protein_dict = rand_initial(protein_list)
    print(protein_dict[0].shape)

    encoder = ablang.pretrained("light") # working with light chains
    encoder.freeze()
    protein_features = encoder(protein_list, mode = 'rescoding')
    print(protein_features[0].shape)
'''

if __name__ == '__main__':
#    dump_features('LICTOR_fasta')
#    dump_features('combine')
#    dump_features(aligned=False)
    dump_features('LICTOR_clinical')
#    dump_features_fasta('RFAmy')
#    dump_features_fasta('Kappa')