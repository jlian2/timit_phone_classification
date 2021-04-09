import numpy as np 
import pandas as pd 
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
import torch.nn as nn

class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

########### model1 ###############
class BasicBlock01(nn.Module):
    def __init__(self, input_dim, output_dim, p=0.5):
        super(BasicBlock01, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
            nn.Dropout(p),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Classifier01(nn.Module):
    def __init__(self, input_dim=429, output_dim=39, hidden_layers=5, hidden_dim=2048):
        super(Classifier01, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock01(input_dim, hidden_dim),
            *[BasicBlock01(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

########### model2 ###############
class BasicBlock02(nn.Module):
    def __init__(self, input_dim, output_dim, p=0.5):
        super(BasicBlock02, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(p),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Classifier02(nn.Module):
    def __init__(self, input_dim=429, output_dim=39, hidden_layers=5, hidden_dim=2048):
        super(Classifier02, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock02(input_dim, hidden_dim),
            *[BasicBlock02(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

########### model3 ###############
class BasicBlock03(nn.Module):
    def __init__(self, input_dim, output_dim, p=0.5):
        super(BasicBlock03, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Classifier03(nn.Module):
    def __init__(self, input_dim=429, output_dim=39, hidden_layers=5, hidden_dim=2048):
        super(Classifier03, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock03(input_dim, hidden_dim),
            *[BasicBlock03(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

def inference(model, device):
    predict = []
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

            for y in test_pred.cpu().numpy():
                predict.append(y)
    return predict

def postprocess(predict):
    cur_phone = predict[0]
    count = 1

    for i in range(1,len(predict)):
        if predict[i] == cur_phone:
            count += 1
        else:
            if count>3:
                count = 1
            else:
                if predict[i]==predict[i-count-1]:
                    predict[i-count:i] = predict[i]
                    tmp = 0
                    k = i-count-2
                    while predict[k] == predict[i]:
                        tmp += 1
                        k -= 1
                    count += tmp
                else:
                    count = 1
            cur_phone = predict[i]
    return predict

def ensemble(preds):
    print('ensemble...')
    new_val = []
    for i in range(test.shape[0]):
        x = preds[:,i]
        m = np.bincount(x).argmax()
        new_val.append(m)
    
    return new_val

def main():
    device = get_device()
    #model1
    net01 = Classifier01().to(device)
    net01.load_state_dict(torch.load('model_01.ckpt'))
    #model2
    net02 = Classifier02().to(device)
    net02.load_state_dict(torch.load('model_02.ckpt'))
    #model3
    net03 = Classifier03().to(device)
    net03.load_state_dict(torch.load('model_03.ckpt'))
    
    print('predicting using model 1...')
    f1 = inference(net01, device)
    f1 = postprocess(np.array(f1))
    print('predicting using model 2...')
    f2 = inference(net02, device)
    f2 = postprocess(np.array(f2))
    print('predicting using model 3...')
    f3 = inference(net03, device)
    f3 = postprocess(np.array(f3))

    new_val = ensemble(np.array([f1,f2,f3]))

    print('writing prediction to output.csv.')
    with open("output.csv", 'w') as f:
        f.write('Id,Class\n')
        for i, y in  enumerate(new_val):
            f.write('{},{}\n'.format(i, y))


if __name__ == '__main__':
    
    BATCH_SIZE = 256
    data_root='../timit_11/'
    test = np.load(data_root + 'test_11.npy')
    test_set = TIMITDataset(test, None)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    main()
