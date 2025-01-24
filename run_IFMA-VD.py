from models import HGNN
from util.common_utils import *
import copy
import os
import torch.optim as optim
from config import get_config
from dhg.structure import Hypergraph
from collections import defaultdict
import torch
from sklearn.preprocessing import LabelEncoder
import csv
from sklearn.preprocessing import normalize
import numpy as np


# Load configuration
cfg = get_config('config/config_IFMA-VD.yaml')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set random seed, the random seed can be adjusted according to the experimental requirements
seed = 321
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def HG_supervised_embedding(X, y, train_index, test_index, G):
    seed = 321
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Transform data to device
    X = torch.Tensor(X).to(device)
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(y)
    # Then convert the integer label array to torch.Tensor
    y = torch.tensor(y_int).long().to(device)
    train_index = torch.Tensor(train_index).long().to(device)
    test_index = torch.Tensor(test_index).long().to(device)
    G = G.to(device)

    # Model initialization
    HGNN_model = HGNN(in_ch=X.shape[1], n_hid=cfg['n_hid'], dropout=cfg['drop_out'])
    HGNN_model = HGNN_model.to(device)
    optimizer = optim.Adam(HGNN_model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
    criterion = torch.nn.CrossEntropyLoss()

    # Model training
    since = time.time()
    num_epochs = cfg['max_epoch']
    print_freq = cfg['print_freq']
    best_model_wts = copy.deepcopy(HGNN_model.state_dict())
    best_f1 = 0.0
    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                HGNN_model.train()  # Set model to training mode
            else:
                HGNN_model.eval()  # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            index = train_index if phase == 'train' else test_index

            # Iterate over data
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs, _ = HGNN_model(X, G)
                loss = criterion(outputs[index], y[index])
                _, preds = torch.max(outputs, 1)

                # Backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Statistics
            running_loss += loss.item() * X.size(0)
            running_corrects += torch.sum(preds[index] == y.data[index])

            epoch_loss = running_loss / len(index)
            epoch_f1 = metrics.f1_score(y.data[index].cpu().detach().numpy(), preds[index].cpu().detach().numpy())

            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(HGNN_model.state_dict())

        if epoch % print_freq == 0:
            print(f'Best val F1: {best_f1:4f}')
            print('-' * 20)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val F1: {best_f1:4f}')

    # Return result
    _, X_embedding = HGNN_model(X, G)

    return X_embedding[train_index].cpu().detach().numpy(), X_embedding[test_index].cpu().detach().numpy()


def load_embeddings_labels_and_G(csv_file_path):
    embeddings = []
    labels = []
    cluster_indices_dict = defaultdict(list)
    train_index = []
    val_index = []
    test_index = []

    with open(csv_file_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)  # Read and store the headers
        split_col_idx = headers.index("split")  # Find the index of the "split" column

        for idx, row in enumerate(csvreader):
            embedding = [float(value) for value in row[3:-1]]
            embeddings.append(embedding)
            labels.append(row[1])
            cluster_labels = row[-1].split(', ')
            for cluster_label in cluster_labels:
                if cluster_label != 'None':
                    cluster_indices_dict[cluster_label.strip()].append(idx)
            # Collect indices based on the values in the "split" column
            if row[split_col_idx] == 'train':
                train_index.append(idx)
            elif row[split_col_idx] == 'test':
                test_index.append(idx)
            elif row[split_col_idx] == 'val':
                val_index.append(idx)
    embeddings = normalize(embeddings, norm='l2')
    X = torch.tensor(embeddings, dtype=torch.float)
    y = np.array(labels)
    ftext_index = test_index

    cluster_indices = [tuple(indices) for indices in cluster_indices_dict.values() if len(indices) > 2]
    G = Hypergraph(num_v=X.shape[0], e_list=cluster_indices)
    return X, y, G, train_index, ftext_index


# Train and test
def train_and_test():
    mcc_list = []
    auc_list = []
    F1_list = []
    precision_list = []
    recall_list = []
    accuracy_list = []

    csv_file_path = 'dataset/' + cfg['project'] + '_with_BHG.csv'

    X, y, G, train_index, test_index = load_embeddings_labels_and_G(csv_file_path)

    X_train, X_test = HG_supervised_embedding(X, y, train_index, test_index, G)

    label_encoder = LabelEncoder()
    # Vulnerability detection
    y_train, y_test = label_encoder.fit_transform(y[train_index]), label_encoder.fit_transform(y[test_index])
    precision, recall, fmeasure, auc, mcc, accuracy = run_evaluation(X_train, y_train, X_test, y_test, cfg)
    mcc_list.append(mcc)
    auc_list.append(auc)
    F1_list.append(fmeasure)
    precision_list.append(precision)
    recall_list.append(recall)
    accuracy_list.append(accuracy)

    avg = []
    avg.append(average_value(precision_list))
    avg.append(average_value(recall_list))
    avg.append(average_value(F1_list))
    avg.append(average_value(auc_list))
    avg.append(average_value(mcc_list))
    avg.append(average_value(accuracy_list))

    name = ['precision', 'recall', 'F1', 'auc', 'mcc', 'accuracy']
    results = []
    results.append(precision_list)
    results.append(recall_list)
    results.append(F1_list)
    results.append(auc_list)
    results.append(mcc_list)
    results.append(accuracy_list)
    df = pd.DataFrame(data=results)
    df.index = name
    df.insert(0, 'avg', avg)

    # If the folder does not exist, create the folder
    save_path = './results/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Record model parameters
    param_suffix = str(cfg['project']) + '_' + str(cfg['n_hid']) + '_' + str(cfg['lr']) + '_' + str(
        cfg['drop_out']) + '_' + str(cfg['max_epoch'])

    df.to_csv(save_path + '/' + param_suffix + '.csv')


opt_project = ['ffmpeg', 'qemu', 'reveal']
opt_n_hid = [32]
opt_lr = [0.001]
opt_drop_out = [0.1]
opt_max_epoch = [100]

import itertools

for params_i in itertools.product(opt_project, opt_n_hid, opt_lr, opt_drop_out, opt_max_epoch):
    cfg['project'] = params_i[0]
    cfg['n_hid'] = params_i[1]
    cfg['lr'] = params_i[2]
    cfg['drop_out'] = params_i[3]
    cfg['max_epoch'] = params_i[4]
    train_and_test()
