#!pip3 install pandas scikit-learn matplotlib
import time
import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch import tensor
import pandas as pd
import pickle as pk
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from config import Net, PRINT_LOGS, MLP_INPUT_COLUMNS, COLS_TO_STANDARDIZE, MLP_CLASSIFIER_INPUT_COLUMNS, SESSION_DATASET_SPLIT
from config import CASE_UNKNOWN, CASE_FIXED_WING, CASE_MAVIC_PRO, CASE_PHANTOM_4_PRO, CASE_MAVIC2, CASE_PHANTOM4PRO_MAVIC2, CASE_PHANTOM4PRO_MAVICPRO, Classifications

categories = [[CASE_UNKNOWN],[CASE_FIXED_WING],[CASE_MAVIC_PRO],[CASE_PHANTOM_4_PRO],[CASE_MAVIC2],[CASE_PHANTOM4PRO_MAVIC2],[CASE_PHANTOM4PRO_MAVICPRO]]
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(categories)

print(ohe.transform(categories))

print(ohe)
print(ohe.categories_)

EPOCHS = 50
LEARNING_RATE=1e-3

MINI_BATCHES = 5

TRAIN = True
PREPROCESSING = True
SAVE_LOG = True

if SAVE_LOG:
    f = open('info.log', 'a')

if PREPROCESSING:
    WEIGHTS_PATH = f"uranus_e{EPOCHS}_m{MINI_BATCHES}_PRE_CLASS_PD.pt"
else:
    WEIGHTS_PATH = f"uranus_e{EPOCHS}_m{MINI_BATCHES}_CLASS_PD.pt"

def saveLog(string):
    if SAVE_LOG:
        f.write(string + "\n")
        f.flush()

class Dataset(Dataset):
    def __init__(self, input_file):
        self.csv_file = input_file
        self.outputs = self.csv_file['scenario_name'].values
        self.coordinates = self.csv_file[['utm_x', 'utm_y', 'utm_x2', 'utm_y2', 'alvira_utm_x', 'alvira_utm_y', 'arcus_utm_x', 'arcus_utm_y']]
        self.inputs = self.csv_file[MLP_INPUT_COLUMNS]

        #self.inputs = torch.from_numpy(self.inputs.astype(float64))
        #self.outputs = torch.from_numpy(self.outputs.astype(float64))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        current_row = self.inputs.iloc[idx]
        result = current_row["reference_classification"]

        result = int(result)
        result_onehot = ohe.transform([[result]])

        if PRINT_LOGS:
            print(result)
            print(Classifications(result).name)
            print(result_onehot)

        return current_row[MLP_CLASSIFIER_INPUT_COLUMNS].values, result_onehot

def train2(dataset, validation_set, model, batch_size, max_epochs):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        generator=torch.Generator(device='cuda'),
        shuffle=False
    )

    val_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        generator=torch.Generator(device='cuda'),
        shuffle=False
    )

    loss_values = []
    loss_values_val = []

    for epoch in range(max_epochs):
        # Print epoch
        print(f'Starting epoch {epoch + 1}')
        saveLog(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0
        current_loss_count = 0

        current_acc = 0.0
        current_acc_count = 0

        current_loss_val = 0.0
        current_loss_val_count = 0

        current_acc_val = 0.0
        current_acc_val_count = 0

        model.train()

        for batch, data in enumerate(dataloader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()

            inputs = inputs.to(device)
            targets = targets.to(device)

            targets = targets.reshape(targets.shape[0],-1)

            # Zero the gradients
            optimizer.zero_grad()

            outputs = model(inputs)

            '''if PRINT_LOGS:
                print("outputs:\n", outputs)
                print("targets:\n",targets)'''

            # Compute loss
            loss = loss_func(outputs, targets)
            #loss_values.append(loss.item())

            # Perform backward pass
            loss.backward()

            clip = 5
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            current_loss_count += 1

            #acc = binary_acc(outputs, targets)
            acc = (torch.argmax(targets, 1) == torch.argmax(outputs, 1)).float().mean()

            current_acc += acc.item() * 100
            current_acc_count += 1

        with torch.no_grad():
            for batch_val, data_val in enumerate(val_loader, 0):
                model.eval()

                inputs_val, targets_val = data_val
                inputs_val, targets_val = inputs_val.float(), targets_val.float()

                inputs_val = inputs_val.to(device)
                targets_val = targets_val.to(device)
                targets_val = targets_val.reshape(targets_val.shape[0], -1)

                y_pred = model(inputs_val)
                loss_val = loss_func(y_pred, targets_val)
                #acc_val = binary_acc(y_pred, targets_val)
                acc_val = (torch.argmax(y_pred, 1) == torch.argmax(targets_val, 1)).float().mean()

                current_loss_val += loss_val.item()
                current_loss_val_count += 1

                current_acc_val += acc_val.item() * 100
                current_acc_val_count += 1

        print('TRAIN Loss after epoch %5d: %.3f, ACCURACY %2d' % (epoch + 1, current_loss / current_loss_count, current_acc / current_acc_count))
        print('VALIDATION Loss after epoch %5d: %.3f, ACCURARY %2d' % (epoch + 1, current_loss_val / current_loss_val_count, current_acc_val / current_loss_val_count))

        saveLog('TRAIN Loss after epoch %5d: %.3f' % (epoch + 1, current_loss / current_loss_count))
        saveLog('VALIDATION Loss after epoch %5d: %.3f' % (epoch + 1, current_loss_val / current_loss_val_count))

        loss_values.append(current_loss / current_loss_count)
        loss_values_val.append(current_loss_val / current_loss_val_count)

    #print("outputs:\n", outputs)
    #print("targets:\n", targets)

    plt.plot(np.array(loss_values), color="red")
    plt.plot(np.array(loss_values_val), color="orange")
    plt.savefig('train.png')
    plt.show()

    pd.DataFrame(loss_values).to_csv("lossess_train.csv")
    pd.DataFrame(loss_values_val).to_csv("lossess_val.csv")

    torch.save(model.state_dict(), WEIGHTS_PATH)

# https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/
# Reproducing same results
SEED = 2019
MIN_LOSS = 0.1

# Torch
torch.manual_seed(SEED)

# Cuda algorithms
torch.backends.cudnn.deterministic = True

torch.autograd.set_detect_anomaly(True)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

torch.set_default_tensor_type('torch.cuda.FloatTensor')

net = Net(n_features=12).to(device) # define the network
#net = MulticlassClassification(num_feature=10, num_class=7).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
loss_func = torch.nn.CrossEntropyLoss() # the target label is NOT an one-hotted

x = tensor([[1., 1., 1., 1.]])
#print(net.forward(x))

torch.set_printoptions(precision=10)

dataset_file = '../datasets/merge/dataset_UTM.csv'

csv_file = pd.read_csv(dataset_file)

AlviraTracksTrack_ClassificationENC = LabelEncoder()
csv_file["AlviraTracksTrack_Classification"] = AlviraTracksTrack_ClassificationENC.fit_transform(csv_file["AlviraTracksTrack_Classification"])

AlviraTracksTrack_AlarmENC = LabelEncoder()
csv_file["AlviraTracksTrack_Alarm"] = AlviraTracksTrack_AlarmENC.fit_transform(csv_file["AlviraTracksTrack_Alarm"])

ArcusTracksTrack_ClassificationENC = LabelEncoder()
csv_file["ArcusTracksTrack_Classification"] = ArcusTracksTrack_ClassificationENC.fit_transform(csv_file["ArcusTracksTrack_Classification"])

ArcusTracksTrack_AlarmENC = LabelEncoder()
csv_file["ArcusTracksTrack_Alarm"] = ArcusTracksTrack_AlarmENC.fit_transform(csv_file["ArcusTracksTrack_Alarm"])

DianaTargetsTargetClassification_typeENC = LabelEncoder()
csv_file["DianaTargetsTargetClassification_type"] = DianaTargetsTargetClassification_typeENC.fit_transform(csv_file["DianaTargetsTargetClassification_type"])

VenusTrigger_VenusNameENC = LabelEncoder()
csv_file["VenusTrigger_VenusName"] = VenusTrigger_VenusNameENC.fit_transform(csv_file["VenusTrigger_VenusName"])

VenusTrigge_FrequencyBandENC = LabelEncoder()
csv_file["VenusTrigge_FrequencyBand"] = VenusTrigge_FrequencyBandENC.fit_transform(csv_file["VenusTrigge_FrequencyBand"])

print("AlviraTracksTrack_Classification: ",AlviraTracksTrack_ClassificationENC.classes_)
print(AlviraTracksTrack_AlarmENC.classes_)
print("ArcusTracksTrack_Classification: ", ArcusTracksTrack_ClassificationENC.classes_)
print(ArcusTracksTrack_AlarmENC.classes_)
print("DianaTargetsTargetClassification_type: ", DianaTargetsTargetClassification_typeENC.classes_)
print("VenusTrigger_VenusName: ", VenusTrigger_VenusNameENC.classes_)
print(VenusTrigge_FrequencyBandENC.classes_)

#self.outputs = torch.from_numpy(self.csv_file.iloc[:, 11:18].to_numpy())
#inputs = csv_file.loc[(csv_file["alvira_utm_x"] > 0) | (csv_file["arcus_utm_x"] > 0)].copy()
inputs = csv_file.copy()

print(inputs.head())

train_set, test_set = train_test_split(csv_file, test_size=0.3, shuffle=True)
test_set, val_set = train_test_split(test_set, test_size=0.05, shuffle=True)
train_set.to_csv("train_set.csv", index=False)
val_set.to_csv("val_set.csv", index=False)
test_set.to_csv("test_set.csv", index=False)

scaler = StandardScaler()  # for the first input elements, aka the output ones!
csv_file[COLS_TO_STANDARDIZE] = scaler.fit_transform(csv_file[COLS_TO_STANDARDIZE])
pk.dump(scaler, open('scaler.pkl', 'wb'))

train_set[COLS_TO_STANDARDIZE] = scaler.transform(train_set[COLS_TO_STANDARDIZE])
val_set[COLS_TO_STANDARDIZE] = scaler.transform(val_set[COLS_TO_STANDARDIZE])
test_set[COLS_TO_STANDARDIZE] = scaler.transform(test_set[COLS_TO_STANDARDIZE])

train_obj = Dataset(train_set)
val_obj = Dataset(val_set)
test_obj = Dataset(test_set)

# Grab Currrent Time Before Running the Code
start = time.time()

if TRAIN:
    train2(train_obj, val_obj, net, MINI_BATCHES, EPOCHS)
else:
    net.load_state_dict(torch.load(WEIGHTS_PATH))
    net.eval()

def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def TestEvaluations(model, dataTest, batch_size):
    print("Testing phase...")

    test_loader = DataLoader(
        dataTest,
        batch_size=batch_size,
        shuffle=False
    )

    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for batch, data in enumerate(test_loader, 0):
            model.eval()

            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape(targets.shape[0], -1)

            inputs = inputs.to(device)

            y_pred = model(inputs)
            y_pred = y_pred.cpu()

            y_pred_tag = torch.argmax(y_pred, 1)
            true_targets = torch.argmax(targets, 1)

            for x in y_pred_tag.cpu().numpy().tolist():
                y_pred_list.append(x)

            for x in true_targets.cpu().numpy().tolist():
                y_true_list.append(x)

    print("\n\n*** CONFUSION MATRIX ***")
    print(confusion_matrix(y_true_list, y_pred_list), "\n")
    print("--- CLASSIFICATION REPORTS ---")
    print(classification_report(y_true_list, y_pred_list))

    conf = confusion_matrix(y_true_list, y_pred_list)
    class_report = classification_report(y_true_list, y_pred_list)

    saveLog("\n\n*** CONFUSION MATRIX ***")
    saveLog(str(conf))
    saveLog("--- CLASSIFICATION REPORTS ---")
    saveLog(class_report)

TestEvaluations(net, test_obj, MINI_BATCHES)

# Grab Currrent Time After Running the Code
end = time.time()

#Subtract Start Time from The End Time
total_time = end - start
print("\nTotal time: " + str(total_time))

if SAVE_LOG:
    f.close()