import os
import torch
import config as cfg
import re
import pickle


def logical_numerical_sort(elem):

    num = int(elem.split(".")[0])
    return num


def recov_from_ckpt():
    if os.path.exists(cfg.ckpt_folder):
        data_list = os.listdir(cfg.ckpt_folder)
        extension = '.pt'
        checkpoints = [ele for ele in data_list if(extension in ele)]
        if len(checkpoints):
            checkpoints.sort(key=lambda x:int((x.split('_')[2]).split('.')[0]))
            if torch.cuda.is_available():
                model = torch.load(os.path.join(cfg.ckpt_folder, checkpoints[-1]))
            else:
                model = torch.load("checkpoints/"+checkpoints[-1], map_location='cpu')

            saved_epoch = int(re.findall(r'\d+', checkpoints[-1])[0])
            print("Resuming from epoch " + str(saved_epoch))
            return [model, saved_epoch]
        else:
            print("No checkpoints available")
    else:
        print("Can't find checkpoints directory")


def append_to_pickle_file(path, item):
    with open(path, 'ab') as file:
        pickle.dump(item, file)

def read_from_pickle_file(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass