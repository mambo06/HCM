

import copy
import time
import gc
from tqdm import tqdm

import yaml

import evaluations as eval
from src.model import CFL
from utils.arguments import get_arguments, get_config, print_config_summary
from utils.load_data import Loader
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims
import numpy as np

import torch

from torch.multiprocessing import Process
import os
# import torch.distributed as dist
import datetime
from itertools import islice

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier



from sklearn.metrics import r2_score
from utils.eval_utils import linear_model_eval, plot_clusters, append_tensors_to_lists, concatenate_lists, aggregate
import json
from pathlib import Path

from utils.projection_utils import head, projection




def run(config, modelCoventional, save_weights=True):
    """Utility function for training and saving the model.
    Args:
        config (dict): Dictionary containing options and arguments.
        data_loader (IterableDataset): Pytorch data loader.
        save_weights (bool): Saves model if True.

    """
    # # Instantiate model
    models = []
    ds_loaders = []
   
    model = CFL(config)
    data = Loader(config, dataset_name=config["dataset"]).train_loader
    model.modelConventional = modelCoventional
    model.save_conventional()

    data_val = Loader(config, dataset_name=config["dataset"]).validation_loader

    data_test = Loader(config, dataset_name=config["dataset"]).test_loader


    loss = {"tloss_b": [], "tloss_e": [], "vloss_e": [],
                     "closs_b": [], "rloss_b": [], "zloss_b": []}
       
   
    print("Training for :",config["epochs"], ' epochs')
    
    clf = model.modelConventional
    best_epoch = 0
    best_score = 1000 if config['task_type'] == 'regression' else 0
    best_loss = 1000
    patient = 0
    
    start0 = True
    for epoch in range(config["epochs"]):
        epoch_loss = []
        r_loss = []
        # start = time.process_time()
        tqdm_bar = tqdm(enumerate(data), 
            total=len(data), 
            leave=True, 
            desc = 'Training on epoch: ' + str(epoch))

        # tqdm_bar = tqdm(range(len(data)), desc = 'Training on epoch: ' + str(epoch))
        if start0 == True : start0 = time.process_time()
        for i, (x, y) in tqdm_bar:
        # for i in tqdm_bar:
            
            model.optimizer_ae.zero_grad()
            
            loss, rLoss = model.fit(x,y)

            model.loss["tloss_o"].append(loss.item())
    

            epoch_loss.append(loss.item())
            r_loss.append(rLoss)
            
            

            loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1.0)

            model.optimizer_ae.step()
            
            
            # description = 'loss {0:.2f}'.format(loss.item())
            # tqdm_bar.set_description(description)
            if (i == len(data)-1): 
                epoch_loss = sum(epoch_loss)/len(epoch_loss)
                factors = epoch_loss * np.mean(r_loss)
                description = 'Epoch {0} loss {1:.1e} error {2:.2f} factors {3:.2f}'.format(
                    str(epoch),
                    epoch_loss, 
                    np.mean(r_loss),
                    factors
                    )
                tqdm_bar.set_description(description)

                

        if config['validation']:
            epoch_val_loss = []

            z_l, clabels_l = [], []
            tqdm_bar__ = tqdm(enumerate(data_val), 
            total=len(data_val), 
            leave=True, 
            desc = 'validation ')
            for i, (x, label) in tqdm_bar__:


                val_loss_s, _= model.fit(x, label)

                epoch_val_loss.append(val_loss_s.item())
            
                description = 'tloss {0:.2f} '.format(val_loss_s.item())
                # tqdm_bar__.set_description(description)

                del val_loss_s

                val_loss = sum(epoch_val_loss)/len(epoch_val_loss)

                if config['validateScore'] : 
                # if validation using score instead of loss
                    x_tilde_list = model.subset_generator(x)

                    latent_list = []
                    

                    # Extract embeddings (i.e. latent) for each subset
                    for xi in x_tilde_list:
                        # Turn xi to tensor, and move it to the device
                        Xbatch = model._tensor(xi)
                        # Extract latent
                        latent,_= model.encoder(Xbatch) # decoded
                        # Collect latent
                        latent_list.append(latent)

                        
                    # Aggregation of latent representations
                    latent = aggregate(latent_list, config)
                    # Append tensors to the corresponding lists as numpy arrays
                    if config['task_type'] == 'regression':
                        label = label
                    else : label = label.int()
                    z_l, clabels_l = append_tensors_to_lists([z_l, clabels_l],
                                                             [latent.detach(), label])

            

            if config['validateScore'] :
                tqdm_bar__ = tqdm_bar__
                z_test = concatenate_lists([z_l])
                y_test = concatenate_lists([clabels_l])

                # y_std = np.std(y_train)
                # clf.fit(z_train, y_train)
                ŷ = model.modelConventional.predict(z_test)
                # scr = clf.score(z_test, y_test)
                scr = np.sqrt(mean_squared_error(y_test, ŷ)) 

                typeTrain = False
                typeTrain = True if ((config['task_type'] == 'regression') and (scr < best_score ) ) else typeTrain
                typeTrain = True if ((config['task_type'] != 'regression') and (scr > best_score ) ) else typeTrain
                if typeTrain :

                    best_score =  scr
                    best_epoch = epoch
                    patient = 0
                    model.saveTrainParams()
                    tqdm_bar__.set_description('validation on epoch {0} with {1} score {2:.2f}'.format(best_epoch, config['task_type'], best_score))
                    
                    # model.saveTrainParams()
                    _ = model.save_weights() if save_weights else None

                    # Save the config file to keep a record of the settings
                    prefix = str(config['epochs']) + "e-" + str(config["dataset"])

                    with open(model._results_path + "/config_"+ prefix +".yml", 'w') as config_file:
                        yaml.dump(config, config_file, default_flow_style=False)

                else:
                    patient += 1

                if patient-1 == config['patient']:
                    tqdm_bar__.set_description('validation exit on epoch {0} with accuracy {1:.2f}'.format(best_epoch, best_score))
                    break

            else:
                tqdm_bar__ = tqdm_bar__
                if best_loss > val_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    model.saveTrainParams()
                    tqdm_bar__.set_description('Validationon epoch {0} with loss {1:.2f}'.format(best_epoch, best_loss))
                    tqdm_bar__.refresh()
                    # print(('Validationon epoch {0} with loss {1:.2f}'.format(best_epoch, best_loss)))
                    # model.saveTrainParams()
                    # Save the model for future use
                    _ = model.save_weights() if save_weights else None

                    # Save the config file to keep a record of the settings
                    prefix = str(config['epochs']) + "e-" + str(config["dataset"])

                    with open(model._results_path + "/config_"+ prefix +".yml", 'w') as config_file:
                        yaml.dump(config, config_file, default_flow_style=False)


        
        _ = model.scheduler.step() if model.options["scheduler"] else None

        if config['reduce_lr']  : 
            model.reducer.step(factors)
            if config['learning_rate_reducer'] != model.reducer.get_last_lr():
                print('Learning Rate :',model.reducer.get_last_lr())
                config['learning_rate_reducer'] = model.reducer.get_last_lr()

        # training_time = time.process_time() - start

    start1 = time.process_time()
    print("Total Training Time :", start1 - start0)

    # if best_epoch < 15 :

    #     _ = model.save_weights() if save_weights else None

    #     # Save the config file to keep a record of the settings
    #     prefix = str(config['epochs']) + "e-" + str(config["dataset"])

    #     with open(model._results_path + "/config_"+ prefix +".yml", 'w') as config_file:
    #         yaml.dump(config, config_file, default_flow_style=False)
    _ = model.save_weights() if save_weights else None

    # Save the config file to keep a record of the settings
    prefix = str(config['epochs']) + "e-" + str(config["dataset"])

    with open(model._results_path + "/config_"+ prefix +".yml", 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

        


 

def main(config):
    """Main wrapper function for training routine.

    Args:
        config (dict): Dictionary containing options and arguments.

    """
    # Set directories (or create if they don't exist)
    set_dirs(config)
    # set model conventional
    if config['task_type'] == 'regression':
        modelCoventional = LinearRegression()
        # modelCoventional = SVR()
        # modelCoventional = KNeighborsRegressor(n_neighbors=90)
    else:
        modelCoventional = LogisticRegression( C=0.01, solver='lbfgs', multi_class='multinomial')
    # Get data loader for first dataset.
    # print('train :',modelCoventional)
    ds_loader = Loader(config, dataset_name=config["dataset"], modelCoventional = modelCoventional)
    modelCoventional = ds_loader.modelCoventional
    # Add the number of features in a dataset as the first dimension of the model
    config = update_config_with_model_dims(ds_loader, config)
    # Start training and save model weights at the end
    run(config,modelCoventional, save_weights=True)


if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
    # Overwrite the parent folder name for saving results
    config["framework"] = config["dataset"]
    config['task_type'] = json.loads(Path('data/'+config["dataset"]+'/info.json').read_text())['task_type']
    config['cat_policy'] = json.loads(Path('data/'+config["dataset"]+'/info.json').read_text())['cat_policy']
    config['norm'] = json.loads(Path('data/'+config["dataset"]+'/info.json').read_text())['norm']
    config['learning_rate_reducer'] = config['learning_rate']
    # Get a copy of autoencoder dimensions
    dims = copy.deepcopy(config["dims"])
    cfg = copy.deepcopy(config)
    main(config)
    eval.main(config)
    

    
    
    

