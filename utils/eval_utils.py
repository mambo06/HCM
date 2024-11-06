

import csv
import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, HuberRegressor, ElasticNet
from sklearn.model_selection import GridSearchCV



from utils.utils import tsne
from utils.colors import get_color_list
import torch as th
from sklearn.svm import SVC, LinearSVC, SVR
import pickle
import json
import xgboost as xgb

from .regression import regressions  as rgs



def linear_model_eval(config, z_train, y_train, suffix , z_test, y_test,z_val, y_val, description="Logistic Reg."):
    """Evaluates representations using Logistic Regression model.
    Args:
        config (dict): Dictionary that defines options to use
        z_train (numpy.ndarray): Embeddings to be used when plotting clusters for training set
        y_train (list): Class labels for training set
        z_test (numpy.ndarray): Embeddings to be used when plotting clusters for test set
        y_test (list): Class labels for test set
        description (str): Used to print out useful description during evaluation

    """
    results_list = []
    
    # Print out a useful description
    print(10 * ">" + description)

    prefix = str(config['epochs']) + "e-" + str(config["dataset"]) + suffix

    file_name = prefix
    
    # Sweep regularization parameter to see what works best for logistic regression
    regularisation_list = [0.01, 0.1 , 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6]
    regularisation_list = [1, 10,1e2,]

    regularisation_list = range(90,110,5)
    # regularisation_list = [1]

    param_grid = {"max_depth":    [ 8,10,],
              "n_estimators": [900, 1000],
              "learning_rate": [0.01, 0.015]}

    regularisation_list = [1] # overide all
    if config['task_type'] == 'regression':
        regularisation_list = range(90,140,10)
        # regularisation_list = [ 0.01, 0.1, 1, 10, 1e2, 1e3, 1e4]
        regularisation_list = [1, 10,1e2,]
        # regularisation_list = range(90,110,5)
        # regularisation_list = [5,7,9]
        
        regularisation_list = [1]

       

    for c in regularisation_list:
        # Initialize Logistic regression
        print(10 * "*" + "parameters=" + str(c) + 10 * "*")
        if config['task_type'] == 'regression':
            # clf = LinearRegression()
            # clf = SVR()
            # clf = ElasticNet(alpha=c)
            # clf = KNeighborsRegressor(n_neighbors = c, )
            # clf = Ridge(alpha=c)
            # clf=HuberRegressor(alpha=c)
            # clf = RandomForestRegressor(max_depth=c)

            # start xgboost
            # param_grid = {"max_depth": [ 8],
            #   "n_estimators": [ 1000,],
            #   "learning_rate": [0.015]}

            # clf = xgb.XGBRegressor(eval_metric='rmse')
            # search = GridSearchCV(clf, param_grid, cv=2,verbose=2, n_jobs=-1).fit(z_train, y_train)
            # print("The best hyperparameters are ",search.best_params_)

            # clf = xgb.XGBRegressor(learning_rate = search.best_params_["learning_rate"],
            #                n_estimators  = search.best_params_["n_estimators"],
            #                max_depth     = search.best_params_["max_depth"],
            #                eval_metric='rmse')
            # clf = xgb.XGBRegressor(learning_rate = param_grid["learning_rate"][-1],
            #                n_estimators  = param_grid["n_estimators"][-1],
            #                max_depth     = param_grid["max_depth"][-1],
            #                eval_metric='rmse')
            # clf.fit(z_train, y_train,  eval_set=[(z_val, y_val)])
            # end xgboost
            clf = config['modelConventional']
            clf.fit(z_train, y_train)

            #  # Score for training set
            # tr_acc = clf.score(z_train, y_train)
            # # # Score for test set
            # te_acc = clf.score(z_test, y_test)
            # # # Score for test set
            # ve_acc = clf.score(z_val, y_val)
            # print(tr_acc,ve_acc,te_acc)

                # Score for training set
            tr_acc = np.sqrt(mean_squared_error( y_train, clf.predict(z_train))) 

            # # Score for test set
            te_acc = np.sqrt(mean_squared_error(y_test, clf.predict(z_test)))

            # # Score for test set
            ve_acc = np.sqrt(mean_squared_error(y_val, clf.predict(z_val))) 
            # * 1.148042
            print( tr_acc,ve_acc,te_acc)

            results_list.append({"model": "LogReg_" + str(c),
                                 "train_acc": tr_acc,
                                 "test_acc": te_acc,
                                 "val_acc": ve_acc})

            # clf = rgs(z_train,y_train,z_val,y_val,z_test,y_test)
            # result = clf.fit()
            # for modelname, scores in result.items():
            #     for minmax, score in scores.items():
            #         # print('score :',score)
            #         if len(score) == 0 : continue
            #         results_list.append({"model": modelname + '_' + minmax,
            #                      "train_acc": score['train'],
            #                      "test_acc": score['test'],
            #                      "val_acc": score['val']})

        else:
            clf = LogisticRegression( solver='lbfgs', C=c, multi_class='multinomial', max_iter=2000,)
            # clf = DecisionTreeClassifier(random_state=0,criterion='entropy',)
            # clf = RandomForestClassifier(criterion='log_loss', n_estimators=c, )
            # clf = Perceptron(tol=1e-3, random_state=0)        
            # clf = SVC(C=c) 
            # clf = LinearSVC(C=c)
            # Fit model to the data
            # clf = KNeighborsClassifier(n_neighbors=c)

            clf = xgb.XGBClassifier()
            # param_grid = {"max_depth":    [8],
            #       "n_estimators": [ 1000],
            #       "learning_rate": [0.015]}
            search = GridSearchCV(clf, param_grid, cv=2,verbose=2, n_jobs=-1).fit(z_train, y_train)
            print("The best hyperparameters are ",search.best_params_)

            clf = xgb.XGBClassifier(learning_rate = search.best_params_["learning_rate"],
                           n_estimators  = search.best_params_["n_estimators"],
                           max_depth     = search.best_params_["max_depth"],
                           eval_metric='logloss')
            # clf = xgb.XGBClassifier(learning_rate = param_grid["learning_rate"][-1],
            #                n_estimators  = param_grid["n_estimators"][-1],
            #                max_depth     = param_grid["max_depth"][-1],
            #                eval_metric='mlogloss',
            #                # early_stopping_rounds = 10,
            #                verbosity = 2)


            # clf.fit(z_train, y_train,  eval_set=[(z_val, y_val)])
            clf = options['modelConventional']
            clf.fit(z_train, y_train)
        
            y_hat_train = clf.predict(z_train)
            y_hat_test = clf.predict(z_test)
            y_hat_val = clf.predict(z_val)
        
            # Print results
            tr_acc =  precision_recall_fscore_support(y_train, y_hat_train, average='macro')
            val_acc =  precision_recall_fscore_support(y_val, y_hat_val, average='macro')
            te_acc =  precision_recall_fscore_support(y_test, y_hat_test, average='macro')
            print("Training score: precision   {}, recall {}, F1 {}, support {}".format(tr_acc[0],tr_acc[1],tr_acc[2],tr_acc[3]) )
            print("Validation score: precision {}, recall {}, F1 {}, support {}".format(val_acc[0],val_acc[1],val_acc[2],val_acc[3]) )
            print("Test score: precision.      {}, recall {}, F1 {}, support {}".format(te_acc[0],te_acc[1],te_acc[2],te_acc[3]) )
        
        # # Score for training set
        # tr_acc = clf.score(z_train, y_train)
        # # # Score for test set
        # te_acc = clf.score(z_test, y_test)
        # # # Score for test set
        # ve_acc = clf.score(z_val, y_val)
        # print(tr_acc,ve_acc,te_acc)
        
        # Record results
            results_list.append({"model": "LogReg_" + str(c),
                                 "train_acc": tr_acc,
                                 "test_acc": te_acc,
                                 "val_acc": val_acc})
    

    # Save results as a csv file
    keys = results_list[0].keys()
    file_path = './results/'+ config['dataset'] +"/" + file_name + '.csv'
    with open(file_path, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results_list)
        print(f"{100 * '='}\n")
        print(f"Training results are saved at: {file_path}")


def plot_clusters(config, z, clabels, suffix , plot_suffix="_inLatentSpace"):
    """Wrapper function to visualise clusters

    Args:
        config (dict): Dictionary that defines options to use
        z (numpy.ndarray): Embeddings to be used when plotting clusters
        clabels (list): Class labels
        plot_suffix (str): Suffix to use for plot name

    """
    # Number of columns for legends, where each column corresponds to a cluster
    ncol = len(list(set(clabels)))
    # clegends = ["A", "B", "C", "D", ...]..choose first ncol characters, one per cluster
    clegends = list("0123456789")[0:ncol]
    # Show clusters only
    visualise_clusters(config, z, clabels, suffix, plt_name="classes" + plot_suffix, legend_title="Classes",
                       legend_labels=clegends)


def visualise_clusters(config, embeddings, labels, suffix, plt_name="test", alpha=1.0, legend_title=None, legend_labels=None,
                       ncol=1):
    """Function to plot clusters using embeddings from t-SNE and PCA

    Args:
        config (dict): Options and arlguments used
        embeddings (ndarray): Embeddings
        labels (list): Class labels
        plt_name (str): Name to be used for the plot when saving.
        alpha (float): Defines transparency of data poinnts in the scatter plot
        legend_title (str): Legend title
        legend_labels ([str]): Defines labels to use for legends
        ncol (int): Defines number of columns to use for legends of the plot

    """
    # Define colors to be used for each class/cluster
    color_list, _ = get_color_list()
    # Used to adjust space for legends based on number of columns in the legend. ncol: subplot_adjust
    legend_space_adjustment = {"1": 0.9, "2": 0.9, "3": 0.75, "4": 0.65, "5": 0.65}
    # Initialize an empty dictionary to hold the mapping for color palette
    palette = {}
    # Map colors to the indexes.
    for i in range(len(color_list)):
        palette[str(i)] = color_list[i]
    # Make sure that the labels are 1D arrays
    y = labels.reshape(-1, )
    # Turn labels to a list
    y = list(map(str, y.tolist()))
    # Define number of sub-plots to draw. In this case, 2, one for PCA, and one for t-SNE
    img_n = 2
    # Initialize subplots
    fig, axs = plt.subplots(1, img_n, figsize=(9, 3.5), facecolor='w', edgecolor='k')
    # Adjust the whitespace around sub-plots
    fig.subplots_adjust(hspace=.1, wspace=.1)
    # adjust the ticks of axis.
    plt.tick_params(axis='both', which='both', left=False, right=False, bottom=False, top=False, labelbottom=False)
    # Flatten axes if we have more than 1 plot. Or, return a list of 2 axs to make it compatible with multi-plot case.
    axs = axs.ravel() if img_n > 1 else [axs, axs]
    # Get 2D embeddings, using PCA
    pca = PCA(n_components=2)
    # Fit training data and transform
    embeddings_pca = pca.fit_transform(embeddings)  # if embeddings.shape[1]>2 else embeddings
    # Set the title of the sub-plot
    axs[0].title.set_text('Embeddings from PCA')
    # Plot samples, using each class label to define the color of the class.
    sns_plt = sns.scatterplot(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1], ax=axs[0], palette=palette, hue=y, s=20,
                              alpha=alpha)
    # Overwrite legend labels
    overwrite_legends(sns_plt, fig, ncol=ncol, labels=legend_labels, title=legend_title)
    # Get 2D embeddings, using t-SNE
    embeddings_tsne = tsne(embeddings)  # if embeddings.shape[1]>2 else embeddings
    # Set the title of the sub-plot
    axs[1].title.set_text('Embeddings from t-SNE')
    # Plot samples, using each class label to define the color of the class.
    sns_plt = sns.scatterplot(x=embeddings_tsne[:, 0], y=embeddings_tsne[:, 1], ax=axs[1], palette=palette, hue=y, s=20,
                              alpha=alpha)
    # Overwrite legend labels
    overwrite_legends(sns_plt, fig, ncol=ncol, labels=legend_labels, title=legend_title)
    # Remove legends in sub-plots
    axs[0].get_legend().remove()
    axs[1].get_legend().remove()
    # Adjust the scaling factor to fit your legend text completely outside the plot
    # (smaller value results in more space being made for the legend)
    plt.subplots_adjust(right=legend_space_adjustment[str(ncol)])
    # Get the path to the project root
    root_path = os.path.dirname(os.path.dirname(__file__))
    # Define the path to save the plot to.
    fig_path = os.path.join(root_path, "results", config["framework"], "evaluation", "clusters", suffix + plt_name + ".png")
    # Define tick params
    plt.tick_params(axis=u'both', which=u'both', length=0)
    # Save the plot
    plt.savefig(fig_path, bbox_inches="tight")
    # plt.show()
    # Clear figure just in case if there is a follow-up plot.
    plt.clf()


def overwrite_legends(sns_plt, fig, ncol, labels, title=None):
    """Overwrites the legend of the plot

    Args:
        sns_plt (object): Seaborn plot object to manage legends
        c2l (dict): Dictionary mapping classes to labels
        fig (object): Figure to be edited
        ncol (int): Number of columns to use for legends
        title (str): Title of legend
        labels (list): Class labels

    """
    # Get legend handles and labels
    handles, legend_txts = sns_plt.get_legend_handles_labels()
    # Turn str to int before sorting ( to avoid wrong sort order such as having '10' in front of '4' )
    legend_txts = [int(d) for d in legend_txts]
    # Sort both handle and texts so that they show up in a alphabetical order on the plot
    legend_txts, handles = (list(t) for t in zip(*sorted(zip(legend_txts, handles))))
    # Define the figure title
    title = title or "Cluster"
    # Overwrite the legend labels and add a title to the legend
    fig.legend(handles, labels, loc="center right", borderaxespad=0.1, title=title, ncol=ncol)
    sns_plt.set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)
    sns_plt.tick_params(top=False, bottom=False, left=False, right=False)


def save_np2csv(np_list, save_as="test.csv"):
    """Saves a list of numpy arrays to a csv file

    Args:
        np_list (list[numpy.ndarray]): List of numpy arrays
        save_as (str): File name to be used when saving

    """
    # Get numpy arrays and label lists
    Xtr, ytr = np_list
    # Turn label lists into numpy arrays
    ytr = np.array(ytr, dtype=np.int8)
    # Get column names
    columns = ["label"] + list(map(str, list(range(Xtr.shape[1]))))
    # Concatenate "scaled" features and labels
    data_tr = np.concatenate((ytr.reshape(-1, 1), Xtr), axis=1)
    # Generate new dataframes with "scaled features" and labels
    df_tr = pd.DataFrame(data=data_tr, columns=columns)
    # Show samples from scaled data
    print("Samples from the dataframe:")
    print(df_tr.head())
    # Save the dataframe as csv file
    df_tr.to_csv(save_as, index=False)
    # Print an informative message
    print(f"The dataframe is saved as {save_as}")


def append_tensors_to_lists(list_of_lists, list_of_tensors):
    """Appends tensors in a list to a list after converting tensors to numpy arrays

    Args:
        list_of_lists (list[lists]): List of lists, each of which holds arrays
        list_of_tensors (list[torch.tensorFloat]): List of Pytorch tensors

    Returns:
        list_of_lists (list[lists]): List of lists, each of which holds arrays

    """
    # Go through each tensor and corresponding list
    for i in range(len(list_of_tensors)):
        # Convert tensor to numpy and append it to the corresponding list
        list_of_lists[i] += [list_of_tensors[i].cpu().numpy()]
    # Return the lists
    return list_of_lists


def concatenate_lists(list_of_lists):
    """Concatenates each list with the main list to a numpy array

    Args:
        list_of_lists (list[lists]): List of lists, each of which holds arrays

    Returns:
        (list[numpy.ndarray]): List containing numpy arrays

    """
    list_of_np_arrs = []
    # Pick a list of numpy arrays ([np_arr1, np_arr2, ...]), concatenate numpy arrs to a single one (np_arr_big),
    # and append it back to the list ([np_arr_big1, np_arr_big2, ...])
    for list_ in list_of_lists:
        list_of_np_arrs.append(np.concatenate(list_))
    # Return numpy arrays
    return list_of_np_arrs[0] if len(list_of_np_arrs) == 1 else list_of_np_arrs


def aggregate(latent_list, config):
    """Aggregates the latent representations of subsets to obtain joint representation

    Args:
        latent_list (list[torch.FloatTensor]): List of latent variables, one for each subset
        config (dict): Dictionary holding the configuration

    Returns:
        (torch.FloatTensor): Joint representation

    """
    # Initialize the joint representation
    latent = None
    
    # Aggregation of latent representations
    if config["aggregation"]=="mean":
        latent = sum(latent_list)/len(latent_list)
    elif config["aggregation"]=="sum":
        latent = sum(latent_list)
    elif config["aggregation"]=="concat":
        latent = th.cat(latent_list, dim=-1)
    elif config["aggregation"]=="max":
        latent = functools.reduce(th.max, latent_list)
    elif config["aggregation"]=="min":
        latent = functools.reduce(th.min, latent_list)
    else:
        print("Proper aggregation option is not provided. Please check the config file.")
        exit()
        
    return latent
