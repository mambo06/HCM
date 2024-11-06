

import gc
import itertools
import os

import numpy as np
import pandas as pd
import torch as th
import torch.optim as optim
import math

# from tqdm import tqdm

from utils.loss_functions import JointLoss
from utils.model_plot import save_loss_plot
from utils.model_utils import AEWrapper
from utils.utils import set_seed, set_dirs

from utils.projection_utils import head, projection
from sklearn.metrics import mean_squared_error

th.autograd.set_detect_anomaly(True)
from torch.optim.lr_scheduler import ReduceLROnPlateau


import random
import pickle


class CFL:

    def __init__(self, options):
        """

        Args:
            options (dict): Configuration dictionary.

        """
        # Get config
        self.options = options
        # Define which device to use: GPU, or CPU
        self.device = options["device"]
        # Create empty lists and dictionary
        self.model_dict, self.summary = {}, {}
        # Set random seed
        set_seed(self.options)
        # Set paths for results and initialize some arrays to collect data during training
        self._set_paths()
        # Set directories i.e. create ones that are missing.
        set_dirs(self.options)
        # Set the condition if we need to build combinations of 2 out of projections. 
        # self.is_combination = self.options["contrastive_loss"] or self.options["distance_loss"]
        self.is_combination = True # set this to true cause z loss 0 ! code realted
        # ------Network---------
        # Instantiate networks
        print("Building TCL model")
        # Set Autoencoders i.e. setting loss, optimizer, and device assignment (GPU, or CPU)
        self.set_autoencoder()
        # Set scheduler (its use is optional)
        self._set_scheduler()
        self._set_reduce_lr()
        # Print out model architecture
        # self.print_model_summary()
        self.loss = {"tloss_b": [], "tloss_e": [], "vloss_e": [],
                     "closs_b": [], "rloss_b": [], "zloss_b": [],
                     "tloss_o": []}
        self.val_loss = []
        self.train_tqdm = None
        self.modelConventional = None

    def get_loss(self):
        return self.loss

    def set_loss(self, loss):
        self.loss = loss

    def set_autoencoder(self):
        """Sets up the autoencoder model, optimizer, and loss"""
        # Instantiate the model for the text Autoencoder
        self.encoder = AEWrapper(self.options)
        # Add the model and its name to a list to save, and load in the future
        self.model_dict.update({"encoder": self.encoder})
        # Assign autoencoder to a device
        for _, model in self.model_dict.items(): model.to(self.device)
        # Get model parameters
        parameters = [model.parameters() for _, model in self.model_dict.items()]
        # Joint loss including contrastive, reconstruction and distance losses
        self.joint_loss = JointLoss(self.options)
        # Set optimizer for autoencoder
        # self.optimizer_ae = self._adam(parameters, lr=self.options["learning_rate"])
        # availabe  'adabelief', 'adam', 'adamw', 'radam','sgd''
        self.optimizer_ae = self._make_optimizer("adamw",parameters, lr=self.options["learning_rate"])
        # Add items to summary to be used for reporting later
        self.summary.update({"recon_loss": []})

    def fit(self, x,y):
        self.set_mode(mode="training") 
        
        Xorig = self.process_batch(x, x)

        # Generate subsets with added noise -- labels are not used
        x_tilde_list = self.subset_generator(x, mode="train") # partion data

        # If we use either contrastive and/or distance loss, we need to use combinations of subsets
        if self.is_combination:
            x_tilde_list = self.get_combinations_of_subsets(x_tilde_list)

        # 0 - Update Autoencoder
        loss = self.calculate_loss(x_tilde_list, y) 

        return loss


    def saveTrainParams(self):
        config = self.options
        prefix = str(config['epochs']) + "e-" + str(config["dataset"])
        

        # Save plot of training and validation losses
        save_loss_plot(self.loss, self._plots_path,prefix)
        # Convert loss dictionary to a dataframe
        loss_df = pd.DataFrame(dict([(k, pd.Series(v,dtype='float64')) for k, v in self.loss.items()]))
        # Save loss dataframe as csv file for later use
        
        loss_df.to_csv(self._loss_path + "/"+  prefix + "-losses.csv")

    
    
    def calculate_loss(self, x_tilde_list, y):
        # xi = xi[0] # single partition
        # print(xi.shape)
        total_loss, recon_loss = [], []

        # pass data through model
        for xi in x_tilde_list:
            # If we are using combination of subsets use xi since it is already a concatenation of two subsets. 
            # Else, concatenate subset with itself just to make the computation of loss compatible with the case, 
            # in which we use the combinations. Note that Xorig is already concatenation of two copies of original input.
            Xinput = xi if self.is_combination else self.process_batch(xi, xi)
            # Xinput.to(self.device).float()
            # Forwards pass
            latent, logits = self.encoder(Xinput)
            latent = latent.detach().numpy()
            #improve this to use torch only, no numpy
            # projections = self.modelConventional.fit(latent,np.hstack((y,y)))
            projections = self.modelConventional.predict(latent) # * self.options['tau']

            recon_loss.append(np.sqrt(mean_squared_error(projections,np.hstack((y,y)))))

            projections = head(
                    projections * self.options['tau'] 
                    ,y
                    ) #return log(label-delta)
            # projections = th.from_numpy(projections)
            
            loss = self.joint_loss(logits, projections)
            # loss = self.joint_loss(logits, projections * self.options['tau'], th.from_numpy( np.hstack((y,y))))
            # Accumulate losses
            total_loss.append(loss)

        # Compute the average of losses
        # print(recon_loss)
        n = len(total_loss)
        total_loss = sum(total_loss)/ n
        recon_loss = sum(recon_loss)
        

        return total_loss,recon_loss
        # print(tloss, closs, rloss, zloss) = tensor(59.7908, grad_fn=<AddBackward0>) tensor(4.1386, grad_fn=<DivBackward0>) tensor(55.6447, grad_fn=<DivBackward0>) tensor(0.0075, grad_fn=<DivBackward0>)


    def update_autoencoder(self, tloss, retain_graph=True): # 6 torch.Size([64, 343]) original mixed 2 partitions, torch.Size([64, 784]) original data stacked

    
        self._update_model(tloss, self.optimizer_ae, retain_graph=retain_graph)


    def get_combinations_of_subsets(self, x_tilde_list):
        """Generate a list of combinations of subsets from the list of subsets

        Args:
            x_tilde_list (list): List of subsets e.g. [x1, x2, x3, ...]
        
        Returns:
            (list): A list of combinations of subsets e.g. [(x1, x2), (x1, x3), ...]

        """        
                            
        # Compute combinations of subsets [(x1, x2), (x1, x3)...]
        subset_combinations = list(itertools.combinations(x_tilde_list, 2))
        # List to store the concatenated subsets
        concatenated_subsets_list = []
        
        # Go through combinations
        for (xi, xj) in subset_combinations:
            # Concatenate xi, and xj, and turn it into a tensor
            Xbatch = self.process_batch(xi, xj)
            # Add it to the list
            concatenated_subsets_list.append(Xbatch)
        
        # Return the list of combination of subsets
        return concatenated_subsets_list
        
        
    def mask_generator(self, p_m, x):
        """Generate mask vector."""
        mask = np.random.binomial(1, p_m, x.shape)
        return mask

    def subset_generator(self, x, mode="test", skip=[-1]):
        # print(x.shape)
        """Generate subsets and adds noise to them

        Args:
            x (np.ndarray): Input data, which is divded to the subsets
            mode (bool): Indicates whether we are training a model, or testing it
            skip (list): List of integers, showing which subsets to skip when training the model
        
        Returns:
            (list): A list of np.ndarrays, each of which is one subset
            (list): A list of lists, each list of which indicates locations of added noise in a subset

        """
        
        n_subsets = self.options["n_subsets"]
        n_column = self.options["dims"][0]
        # n_column = x.shape[-1]
        overlap = self.options["overlap"]
        n_column_subset = int(n_column / n_subsets)
        # Number of overlapping features between subsets
        n_overlap = int(overlap * n_column_subset)

       
        subset_column_list = [x.clone() for n in range(n_subsets)] # for 3 noises layer

        x_tilde_list = []
        for z, subset_column in enumerate(subset_column_list):
            rvs = int(z%2)
            x_bar = subset_column #[:, subset_column_idx]
            # Add noise to cropped columns - Noise types: Zero-out, Gaussian, or Swap noise
            # print('noise mark',self.options["add_noise"])
            # if not self.options["add_noise"]: print('No noise generated')
            if self.options["add_noise"]:
                x_bar_noisy = self.generate_noisy_xbar(x_bar,rvs ) #,["swap_noise", "gaussian_noise", "zero_out"][z])

                # Generate binary mask
                p_m = self.options["masking_ratio"][rvs]
                mask = np.random.binomial(1, p_m, x_bar.shape)

                # Replace selected x_bar features with the noisy ones
                x_bar = x_bar * (1 - mask) + x_bar_noisy * mask

            # Add the subset to the list   
            x_tilde_list.append(x_bar)

        return x_tilde_list

    def generate_noisy_xbar(self, x, rvs):
        """Generates noisy version of the samples x
        
        Args:
            x (np.ndarray): Input data to add noise to
        
        Returns:
            (np.ndarray): Corrupted version of input x
            
        """
        # Dimensions
        no, dim = x.shape

        # Get noise type
        noise_type = self.options["noise_type"]
        noise_level = self.options["noise_level"][rvs]

        # Initialize corruption array
        x_bar = np.zeros([no, dim])

        # Randomly (and column-wise) shuffle data
        if noise_type == "swap_noise":
            for i in range(dim):
                idx = np.random.permutation(no)
                x_bar[:, i] = x[idx, i]
        # Elif, overwrite x_bar by adding Gaussian noise to x

        elif noise_type == "gaussian_noise":
            x_bar = x + np.random.normal(float(th.mean(x)), noise_level, x.shape)

        else:
            x_bar = x_bar

        return x_bar

    def clean_up_memory(self, losses):
        """Deletes losses with attached graph, and cleans up memory"""
        for loss in losses: del loss
        gc.collect()

    def process_batch(self, xi, xj):
        """Concatenates two transformed inputs into one, and moves the data to the device as tensor"""
        # Combine xi and xj into a single batch
        Xbatch = np.concatenate((xi, xj), axis=0)
        # Convert the batch to tensor and move it to where the model is
        Xbatch = self._tensor(Xbatch)
        # Return batches
        return Xbatch

    def update_log(self, epoch, batch):
        """Updates the messages displayed during training and evaluation"""
        # For the first epoch, add losses for batches since we still don't have loss for the epoch
        if epoch < 1:
            description = f"Losses per batch - Total:{self.loss['tloss_b'][-1]:.4f}"
            description += f", X recon:{self.loss['rloss_b'][-1]:.4f}"
            if self.options["contrastive_loss"]:
                description += f", contrastive:{self.loss['closs_b'][-1]:.4f}"
            if self.options["distance_loss"]:
                description += f", z distance:{self.loss['zloss_b'][-1]:.6f}, Progress"
        # For sub-sequent epochs, display only epoch losses.
        else:
            description = f"Epoch-{epoch} Total training loss:{self.loss['tloss_e'][-1]:.4f}"
            description += f", val loss:{self.loss['vloss_e'][-1]:.4f}" if self.options["validate"] else ""
            description += f" | Losses per batch - X recon:{self.loss['rloss_b'][-1]:.4f}"
            if self.options["contrastive_loss"]:
                description += f", contrastive:{self.loss['closs_b'][-1]:.4f}"
            if self.options["distance_loss"]:
                description += f", z distance:{self.loss['zloss_b'][-1]:.6f}, Progress"
        # return description
        # Update the displayed message
        self.train_tqdm.set_description(description)

    def set_mode(self, mode="training"):
        """Sets the mode of the models, either as .train(), or .eval()"""
        for _, model in self.model_dict.items():
            model.train() if mode == "training" else model.eval()

    def save_weights(self):
        config = self.options

        prefix = str(config['epochs']) + "e-" + str(config["dataset"]) 

        """Used to save weights."""
        for model_name in self.model_dict:
            th.save(self.model_dict[model_name], self._model_path + "/" + model_name + "_"+ prefix + ".pt")
        print("Done with saving models.")

    def load_models(self):
        config = self.options

        prefix = str(config['epochs']) + "e-" + str(config["dataset"])

        

        """Used to load weights saved at the end of the training."""
        for model_name in self.model_dict:
            model = th.load(self._model_path + "/" + model_name + "_"+ prefix + ".pt", map_location=self.device)
            setattr(self, model_name, model.eval())
            print(f"--{model_name} is loaded")
        print("Done with loading models.")

    def save_conventional(self):
        config = self.options

        prefix = str(config['epochs']) + "e-" + str(config["dataset"]) 
        with open(self._model_path + '/' + prefix + 'CM.pickle', 'wb') as file:
            pickle.dump(self.modelConventional, file)
        print("Done with saving conventional models.")

    def load_conventional(self):
        config = self.options

        prefix = str(config['epochs']) + "e-" + str(config["dataset"])
        with open(self._model_path + '/' + prefix + 'CM.pickle', 'rb') as file:
            self.modelConventional = pickle.load(file)

        print("Done with loading conventional models.")

    def print_model_summary(self):
        """Displays model architectures as a sanity check to see if the models are constructed correctly."""
        # Summary of the model
        description = f"{40 * '-'}Summary of the models (an Autoencoder and Projection network):{40 * '-'}\n"
        description += f"{34 * '='}{self.options['model_mode'].upper().replace('_', ' ')} Model{34 * '='}\n"
        description += f"{self.encoder}\n"
        # Print model architecture
        print(description)

    def _update_model(self, loss, optimizer, retain_graph=True):
        """Does backprop, and updates the model parameters

        Args:
            loss (): Loss containing computational graph
            optimizer (th.optim): Optimizer used during training
            retain_graph (bool): If True, retains graph. Otherwise, it does not.

        """
        # Reset optimizer
        optimizer.zero_grad()
        # Backward propagation to compute gradients
        loss.backward(retain_graph=retain_graph)
        # Update weights
        optimizer.step()

    def _set_scheduler(self):
        """Sets a scheduler for learning rate of autoencoder"""
        # Set scheduler (Its use will be optional)
        self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer_ae, step_size=0.1, gamma=0.99)

    def _set_reduce_lr(self):
        self.reducer = ReduceLROnPlateau(self.optimizer_ae, 'min', patience=5, factor=0.1,)


    def _set_paths(self):
        """ Sets paths to bse used for saving results at the end of the training"""
        # Top results directory
        self._results_path = os.path.join(self.options["paths"]["results"], self.options["framework"])
        # Directory to save model
        self._model_path = os.path.join(self._results_path, "training", self.options["model_mode"], "model")
        # Directory to save plots as png files
        self._plots_path = os.path.join(self._results_path, "training", self.options["model_mode"], "plots")
        # Directory to save losses as csv file
        self._loss_path = os.path.join(self._results_path, "training", self.options["model_mode"], "loss")

    def _make_optimizer( self, optimizer: str, params, lr, weight_decay= 0,) -> optim.Optimizer:
        Optimizer = {
            'adabelief': AdaBelief,
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'radam': RAdam,
            'sgd': optim.SGD,
        }[optimizer]
        momentum = (0.9,) if Optimizer is optim.SGD else ()
        return Optimizer(itertools.chain(*params), lr, *momentum, weight_decay=weight_decay)

    def _adam(self, params, lr=1e-4):
        """Sets up AdamW optimizer using model params"""
        return th.optim.AdamW(itertools.chain(*params), lr=lr, betas=(0.9, 0.999), eps=1e-07)

    def _lbfgs(self, params, lr=1e-4):
        """Sets up AdamW optimizer using model params"""
        return th.optim.ASGD(itertools.chain(*params), lr=lr,)

    def _tensor(self, data):
        """Turns numpy arrays to torch tensors"""
        if type(data).__module__ == np.__name__:
            data = np.float32(data) # support mps
            data = th.from_numpy(data)

        # return data
        return data.to(self.device).float()
# Source: https://github.com/LiyuanLucasLiu/RAdam
class RAdam(optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        degenerated_to_sgd=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if 'betas' in param and (
                    param['betas'][0] != betas[0] or param['betas'][1] != betas[1]
                ):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = th.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = th.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(
                            -group['weight_decay'] * group['lr'], p_data_fp32
                        )
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(
                            -group['weight_decay'] * group['lr'], p_data_fp32
                        )
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


version_higher = th.__version__ >= "1.5.0"


# Source: https://github.com/juntang-zhuang/Adabelief-Optimizer
class AdaBelief(optim.Optimizer):
    r"""Implements AdaBelief algorithm. Modified from Adam in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-16)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: True) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: True) If set as True, then perform the rectified
            update similar to RAdam
        degenerated_to_sgd (boolean, optional) (default:True) If set as True, then perform SGD update
            when variance of gradient is high
        print_change_log (boolean, optional) (default: True) If set as True, print the modifcation to
            default hyper-parameters
    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients, NeurIPS 2020
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-16,
        weight_decay=0,
        amsgrad=False,
        weight_decouple=True,
        fixed_decay=False,
        rectify=True,
        degenerated_to_sgd=True,
        print_change_log=True,
    ):

        # ------------------------------------------------------------------------------
        # Print modifications to default arguments
        if print_change_log:
            print(
                'Please check your arguments if you have upgraded adabelief-pytorch from version 0.0.5.'
            )
            print('Modifications to default arguments:')
            default_table = [
                ['eps', 'weight_decouple', 'rectify'],
                ['adabelief-pytorch=0.0.5', '1e-8', 'False', 'False'],
                ['>=0.1.0 (Current 0.2.0)', '1e-16', 'True', 'True'],
            ]
            print(default_table)

            recommend_table = [
                [
                    'SGD better than Adam (e.g. CNN for Image Classification)',
                    'Adam better than SGD (e.g. Transformer, GAN)',
                ],
                ['Recommended eps = 1e-8', 'Recommended eps = 1e-16'],
            ]
            print(recommend_table)

            print('For a complete table of recommended hyperparameters, see')
            print('https://github.com/juntang-zhuang/Adabelief-Optimizer')

            print(
                'You can disable the log message by setting "print_change_log = False", though it is recommended to keep as a reminder.'
            )
        # ------------------------------------------------------------------------------

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if 'betas' in param and (
                    param['betas'][0] != betas[0] or param['betas'][1] != betas[1]
                ):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(AdaBelief, self).__init__(params, defaults)

        self.degenerated_to_sgd = degenerated_to_sgd
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print('Weight decoupling enabled in AdaBelief')
            if self.fixed_decay:
                print('Weight decay fixed')
        if self.rectify:
            print('Rectification enabled in AdaBelief')
        if amsgrad:
            print('AMSGrad enabled in AdaBelief')

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = (
                    th.zeros_like(p.data, memory_format=th.preserve_format)
                    if version_higher
                    else th.zeros_like(p.data)
                )

                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = (
                    th.zeros_like(p.data, memory_format=th.preserve_format)
                    if version_higher
                    else th.zeros_like(p.data)
                )

                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_var'] = (
                        th.zeros_like(p.data, memory_format=th.preserve_format)
                        if version_higher
                        else th.zeros_like(p.data)
                    )

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # cast data type
                half_precision = False
                if p.data.dtype == th.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaBelief does not support sparse gradients, please consider SparseAdam instead'
                    )
                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = (
                        th.zeros_like(p.data, memory_format=th.preserve_format)
                        if version_higher
                        else th.zeros_like(p.data)
                    )
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = (
                        th.zeros_like(p.data, memory_format=th.preserve_format)
                        if version_higher
                        else th.zeros_like(p.data)
                    )
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = (
                            th.zeros_like(
                                p.data, memory_format=th.preserve_format
                            )
                            if version_higher
                            else th.zeros_like(p.data)
                        )

                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(
                    grad_residual, grad_residual, value=1 - beta2
                )

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    th.max(
                        max_exp_avg_var,
                        exp_avg_var.add_(group['eps']),
                        out=max_exp_avg_var,
                    )

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(
                        group['eps']
                    )
                else:
                    denom = (
                        exp_avg_var.add_(group['eps']).sqrt()
                        / math.sqrt(bias_correction2)
                    ).add_(group['eps'])

                # update
                if not self.rectify:
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                else:  # Rectified update, forked from RAdam
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma

                        # more conservative since it's an approximated value
                        if N_sma >= 5:
                            step_size = math.sqrt(
                                (1 - beta2_t)
                                * (N_sma - 4)
                                / (N_sma_max - 4)
                                * (N_sma - 2)
                                / N_sma
                                * N_sma_max
                                / (N_sma_max - 2)
                            ) / (1 - beta1 ** state['step'])
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / (1 - beta1 ** state['step'])
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    if N_sma >= 5:
                        denom = exp_avg_var.sqrt().add_(group['eps'])
                        p.data.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    elif step_size > 0:
                        p.data.add_(exp_avg, alpha=-step_size * group['lr'])

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half()

        return loss
