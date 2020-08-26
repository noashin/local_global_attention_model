## Should be run from its folder

import os
import sys
import time
import pickle

import yaml
import click
import numpy as np

sys.path.append('./../../')

from src.FullModel.model import Model as model
from src.LocalGlobalAttentionModel.b_param import BParam as b_param
from src.LocalGlobalAttentionModel.s0_param import S0Param as s0_param
from src.FullModel.epsilon_param import EpsParam as eps_param
from src.FullModel.xi_param import XiParam as xi_param

@click.command()
@click.option('--yml_file', type=click.STRING,
              help='path to the yaml file with the input')
@click.option('--output_path', type=click.STRING, default='./')
@click.option('--fold_index', type=click.INT, default=0)
def main(yml_file, output_path, fold_index):
    """
    This script performs generates samples for the model parameters given experimental data.
    It is intended to be used for cross validation, where the input is chunked already to batches.
    :param yml_file: path to a yaml file containing the settings for the model
    :param output_path: path to an existing directory to store the experiment results
    :param fold_index: index of one of the cross validation folds
    """
    # read config file
    secs = np.random.randint(20)
    time.sleep(secs)
    config_stream = open(yml_file, 'r')
    config = yaml.load(config_stream, Loader=yaml.FullLoader)

    # create folder for the output of the experiment and save there the configuration file
    output_folder = os.path.join(output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(os.path.join(output_folder, 'input.yml'), 'w') as f:
        yaml.dump(config, f)
    output_fold_folder = os.path.join(output_folder, 'fold_ind_' + str(fold_index))
    if not os.path.exists(output_fold_folder):
        os.makedirs(output_fold_folder)

    # read configurations into parameter
    s_0_config = config['s_0']
    s_0_ob = s0_param(s_0_config['is_fixed'], s_0_config['m_p'], s_0_config['s_p'])

    b_config = config['b']
    b_ob = b_param(b_config['is_fixed'], b_config['m_p'], b_config['s_p'])

    eps_config = config['epsilon']
    eps_ob = eps_param(eps_config['is_fixed'], np.array(eps_config['alpha']), np.array(eps_config['betta']), 1)

    xi_config = config['xi']
    xi_ob = xi_param(xi_config['is_fixed'], np.array(xi_config['alpha']), np.array(xi_config['betta']), 1)

    cov_ratio = config['cov_ratio']
    num_subjects = config['num_subjects']

    b_ob.set_init_value(b_ob.prior())
    s_0_ob.set_init_value(s_0_ob.prior())
    eps_ob.set_init_value(eps_ob.prior())
    xi_ob.set_init_value(xi_ob.prior())

    # sampler parameters
    sampler_parameters = config['sampler_parameters']
    save_steps = sampler_parameters['save_steps']
    num_samples = sampler_parameters['num_samples']

    # paths
    paths = config['paths']
    saliency_path = paths['saliency_path']
    fixation_path = paths['fixations_path']
    results_path = os.path.join(output_fold_folder, paths['output_path'])
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # read the files containing the saliencies and fixations locations
    with open(saliency_path, 'rb') as f:
        saliencies = pickle.load(f)
    with open(fixation_path, 'rb') as f:
        fixations = pickle.load(f)

    # get the data for the specific fold
    train_sals = saliencies[fold_index]
    train_fixs = fixations[fold_index]

    model_ob = model(train_sals, s_0_ob, b_ob, eps_ob, xi_ob, cov_ratio)

    # This script is intended to run on a slurm cluster.
    # we use the task id to determine which subject to fit.
    # when run locally this should be changed manually to a umber between 1 and 35
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    sub = (task_id - 1) % num_subjects
    # get the data for the specific subject
    fixs = [[train_fixs[j][sub][:]] for j in range(len(train_fixs))]

    #  set stuff in the model
    model_ob.set_fixations(fixs)
    model_ob.set_saliencies_ts()
    model_ob.set_sal_ts_ratio()
    model_ob.set_fix_dist_2()
    model_ob.set_dist_mat_per_fix()

    # path to file to store the sampled chains
    file_name = '%s_subject_%s.p' % (time.strftime("%Y%m%d-%H%M%S"), sub)
    file_path = os.path.join(results_path, file_name)

    samples_mu0, samples_b, samples_epsilon, samples_xi = model_ob.sample(num_samples, save_steps, file_path, True)


if __name__ == '__main__':
    main()
