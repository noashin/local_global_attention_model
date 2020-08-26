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
def main(yml_file, output_path):
    """
    This script generates data from the model with certain parameter values
    and performs inference on the generated data to recover the parameters values.
    :param yml_file: path to a yaml file containing the settings for the model
    :param output_path: path to an existing directory to store the experiment results
    """

    # read config file
    secs = np.random.randint(20)
    time.sleep(secs)
    config_stream = open(yml_file, 'r')
    config = yaml.load(config_stream, Loader=yaml.FullLoader)

    # create folder for the output of the experiment and save there the configuration file
    # one can run multiple jobs of this script using a job array and each will get its own folder
    # if the job is run on a slurm cluster one can use the  SLURM_ARRAY_TASK_ID
    # task_id = os.environ['SLURM_ARRAY_TASK_ID']
    task_id = 0
    output_folder = output_path + '_' + str(task_id)
    output_folder = os.path.join(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(os.path.join(output_folder, 'input.yml'), 'w') as f:
        yaml.dump(config, f)

    # read configurations into parameter
    eps_config = config['epsilon']
    eps_ob = eps_param(eps_config['is_fixed'], np.array(eps_config['alpha']), np.array(eps_config['betta']))
    eps_ob.set_value(np.array(eps_config['value']).astype(float))

    b_config = config['b']
    b_ob = b_param(b_config['is_fixed'], b_config['m_p'], b_config['s_p'])
    b_ob.set_value(b_config['value'])

    s_0_config = config['s_0']
    s_ob = s0_param(s_0_config['is_fixed'], s_0_config['m_p'], s_0_config['s_p'])
    s_ob.set_value(s_0_config['value'])

    xi_config = config['xi']
    xi_ob = xi_param(xi_config['is_fixed'], np.array(xi_config['alpha']).astype(float),
                     np.array(xi_config['betta']).astype(float))
    xi_ob.set_value(np.array(xi_config['value']).astype(float))

    cov_ratio = config['cov_ratio']

    # sampler parameters
    sampler_parameters = config['sampler_parameters']
    save_steps = sampler_parameters['save_steps']
    num_samples = sampler_parameters['num_samples']

    # paths
    paths = config['paths']
    saliency_path = paths['saliency_path']
    results_path = os.path.join(output_folder, paths['output_path'])

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with open(saliency_path, 'rb') as f:
        saliencies = pickle.load(f)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # create the model object
    model_ob = model(saliencies, s_ob, b_ob, eps_ob, xi_ob, cov_ratio)

    # one can either perform the inference on pre-generated data or generate a new data set and save it
    if 'data_path' in paths.keys():
        data_path = paths['data_path']
        with open(os.path.join(data_path, 'gen_data.p'), 'rb') as f:
            res = pickle.load(f)
            fixs = res[0]
            gammas = res[1]
    else:
        gammas, fixs = model_ob.generate_dataset(30, 1)
        with open(os.path.join(output_folder, 'gen_data.p'), 'wb') as f:
            pickle.dump([fixs, gammas], f)

    #  set stuff in the model
    model_ob.set_fixations(fixs)
    model_ob.set_gammas(gammas)
    model_ob.set_saliencies_ts()
    model_ob.set_sal_ts_ratio()
    model_ob.set_fix_dist_2()
    model_ob.set_dist_mat_per_fix()

    # sample random starting points to sample the parameters from their priors
    s_ob.set_init_value(s_ob.prior())
    b_ob.set_init_value(b_ob.prior())
    eps_ob.set_init_value(eps_ob.prior())
    xi_ob.set_init_value(xi_ob.prior())

    # path to file to store the sampled chains
    file_name = '%s_subject_%s.p' % (time.strftime("%Y%m%d-%H%M%S"), str(0))
    file_path = os.path.join(results_path, file_name)

    samples_s_0, samples_b, samples_epsilon, sample_xi = model_ob.sample(num_samples, save_steps, file_path, True)


if __name__ == '__main__':
    main()
