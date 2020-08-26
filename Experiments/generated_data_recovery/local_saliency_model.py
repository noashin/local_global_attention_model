## Should be run from its folder

import os
import sys
import time
import pickle

import yaml
import click
import numpy as np

sys.path.append('./../../')

from src.LocalSaliencyModel.model import Model as model
from src.LocalSaliencyModel.xi_param import XiParam as xi_param


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
    output_folder = os.path.join(output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(os.path.join(output_folder, 'input.yml'), 'w') as f:
        yaml.dump(config, f)

    # read configurations into parameter
    xi_config = config['xi']
    xi_ob = xi_param(xi_config['is_fixed'], np.array(xi_config['alpha']), np.array(xi_config['betta']))
    xi_ob.set_value(np.array(xi_config['value']).astype(float))
    xi_ob.set_init_value(xi_ob.prior())

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

    model_ob = model(saliencies, xi_ob)

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
    model_ob.set_saliencies_ts()
    model_ob.set_fix_dist_2()
    model_ob.set_dist_mat_per_fix()

    # sample random starting points to sample the parameters from their priors
    xi_ob.set_init_value(xi_ob.prior())

    # path to file to store the sampled chains
    file_name = '%s_subject_%s.p' % (time.strftime("%Y%m%d-%H%M%S"), str(0))
    file_path = os.path.join(results_path, file_name)

    samples_xi = model_ob.sample(num_samples, save_steps, file_path)


if __name__ == '__main__':
    main()
