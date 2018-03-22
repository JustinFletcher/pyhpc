
import os

import pytest
import mock

from pyhpc import cluster_experiment


def test_launch_experiment_01():
    '''Check that launching an experiment sends a correct job script to PBS'''
    experiment = cluster_experiment.ClusterExperiment()
    experiment.add_design('parameter', [1])
    
    with mock.patch('pyhpc.cluster_experiment.subprocess.Popen') as Popen:
        main_dir = os.path.join('/home', 'pyhpc')
        experiment.launch_experiment('experiment.py', main_dir, shuffle_job_order=False)

        script_template = cluster_experiment.JOB_SCRIPT_TEMPLATE
        expected_command = 'python experiment.py --parameter=1 --log_dir={} --log_filename={}'.format(os.path.join(main_dir, 'templog_0'), 'templog_0')
        expected_prologue = "module load anaconda2/5.0.1 gcc/5.3.0 cudnn/6.0"
        expected_epilogue = 'wait'
        expected_script = script_template.format(job_name='dist_ex_0',
                                                 walltime='4:00:00',
                                                 select='1:ncpus=20:mpiprocs=20',
                                                 command=expected_command,
                                                 command_prologue=expected_prologue,
                                                 command_epilogue=expected_epilogue,
                                                 account=experiment.account,
                                                 queue=experiment.queue)
        assert Popen.mock_calls[1][1][0] == expected_script


def test_non_pbs_manager():
    '''Check that when a non PBS queue manager is requested, that experiment launching fails.

    As of 2018-03-21, only PBS is supported.
    '''
    experiment = cluster_experiment.ClusterExperiment()
    experiment.add_design('parameter', [1])

    with pytest.raises(ValueError):
        experiment.launch_experiment('experiment.py', '/tmp', manager='LSF')
