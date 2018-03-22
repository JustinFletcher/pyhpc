
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
        expected_command = 'CUDA_VISIBLE_DEVICES=0 python experiment.py --parameter=1 --log_dir={} --log_filename={}'.format(os.path.join(main_dir, 'templog_0'), 'templog_0')
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


def test_launch_batched_experiments():
    """Test that launching 10 experiments with a batch size of 4 makes three job scripts"""
    
    experiment = cluster_experiment.ClusterExperiment()
    # Now make 10 experiments
    experiment.add_design('parameter', list(range(10)))
     
    main_dir = os.path.join('/home', 'pyhpc')

    # Update the expected commands 3 times, one for each
    #   script that should be submitted.
    # FIXME: Using i%4 for the CUDA device is pretty yucky
    #    I'm sure there is a better way.
    expected_commands = ['CUDA_VISIBLE_DEVICES={} python experiment.py --parameter={} --log_dir={} --log_filename={}'.format(i%4, i, os.path.join(main_dir, 'templog_0'), 'templog_0') for i in range(4)]
    expected_commands.extend(['CUDA_VISIBLE_DEVICES={} python experiment.py --parameter={} --log_dir={} --log_filename={}'.format(i%4, i, os.path.join(main_dir, 'templog_1'), 'templog_1') for i in range(4, 8)])
    expected_commands.extend(['CUDA_VISIBLE_DEVICES={} python experiment.py --parameter={} --log_dir={} --log_filename={}'.format(i%4, i, os.path.join(main_dir, 'templog_2'), 'templog_2') for i in range(8, 10)])
    
    with mock.patch('pyhpc.cluster_experiment.subprocess.Popen') as Popen:
        experiment.launch_experiment('experiment.py', main_dir, shuffle_job_order=False, experiments_per_job=4)

        script_template = cluster_experiment.JOB_SCRIPT_TEMPLATE
        expected_prologue = "module load anaconda2/5.0.1 gcc/5.3.0 cudnn/6.0"
        expected_epilogue = 'wait'
        expected_script1 = script_template.format(job_name='dist_ex_0',
                                                  walltime='4:00:00',
                                                  select='1:ncpus=20:mpiprocs=20',
                                                  command=' &\n'.join(expected_commands[0:4]),
                                                  command_prologue=expected_prologue,
                                                  command_epilogue=expected_epilogue,
                                                  account=experiment.account,
                                                  queue=experiment.queue)
        
        expected_script2 = script_template.format(job_name='dist_ex_1',
                                                  walltime='4:00:00',
                                                  select='1:ncpus=20:mpiprocs=20',
                                                  command=' &\n'.join(expected_commands[4:8]),
                                                  command_prologue=expected_prologue,
                                                  command_epilogue=expected_epilogue,
                                                  account=experiment.account,
                                                  queue=experiment.queue)
        
        expected_script3 = script_template.format(job_name='dist_ex_2',
                                                  walltime='4:00:00',
                                                  select='1:ncpus=20:mpiprocs=20',
                                                  command=' &\n'.join(expected_commands[8:]),
                                                  command_prologue=expected_prologue,
                                                  command_epilogue=expected_epilogue,
                                                  account=experiment.account,
                                                  queue=experiment.queue)
        
        assert Popen.mock_calls[1][1][0] == expected_script1
        assert Popen.mock_calls[4][1][0] == expected_script2
        assert Popen.mock_calls[7][1][0] == expected_script3

