#!/usr/bin/python
# Example PBS cluster job submission in Python

import os
import sys
import argparse
import tensorflow as tf

# import pyhpc as hpc
# hpc.ClusterExperiment()

sys.path.insert(1, os.path.join(sys.path[0], '../../pyhpc'))

from cluster_experiment import ClusterExperiment


def main(FLAGS):

    # Clear and remake the log directory.
    if tf.gfile.Exists(FLAGS.log_dir) and FLAGS.delete_log_dir:

        if not FLAGS.delete_without_asking:
            delete_log_dir = input("Recursively delete directory {} (y/N)?".format(FLAGS.log_dir))
            
        if delete_log_dir.lower() == 'y':
            import pdb
            pdb.set_trace()
            tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Instantiate an experiment.
    exp = ClusterExperiment()
    
    # Set the number of reps for each config.
    # Set the number of reps for each config.
    exp.set_rep_count(5)

    # Set independent parameters.
    exp.add_design('train_batch_size', [128, 256])
    # exp.add_design('batch_interval', [1, 2, 4, 8, 16, 32, 64, 128])
    # exp.add_design('train_enqueue_threads', [1, 2, 4, 8, 16, 32, 64, 128])
    exp.add_design('learning_rate', [0.0001])
    exp.add_design('max_steps', [10000])
    exp.add_design('test_interval', [100])
    exp.add_design('pause_time', [10])

    # Launch the experiment.
    exp.launch_experiment(exp_filename=FLAGS.experiment_py_file,
                          log_dir=FLAGS.log_dir,
                          account='MHPCC96650DE1' if not FLAGS.account else FLAGS.account,
                          queue='standard',
                          command_prologue='module load anaconda3/5.0.1 tensorflow',
                          manager='pbs',
                          shuffle_job_order=True,
                          experiments_per_job=FLAGS.experiments_per_job)

    # Manually note response varaibles.
    response_labels = ['step_num',
                       'global_step_per_sec',
                       'loss']

    # Wait for the output to return.
    exp.join_job_output(FLAGS.log_dir,
                        FLAGS.log_filename,
                        FLAGS.max_runtime,
                        response_labels)

    print("All jobs complete. Exiting.")


if __name__ == '__main__':

    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str,
                        default='/gpfs/projects/ml/log/resnet_cifar_study/',
                        help='Summaries log directory.')
    parser.add_argument('--delete_log_dir', default=False, help='Recursively delete the log dir if it exists')
    parser.add_argument('--delete_without_asking', default=False, help='If deleting the log dir, remove it without asking')

    parser.add_argument('--log_filename', type=str,
                        default='resnet_cifar_study.csv',
                        help='Merged output filename.')

    parser.add_argument('--max_runtime', type=int,
                        default=3600000,
                        help='Number of seconds to run before giving up.')

    parser.add_argument('--experiment_py_file', type=str,
                        default='/gpfs/projects/ml/hpc-tensorflow/resnet_cifar_study/resnet_cifar_experiment.py',
                        help='Number of seconds to run before giving up.')

    parser.add_argument('--account', type=str, default=None, help="The PBS Account that this set of experiments should run under.")
    parser.add_argument('--experiments_per_job', type=int, default=1, help='How many experiments should be batched into a single job')
    
    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
    
