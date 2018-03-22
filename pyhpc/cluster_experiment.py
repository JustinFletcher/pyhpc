#!/usr/bin/env python
# Example PBS cluster job submission in Python

from __future__ import print_function

import os
import csv
import time
import random
import itertools
import subprocess

JOB_SCRIPT_TEMPLATE = """\
#!/bin/bash
#PBS -N {job_name}
#PBS -l walltime={walltime}
#PBS -l select={select}
#PBS -o ~/log/output/{job_name}.out
#PBS -e ~/log/error/{job_name}.err
#PBS -A {account}
#PBS -q {queue}
cd $PBS_O_WORKDIR
{command_prologue}
{command}
{command_epilogue}
"""


class ClusterExperiment(object):
    account = ''
    queue = 'standard'
    walltime = '4:00:00'
    resources = '1:ncpus=20:mpiprocs=20'
    
    def __init__(self):
        self.experimental_configs = []
        self.independent_designs = []
        self.coupled_designs = []
        self.parameter_labels = []
        self._input_output_maps = []
        self._job_ids = []

    def add_design(self, flag, value_it):
        """Add a range of values for the experiment with the range of values.

        Parameters
        ----------
        flag (str) - 
        """
        self.independent_designs.append((flag, list(value_it)))

    def add_coupled_design(self, coupled_design):
        self.coupled_designs.append(coupled_design)

    def set_rep_count(self, num_reps):
        self.add_design('rep_num', range(num_reps))

    def get_configs(self):
        # Translate the design structure into flag strings.
        exp_flag_strings = [['--{}={}'.format(f, v) for v in r] for (f, r) in self.independent_designs]

        # Produce the Cartesian set of configurations.
        indep_experimental_configs = list(itertools.product(*exp_flag_strings))
        coupled_configs = []
        # Scope this variable higher due to write-out coupling.
        coupled_flag_strs = []

        for coupled_design in self.coupled_designs:
            for d in coupled_design:
                coupled_flag_strs = [['--{}={}'.format(f, v) for v in r] for (f, r) in d]
                coupled_configs += list(itertools.product(*coupled_flag_strs))

        # Join each coupled config to each independent config
        if coupled_configs:
            experimental_configs = []
            for e in indep_experimental_configs:
                for c in coupled_configs:
                    experimental_configs.append(e + tuple(c))

        else:
            experimental_configs = indep_experimental_configs

        return experimental_configs

    def get_parameter_labels(self):
        parameter_labels = [f for (f, _) in self.independent_designs]

        for coupled_design in self.coupled_designs:
            coupled_flag_strs = []
            parameter_labels.extend([f for (f, _) in d for d in coupled_design])

        return(parameter_labels)

    def launch_experiment(self, exp_filename, log_dir,
                          manager='pbs', shuffle_job_order=True, job_fmt='dist_ex_{}',
                          walltime=None, select=None,
                          account=None, queue=None,
                          experiments_per_job=1):
        """Submit all the experiments to the batch scheduler.

        Parameters
        ----------
        exp_filename (str)
        log_dir (str)
        manager (str, one of ['pbs'])
        shuffle_job_order (bool)
        job_fmt (format str)
        walltime (None or 'HH:MM:SS')
        select (None or str)
        account (None or str)
        queue (None or str)
        experiments_per_job (int)
          - denotes how many experiments will be batched into each job submitted to the scheduler.
        """

        if account is None:
            account = self.account

        if queue is None:
            queue = self.queue

        if walltime is None:
            walltime = self.walltime

        if select is None:
            select = self.resources
            
        experimental_configs = self.get_configs()

        if shuffle_job_order:
            # Shuffle the submission order of configs to avoid asymetries.
            random.shuffle(experimental_configs)

        # Iterate over each experimental configuration, launching jobs.
        for job_idx, chunk in enumerate(chunker(experimental_configs, experiments_per_job)):
            if manager.lower() == 'pbs':
                # Use subproces to command qsub to submit a job.
                p = subprocess.Popen('qsub', stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)

                # Customize your options here.
                job_name = job_fmt.format(job_idx)
                temp_log_dir = os.path.join(log_dir, 'templog_' + str(job_idx))
                log_filename = 'templog_' + str(job_idx)

                command_prologue = "module load anaconda2/5.0.1 gcc/5.3.0 cudnn/6.0"
                command_epilogue = "wait"
                command = []
                for i, experimental_config in enumerate(chunk):
                    _cmd = "CUDA_VISIBLE_DEVICES={device} python {filename} {config} --log_dir={log_dir} --log_filename={log_filename}"
                    command.append(_cmd.format(device=i,
                                               filename=exp_filename,
                                               config=' '.join(experimental_config),
                                               log_dir=temp_log_dir,
                                               log_filename=log_filename,
                    ))
                command = ' &\n'.join(command)

                # Build IO maps.
                input_output_map = (experimental_config,
                                    temp_log_dir,
                                    log_filename)

                self._input_output_maps.append(input_output_map)

                job_script = JOB_SCRIPT_TEMPLATE.format(job_name=job_name,
                                                        walltime=walltime,
                                                        select=select,
                                                        command=command,
                                                        account=account,
                                                        queue=queue,
                                                        command_prologue=command_prologue,
                                                        command_epilogue=command_epilogue)

                print(job_script)

                # Send job_string to qsub.
                self._job_ids.append(p.communicate(job_script)[0])
                time.sleep(1)

            else:
                raise ValueError("Unknown manager '{}' supplied to launch_experiment(). Allowed managers: 'pbs'".format(manager))

    def join_job_output(self, log_dir, log_filename, max_runtime, job_ids):
        '''Wait until all the the jobs complete
        '''

        jobs_complete = False
        timeout = False
        elapsed_time = 0

        # Loop until timeout or all jobs complete.
        while not(jobs_complete) and not(timeout):
            print("-----------------")
            print('Time elapsed: ' + str(elapsed_time) + ' seconds.')
            time.sleep(10)
            elapsed_time += 10

            # Create a list to hold the Bool job complete flags
            job_complete_flags = []
            # Iterate over each job id string.
            for job_id in self._job_ids:
                # TODO: Handle job completion gracefully.
                # Issue qstat command to get job status.
                p = subprocess.Popen('qstat -r ' + job_id,
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     shell=True)

                output = p.communicate()

                try:
                    # Read the qstat out, parse the state, and conv to Boolean.
                    job_complete = output[0].split()[-2] == 'E'
                except:
                    job_complete = True

                # Print a diagnostic.
                print('Job', job_id[:-1], 'complete?', str(job_complete))

                job_complete_flags.append(job_complete)

                if job_complete:
                    p = subprocess.Popen('qdel -Wforce ' + job_id,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         shell=True)

            # And the job complete flags together.
            jobs_complete = (all(job_complete_flags))

            # Check if we've reached timeout.
            timeout = (elapsed_time > max_runtime)

            # # Accomodate Python 3+
            # with open(FLAGS.log_dir '/' + FLAGS.log_filename, 'w') as csvfile:

            # Accomodate Python 2.7 on Hokulea.
            with open(log_dir + '/' + log_filename, 'wb') as csvfile:
                # Manually note response varaibles.
                response_labels = ['step_num',
                                   'train_loss',
                                   'train_error',
                                   'val_loss',
                                   'val_error',
                                   'mean_running_time']

                # Join lists.
                headers = self.get_parameter_labels() + response_labels

                # Open a writer and write the header.
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(headers)

                # Iterate over each eperimental mapping and write out.
                for (input_flags,
                     output_dir,
                     output_filename) in self._input_output_maps:

                    input_row = []

                    # Process the flags into output values.
                    for flag in input_flags:
                        flag_val = flag.split('=')[1]
                        input_row.append(flag_val)

                    output_file = output_dir + output_filename
                    print("output_file")

                    # Check if the output file has been written.
                    if os.path.exists(output_file):
                        with open(output_file, 'rb') as f:
                            reader = csv.reader(f)
                            for output_row in reader:
                                csvwriter.writerow(input_row + output_row)
                        print("---------------------------------------")

                    else:
                        print("output filename not found: " + output_filename)

            print("-----------------")


def chunker(iterable, n=1):
    """Split the iterable into chunks of size n, yielding iterables of each new chunk"""
    if n < 1:
        raise ValueError("The size of the chunk 'n' must be 1 or greater")
    it = iter(iterable)
    while True:
        chunk = itertools.chain([next(it)], itertools.islice(it, 0, n-1))
        yield chunk
    
