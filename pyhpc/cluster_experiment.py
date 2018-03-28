#!/usr/bin/env python
# Example PBS cluster job submission in Python

from __future__ import print_function

import os
import csv
import time
import random
import itertools
import subprocess

JOB_SCRIPT_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
JOB_SCRIPT_TEMPLATE = open(os.path.join(JOB_SCRIPT_TEMPLATE_DIR, 'default_job_template.pbs'), 'r').read()


class ClusterExperiment(object):
    account = 'MHPCC96670DA1'
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
        flag (str)
          This will be turned into the parameter name passed to the experiment script.
          i.e. "batch_size" becomes "--batch_size=..." in the job script
        value_it (iterable of values for the flag)
          This will be expanded to have one value per experiment.
          e.g. add_design('batch_size', [128, 256]) becomes two different experiments,
            one for --batch_size=128, and another for --batch_size=256
        """
        self.independent_designs.append((flag, list(value_it)))

    def add_coupled_design(self, coupled_design):
        self.coupled_designs.append(coupled_design)

    def set_rep_count(self, num_reps):
        self.add_design('rep_num', range(num_reps))

    def get_configs(self):
        # Translate the design structure into flag strings.

        # Translate the design structure into flag strings.
        exp_flag_strings = [['--' + f + '=' + str(v) for v in r]
                            for (f, r) in self.independent_designs]

        # Produce the Cartesian set of configurations.
        indep_experimental_configs = list(itertools.product(*exp_flag_strings))

        # Initialize a list to store coupled configurations.
        coupled_configs = []

        # Scope this variable higher due to write-out coupling.
        coupled_flag_strs = []

        for coupled_design in self.coupled_designs:

            for d in coupled_design:

                coupled_flag_strs = [['--' + f + '=' + str(v) for v in r]
                                     for (f, r) in d]

                coupled_configs += list(itertools.product(*coupled_flag_strs))

        # Initialize empty experimental configs list...
        experimental_configs = []

        # ...and if there are coupled configs...
        if coupled_configs:

            # ...iterate over each independent config...
            for e in indep_experimental_configs:

                    # ...then for each coupled config...
                    for c in coupled_configs:

                        # ...join the coupled config to the independent one.
                        experimental_configs.append(e + tuple(c))

        # Otherwise, ....
        else:

            # ...just pass the independent experiments through.
            experimental_configs = indep_experimental_configs

        return(experimental_configs)

    def get_parameter_labels(self):

        parameter_labels = []

        for (f, _) in self.independent_designs:

            parameter_labels.append(f)

        for coupled_design in self.coupled_designs:

            coupled_flag_strs = []

            for d in coupled_design:

                coupled_flag_strs = [f for (f, _) in d]

            parameter_labels += coupled_flag_strs

        return(parameter_labels)

    def launch_experiment(self,
                          exp_filename,
                          log_dir,
                          manager='pbs',
                          shuffle_job_order=True,
                          job_fmt='dist_ex_{}',
                          command_prologue="",
                          walltime=None,
                          select=None,
                          account=None,
                          queue=None,
                          experiments_per_job=1,
                          dry_run=False):
        """Submit all the experiments to the batch scheduler.

        Parameters
        ----------
        exp_filename (str)
        log_dir (str)
        manager (str, one of ['pbs'])
        shuffle_job_order (bool)
        job_fmt (format str)
        command_prologue (str, e.g. 'module load <module>')
        walltime (None or 'HH:MM:SS')
        select (None or str)
        account (None or str)
        queue (None or str)
        experiments_per_job (int)
          - denotes how many experiments will be batched into each job submitted to the scheduler.
        dry_run (bool)
          - Tells whether or not the script will actually be submitted to the scheduler,
            or if we simply print what would have been submitted.
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
                if not dry_run:
                    p = subprocess.Popen('qsub',
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         shell=True)

                # Customize your options here.
                job_name = job_fmt.format(job_idx)
                temp_log_dir = os.path.join(log_dir, 'templog_' + str(job_idx))
                log_filename = 'templog_' + str(job_idx)

                # TODO: Command prologue and epilogue should be configurable.
                # command_prologue = "module load anaconda3/5.0.1 tensorflow"
                command_epilogue = "wait"
                command = []

                # Build the commands that will be run simultaneously in the PBS script.
                # This entails running them all in the background and waiting for them to complete
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

                print("----Start Job Script----")
                print(job_script)
                print("----End Job Script----")

                # Send job_script to qsub.
                if not dry_run:

                    # Convert the job_script str to bytes for p.
                    job_script = bytes(job_script, 'utf-8')

                    # Send job_string to qsub, returning a job ID in bytes.
                    job_id = p.communicate(job_script)[0]

                    # Convert the bytes to an ID string.
                    job_id = str(job_id)[2:-2]

                    self._job_ids.append(job_id)
                    time.sleep(1)

            else:
                raise ValueError("Unknown manager '{}' supplied to launch_experiment(). Allowed managers: 'pbs'".format(manager))

    def join_job_output(self, log_dir, log_filename, max_runtime, response_labels):
        '''Wait until all the the jobs complete while writing partial results.
        '''

        jobs_complete = False
        timeout = False
        elapsed_time = 0

        # Loop until timeout or all jobs complete.
        while not(jobs_complete) and not(timeout):

            print("----Writing Partial Results----")
            print('Time elapsed: ' + str(elapsed_time) + ' seconds.')

            # TODO: Make a correct counter.
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
                print(job_id)

                # Get the subprocess output from qstat.
                output = p.communicate()

                # Compute the job completion flag.
                try:

                    # Read the qstat out, parse the state, and conv to Boolean.
                    job_complete = output[0].split()[-2] == 'E'

                except:

                    job_complete = True

                # Print a diagnostic.
                print('Job ' +
                      job_id +
                      ' complete? ' +
                      str(job_complete) +
                      '.')

                # Append the completion flag.
                job_complete_flags.append(job_complete)

                # If the job is complete...
                if job_complete:

                    # ...clear it form the queue.
                    p = subprocess.Popen('qdel -Wforce ' + job_id,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         shell=True)

            # And the job complete flags together.
            jobs_complete = (all(job_complete_flags))

            # Check if we've reached timeout.
            timeout = (elapsed_time > max_runtime)

            # Open a csv for writeout.
            with open(log_dir + '/' + log_filename, 'w') as csvfile:

                # Join the parameter labels and respons labels, making a header.
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

                    # Check if the output file has been written.
                    if os.path.exists(output_file):

                        with open(output_file, 'rb') as f:

                            reader = csv.reader(f)

                            for output_row in reader:

                                csvwriter.writerow(input_row + output_row)

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
    
