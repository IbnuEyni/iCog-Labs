# import hexaly.optimizer
import sys


# The input files follow the "Taillard" format
def read_instance(filename):
    with open(filename) as f:
        lines = f.readlines()

    first_line = lines[1].split()
    # Number of jobs
    nb_jobs = int(first_line[0])
    # Number of machines
    nb_machines = int(first_line[1])

    # Processing times for each job on each machine (given in the processing order)
    processing_times_in_processing_order = [[int(lines[i].split()[j])
                                             for j in range(nb_machines)]
                                            for i in range(3, 3 + nb_jobs)]

    # Processing order of machines for each job
    machine_order = [[int(lines[i].split()[j]) - 1 for j in range(nb_machines)]
                     for i in range(4 + nb_jobs, 4 + 2 * nb_jobs)]

    # Reorder processing times: processing_time[j][m] is the processing time of the
    # task of job j that is processed on machine m
    processing_time = [[processing_times_in_processing_order[j][machine_order[j].index(m)]
                        for m in range(nb_machines)]
                       for j in range(nb_jobs)]

    # Trivial upper bound for the start times of the tasks
    max_start = sum(sum(processing_time[j]) for j in range(nb_jobs))

    return nb_jobs, nb_machines, processing_time, machine_order, max_start


def main(instance_file, output_file, time_limit):
    nb_jobs, nb_machines, processing_time, machine_order, max_start = read_instance(instance_file)

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Interval decisions: time range of each task
        # tasks[j][m] is the interval of time of the task of job j which is processed
        # on machine m
        tasks = [[model.interval(0, max_start) for m in range(nb_machines)]
                 for j in range(nb_jobs)]

        # Task duration constraints
        for j in range(nb_jobs):
            for m in range(0, nb_machines):
                model.constraint(model.length(tasks[j][m]) == processing_time[j][m])

        # Create an Hexaly array in order to be able to access it with "at" operators
        task_array = model.array(tasks)

        # Precedence constraints between the tasks of a job
        for j in range(nb_jobs):
            for k in range(nb_machines - 1):
                model.constraint(
                    tasks[j][machine_order[j][k]] < tasks[j][machine_order[j][k + 1]])

        # Sequence of tasks on each machine
        jobs_order = [model.list(nb_jobs) for m in range(nb_machines)]

        for m in range(nb_machines):
            # Each job has a task scheduled on each machine
            sequence = jobs_order[m]
            model.constraint(model.eq(model.count(sequence), nb_jobs))

            # Disjunctive resource constraints between the tasks on a machine
            sequence_lambda = model.lambda_function(
                lambda i: model.lt(model.at(task_array, sequence[i], m),
                                   model.at(task_array, sequence[i + 1], m)))
            model.constraint(model.and_(model.range(0, nb_jobs - 1), sequence_lambda))

        # Minimize the makespan: end of the last task of the last job
        makespan = model.max([model.end(tasks[j][machine_order[j][nb_machines - 1]])
                             for j in range(nb_jobs)])
        model.minimize(makespan)

        model.close()

        # Parameterize the optimizer
        optimizer.param.time_limit = time_limit

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        # - for each machine, the job sequence
        #
        if output_file != None:
            final_jobs_order = [list(jobs_order[m].value) for m in range(nb_machines)]
            with open(output_file, "w") as f:
                print("Solution written in file ", output_file)
                for m in range(nb_machines):
                    for j in range(nb_jobs):
                        f.write(str(final_jobs_order[m][j]) + " ")
                    f.write("\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python jobshop.py instance_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 60
    main(instance_file, output_file, time_limit)
