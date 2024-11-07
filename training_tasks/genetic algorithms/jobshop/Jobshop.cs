using System;
using System.IO;
using Hexaly.Optimizer;

public class Jobshop : IDisposable
{
    // Number of jobs
    private int nbJobs;

    // Number of machines
    private int nbMachines;

    // Processing order of machines for each job
    private int[,] machineOrder;

    // Processing time on each machine for each job (given in the machine order)
    private long[,] processingTime;

    // Trivial upper bound for the start times of the tasks
    private long maxStart;

    // Hexaly Optimizer
    private HexalyOptimizer optimizer;

    // Decision variables: time range of each task
    private HxExpression[,] tasks;

    // Decision variables: sequence of tasks on each machine
    private HxExpression[] jobsOrder;

    // Objective = minimize the makespan: end of the last task of the last job
    private HxExpression makespan;

    public Jobshop()
    {
        optimizer = new HexalyOptimizer();
    }

    // The input files follow the "Taillard" format
    public void ReadInstance(string fileName)
    {
        using (StreamReader input = new StreamReader(fileName))
        {
            input.ReadLine();
            string[] splitted = input.ReadLine().Split(' ');
            nbJobs = int.Parse(splitted[0]);
            nbMachines = int.Parse(splitted[1]);

            // Processing times for each job on each machine (given in the processing order)
            input.ReadLine();
            long[,] processingTimesInProcessingOrder = new long[nbJobs, nbMachines];
            for (int j = 0; j < nbJobs; ++j)
            {
                splitted = input.ReadLine().Trim().Split(' ');
                for (int m = 0; m < nbMachines; ++m)
                    processingTimesInProcessingOrder[j, m] = long.Parse(splitted[m]);
            }

            // Processing order of machines for each job
            input.ReadLine();
            machineOrder = new int[nbJobs, nbMachines];
            for (int j = 0; j < nbJobs; ++j)
            {
                splitted = input.ReadLine().Trim().Split(' ');
                for (int m = 0; m < nbMachines; ++m)
                    machineOrder[j, m] = int.Parse(splitted[m]) - 1;
            }

            // Reorder processing times: processingTime[j, m] is the processing time of the
            // task of job j that is processed on machine m
            processingTime = new long[nbJobs, nbMachines];
            // Trivial upper bound for the start times of the tasks
            maxStart = 0;
            for (int j = 0; j < nbJobs; ++j)
            {
                for (int m = 0; m < nbMachines; ++m)
                {
                    int machineIndex = nbMachines;
                    for (int k = 0; k < nbMachines; ++k)
                    {
                        if (machineOrder[j, k] == m)
                        {
                            machineIndex = k;
                            break;
                        }
                    }
                    processingTime[j, m] = processingTimesInProcessingOrder[j, machineIndex];
                    maxStart += processingTime[j, m];
                }
            }
        }
    }

    public void Dispose()
    {
        optimizer.Dispose();
    }

    public void Solve(int timeLimit)
    {
        // Declare the optimization model
        HxModel model = optimizer.GetModel();

        // Interval decisions: time range of each task
        // tasks[j][m] is the interval of time of the task of job j which is processed on machine m
        tasks = new HxExpression[nbJobs, nbMachines];
        for (int j = 0; j < nbJobs; ++j)
        {
            for (int m = 0; m < nbMachines; ++m)
            {
                tasks[j, m] = model.Interval(0, maxStart);

                // Task duration constraints
                model.Constraint(model.Length(tasks[j, m]) == processingTime[j, m]);
            }
        }

        // Create a HexalyOptimizer array in order to be able to access it with "at" operators
        HxExpression taskArray = model.Array(tasks);

        // Precedence constraints between the tasks of a job
        for (int j = 0; j < nbJobs; ++j)
        {
            for (int k = 0; k < nbMachines - 1; ++k)
            {
                model.Constraint(tasks[j, machineOrder[j, k]] < tasks[j, machineOrder[j, k + 1]]);
            }
        }

        // Sequence of tasks on each machine
        jobsOrder = new HxExpression[nbMachines];
        for (int m = 0; m < nbMachines; ++m)
            jobsOrder[m] = model.List(nbJobs);

        for (int m = 0; m < nbMachines; ++m)
        {
            // Each job has a task scheduled on each machine
            HxExpression sequence = jobsOrder[m];
            model.Constraint(model.Count(sequence) == nbJobs);

            // Disjunctive resource constraints between the tasks on a machine
            HxExpression sequenceLambda = model.LambdaFunction(
                i => taskArray[sequence[i], m] < taskArray[sequence[i + 1], m]
            );
            model.Constraint(model.And(model.Range(0, nbJobs - 1), sequenceLambda));
        }

        // Minimize the makespan: end of the last task of the last job
        makespan = model.Max();
        for (int j = 0; j < nbJobs; ++j)
            makespan.AddOperand(model.End(tasks[j, machineOrder[j, nbMachines - 1]]));
        model.Minimize(makespan);

        model.Close();

        // Parameterize the optimizer
        optimizer.GetParam().SetTimeLimit(timeLimit);

        optimizer.Solve();
    }

    /* Write the solution in a file with the following format:
     *  - for each machine, the job sequence */
    public void WriteSolution(string fileName)
    {
        using (StreamWriter output = new StreamWriter(fileName))
        {
            Console.WriteLine("Solution written in file " + fileName);
            for (int m = 0; m < nbMachines; ++m)
            {
                HxCollection finalJobsOrder = jobsOrder[m].GetCollectionValue();
                for (int i = 0; i < nbJobs; ++i)
                {
                    int j = (int)finalJobsOrder.Get(i);
                    output.Write(j + " ");
                }
                output.WriteLine();
            }
        }
    }

    public static void Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.WriteLine("Usage: Jobshop instanceFile [outputFile] [timeLimit]");
            System.Environment.Exit(1);
        }

        string instanceFile = args[0];
        string outputFile = args.Length > 1 ? args[1] : null;
        string strTimeLimit = args.Length > 2 ? args[2] : "60";

        using (Jobshop model = new Jobshop())
        {
            model.ReadInstance(instanceFile);
            model.Solve(int.Parse(strTimeLimit));
            if (outputFile != null)
                model.WriteSolution(outputFile);
        }
    }
}
