import java.util.*;
import java.io.*;
import com.hexaly.optimizer.*;

public class Jobshop {
    // Number of jobs
    private int nbJobs;
    // Number of machines
    private int nbMachines;
    // Processing time on each machine for each job (given in the machine order)
    private long[][] processingTime;
    // Processing order of machines for each job
    private int[][] machineOrder;
    // Trivial upper bound for the start times of the tasks
    private long maxStart;

    // Hexaly Optimizer
    final HexalyOptimizer optimizer;
    // Decision variables: time range of each task
    private HxExpression[][] tasks;
    // Decision variables: sequence of tasks on each machine
    private HxExpression[] jobsOrder;
    // Objective = minimize the makespan: end of the last task of the last job
    private HxExpression makespan;

    public Jobshop(HexalyOptimizer optimizer) throws IOException {
        this.optimizer = optimizer;
    }

    // The input files follow the "Taillard" format
    public void readInstance(String fileName) throws IOException {
        try (Scanner input = new Scanner(new File(fileName))) {
            input.nextLine();
            nbJobs = input.nextInt();
            nbMachines = input.nextInt();

            input.nextLine();
            input.nextLine();
            // Processing times for each job on each machine (given in the processing order)
            long[][] processingTimesInProcessingOrder = new long[nbJobs][nbMachines];
            for (int j = 0; j < nbJobs; ++j) {
                for (int m = 0; m < nbMachines; ++m) {
                    processingTimesInProcessingOrder[j][m] = input.nextInt();
                }
            }
            // Processing order of machines for each job
            input.nextLine();
            input.nextLine();
            machineOrder = new int[nbJobs][nbMachines];
            for (int j = 0; j < nbJobs; ++j) {
                for (int m = 0; m < nbMachines; ++m) {
                    machineOrder[j][m] = input.nextInt() - 1;
                }
            }
            // Reorder processing times: processingTime[j][m] is the processing time of the
            // task of job j that is processed on machine m
            processingTime = new long[nbJobs][nbMachines];
            // Trivial upper bound for the start times of the tasks
            maxStart = 0;
            for (int j = 0; j < nbJobs; ++j) {
                for (int m = 0; m < nbMachines; ++m) {
                    int machineIndex = nbMachines;
                    for (int k = 0; k < nbMachines; ++k) {
                        if (machineOrder[j][k] == m) {
                            machineIndex = k;
                            break;
                        }
                    }
                    processingTime[j][m] = processingTimesInProcessingOrder[j][machineIndex];
                    maxStart += processingTime[j][m];
                }
            }
        }
    }

    public void solve(int timeLimit) {
        // Declare the optimization model
        HxModel model = optimizer.getModel();

        // Interval decisions: time range of each task
        // tasks[j][m] is the interval of time of the task of job j which is processed
        // on machine m
        tasks = new HxExpression[nbJobs][nbMachines];
        for (int j = 0; j < nbJobs; ++j) {
            for (int m = 0; m < nbMachines; ++m) {
                tasks[j][m] = model.intervalVar(0, maxStart);

                // Task duration constraints
                model.constraint(model.eq(model.length(tasks[j][m]), processingTime[j][m]));
            }
        }

        // Create a HexalyOptimizer array in order to be able to access it with "at"
        // operators
        HxExpression taskArray = model.array(tasks);

        // Precedence constraints between the tasks of a job
        for (int j = 0; j < nbJobs; ++j) {
            for (int k = 0; k < nbMachines - 1; ++k) {
                model.constraint(model.lt(tasks[j][machineOrder[j][k]], tasks[j][machineOrder[j][k + 1]]));
            }
        }

        // Sequence of tasks on each machine
        jobsOrder = new HxExpression[nbMachines];
        for (int m = 0; m < nbMachines; ++m) {
            jobsOrder[m] = model.listVar(nbJobs);
        }

        for (int m = 0; m < nbMachines; ++m) {
            // Each job has a task scheduled on each machine
            HxExpression sequence = jobsOrder[m];
            model.constraint(model.eq(model.count(sequence), nbJobs));

            // Disjunctive resource constraints between the tasks on a machine
            HxExpression mExpr = model.createConstant(m);
            HxExpression sequenceLambda = model
                    .lambdaFunction(i -> model.lt(model.at(taskArray, model.at(sequence, i), mExpr),
                            model.at(taskArray, model.at(sequence, model.sum(i, 1)), mExpr)));
            model.constraint(model.and(model.range(0, nbJobs - 1), sequenceLambda));
        }

        // Minimize the makespan: end of the last task of the last job
        makespan = model.max();
        for (int j = 0; j < nbJobs; ++j) {
            makespan.addOperand(model.end(tasks[j][machineOrder[j][nbMachines - 1]]));
        }
        model.minimize(makespan);

        model.close();

        // Parameterize the optimizer
        optimizer.getParam().setTimeLimit(timeLimit);

        optimizer.solve();
    }

    /*
     * Write the solution in a file with the following format:
     * - for each machine, the job sequence
     */
    public void writeSolution(String fileName) throws IOException {
        try (PrintWriter output = new PrintWriter(fileName)) {
            System.out.println("Solution written in file " + fileName);

            for (int m = 0; m < nbMachines; ++m) {
                HxCollection finalJobsOrder = jobsOrder[m].getCollectionValue();
                for (int i = 0; i < nbJobs; ++i) {
                    int j = Math.toIntExact(finalJobsOrder.get(i));
                    output.write(j + " ");
                }
                output.write("\n");
            }
        }
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: java Jobshop instanceFile [outputFile] [timeLimit]");
            System.exit(1);
        }

        String instanceFile = args[0];
        String outputFile = args.length > 1 ? args[1] : null;
        String strTimeLimit = args.length > 2 ? args[2] : "60";

        try (HexalyOptimizer optimizer = new HexalyOptimizer()) {
            Jobshop model = new Jobshop(optimizer);
            model.readInstance(instanceFile);
            model.solve(Integer.parseInt(strTimeLimit));
            if (outputFile != null) {
                model.writeSolution(outputFile);
            }
        } catch (Exception ex) {
            System.err.println(ex);
            ex.printStackTrace();
            System.exit(1);
        }
    }
}
