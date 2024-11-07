#include "optimizer/hexalyoptimizer.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

using namespace hexaly;
using namespace std;

class Jobshop {
private:
    // Number of jobs
    int nbJobs;
    // Number of machines
    int nbMachines;
    // Processing order of machines for each job
    vector<vector<int>> machineOrder;
    // Processing time on each machine for each job (given in the machine order)
    vector<vector<int>> processingTime;
    // Trivial upper bound for the start times of the tasks
    int maxStart;

    // Hexaly Optimizer
    HexalyOptimizer optimizer;
    // Decision variables: time range of each task
    vector<vector<HxExpression>> tasks;
    // Decision variables: sequence of tasks on each machine
    vector<HxExpression> jobsOrder;
    // Objective = minimize the makespan: end of the last task of the last job
    HxExpression makespan;

public:
    Jobshop() : optimizer() {}

    // The input files follow the "Taillard" format
    void readInstance(const string& fileName) {
        ifstream infile;
        infile.exceptions(ifstream::failbit | ifstream::badbit);
        infile.open(fileName.c_str());

        infile.ignore(numeric_limits<streamsize>::max(), '\n');
        infile >> nbJobs;
        infile >> nbMachines;
        infile.ignore(numeric_limits<streamsize>::max(), '\n');

        // Processing times for each job on each machine (given in the processing order)
        infile.ignore(numeric_limits<streamsize>::max(), '\n');
        vector<vector<int>> processingTimesInProcessingOrder = vector<vector<int>>(nbJobs, vector<int>(nbMachines));
        for (int j = 0; j < nbJobs; ++j) {
            for (int m = 0; m < nbMachines; ++m) {
                infile >> processingTimesInProcessingOrder[j][m];
            }
        }

        // Processing order of machines for each job
        infile.ignore(numeric_limits<streamsize>::max(), '\n');
        infile.ignore(numeric_limits<streamsize>::max(), '\n');
        machineOrder.resize(nbJobs);
        for (int j = 0; j < nbJobs; ++j) {
            machineOrder[j].resize(nbMachines);
            for (int m = 0; m < nbMachines; ++m) {
                int x;
                infile >> x;
                machineOrder[j][m] = x - 1;
            }
        }

        // Reorder processing times: processingTime[j][m] is the processing time of the
        // task of job j that is processed on machine m
        for (int j = 0; j < nbJobs; ++j) {
            processingTime.resize(nbJobs);
            for (int m = 0; m < nbMachines; ++m) {
                processingTime[j].resize(nbMachines);
                vector<int>::iterator findM = find(machineOrder[j].begin(), machineOrder[j].end(), m);
                unsigned int k = distance(machineOrder[j].begin(), findM);
                processingTime[j][m] = processingTimesInProcessingOrder[j][k];
            }
        }

        // Trivial upper bound for the start times of the tasks
        maxStart = 0;
        for (int j = 0; j < nbJobs; ++j)
            maxStart += accumulate(processingTime[j].begin(), processingTime[j].end(), 0);

        infile.close();
    }

    void solve(int timeLimit) {
        // Declare the optimization model
        HxModel model = optimizer.getModel();

        // Interval decisions: time range of each task
        // tasks[j][m] is the interval of time of the task of job j which is processed on machine m
        tasks.resize(nbJobs);
        for (unsigned int j = 0; j < nbJobs; ++j) {
            tasks[j].resize(nbMachines);
            for (unsigned int m = 0; m < nbMachines; ++m) {
                tasks[j][m] = model.intervalVar(0, maxStart);

                // Task duration constraints
                model.constraint(model.length(tasks[j][m]) == processingTime[j][m]);
            }
        }

        // Create an Hexaly array in order to be able to access it with "at" operators
        HxExpression taskArray = model.array();
        for (int j = 0; j < nbJobs; ++j) {
            taskArray.addOperand(model.array(tasks[j].begin(), tasks[j].end()));
        }

        // Precedence constraints between the tasks of a job
        for (int j = 0; j < nbJobs; ++j) {
            for (int k = 0; k < nbMachines - 1; ++k) {
                model.constraint(tasks[j][machineOrder[j][k]] < tasks[j][machineOrder[j][k + 1]]);
            }
        }

        // Sequence of tasks on each machine
        jobsOrder.resize(nbMachines);
        for (int m = 0; m < nbMachines; ++m) {
            jobsOrder[m] = model.listVar(nbJobs);
        }

        for (int m = 0; m < nbMachines; ++m) {
            // Each job has a task scheduled on each machine
            HxExpression sequence = jobsOrder[m];
            model.constraint(model.eq(model.count(sequence), nbJobs));

            // Disjunctive resource constraints between the tasks on a machine
            HxExpression sequenceLambda = model.createLambdaFunction([&](HxExpression i) {
                return model.at(taskArray, sequence[i], m) < model.at(taskArray, sequence[i + 1], m);
            });
            model.constraint(model.and_(model.range(0, nbJobs - 1), sequenceLambda));
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

    /* Write the solution in a file with the following format:
     *  - for each machine, the job sequence */
    void writeSolution(const string& fileName) {
        ofstream outfile;
        outfile.exceptions(ofstream::failbit | ofstream::badbit);
        outfile.open(fileName.c_str());
        cout << "Solution written in file " << fileName << endl;

        for (int m = 0; m < nbMachines; ++m) {
            HxCollection finalJobsOrder = jobsOrder[m].getCollectionValue();
            for (int j = 0; j < nbJobs; ++j) {
                outfile << finalJobsOrder.get(j) << " ";
            }
            outfile << endl;
        }
        outfile.close();
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: jobshop instanceFile [outputFile] [timeLimit]" << endl;
        exit(1);
    }

    const char* instanceFile = argv[1];
    const char* outputFile = argc > 2 ? argv[2] : NULL;
    const char* strTimeLimit = argc > 3 ? argv[3] : "60";

    Jobshop model;
    try {
        model.readInstance(instanceFile);
        const int timeLimit = atoi(strTimeLimit);
        model.solve(timeLimit);
        if (outputFile != NULL)
            model.writeSolution(outputFile);
        return 0;
    } catch (const exception& e) {
        cerr << "An error occurred: " << e.what() << endl;
        return 1;
    }
}
