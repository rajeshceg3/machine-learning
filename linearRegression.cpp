#include <stdio.h>
#include <shark/Data/Csv.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Algorithms/Trainers/LinearRegression.h>

// Library dependencies -lboost_serialization -lshark -lcblas
#include <iostream>
using namespace std;
using namespace shark;

int main (int argc, char ** argv) {
    Data<RealVector> inputs;
    Data<RealVector> labels;
    importCSV(inputs, "inputs.csv");
    importCSV(labels, "labels.csv");
	
	  RegressionDataset data(inputs, labels);
	
	  // Training the model
    LinearRegression trainer;
    LinearModel<> model;
 
    trainer.train(model, data);
  
    // Model parameters
    cout << "Intercept: "<<model.offset() << endl;
    cout << "Matrix: "<< model.matrix() << endl;
 
    SquaredLoss<> loss;
    Data<RealVector> prediction = model(data.inputs()); 
    cout << "Squared Loss: " << loss(data.labels(), prediction) << endl;

  	return 0;
}
