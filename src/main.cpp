#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"

int main(int argc, char** argv) {

	std::string PathGraph = "/Users/teun/Dropbox/masterproject/betheequationsolver/src/SaveFiles/frozen_graph.pbXb";

	//Setup Input Tensors 
	tensorflow::Tensor Input1(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,1}));
	tensorflow::Tensor Input0(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,1}));
	// Output
	std::vector<tensorflow::Tensor> output;
	Input1.scalar<float>()() = 1.0;
	Input0.scalar<float>()() = 0.0;

	//initial declaration Tensorflow
	tensorflow::Session* session;
	tensorflow::Status status;
	status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
	if (!status.ok()) {
   		std::cout << status.ToString() << "\n";
    	return 1;
    }
    // Define Graph
	tensorflow::GraphDef graph_def;
	status = ReadBinaryProto(tensorflow::Env::Default(),PathGraph, &graph_def);
	
	if (!status.ok()) {
    	std::cout << status.ToString() << "\n";
     	return 1;
   	}

   	// Add the graph to the session
  	status = session->Create(graph_def);
    if (!status.ok()) {
    	std::cout << status.ToString() << "\n";
        return 1;
    }
 
    // Feed dict
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
        { "Input:0", Input0},
    };
    status = session->Run(inputs, {"Layer2/Output"},{}, &output);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
   		return 1;
    }
    auto Result = output[0].matrix<float>();
    std::cout << "Input: 0 | Output: "<< Result(0,0) << std::endl;
	
	inputs = {
        { "Input:0", Input1},
    };
    status = session->Run(inputs, {"Layer2/Output"},{}, &output);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
   		return 1;
    }
    auto Result1 = output[0].matrix<float>();
    std::cout << "Input: 1 | Output: "<< Result1(0,0) << std::endl;	
		
}


// #include "constants.hpp"
// #include "lieb_liniger_state.hpp"

// #include <Eigen/Dense>

// #include <ctime>
// #include <iomanip>
// #include <iostream>
    

// int main() {
//     clock_t time_a = clock();
//     for (int n = 0; n < 10000; n++) {
//         Eigen::VectorXd bethe_numbers = generate_bethe_numbers(100);
//         // std::cout << bethe_numbers << std::endl;
//         lieb_liniger_state llstate(1, 100, 100, bethe_numbers);
//         llstate.find_rapidities();
//         // std::cout << llstate.lambdas << std::endl;
//     }
//     clock_t time_b = clock();

//     std::cout << "time: " << (time_b - time_a) / (double)CLOCKS_PER_SEC << std::endl;


//     // Eigen::VectorXd bethe_numbers = generate_bethe_numbers(10);
    
//     // lieb_liniger_state llstate2(1, 100, 5);
//     // llstate2.find_rapidities(false);
//     // std::cout << llstate2.lambdas << std::endl;

//     // lieb_liniger_state llstate3(1, 200, 10, bethe_numbers);
//     // llstate3.find_rapidities(false);
//     // std::cout << llstate3.lambdas << std::endl;

//     // std::cout << llstate2.Is << std::endl;

// }

// // int main() {
// //     std::cout << "Hello" << std::endl;
// //     std::cout << "Machine epsilon " <<MACHINE_EPS << std::endl;

// //     std::cout << "Machine epsilon square" <<MACHINE_EPS_SQUARE << std::endl;
// // }
