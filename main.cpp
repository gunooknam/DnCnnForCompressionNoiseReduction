#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h> // One-stop header.
#include <cstdio>
#include "cxxopts.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

inline bool file_exists(const std::string &name) {
	std::cout << "checking file exists " << name << std::endl;
	std::ifstream f(name.c_str());
	return f.good();
}

cxxopts::ParseResult parse(int argc, char *argv[]) {
	try {
		cxxopts::Options options(argv[0], " - SR command line options");

		options.add_options()
			("help", "Print help")
			("w, weights", "weights file path", cxxopts::value<std::string>())
			("i, input", "input image file path", cxxopts::value<std::string>())
			("o, output", "output image file path", cxxopts::value<std::string>());

		auto result = options.parse(argc, argv);

		if (result.count("help") || result.arguments().empty()) {
			std::cout << options.help() << std::endl;
			exit(0);
		}

		bool missing = false;
		for (auto o : { "w", "i", "o" }) {
			if (result.count(o) == 0) {
				std::cerr << "missing arg " << o << std::endl;
				missing = true;
			}
		}
		if (missing) exit(-1);

		missing = false;
		for (auto o : { "w", "i" }) {
			if (!file_exists(result[o].as<std::string>())) {
				std::cerr << "missing file " << o << std::endl;
				missing = true;
			}
		}
		if (missing) exit(-1);

		std::cout << "weights = " << result["weights"].as<std::string>()
			<< std::endl;

		std::cout << "input = " << result["input"].as<std::string>()
			<< std::endl;

		std::cout << "output = " << result["output"].as<std::string>()
			<< std::endl;

		return result;

	}
	catch (const cxxopts::OptionException &e) {
		std::cout << "error parsing options: " << e.what() << std::endl;
		exit(1);
	}
}

void check_cuda() {
	int count = 0;
	if (cudaGetDeviceCount(&count) == cudaError::cudaSuccess) {
		std::printf("%d.%d", CUDA_VERSION / 1000, (CUDA_VERSION / 10) % 100);
		if (count == 0) {
			std::cerr << "couldn't get number of gpus";
			exit(-1);
		}
	}
	else {
		std::cerr << "couldn't get cuda device count";
		exit(-1);
	}
}

at::Tensor cv2_to_torch(cv::Mat frame) {
	frame.convertTo(frame, CV_32FC1, 1.0f / 255.0f);
	at::Tensor input_tensor = torch::from_blob(
		frame.ptr<float>(),
		{ 1, frame.size().height, frame.size().width, frame.channels() }
	);
	input_tensor = input_tensor.permute({ 0, 3, 1, 2 });
	return input_tensor.clone();
}

cv::Mat cv2_image(const std::string &fp) {
	cv::Mat image = imread(fp, cv::IMREAD_GRAYSCALE);
	return image;
}

void display_cv_image(cv::Mat image) {
	namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	imshow("Display window", image);
	cv::waitKey(0);
}

cv::Mat torch_to_cv2(at::Tensor tensor) {
	tensor = tensor.detach().permute({ 1, 2, 0 }); // detach() 
	tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
	tensor = tensor.to(torch::kCPU);
	cv::Mat result_img(tensor.size(0), tensor.size(1), CV_8UC1);
	// -------------------------------------------------------------------------------------- //
	std::memcpy((void *)result_img.data, tensor.data_ptr(), sizeof(torch::kU8) * tensor.numel());
	return result_img;
	// --------------------------------------------------------------------------------------- //
}

int main(int argc, char *argv[]) {
	torch::manual_seed(1);
	check_cuda();

	auto result = parse(argc, argv);
	const auto &arguments = result.arguments();

	torch::jit::script::Module model;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		model = torch::jit::load(result["weights"].as<std::string>());
		model.to(at::kCUDA);
		
	}
	catch (const c10::Error &e) {
		std::cerr << "error loading the model\n";
		std::cerr << result["weights"].as<std::string>();
		return -1;
	}

	// Create a vector of inputs.
	std::vector<torch::jit::IValue> inputs;

	cv::Mat img = cv2_image(result["input"].as<std::string>());
	std::cout << img.size() << std::endl;
	//image.reshape( 1, image.cols * image.rows );
	at::Tensor t_img = cv2_to_torch(img); // cv to torch code 구현
	t_img = t_img.to(at::kCUDA); // input에 cuda를 붙여줘야 한다. 불일치시 에러
	
	inputs.emplace_back(t_img); 
	at::Tensor output = model.forward(inputs).toTensor();

	cv::Mat sr_img = torch_to_cv2(output[0]);
    display_cv_image(sr_img);
	cv::imwrite(result["output"].as<std::string>(), sr_img);
}

//for (const auto & pair : model.named_parameters()) {
//	//pair.value
//	std::cout<< pair.name<<", " <<pair.value.sizes()<< std::endl;
//}
// Execute the model and turn its output into a tensor.
//auto data = batch.data.to(device), targets = batch.target.to(device);
//optimizer.zero_grad();
//	auto output = model.forward(data);