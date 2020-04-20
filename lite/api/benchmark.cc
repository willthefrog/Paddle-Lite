// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <sys/time.h>
#include <ctime>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <ctime>
#include <numeric>
#include <string>
#include <vector>
#include <iterator>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include "lite/api/paddle_api.h"
#include "lite/core/device_info.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/string.h"
#include <cstdint>

DEFINE_string(optimized_model_path,
              "",
              "the path of the model that is optimized by opt.");
DEFINE_string(model_dir,
              "",
              "the path of the model, the model and param files is under "
              "model_dir.");
DEFINE_string(model_path,
              "",
              "model file path.");
DEFINE_string(model_filename,
              "",
              "the filename of model file. When the model is combined formate, "
              "please set model_file.");
DEFINE_string(param_filename,
              "",
              "the filename of param file, set param_file when the model is "
              "combined formate.");
DEFINE_bool(run_model_optimize,
            false,
            "if set true, apply model_optimize_tool to "
            "model and use optimized model to test. ");
DEFINE_string(input_file, "", "path to input file");
DEFINE_int32(warmup, 5, "warmup times");
DEFINE_int32(repeats, 20, "repeats times");
DEFINE_int32(power_mode,
             3,
             "arm power mode: "
             "0 for big cluster, "
             "1 for little cluster, "
             "2 for all cores, "
             "3 for no bind");
DEFINE_int32(threads, 1, "threads num");
DEFINE_bool(is_quantized_model,
            false,
            "if set true, "
            "test the performance of the quantized model. ");

namespace paddle {
namespace lite_api {

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

template <typename T>
inline T prod(std::vector<T> input)
{
  return std::accumulate(
    input.begin(), input.end(), 1, std::multiplies<T>());
}

void OutputOptModel(const std::string& save_optimized_model_dir) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  if (!FLAGS_model_filename.empty() && !FLAGS_param_filename.empty()) {
    config.set_model_file(FLAGS_model_dir + "/" + FLAGS_model_filename);
    config.set_param_file(FLAGS_model_dir + "/" + FLAGS_param_filename);
  }
  std::vector<Place> vaild_places = {
      Place{TARGET(kARM), PRECISION(kFloat)},
  };
  config.set_valid_places(vaild_places);
  auto predictor = lite_api::CreatePaddlePredictor(config);

  int ret = system(
      paddle::lite::string_format("rm -rf %s", save_optimized_model_dir.c_str())
          .c_str());
  if (ret == 0) {
    LOG(INFO) << "Delete old optimized model " << save_optimized_model_dir;
  }
  predictor->SaveOptimizedModel(save_optimized_model_dir,
                                LiteModelType::kNaiveBuffer);
  LOG(INFO) << "Load model from " << FLAGS_model_dir;
  LOG(INFO) << "Save optimized model to " << save_optimized_model_dir;
}

int64_t ShapeProduction(const std::vector<int64_t>& shape) {
  int64_t num = 1;
  for (auto i : shape) {
    num *= i;
  }
  return num;
}

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
void Run(const std::string& model_path,
         const std::string& input_file) {
  // set config and create predictor
  lite_api::MobileConfig config;
  config.set_threads(FLAGS_threads);
  config.set_power_mode(static_cast<PowerMode>(FLAGS_power_mode));
  config.set_model_from_file(model_path);

  auto predictor = lite_api::CreatePaddlePredictor(config);
  auto input_names = predictor->GetInputNames();
  auto output_names = predictor->GetOutputNames();

  int64_t num_inputs = 0;
  std::ifstream fin(input_file, std::ios::binary);
  if (!fin.is_open()) {
    LOG(FATAL) << "open input file " << input_file << " error.";
  }
  fin.read(reinterpret_cast<char*>(&num_inputs), sizeof(int64_t));
  assert(num_inputs == input_names.size() && "invalid number of inputs");

  // set input
  for (size_t i = 0; i < num_inputs; i++) {
    std::cout << "read input for " << input_names[i] << "\n";
    auto input_tensor = predictor->GetInput(i);
    int64_t num_dims = 0;
    fin.read(reinterpret_cast<char*>(&num_dims), sizeof(int64_t));

    std::vector<int64_t> shape(num_dims);
    int64_t tmp;
    for (size_t j = 0; j < num_dims; j++) {
        fin.read(reinterpret_cast<char*>(&tmp), sizeof(int64_t));
        shape[j] = tmp;
    }

    input_tensor->Resize(shape);
    auto input_data = input_tensor->mutable_data<float>();
    fin.read(reinterpret_cast<char*>(input_data),
             sizeof(float) * prod(input_tensor->shape()));
  }

  // warmup
  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }

  // run
  std::vector<float> perf_vct;
  for (int i = 0; i < FLAGS_repeats; ++i) {
    auto start = GetCurrentUS();
    predictor->Run();
    auto end = GetCurrentUS();
    perf_vct.push_back((end - start) / 1000.0);
  }
  std::sort(perf_vct.begin(), perf_vct.end());
  float min_res = perf_vct.back();
  float max_res = perf_vct.front();
  float total_res = std::accumulate(perf_vct.begin(), perf_vct.end(), 0.0);
  float avg_res = total_res / FLAGS_repeats;

  // print result
  std::setprecision(5);
  std::cout << std::setw(30) << std::fixed << std::left << model_path << " ";
  std::cout << "min = " << std::setw(12) << min_res;
  std::cout << "max = " << std::setw(12) << max_res;
  std::cout << "average = " << std::setw(12) << avg_res;
  std::cout << "\n";


  for (size_t i = 0; i < output_names.size(); i++) {
    auto output_tensor = predictor->GetOutput(0);
    std::cout << output_names[i] << " shape is: [";
    auto shape = output_tensor->shape();
    for (size_t d = 0; d < shape.size(); d++) {
      std::cout << shape[d];
      if (d != shape.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]\n";
    auto numel = prod(output_tensor->shape());
    std::cout << "data is \n[";
    for (size_t j = 0; j < numel; j++) {
      std::cout << output_tensor->mutable_data<float>()[j] << " ";
      if (j != numel - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]\n";
  }
}
#endif

}  // namespace lite_api
}  // namespace paddle

int main(int argc, char** argv) {
  // Check inputs
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_path == "" || FLAGS_input_file == "") {
    LOG(INFO) << "please run ./benchmark_bin --help to obtain usage.";
    exit(0);
  }

  if (FLAGS_model_dir.back() == '/') {
    FLAGS_model_dir.pop_back();
  }

  std::string save_optimized_model_dir = FLAGS_model_dir + "_opt2";
  // Output optimized model if needed
  if (FLAGS_run_model_optimize) {
    paddle::lite_api::OutputOptModel(save_optimized_model_dir);
  }

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
  // Run inference using optimized model
  paddle::lite_api::Run(FLAGS_model_path, FLAGS_input_file);
#endif
  return 0;
}
