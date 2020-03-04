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
#if !defined(_WIN32)
#include <sys/time.h>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <string>
#include <vector>
#include <iostream>
#include <iterator>
#include <sstream>
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
DEFINE_string(model_filename,
              "",
              "the filename of model file. When the model is combined formate, "
              "please set model_file.");
DEFINE_string(param_filename,
              "",
              "the filename of param file, set param_file when the model is "
              "combined formate.");
DEFINE_string(input_shape,
              "1,3,224,224",
              "set input shapes according to the model, "
              "separated by colon and comma, "
              "such as 1,3,244,244:1,3,300,300.");
DEFINE_string(input_path,
              "",
              "path of input files, "
              "separated by colons, such as input1_file:input2_file");
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

using shape_list = std::vector<std::vector<size_t>>;
using string_list = std::vector<std::string>;

namespace paddle {
namespace lite_api {

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
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
void Run(::shape_list& input_shapes,
         ::string_list& input_files,
         const std::string& model_dir,
         const std::string model_name) {
  // set config and create predictor
  lite_api::MobileConfig config;
  config.set_threads(FLAGS_threads);
  config.set_power_mode(static_cast<PowerMode>(FLAGS_power_mode));
  config.set_model_from_file(model_path);

  auto predictor = lite_api::CreatePaddlePredictor(config);

  // set input
  for (size_t i = 0; i < input_shapes.size(); i++) {
    auto input_tensor = predictor->GetInput(i);
    auto dim = input_shapes[i];
    input_tensor->Resize(dim);
    auto input_data = input_tensor->mutable_data<float>();
    auto numel = std::accumulate(
      dim.begin(), dim.end(), 1, std::multiplies<size_t>());

    std::ifstream fin(input_files[i], ios::binary | ios::ate);
    if (!fin.is_open()) {
      LOG(FATAL) << "open input image " << input_files[i] << " error.";
    }
    if (fin.tellg() != numel) {
      LOG(FATAL) << "size not matching shape: " << input_files[i];
    }
    fin.seekg(0);
    float f;
    while (fin.read(reinterpret_cast<char*>(&f), sizeof(float)))
        input_data[j] = f;
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
  float total_res = accumulate(perf_vct.begin(), perf_vct.end(), 0.0);
  float avg_res = total_res / FLAGS_repeats;

  // print result
  std::setprecision(5);
  std::cout << std::setw(30) << std::fixed << std::left << model_name << " ";
  std::cout << "min = " << std::setw(12) << min_res;
  std::cout << "max = " << std::setw(12) << max_res;
  std::cout << "average = " << std::setw(12) << avg_res;
  std::cout << "\n";
}
#endif

}  // namespace lite_api
}  // namespace paddle

template <char delim>
class string_segment : public std::string {};

template <char delim>
std::istream& operator>>(std::istream& is, comma_segment& output)
{
  std::getline(is, output, delim);
  return is;
}

template <typename T>
string_list split_string(std::string& input)
{
  std::istringstream iss(input);
  string_list results(
    std::istream_iterator<T>{iss},
    std::istream_iterator<T>());
  return results;
}

shape_list parse_input_shapes(std::string& input_shapes)
{
  string_list shape_strings = split_string<string_segment<':'>>(input_shapes);
  shape_list results(shape_strings.size());
  std::transform(shape_strings.begin(), shape_strings.end(), results.begin(),
                 [](std::string& input) -> std::vector<size_t> {
                   auto dim_strings = split_string<string_segment<','>>(input);
                   std::vector<size_t> dims(dim_strings.size());
                   std::transform(
                     dim_strings.begin(), dim_strings.end(), dims.begin(),
                     [](std::string& dim_s) -> std::size_t {
                       return std::atoi(dim_s.data());
                     }
                     );
                   return dims;
                 });
  return results;
}

int main(int argc, char** argv) {
  // Check inputs
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir == "") {
    LOG(INFO) << "please run ./benchmark_bin --help to obtain usage.";
    exit(0);
  }

  if (FLAGS_model_dir.back() == '/') {
    FLAGS_model_dir.pop_back();
  }
  std::size_t found = FLAGS_model_dir.find_last_of("/");
  std::string model_name = FLAGS_model_dir.substr(found + 1);

  shape_list input_shapes = parse_input_shapes(FLAGS_input_shape);
  string_list input_files = split_string<string_segment<':'>>(FLAGS_input_path);

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
  // Run inference using optimized model
  paddle::lite_api::Run(input_shapes, input_files, run_model_dir, model_name);
#endif
  return 0;
}
