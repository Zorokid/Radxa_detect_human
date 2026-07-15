// QnnRuntime.hpp — project copy of the qnn_py runtime wrapper, extended to
// carry per-tensor quantization params (scale/offset) so the app can
// quantize/dequantize the NPU's native UFIXED_POINT_8 I/O itself.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "QnnSampleApp.hpp"

namespace qnn_py {

struct TensorInfo {
  std::string name;
  std::vector<uint32_t> dims;
  uint32_t data_type = 0;
  size_t byte_size = 0;
  double scale = 1.0;
  int32_t offset = 0;
};

class QnnRuntime {
 public:
  QnnRuntime(std::string backend_path,
             std::string system_library_path,
             std::string context_bin_path,
             std::string op_package_paths = "");

  ~QnnRuntime();

  QnnRuntime(const QnnRuntime&) = delete;
  QnnRuntime& operator=(const QnnRuntime&) = delete;

  void init();
  bool initialized() const { return initialized_; }

  std::vector<std::vector<uint8_t>> runRawPtrs(const std::vector<const uint8_t*>& input_ptrs,
                                               const std::vector<size_t>& input_sizes,
                                               uint32_t graph_idx = 0);

  std::vector<TensorInfo> inputInfo(uint32_t graph_idx = 0) const;
  std::vector<TensorInfo> outputInfo(uint32_t graph_idx = 0) const;

  void close();

 private:
  void ensureInitialized() const;

  std::string backend_path_;
  std::string system_library_path_;
  std::string context_bin_path_;
  std::string op_package_paths_;

  qnn::tools::sample_app::QnnFunctionPointers qnn_function_pointers_{};
  void* backend_handle_ = nullptr;

  std::unique_ptr<qnn::tools::sample_app::QnnSampleApp> app_;

  bool initialized_ = false;
  bool device_created_ = false;
  bool context_created_ = false;
  bool closed_ = false;

  mutable std::mutex mutex_;
};

}  // namespace qnn_py
