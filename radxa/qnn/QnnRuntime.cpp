// QnnRuntime.cpp — project copy of the qnn_py runtime wrapper (see .hpp).
// Identical to the qnn_py_runtime original except convertInfos() also carries
// the per-tensor scale/offset now exposed by the patched RuntimeTensorInfo.
#include "qnn/QnnRuntime.hpp"

#include <stdexcept>
#include <utility>

#include "DynamicLoadUtil.hpp"
#include "Logger.hpp"
#include "PAL/DynamicLoading.hpp"

namespace qnn_py {
namespace {

using qnn::tools::dynamicloadutil::StatusCode;
using qnn::tools::sample_app::ProfilingLevel;
using qnn::tools::sample_app::QnnSampleApp;
using qnn::tools::sample_app::QnnFunctionPointers;

static void throwIfSampleFailure(qnn::tools::sample_app::StatusCode code,
                                 const char* message) {
  if (code != qnn::tools::sample_app::StatusCode::SUCCESS) {
    throw std::runtime_error(message);
  }
}

static std::vector<TensorInfo> convertInfos(const std::vector<QnnSampleApp::RuntimeTensorInfo>& in) {
  std::vector<TensorInfo> out;
  out.reserve(in.size());
  for (const auto& t : in) {
    TensorInfo ti;
    ti.name = t.name;
    ti.dims = t.dims;
    ti.data_type = static_cast<uint32_t>(t.dataType);
    ti.byte_size = t.byteSize;
    ti.scale = t.scale;
    ti.offset = t.offset;
    out.push_back(std::move(ti));
  }
  return out;
}

}  // namespace

QnnRuntime::QnnRuntime(std::string backend_path,
                       std::string system_library_path,
                       std::string context_bin_path,
                       std::string op_package_paths)
    : backend_path_(std::move(backend_path)),
      system_library_path_(std::move(system_library_path)),
      context_bin_path_(std::move(context_bin_path)),
      op_package_paths_(std::move(op_package_paths)) {}

QnnRuntime::~QnnRuntime() {
  try {
    close();
  } catch (...) {
  }
}

void QnnRuntime::init() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) {
    return;
  }
  if (backend_path_.empty()) {
    throw std::invalid_argument("backend_path is empty");
  }
  if (system_library_path_.empty()) {
    throw std::invalid_argument("system_library_path is empty; context.bin loading requires libQnnSystem.so");
  }
  if (context_bin_path_.empty()) {
    throw std::invalid_argument("context_bin_path is empty");
  }

  if (!qnn::log::isLogInitialized()) {
    if (!qnn::log::initializeLogging()) {
      throw std::runtime_error("Unable to initialize QNN sample logging");
    }
  }

  auto dynStatus = qnn::tools::dynamicloadutil::getQnnFunctionPointers(
      backend_path_,
      "",
      &qnn_function_pointers_,
      &backend_handle_,
      false,
      nullptr);
  if (dynStatus != StatusCode::SUCCESS) {
    throw std::runtime_error("getQnnFunctionPointers failed for backend: " + backend_path_);
  }

  dynStatus = qnn::tools::dynamicloadutil::getQnnSystemFunctionPointers(
      system_library_path_, &qnn_function_pointers_);
  if (dynStatus != StatusCode::SUCCESS) {
    throw std::runtime_error("getQnnSystemFunctionPointers failed for system library: " +
                             system_library_path_);
  }

  app_.reset(new QnnSampleApp(qnn_function_pointers_,
                              "",
                              op_package_paths_,
                              backend_handle_,
                              "",
                              false,
                              qnn::tools::iotensor::OutputDataType::FLOAT_ONLY,
                              qnn::tools::iotensor::InputDataType::FLOAT,
                              ProfilingLevel::OFF,
                              false,
                              context_bin_path_,
                              "",
                              1,
                              false,
                              ""));

  throwIfSampleFailure(app_->initializeRuntimeOnly(), "initializeRuntimeOnly failed");
  throwIfSampleFailure(app_->initializeBackend(), "initializeBackend failed");

  auto deviceSupport = app_->isDevicePropertySupported();
  if (deviceSupport != qnn::tools::sample_app::StatusCode::FAILURE) {
    throwIfSampleFailure(app_->createDevice(), "createDevice failed");
    device_created_ = true;
  }

  throwIfSampleFailure(app_->initializeProfiling(), "initializeProfiling failed");

  if (!op_package_paths_.empty()) {
    throwIfSampleFailure(app_->registerOpPackages(), "registerOpPackages failed");
  }

  throwIfSampleFailure(app_->createFromBinary(), "createFromBinary failed");
  context_created_ = true;

  if (app_->isFinalizeDeserializedGraphSupported() ==
      qnn::tools::sample_app::StatusCode::SUCCESS) {
    throwIfSampleFailure(app_->finalizeGraphs(), "finalizeGraphs for deserialized graph failed");
  }

  throwIfSampleFailure(app_->preparePersistentIo(), "preparePersistentIo failed");
  initialized_ = true;
  closed_ = false;
}

void QnnRuntime::ensureInitialized() const {
  if (!initialized_ || !app_) {
    throw std::runtime_error("QnnRuntime is not initialized");
  }
}

std::vector<std::vector<uint8_t>> QnnRuntime::runRawPtrs(
    const std::vector<const uint8_t*>& input_ptrs,
    const std::vector<size_t>& input_sizes,
    uint32_t graph_idx) {
  std::lock_guard<std::mutex> lock(mutex_);
  ensureInitialized();
  std::vector<std::vector<uint8_t>> outputs;
  auto status = app_->runFromHostBuffers(input_ptrs, input_sizes, outputs, graph_idx);
  if (status != qnn::tools::sample_app::StatusCode::SUCCESS) {
    throw std::runtime_error("QNN graphExecute failed");
  }
  return outputs;
}

std::vector<TensorInfo> QnnRuntime::inputInfo(uint32_t graph_idx) const {
  std::lock_guard<std::mutex> lock(mutex_);
  ensureInitialized();
  return convertInfos(app_->getInputTensorInfo(graph_idx));
}

std::vector<TensorInfo> QnnRuntime::outputInfo(uint32_t graph_idx) const {
  std::lock_guard<std::mutex> lock(mutex_);
  ensureInitialized();
  return convertInfos(app_->getOutputTensorInfo(graph_idx));
}

void QnnRuntime::close() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    return;
  }

  if (app_) {
    app_->releasePersistentIo();
    if (context_created_) {
      app_->freeContext();
      context_created_ = false;
    }
    if (device_created_) {
      app_->freeDevice();
      device_created_ = false;
    }
    app_->terminateBackend();
    app_.reset();
  }

  if (backend_handle_) {
    pal::dynamicloading::dlClose(backend_handle_);
    backend_handle_ = nullptr;
  }

  initialized_ = false;
  closed_ = true;
}

}  // namespace qnn_py
