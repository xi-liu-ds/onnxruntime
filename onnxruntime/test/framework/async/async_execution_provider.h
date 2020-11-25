// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/allocatormgr.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {

constexpr const char* kAsyncFuseExecutionProvider = "AsyncFuseExecutionProvider";

class AsyncFuseExecutionProvider : public IExecutionProvider {
 public:
  explicit AsyncFuseExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
};

}  // namespace onnxruntime