// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/session/inference_session.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "async_execution_provider.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
class AsyncFuseAdd : public OpKernel {
 public:
  explicit AsyncFuseAdd(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto X = context->Input<Tensor>(0);
    auto Y = context->Input<Tensor>(1);
    auto Z = context->Input<Tensor>(2);
    auto& shape = X->Shape();
    auto M = context->Output(0, shape)->template MutableData<float>();
    for (int i = 0; i < shape.Size(); ++i) {
      *(M + i) = *(X->template Data<float>() + i) + *(Y->template Data<float>() + i) + *(Z->template Data<float>() + i);
    }
    return Status::OK();
  }
};

constexpr const char* kAsyncFuseTest = "AsyncFuseTest";
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kFuseExecutionProvider, kFuseTest, 1, FuseAdd);
ONNX_OPERATOR_KERNEL_EX(AsyncFuseAdd,
                        kAsyncFuseTest,
                        1,
                        kAsyncFuseExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        AsyncFuseAdd);

AsyncFuseExecutionProvider::AsyncFuseExecutionProvider() : IExecutionProvider{kAsyncFuseExecutionProvider} {
  AllocatorCreationInfo device_info{
      [](int) {
        return onnxruntime::make_unique<CPUAllocator>(OrtMemoryInfo("AsyncFuse", OrtAllocatorType::OrtDeviceAllocator));
      }};
  InsertAllocator(device_info.device_alloc_factory(0));
}

std::vector<std::unique_ptr<ComputeCapability>>
AsyncFuseExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                          const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  // Fuse two add into one.
  std::vector<std::unique_ptr<ComputeCapability>> result;
  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
  for (auto& node : graph.Nodes()) {
    sub_graph->nodes.push_back(node.Index());
  }
  auto meta_def = onnxruntime::make_unique<IndexedSubGraph::MetaDef>();
  meta_def->name = "FuseAdd";
  meta_def->domain = "FuseTest";
  meta_def->inputs = {"X", "Y", "Z"};
  meta_def->outputs = {"M"};
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  sub_graph->SetMetaDef(std::move(meta_def));
  result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
  return result;
}

std::shared_ptr<KernelRegistry> AsyncFuseExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry;
  if (kernel_registry == nullptr) {
    kernel_registry = std::make_shared<KernelRegistry>();
    ORT_ENFORCE(kernel_registry->Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAsyncFuseExecutionProvider, kAsyncFuseTest, 1, AsyncFuseAdd)>()).IsOK());
  }
  return kernel_registry;
}
};  // namespace onnxruntime
