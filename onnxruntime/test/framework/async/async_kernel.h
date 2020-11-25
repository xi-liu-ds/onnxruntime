// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/codegen/common/common.h"
#include "core/codegen/passes/utils/codegen_context.h"
#include "core/graph/graph.h"
#include "core/providers/nuphar/common/nuphar_subgraph.h"
#include "core/providers/nuphar/compiler/nuphar_compiler.h"
#include "core/providers/nuphar/compiler/initializer_info.h"
#include "core/providers/nuphar/runtime/compute_ctx.h"
#include "core/providers/nuphar/runtime/exec_block.h"

#include <map>
#include <unordered_map>

namespace onnxruntime {

class AsyncExecutionProvider;

namespace async_exec {

class AsyncKernelState;
using AsyncFuncStateToComputeCtxMap =
    std::unordered_map<const AsyncKernelState*, std::unique_ptr<KernelComputeCtx>>;

class AsyncKernelState {
 public:
  explicit AsyncKernelState(
      const Node& fused_node,
      const ComputeContext& ctx,
      const AsyncExecutionProvider& provider);

  ~AsyncKernelState();

  Status Compute(OpKernelContext* op_kernel_context) const;

  void Compile(const SubgraphUnit& subgraph);

  void BuildExecBlocksAndCalls(const std::vector<SubgraphUnit>& subgraphs);

 private:
  const AsyncExecutionProvider& provider_;

  Status codegen_status_;

  // Hold Partition_info for codegen
  std::unique_ptr<OrtSubgraphAllocationInfo> partition_info_;

  // Hold NupharFuncInfo from codegen.
  std::vector<std::unique_ptr<NupharFuncInfo>> func_infos_;

  // ExecBlocks of runtime
  // Ownership of ExecBlock
  std::vector<std::unique_ptr<ExecBlock>> exec_blocks_;

  // Calls
  std::vector<ExecBlock*> exec_block_calls_;

  // Here ComputeContext of Ort is used for allocator
  ComputeContext ctx_;  // the compute context from IExecutionProvider::Compile interface

  static thread_local std::unique_ptr<NupharFuncStateToComputeCtxMap> nuphar_compute_ctx_map_;
};

#define LIST_NUPHAR_OPS()                                    \
  NUPHAR_OP(Add, 7, DataTypeImpl::AllFixedSizeTensorTypes()) \
  NUPHAR_OP(Mul, 7, DataTypeImpl::AllFixedSizeTensorTypes())

}  // namespace async_exec
}  // namespace onnxruntime