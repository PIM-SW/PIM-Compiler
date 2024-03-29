//===--------------------- Passes.h - PIM Ops Header ----------------------===//
//
//===-------------------------- corelab heelim ----------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PIM_PASSES_H_
#define MLIR_PIM_PASSES_H_
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

#include <functional>
#include <memory>

namespace mlir {
class Pass;

class FuncOp;
class ModuleOp;
template <typename T>
class OperationPass;

std::unique_ptr<Pass> replaceCallToPIMOps();

std::unique_ptr<Pass> createConvertPIMToPNMPass();
std::unique_ptr<Pass> createConvertPNMToLLVMPass();

std::unique_ptr<Pass> createConvertPIMToAPIMPass();
std::unique_ptr<Pass> createConvertAPIMToLLVMPass();

std::unique_ptr<Pass> createConvertPIMToDPIMPass();
std::unique_ptr<Pass> createConvertDPIMToLLVMPass();

} // end namespace mlir

#endif
