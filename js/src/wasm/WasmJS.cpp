/*
 * Copyright 2016 Mozilla Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "wasm/WasmJS.h"

#include "mozilla/Maybe.h"
#include "mozilla/TimeStamp.h"  // JS->wasm JIT deopt logging (GECKO_DEBUG_JIT)

#include <algorithm>
#include <cstdint>
#include <functional>     // JS->wasm JIT (Ion front-end) recursive inliner
#include <unordered_map>  // JS->wasm JIT script cache
#include <unordered_set>  // JS->wasm JIT (Ion front-end) loop-head set
#include <vector>         // JS->wasm JIT basic-block analysis

#include "jsapi.h"

#include "js/GCAPI.h"                  // JS_AddFinalizeCallback (JS->wasm JIT GC safety)
#include "js/Conversions.h"            // JS::ToObject/ToNumber (Mode VS helper)
#include "js/TracingAPI.h"             // JS::TraceRoot/JSTracer (Mode VS GC roots)
#include "js/CallAndConstruct.h"       // JS::Call (Mode VS generic call helper)
#include "ds/IdValuePair.h"            // js::IdValuePair
#include "frontend/FrontendContext.h"  // AutoReportFrontendContext
#include "gc/GCContext.h"
#include "jit/AtomicOperations.h"
#include "jit/CompileInfo.h"          // ION-REWRITE: bytecode->MIR front-end + OptimizeMIR reuse
#include "jit/FlushICache.h"
#include "jit/Ion.h"                  // ION-REWRITE: OptimizeMIR
#include "jit/IonOptimizationLevels.h"  // ION-REWRITE: IonOptimizations / OptimizationLevel::Wasm
#include "jit/JitContext.h"
#include "jit/JitOptions.h"
#include "jit/JitScript.h"  // PHASE F: pre-create JitScript at compile for deopt-resume
#include "jit/BaselineIC.h"      // ION-REWRITE: read Baseline ICs (decoupled oracle)
#include "jit/CacheIR.h"         // ION-REWRITE: CacheOp / CacheIROpInfos
#include "jit/CacheIRReader.h"   // ION-REWRITE: iterate CacheIR ops
#include "jit/CacheIRCompiler.h"  // ION-REWRITE: CacheIRStubInfo stub-field access
#include "jit/MIR-wasm.h"             // ION-REWRITE: MWasm* nodes
#include "jit/MIR.h"                  // ION-REWRITE: MConstant/MAdd/MUnreachable/...
#include "jit/MIRGenerator.h"         // ION-REWRITE: MIRGenerator
#include "jit/MIRGraph.h"             // ION-REWRITE: MIRGraph/MBasicBlock
#include "jit/DominatorTree.h"        // ION-REWRITE: BuildDominatorTree (validate)
#include "jit/IonAnalysis.h"          // ION-REWRITE: RenumberBlocks (validate)
#include "jit/Simulator.h"
#include "js/ColumnNumber.h"  // JS::ColumnNumberOneOrigin
#include "js/ForOfIterator.h"
#include "js/friend/ErrorMessages.h"  // js::GetErrorMessage, JSMSG_*
#include "js/Printf.h"
#include "js/PropertyAndElement.h"  // JS_DefineProperty, JS_GetProperty
#include "js/PropertySpec.h"        // JS_{PS,FN}{,_END}
#include "js/Stack.h"               // BuildStackString
#include "js/StreamConsumer.h"
#include "util/StringBuilder.h"
#include "util/Text.h"
#include "vm/ErrorObject.h"
#include "vm/FunctionFlags.h"      // js::FunctionFlags
#include "vm/EnvironmentObject.h"  // GlobalLexicalEnvironmentObject (Mode VS GetGName)
#include "vm/GlobalObject.h"       // js::GlobalObject
#include "vm/HelperThreadState.h"  // js::PromiseHelperTask
#include "vm/EqualityOperations.h"  // JS->wasm JIT Mode VS helper (Eq/StrictEq)
#include "vm/Interpreter.h"
#include "vm/JSFunction.h"
#include "vm/JSScript.h"       // JSScript (JS->wasm JIT)
#include "vm/BytecodeUtil.h"   // JSOp, GET_*, GetBytecodeLength (JS->wasm JIT)
#include "vm/ArrayObject.h"    // ArrayObject (JS->wasm JIT dense element IC)
#include "vm/ConstantCompareOperand.h"  // StrictConstantEq/Ne operand (JS->wasm JIT)
#include "vm/TypeofEqOperand.h"          // TypeofEq operand (JS->wasm JIT)
#include "vm/NativeObject.h"   // NativeObject::lookupPure (JS->wasm JIT prop IC)
#include "vm/PropertyInfo.h"   // PropertyInfo (JS->wasm JIT prop IC)
#include "vm/PlainObject.h"    // js::PlainObject
#include "vm/PromiseObject.h"  // js::PromiseObject
#include "vm/SharedArrayObject.h"
#include "vm/StringType.h"
#include "vm/TypedArrayObject.h"  // TypedArrayObject (JS->wasm JIT typed-array element IC)
#include "vm/Warnings.h"  // js::WarnNumberASCII
#include "wasm/WasmBaselineCompile.h"
#include "wasm/WasmBuiltinModule.h"
#include "wasm/WasmBuiltins.h"
#include "wasm/WasmCompile.h"
#include "wasm/WasmDebug.h"
#include "wasm/WasmFeatures.h"
#include "wasm/WasmInstance.h"
#include "wasm/WasmIonCompile.h"
#include "wasm/WasmMemory.h"
#include "wasm/WasmModule.h"
#include "wasm/WasmPI.h"
#include "wasm/WasmProcess.h"
#include "wasm/WasmSignalHandlers.h"
#include "wasm/WasmStubs.h"
#include "wasm/WasmValidate.h"

#include "gc/GCContext-inl.h"
#include "gc/StableCellHasher-inl.h"
#include "gc/StoreBuffer-inl.h"  // inline StoreBuffer::putWholeCell for the OBJSET object-store barrier
#include "jit/JitScript-inl.h"  // PHASE F: JSScript::ensureHasJitScript
#include "vm/Realm-inl.h"       // PHASE F: js::AutoRealm for safe JitScript creation
#include "vm/ArrayBufferObject-inl.h"
#include "vm/JSObject-inl.h"
#include "vm/NativeObject-inl.h"
#include "wasm/WasmInstance-inl.h"

/*
 * [SMDOC] WebAssembly code rules (evolving)
 *
 * TlsContext.get() is only to be invoked from functions that have been invoked
 *   _directly_ by generated code as cold(!) Builtin calls, from code that is
 *   only used by signal handlers, or from helper functions that have been
 *   called _directly_ from a simulator.  All other code shall pass in a
 *   JSContext* to functions that need it, or an Instance* or Instance* since
 * the context is available through them.
 *
 *   Code that uses TlsContext.get() shall annotate each such call with the
 *   reason why the call is OK.
 */

using namespace js;
using namespace js::jit;
using namespace js::wasm;

using mozilla::MakeStringSpan;
using mozilla::Maybe;
using mozilla::Nothing;
using mozilla::Some;
using mozilla::Span;

static bool ThrowCompileOutOfMemory(JSContext* cx) {
  // Most OOMs during compilation are due to large contiguous allocations,
  // and future allocations are likely to succeed. Throwing a proper error
  // object is nicer for users in these circumstances.
  JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr, JSMSG_OUT_OF_MEMORY);
  return false;
}

// ============================================================================
// Imports

static bool ThrowBadImportArg(JSContext* cx) {
  JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                           JSMSG_WASM_BAD_IMPORT_ARG);
  return false;
}

static bool ThrowBadImportType(JSContext* cx, const CacheableName& field,
                               const char* str) {
  UniqueChars fieldQuoted = field.toQuotedString(cx);
  if (!fieldQuoted) {
    ReportOutOfMemory(cx);
    return false;
  }
  JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                           JSMSG_WASM_BAD_IMPORT_TYPE, fieldQuoted.get(), str);
  return false;
}

// For now reject cross-compartment wrappers. These have more complicated realm
// semantics (we use nonCCWRealm in a few places) and may require unwrapping to
// test for specific function types.
static bool IsCallableNonCCW(const Value& v) {
  return IsCallable(v) && !IsCrossCompartmentWrapper(&v.toObject());
}

static bool IsWasmSuspendingWrapper(const Value& v) {
  return v.isObject() && js::IsWasmSuspendingObject(&v.toObject());
}

bool js::wasm::GetImports(JSContext* cx, const Module& module,
                          HandleObject importObj, ImportValues* imports) {
  const ModuleMetadata& moduleMeta = module.moduleMeta();
  const CodeMetadata& codeMeta = module.codeMeta();
  const BuiltinModuleIds& builtinModules = codeMeta.features().builtinModules;

  if (!moduleMeta.imports.empty() && !importObj) {
    return ThrowBadImportArg(cx);
  }

  BuiltinModuleInstances builtinInstances(cx);
  RootedValue importModuleValue(cx);
  RootedObject importModuleObject(cx);
  bool isImportedStringModule = false;
  RootedValue importFieldValue(cx);

  uint32_t tagIndex = 0;
  const TagDescVector& tags = codeMeta.tags;
  uint32_t globalIndex = 0;
  const GlobalDescVector& globals = codeMeta.globals;
  uint32_t tableIndex = 0;
  const TableDescVector& tables = codeMeta.tables;
  for (const Import& import : moduleMeta.imports) {
    Maybe<BuiltinModuleId> builtinModule =
        ImportMatchesBuiltinModule(import, builtinModules);
    if (builtinModule) {
      if (*builtinModule == BuiltinModuleId::JSStringConstants) {
        isImportedStringModule = true;
        importModuleObject = nullptr;
      } else {
        MutableHandle<JSObject*> builtinInstance =
            builtinInstances[*builtinModule];

        // If this module has not been instantiated yet, do so now.
        if (!builtinInstance) {
          // Use the first imported memory, if it exists, when compiling the
          // builtin module.
          const Import* firstMemoryImport =
              (codeMeta.memories.empty() ||
               codeMeta.memories[0].importIndex.isNothing())
                  ? nullptr
                  : &moduleMeta.imports[*codeMeta.memories[0].importIndex];

          // Compile and instantiate the builtin module. This uses our module's
          // importObj so that it can read the memory import that we provided
          // above.
          if (!wasm::InstantiateBuiltinModule(cx, *builtinModule,
                                              firstMemoryImport, importObj,
                                              builtinInstance)) {
            return false;
          }
        }
        isImportedStringModule = false;
        importModuleObject = builtinInstance;
      }
    } else {
      RootedId moduleName(cx);
      if (!import.module.toPropertyKey(cx, &moduleName)) {
        return false;
      }

      if (!GetProperty(cx, importObj, importObj, moduleName,
                       &importModuleValue)) {
        return false;
      }

      if (!importModuleValue.isObject()) {
        UniqueChars moduleQuoted = import.module.toQuotedString(cx);
        if (!moduleQuoted) {
          ReportOutOfMemory(cx);
          return false;
        }
        JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                                 JSMSG_WASM_BAD_IMPORT_FIELD,
                                 moduleQuoted.get());
        return false;
      }

      isImportedStringModule = false;
      importModuleObject = &importModuleValue.toObject();
    }
    MOZ_RELEASE_ASSERT(!isImportedStringModule ||
                       import.kind == DefinitionKind::Global);

    if (isImportedStringModule) {
      RootedString stringConstant(cx, import.field.toJSString(cx));
      if (!stringConstant) {
        ReportOutOfMemory(cx);
        return false;
      }
      importFieldValue = StringValue(stringConstant);
    } else {
      RootedId fieldName(cx);
      if (!import.field.toPropertyKey(cx, &fieldName)) {
        return false;
      }
      if (!GetProperty(cx, importModuleObject, importModuleObject, fieldName,
                       &importFieldValue)) {
        return false;
      }
    }

    switch (import.kind) {
      case DefinitionKind::Function: {
        if (!IsCallableNonCCW(importFieldValue) &&
            !IsWasmSuspendingWrapper(importFieldValue)) {
          return ThrowBadImportType(cx, import.field, "Function");
        }

        if (!imports->funcs.append(&importFieldValue.toObject())) {
          ReportOutOfMemory(cx);
          return false;
        }

        break;
      }
      case DefinitionKind::Table: {
        const uint32_t index = tableIndex++;
        if (!importFieldValue.isObject() ||
            !importFieldValue.toObject().is<WasmTableObject>()) {
          return ThrowBadImportType(cx, import.field, "Table");
        }

        Rooted<WasmTableObject*> obj(
            cx, &importFieldValue.toObject().as<WasmTableObject>());
        if (obj->table().elemType() != tables[index].elemType()) {
          JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                                   JSMSG_WASM_BAD_TBL_TYPE_LINK);
          return false;
        }

        if (!imports->tables.append(obj)) {
          ReportOutOfMemory(cx);
          return false;
        }
        break;
      }
      case DefinitionKind::Memory: {
        if (!importFieldValue.isObject() ||
            !importFieldValue.toObject().is<WasmMemoryObject>()) {
          return ThrowBadImportType(cx, import.field, "Memory");
        }

        if (!imports->memories.append(
                &importFieldValue.toObject().as<WasmMemoryObject>())) {
          ReportOutOfMemory(cx);
          return false;
        }
        break;
      }
      case DefinitionKind::Tag: {
        const uint32_t index = tagIndex++;
        if (!importFieldValue.isObject() ||
            !importFieldValue.toObject().is<WasmTagObject>()) {
          return ThrowBadImportType(cx, import.field, "Tag");
        }

        Rooted<WasmTagObject*> obj(
            cx, &importFieldValue.toObject().as<WasmTagObject>());

        // Checks whether the signature of the imported exception object matches
        // the signature declared in the exception import's TagDesc.
        if (!TagType::matches(*obj->tagType(), *tags[index].type)) {
          UniqueChars fieldQuoted = import.field.toQuotedString(cx);
          UniqueChars moduleQuoted = import.module.toQuotedString(cx);
          if (!fieldQuoted || !moduleQuoted) {
            ReportOutOfMemory(cx);
            return false;
          }
          JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                                   JSMSG_WASM_BAD_TAG_SIG, moduleQuoted.get(),
                                   fieldQuoted.get());
          return false;
        }

        if (!imports->tagObjs.append(obj)) {
          ReportOutOfMemory(cx);
          return false;
        }
        break;
      }
      case DefinitionKind::Global: {
        const uint32_t index = globalIndex++;
        const GlobalDesc& global = globals[index];
        MOZ_ASSERT(global.importIndex() == index);

        RootedVal val(cx);
        if (importFieldValue.isObject() &&
            importFieldValue.toObject().is<WasmGlobalObject>()) {
          Rooted<WasmGlobalObject*> obj(
              cx, &importFieldValue.toObject().as<WasmGlobalObject>());

          if (obj->isMutable() != global.isMutable()) {
            JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                                     JSMSG_WASM_BAD_GLOB_MUT_LINK);
            return false;
          }

          bool matches = global.isMutable()
                             ? obj->type() == global.type()
                             : ValType::isSubTypeOf(obj->type(), global.type());
          if (!matches) {
            JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                                     JSMSG_WASM_BAD_GLOB_TYPE_LINK);
            return false;
          }

          if (imports->globalObjs.length() <= index &&
              !imports->globalObjs.resize(index + 1)) {
            ReportOutOfMemory(cx);
            return false;
          }
          imports->globalObjs[index] = obj;
          val = obj->val();
        } else {
          if (!global.type().isRefType()) {
            if (global.type() == ValType::I64 && !importFieldValue.isBigInt()) {
              return ThrowBadImportType(cx, import.field, "BigInt");
            }
            if (global.type() != ValType::I64 && !importFieldValue.isNumber()) {
              return ThrowBadImportType(cx, import.field, "Number");
            }
          }

          if (global.isMutable()) {
            JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                                     JSMSG_WASM_BAD_GLOB_MUT_LINK);
            return false;
          }

          if (!Val::fromJSValue(cx, global.type(), importFieldValue, &val)) {
            return false;
          }
        }

        if (!imports->globalValues.append(val)) {
          ReportOutOfMemory(cx);
          return false;
        }

        break;
      }
    }
  }

  MOZ_ASSERT(globalIndex == globals.length() ||
             !globals[globalIndex].isImport());

  return true;
}

static bool DescribeScriptedCaller(JSContext* cx, ScriptedCaller* caller,
                                   const char* introducer) {
  // Note: JS::DescribeScriptedCaller returns whether a scripted caller was
  // found, not whether an error was thrown. This wrapper function converts
  // back to the more ordinary false-if-error form.

  JS::AutoFilename af;
  if (JS::DescribeScriptedCaller(&af, cx, &caller->line)) {
    caller->source =
        FormatIntroducedFilename(af.get(), caller->line, introducer);
    if (!caller->source) {
      ReportOutOfMemory(cx);
      return false;
    }
  }

  return true;
}

static bool CreateCompileError(JSContext* cx, const ScriptedCaller& caller,
                               HandleObject stack, const char* error,
                               MutableHandleObject errorObj) {
  RootedString fileName(cx);
  if (const char* fn = caller.source.get()) {
    fileName = JS_NewStringCopyUTF8N(cx, JS::UTF8Chars(fn, strlen(fn)));
  } else {
    fileName = JS_GetEmptyString(cx);
  }
  if (!fileName) {
    return false;
  }

  UniqueChars str(JS_smprintf("wasm validation error: %s", error));
  if (!str) {
    return false;
  }

  RootedString message(cx,
                       NewStringCopyN<CanGC>(cx, str.get(), strlen(str.get())));
  if (!message) {
    return false;
  }

  auto cause = JS::NothingHandleValue;
  errorObj.set(ErrorObject::create(cx, JSEXN_WASMCOMPILEERROR, stack, fileName,
                                   0, caller.line, JS::ColumnNumberOneOrigin(),
                                   nullptr, message, cause));
  return !!errorObj;
}

static SharedCompileArgs InitCompileArgs(JSContext* cx,
                                         const FeatureOptions& options,
                                         const char* introducer) {
  ScriptedCaller scriptedCaller;
  if (!DescribeScriptedCaller(cx, &scriptedCaller, introducer)) {
    return nullptr;
  }

  return CompileArgs::buildAndReport(cx, std::move(scriptedCaller), options);
}

// ============================================================================
// Testing / Fuzzing support

bool wasm::Eval(JSContext* cx, Handle<TypedArrayObject*> code,
                HandleObject importObj,
                MutableHandle<WasmInstanceObject*> instanceObj) {
  if (!GlobalObject::ensureConstructor(cx, cx->global(), JSProto_WebAssembly)) {
    return false;
  }

  FeatureOptions options;
  SharedCompileArgs compileArgs = InitCompileArgs(cx, options, "wasm_eval");
  if (!compileArgs) {
    return false;
  }

  BytecodeSource source((uint8_t*)code->dataPointerEither().unwrap(),
                        code->byteLength().valueOr(0));
  UniqueChars error;
  UniqueCharsVector warnings;
  // TODO(wasm-cm): Support components
  SharedModule module = CompileModule(
      *compileArgs, BytecodeBufferOrSource(source), &error, &warnings, nullptr);
  if (!module) {
    if (error) {
      JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                               JSMSG_WASM_COMPILE_ERROR, error.get());
      return false;
    }
    return ThrowCompileOutOfMemory(cx);
  }

  Rooted<ImportValues> imports(cx);
  if (!GetImports(cx, *module, importObj, imports.address())) {
    return false;
  }

  return module->instantiate(cx, imports.get(), nullptr, instanceObj);
}

struct MOZ_STACK_CLASS SerializeListener : JS::OptimizedEncodingListener {
  // MOZ_STACK_CLASS means these can be nops.
  MozExternalRefCountType MOZ_XPCOM_ABI AddRef() override { return 0; }
  MozExternalRefCountType MOZ_XPCOM_ABI Release() override { return 0; }

  mozilla::DebugOnly<bool> called = false;
  Bytes* serialized;
  explicit SerializeListener(Bytes* serialized) : serialized(serialized) {}

  void storeOptimizedEncoding(const uint8_t* bytes, size_t length) override {
    MOZ_ASSERT(!called);
    called = true;
    if (serialized->resizeUninitialized(length)) {
      memcpy(serialized->begin(), bytes, length);
    }
  }
};

bool wasm::CompileAndSerialize(JSContext* cx,
                               const BytecodeSource& bytecodeSource,
                               Bytes* serialized) {
  // The caller must check that code caching is available
  MOZ_ASSERT(CodeCachingAvailable(cx));

  // Create and manually fill in compile args for code caching
  MutableCompileArgs compileArgs = js_new<CompileArgs>();
  if (!compileArgs) {
    return false;
  }

  // The caller has ensured CodeCachingAvailable(). Moreover, we want to ensure
  // we go straight to tier-2 so that we synchronously call
  // JS::OptimizedEncodingListener::storeOptimizedEncoding().
  compileArgs->baselineEnabled = false;
  compileArgs->forceTiering = false;

  // We always pick Ion here, and we depend on CodeCachingAvailable() having
  // determined that Ion is available, see comments at CodeCachingAvailable().
  // To do better, we need to pass information about which compiler that should
  // be used into CompileAndSerialize().
  compileArgs->ionEnabled = true;

  // Select features that are enabled. This is guaranteed to be consistent with
  // our compiler selection, as code caching is only available if ion is
  // available, and ion is only available if it's not disabled by enabled
  // features.
  compileArgs->features = FeatureArgs::build(cx, FeatureOptions());

  SerializeListener listener(serialized);

  UniqueChars error;
  UniqueCharsVector warnings;
  // TODO(wasm-cm): Support components
  SharedModule module =
      CompileModule(*compileArgs, BytecodeBufferOrSource(bytecodeSource),
                    &error, &warnings, &listener);
  if (!module) {
    fprintf(stderr, "Compilation error: %s\n", error ? error.get() : "oom");
    return false;
  }

  MOZ_ASSERT(module->code().hasCompleteTier(Tier::Serialized));
  MOZ_ASSERT(listener.called);
  return !listener.serialized->empty();
}

bool wasm::DeserializeModule(JSContext* cx, const Bytes& serialized,
                             MutableHandleObject moduleObj) {
  MutableModule module =
      Module::deserialize(serialized.begin(), serialized.length());
  if (!module) {
    ReportOutOfMemory(cx);
    return false;
  }

  moduleObj.set(module->createObject(cx));
  return !!moduleObj;
}

static bool ReportCompileWarnings(JSContext* cx,
                                  const UniqueCharsVector& warnings) {
  // Avoid spamming the console.
  size_t numWarnings = std::min<size_t>(warnings.length(), 3);

  for (size_t i = 0; i < numWarnings; i++) {
    if (!WarnNumberASCII(cx, JSMSG_WASM_COMPILE_WARNING, warnings[i].get())) {
      return false;
    }
  }

  if (warnings.length() > numWarnings) {
    if (!WarnNumberASCII(cx, JSMSG_WASM_COMPILE_WARNING,
                         "other warnings suppressed")) {
      return false;
    }
  }

  return true;
}

// https://webassembly.github.io/esm-integration/js-api/index.html#esm-integration
bool js::wasm::CompileForESM(JSContext* cx,
                             const JS::ReadOnlyCompileOptions& options,
                             const BytecodeSource& bytecodeSource,
                             MutableHandleObject moduleObj) {
  // Step 1. Let stableBytes be a copy of the bytes held by the buffer bytes.
  // (Performed by caller)

  FeatureOptions featureOptions;
  // Step 4 (reordered). Let builtinSetNames be « "js-string" ».
  featureOptions.jsStringBuiltins = true;
  // Step 5 (reordered). Let importedStringModule be "wasm:js/string-constants".
  featureOptions.jsStringConstants = true;
  UniqueChars ns = DuplicateString(cx, "wasm:js/string-constants");
  if (!ns) {
    return false;
  }
  featureOptions.jsStringConstantsNamespace =
      cx->new_<ShareableChars>(std::move(ns));
  if (!featureOptions.jsStringConstantsNamespace) {
    return false;
  }

  // Step 2. Compile the WebAssembly module stableBytes and store the result
  //         as module.
  ScriptedCaller scriptedCaller;
  if (options.filename()) {
    scriptedCaller.source = DuplicateString(cx, options.filename().c_str());
    if (!scriptedCaller.source) {
      return false;
    }
    scriptedCaller.kind = ScriptedCallerKind::Url;
  }
  SharedCompileArgs compileArgs = CompileArgs::buildAndReport(
      cx, std::move(scriptedCaller), featureOptions, /* reportOOM */ true);
  if (!compileArgs) {
    return false;
  }

  UniqueChars error;
  UniqueCharsVector warnings;
  SharedModule module =
      CompileModule(*compileArgs, BytecodeBufferOrSource(bytecodeSource),
                    &error, &warnings, nullptr);

  if (!ReportCompileWarnings(cx, warnings)) {
    return false;
  }

  // Step 3. If module is error, throw a CompileError exception.
  if (!module) {
    if (!error) {
      JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                               JSMSG_OUT_OF_MEMORY);
      return false;
    }
    RootedObject errorObj(cx);
    RootedObject nullStack(cx, nullptr);
    if (!CreateCompileError(cx, compileArgs->scriptedCaller, nullStack,
                            error.get(), &errorObj)) {
      return false;
    }
    RootedValue errorVal(cx, ObjectValue(*errorObj));
    cx->setPendingException(errorVal, js::ShouldCaptureStack::Maybe);
    return false;
  }

  // Step 6. Construct a WebAssembly module object from module, bytes,
  //         builtinSetNames and importedStringModule, and let module
  //         be the result.
  RootedObject proto(
      cx, GlobalObject::getOrCreatePrototype(cx, JSProto_WasmModule));
  if (!proto) {
    return false;
  }

  RootedObject wasmModuleObject(cx,
                                WasmModuleObject::create(cx, *module, proto));
  if (!wasmModuleObject) {
    return false;
  }

  // Step 7. Let requestedModules be a set.
  // TODO: Populate requestedModules for evaluation phase imports (Bug 2030454).

  const ModuleMetadata& moduleMeta = module->moduleMeta();
  const CodeMetadata& codeMeta = module->codeMeta();

  // Step 8. For each (moduleName, name, type) in
  // module_imports(module.[[Module]]),
  for (const Import& import : moduleMeta.imports) {
    Span<const char> moduleName = import.module.utf8Bytes();
    Span<const char> name = import.field.utf8Bytes();

    // Step 8.1. If moduleName starts with the prefix "wasm-js:",
    if (CharsStartsWith(moduleName, MakeStringSpan("wasm-js:"))) {
      // Step 8.1.1. Throw a LinkError exception.
      UniqueChars moduleNameQuoted = import.module.toQuotedString(cx);
      if (!moduleNameQuoted) {
        ReportOutOfMemory(cx);
        return false;
      }
      JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                               JSMSG_WASM_ESM_RESERVED_MODULE_NAME,
                               moduleNameQuoted.get());
      return false;
    }

    // Step 8.2. If name starts with the prefix "wasm:" or "wasm-js:",
    if (CharsStartsWith(name, MakeStringSpan("wasm:")) ||
        CharsStartsWith(name, MakeStringSpan("wasm-js:"))) {
      // Step 8.2.1. Throw a LinkError exception.
      UniqueChars nameQuoted = import.field.toQuotedString(cx);
      if (!nameQuoted) {
        ReportOutOfMemory(cx);
        return false;
      }
      JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                               JSMSG_WASM_ESM_RESERVED_FIELD_NAME,
                               nameQuoted.get());
      return false;
    }

    // Step 8.3. Note: The following step only applies when integrating with the
    // JS String Builtins proposal.

    // Step 8.4. If Find a builtin with (moduleName, name, type) and builtins
    // module.[[BuiltinSets]] is not null, then continue.
    if (ImportMatchesBuiltinModule(moduleName,
                                   codeMeta.features().builtinModules)) {
      continue;
    }

    // Step 8.4.1. Append moduleName to requestedModules.
    // TODO: Populate requestedModules for evaluation phase imports (Bug
    // 2030454).
  }

  // Step 9. For each (name, type) in module_exports(module.[[Module]])
  for (const Export& exp : moduleMeta.exports) {
    Span<const char> name = exp.fieldName().utf8Bytes();

    // Step 9.1. If name starts with the prefix "wasm:" or "wasm-js:",
    if (CharsStartsWith(name, MakeStringSpan("wasm:")) ||
        CharsStartsWith(name, MakeStringSpan("wasm-js:"))) {
      // Step 9.1.1. Throw a LinkError exception.
      UniqueChars nameQuoted = exp.fieldName().toQuotedString(cx);
      if (!nameQuoted) {
        ReportOutOfMemory(cx);
        return false;
      }
      JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                               JSMSG_WASM_ESM_RESERVED_EXPORT_NAME,
                               nameQuoted.get());
      return false;
    }
  }

  moduleObj.set(wasmModuleObject);
  return true;
}

// ============================================================================
// Common functions

// '[EnforceRange] unsigned long' types are coerced with
//    ConvertToInt(v, 32, 'unsigned')
// defined in Web IDL Section 3.2.4.9.
//
// This just generalizes that to an arbitrary limit that is representable as an
// integer in double form.

static bool EnforceRange(JSContext* cx, HandleValue v, const char* kind,
                         const char* noun, uint64_t max, uint64_t* val) {
  // Step 4.
  double x;
  if (!ToNumber(cx, v, &x)) {
    return false;
  }

  // Step 5.
  if (mozilla::IsNegativeZero(x)) {
    x = 0.0;
  }

  // Step 6.1.
  if (!std::isfinite(x)) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_ENFORCE_RANGE, kind, noun);
    return false;
  }

  // Step 6.2.
  x = JS::ToInteger(x);

  // Step 6.3.
  if (x < 0 || x > double(max)) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_ENFORCE_RANGE, kind, noun);
    return false;
  }

  *val = uint64_t(x);
  MOZ_ASSERT(double(*val) == x);
  return true;
}

static bool EnforceRangeU32(JSContext* cx, HandleValue v, const char* kind,
                            const char* noun, uint32_t* u32) {
  uint64_t u64 = 0;
  if (!EnforceRange(cx, v, kind, noun, uint64_t(UINT32_MAX), &u64)) {
    return false;
  }
  *u32 = uint32_t(u64);
  return true;
}

static bool EnforceRangeU64(JSContext* cx, HandleValue v, const char* kind,
                            const char* noun, uint64_t* u64) {
  // The max is Number.MAX_SAFE_INTEGER
  return EnforceRange(cx, v, kind, noun, (1LL << 53) - 1, u64);
}

static bool EnforceRangeBigInt64(JSContext* cx, HandleValue v, const char* kind,
                                 const char* noun, uint64_t* u64) {
  RootedBigInt bi(cx, ToBigInt(cx, v));
  if (!bi) {
    return false;
  }
  if (!BigInt::isUint64(bi, u64)) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_ENFORCE_RANGE, kind, noun);
    return false;
  }
  return true;
}

static bool EnforceAddressValue(JSContext* cx, HandleValue v,
                                AddressType addressType, const char* kind,
                                const char* noun, uint64_t* result) {
  switch (addressType) {
    case AddressType::I32: {
      uint32_t result32;
      if (!EnforceRangeU32(cx, v, kind, noun, &result32)) {
        return false;
      }
      *result = uint64_t(result32);
      return true;
    }
    case AddressType::I64:
      return EnforceRangeBigInt64(cx, v, kind, noun, result);
    default:
      MOZ_CRASH("unknown address type");
  }
}

// The AddressValue typedef, a union of number and bigint, is used in the JS API
// spec for memory and table arguments, where number is used for memory32 and
// bigint is used for memory64.
[[nodiscard]] static bool CreateAddressValue(JSContext* cx, uint64_t value,
                                             AddressType addressType,
                                             MutableHandleValue addressValue) {
  switch (addressType) {
    case AddressType::I32:
      MOZ_ASSERT(value <= UINT32_MAX);
      addressValue.set(NumberValue(value));
      return true;
    case AddressType::I64: {
      BigInt* bi = BigInt::createFromUint64(cx, value);
      if (!bi) {
        return false;
      }
      addressValue.set(BigIntValue(bi));
      return true;
    }
    default:
      MOZ_CRASH("unknown address type");
  }
}

// Gets an AddressValue property ("initial" or "maximum") from a
// MemoryDescriptor or TableDescriptor. The values returned by this should be
// run through CheckLimits to enforce the validation limits prescribed by the
// spec.
static bool GetDescriptorAddressValue(JSContext* cx, HandleObject obj,
                                      const char* name, const char* noun,
                                      const char* msg, AddressType addressType,
                                      bool* found, uint64_t* value) {
  JSAtom* atom = Atomize(cx, name, strlen(name));
  if (!atom) {
    return false;
  }
  RootedId id(cx, AtomToId(atom));

  RootedValue val(cx);
  if (!GetProperty(cx, obj, obj, id, &val)) {
    return false;
  }

  if (val.isUndefined()) {
    *found = false;
    return true;
  }
  *found = true;

  return EnforceAddressValue(cx, val, addressType, noun, msg, value);
}

static bool GetLimits(JSContext* cx, HandleObject obj, LimitsKind kind,
                      Limits* limits) {
  limits->addressType = AddressType::I32;

  // Limits may specify an alternate address type, and we need this to check the
  // ranges for initial and maximum, so look for the address type first.
  // Get the address type field
  JSAtom* addressTypeAtom = Atomize(cx, "address", strlen("address"));
  if (!addressTypeAtom) {
    return false;
  }
  RootedId addressTypeId(cx, AtomToId(addressTypeAtom));
  RootedValue addressTypeVal(cx);
  if (!GetProperty(cx, obj, obj, addressTypeId, &addressTypeVal)) {
    return false;
  }

  // The address type has a default value
  if (!addressTypeVal.isUndefined()) {
    if (!ToAddressType(cx, addressTypeVal, &limits->addressType)) {
      return false;
    }
  }

  const char* noun = ToString(kind);
  uint64_t limit = 0;

  bool haveInitial = false;
  if (!GetDescriptorAddressValue(cx, obj, "initial", noun, "initial size",
                                 limits->addressType, &haveInitial, &limit)) {
    return false;
  }
  if (haveInitial) {
    limits->initial = limit;
  }

  bool haveMinimum = false;
#ifdef ENABLE_WASM_TYPE_REFLECTIONS
  if (!GetDescriptorAddressValue(cx, obj, "minimum", noun, "initial size",
                                 limits->addressType, &haveMinimum, &limit)) {
    return false;
  }
  if (haveMinimum) {
    limits->initial = limit;
  }
#endif

  if (!(haveInitial || haveMinimum)) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_MISSING_REQUIRED, "initial");
    return false;
  }
  if (haveInitial && haveMinimum) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_SUPPLY_ONLY_ONE, "minimum", "initial");
    return false;
  }

  bool haveMaximum = false;
  if (!GetDescriptorAddressValue(cx, obj, "maximum", noun, "maximum size",
                                 limits->addressType, &haveMaximum, &limit)) {
    return false;
  }
  if (haveMaximum) {
    limits->maximum = Some(limit);
  }

  limits->shared = Shareable::False;

  // Memory limits may be shared.
  if (kind == LimitsKind::Memory) {
    // Get the shared field
    JSAtom* sharedAtom = Atomize(cx, "shared", strlen("shared"));
    if (!sharedAtom) {
      return false;
    }
    RootedId sharedId(cx, AtomToId(sharedAtom));

    RootedValue sharedVal(cx);
    if (!GetProperty(cx, obj, obj, sharedId, &sharedVal)) {
      return false;
    }

    // shared's default value is false, which is already the value set above.
    if (!sharedVal.isUndefined()) {
      limits->shared =
          ToBoolean(sharedVal) ? Shareable::True : Shareable::False;

      if (limits->shared == Shareable::True) {
        if (!haveMaximum) {
          JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                                    JSMSG_WASM_MISSING_MAXIMUM, noun);
          return false;
        }

        if (!cx->realm()
                 ->creationOptions()
                 .getSharedMemoryAndAtomicsEnabled()) {
          JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                                    JSMSG_WASM_NO_SHMEM_LINK);
          return false;
        }
      }
    }

    // TODO: Should be updated when the JS API for custom page sizes
    // is finalized, see https://bugzilla.mozilla.org/show_bug.cgi?id=1985679
    limits->pageSize = PageSize::Standard;
  }

  return true;
}

static bool CheckLimits(JSContext* cx, uint64_t validationMax, LimitsKind kind,
                        Limits* limits) {
  const char* noun = ToString(kind);

  // There are several layers of validation and error-throwing here, including
  // one which is currently not defined by the JS API spec:
  //
  // - [EnforceRange] on parameters (must be TypeError)
  // - A check that initial <= maximum (must be RangeError)
  // - Either a mem_alloc or table_alloc operation, which has two components:
  //   - A pre-condition that the given memory or table type is valid
  //     (not specified, RangeError in practice)
  //   - The actual allocation (should report OOM if it fails)
  //
  // There are two questions currently left open by the spec: when is the memory
  // or table type validated, and if it is invalid, what type of exception does
  // it throw? In practice, all browsers throw RangeError, and by the time you
  // read this the spec will hopefully have been updated to reflect this. See
  // the following issue: https://github.com/WebAssembly/spec/issues/1792

  // Check that initial <= maximum
  if (limits->maximum.isSome() && *limits->maximum < limits->initial) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_MAX_LT_INITIAL, noun);
    return false;
  }

  // Check wasm validation limits
  if (limits->initial > validationMax) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr, JSMSG_WASM_BAD_RANGE,
                             noun, "initial size");
    return false;
  }
  if (limits->maximum.isSome() && *limits->maximum > validationMax) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr, JSMSG_WASM_BAD_RANGE,
                             noun, "maximum size");
    return false;
  }

  return true;
}

template <class Class, const char* name>
static JSObject* CreateWasmConstructor(JSContext* cx, JSProtoKey key) {
  Rooted<JSAtom*> className(cx, Atomize(cx, name, strlen(name)));
  if (!className) {
    return nullptr;
  }

#ifdef NIGHTLY_BUILD
  if (JS::Prefs::experimental_wasm_esm_integration()) {
    if constexpr (std::is_same_v<Class, WasmModuleObject>) {
      RootedObject proto(cx, GlobalObject::getOrCreateConstructor(
                                 cx, JSProto_AbstractModuleSource));
      if (!proto) {
        return nullptr;
      }
      return NewFunctionWithProto(
          cx, Class::construct, 1, FunctionFlags::NATIVE_CTOR, nullptr,
          className, proto, gc::AllocKind::FUNCTION, TenuredObject);
    }
  }
#endif
  return NewNativeConstructor(cx, Class::construct, 1, className);
}

static JSObject* GetWasmConstructorPrototype(JSContext* cx,
                                             const CallArgs& callArgs,
                                             JSProtoKey key) {
  RootedObject proto(cx);
  if (!GetPrototypeFromBuiltinConstructor(cx, callArgs, key, &proto)) {
    return nullptr;
  }
  if (!proto) {
    proto = GlobalObject::getOrCreatePrototype(cx, key);
  }
  return proto;
}

[[nodiscard]] static bool ParseValTypes(JSContext* cx, HandleValue src,
                                        ValTypeVector& dest) {
  JS::ForOfIterator iterator(cx);

  if (!iterator.init(src, JS::ForOfIterator::ThrowOnNonIterable)) {
    return false;
  }

  RootedValue nextParam(cx);
  while (true) {
    bool done;
    if (!iterator.next(&nextParam, &done)) {
      return false;
    }
    if (done) {
      break;
    }

    ValType valType;
    if (!ToValType(cx, nextParam, &valType) || !dest.append(valType)) {
      return false;
    }
  }
  return true;
}

#ifdef ENABLE_WASM_TYPE_REFLECTIONS
template <typename T>
static JSString* TypeToString(JSContext* cx, T type) {
  UniqueChars chars = ToString(type, nullptr);
  if (!chars) {
    return nullptr;
  }
  return NewStringCopyUTF8Z(
      cx, JS::ConstUTF8CharsZ(chars.get(), strlen(chars.get())));
}

static JSString* AddressTypeToString(JSContext* cx, AddressType type) {
  return JS_NewStringCopyZ(cx, ToString(type));
}

[[nodiscard]] static JSObject* ValTypesToArray(JSContext* cx,
                                               const ValTypeVector& valTypes) {
  Rooted<ArrayObject*> arrayObj(cx, NewDenseEmptyArray(cx));
  if (!arrayObj) {
    return nullptr;
  }
  for (ValType valType : valTypes) {
    RootedString type(cx, TypeToString(cx, valType));
    if (!type) {
      return nullptr;
    }
    if (!NewbornArrayPush(cx, arrayObj, StringValue(type))) {
      return nullptr;
    }
  }
  return arrayObj;
}

static JSObject* FuncTypeToObject(JSContext* cx, const FuncType& type) {
  Rooted<IdValueVector> props(cx, IdValueVector(cx));

  RootedObject parametersObj(cx, ValTypesToArray(cx, type.args()));
  if (!parametersObj ||
      !props.append(IdValuePair(NameToId(cx->names().parameters),
                                ObjectValue(*parametersObj)))) {
    ReportOutOfMemory(cx);
    return nullptr;
  }

  RootedObject resultsObj(cx, ValTypesToArray(cx, type.results()));
  if (!resultsObj || !props.append(IdValuePair(NameToId(cx->names().results),
                                               ObjectValue(*resultsObj)))) {
    ReportOutOfMemory(cx);
    return nullptr;
  }

  return NewPlainObjectWithUniqueNames(cx, props);
}

static JSObject* TableTypeToObject(JSContext* cx, AddressType addressType,
                                   RefType type, uint64_t initial,
                                   Maybe<uint64_t> maximum) {
  Rooted<IdValueVector> props(cx, IdValueVector(cx));

  RootedString elementType(cx, TypeToString(cx, type));
  if (!elementType || !props.append(IdValuePair(NameToId(cx->names().element),
                                                StringValue(elementType)))) {
    ReportOutOfMemory(cx);
    return nullptr;
  }

  if (maximum.isSome()) {
    RootedId maximumId(cx, NameToId(cx->names().maximum));
    RootedValue maximumValue(cx);
    if (!CreateAddressValue(cx, maximum.value(), addressType, &maximumValue)) {
      ReportOutOfMemory(cx);
      return nullptr;
    }
    if (!props.append(IdValuePair(maximumId, maximumValue))) {
      ReportOutOfMemory(cx);
      return nullptr;
    }
  }

  RootedId minimumId(cx, NameToId(cx->names().minimum));
  RootedValue minimumValue(cx);
  if (!CreateAddressValue(cx, initial, addressType, &minimumValue)) {
    ReportOutOfMemory(cx);
    return nullptr;
  }
  if (!props.append(IdValuePair(minimumId, minimumValue))) {
    ReportOutOfMemory(cx);
    return nullptr;
  }

  RootedString at(cx, AddressTypeToString(cx, addressType));
  if (!at) {
    return nullptr;
  }
  if (!props.append(
          IdValuePair(NameToId(cx->names().address), StringValue(at)))) {
    ReportOutOfMemory(cx);
    return nullptr;
  }

  return NewPlainObjectWithUniqueNames(cx, props);
}

static JSObject* MemoryTypeToObject(JSContext* cx, bool shared,
                                    wasm::AddressType addressType,
                                    wasm::Pages minPages,
                                    Maybe<wasm::Pages> maxPages) {
  Rooted<IdValueVector> props(cx, IdValueVector(cx));
  if (maxPages) {
    RootedId maximumId(cx, NameToId(cx->names().maximum));
    RootedValue maximumValue(cx);
    if (!CreateAddressValue(cx, maxPages.value().pageCount(), addressType,
                            &maximumValue)) {
      ReportOutOfMemory(cx);
      return nullptr;
    }
    if (!props.append(IdValuePair(maximumId, maximumValue))) {
      ReportOutOfMemory(cx);
      return nullptr;
    }
  }

  RootedId minimumId(cx, NameToId(cx->names().minimum));
  RootedValue minimumValue(cx);
  if (!CreateAddressValue(cx, minPages.pageCount(), addressType,
                          &minimumValue)) {
    ReportOutOfMemory(cx);
    return nullptr;
  }
  if (!props.append(IdValuePair(minimumId, minimumValue))) {
    ReportOutOfMemory(cx);
    return nullptr;
  }

  RootedString at(cx, AddressTypeToString(cx, addressType));
  if (!at) {
    return nullptr;
  }
  if (!props.append(
          IdValuePair(NameToId(cx->names().address), StringValue(at)))) {
    ReportOutOfMemory(cx);
    return nullptr;
  }

  if (!props.append(
          IdValuePair(NameToId(cx->names().shared), BooleanValue(shared)))) {
    ReportOutOfMemory(cx);
    return nullptr;
  }

  return NewPlainObjectWithUniqueNames(cx, props);
}

static JSObject* GlobalTypeToObject(JSContext* cx, ValType type,
                                    bool isMutable) {
  Rooted<IdValueVector> props(cx, IdValueVector(cx));

  if (!props.append(IdValuePair(NameToId(cx->names().mutable_),
                                BooleanValue(isMutable)))) {
    ReportOutOfMemory(cx);
    return nullptr;
  }

  RootedString valueType(cx, TypeToString(cx, type));
  if (!valueType || !props.append(IdValuePair(NameToId(cx->names().value),
                                              StringValue(valueType)))) {
    ReportOutOfMemory(cx);
    return nullptr;
  }

  return NewPlainObjectWithUniqueNames(cx, props);
}

static JSObject* TagTypeToObject(JSContext* cx,
                                 const wasm::ValTypeVector& params) {
  Rooted<IdValueVector> props(cx, IdValueVector(cx));

  RootedObject parametersObj(cx, ValTypesToArray(cx, params));
  if (!parametersObj ||
      !props.append(IdValuePair(NameToId(cx->names().parameters),
                                ObjectValue(*parametersObj)))) {
    ReportOutOfMemory(cx);
    return nullptr;
  }

  return NewPlainObjectWithUniqueNames(cx, props);
}
#endif  // ENABLE_WASM_TYPE_REFLECTIONS

// ============================================================================
// WebAssembly.Module class and methods

const JSClassOps WasmModuleObject::classOps_ = {
    .finalize = WasmModuleObject::finalize,
};

const JSClass WasmModuleObject::class_ = {
    "WebAssembly.Module",
    JSCLASS_DELAY_METADATA_BUILDER |
        JSCLASS_HAS_RESERVED_SLOTS(WasmModuleObject::RESERVED_SLOTS) |
        JSCLASS_FOREGROUND_FINALIZE,
    &WasmModuleObject::classOps_,
    &WasmModuleObject::classSpec_,
};

const JSClass& WasmModuleObject::protoClass_ = PlainObject::class_;

static constexpr char WasmModuleName[] = "Module";

// https://webassembly.github.io/esm-integration/js-api/index.html#modules
static JSObject* CreateWasmModulePrototype(JSContext* cx, JSProtoKey key) {
#ifdef NIGHTLY_BUILD
  if (JS::Prefs::experimental_wasm_esm_integration()) {
    RootedObject abstractModuleSourceProto(
        cx,
        GlobalObject::getOrCreatePrototype(cx, JSProto_AbstractModuleSource));
    if (!abstractModuleSourceProto) {
      return nullptr;
    }
    return GlobalObject::createBlankPrototypeInheriting(
        cx, &WasmModuleObject::protoClass_, abstractModuleSourceProto);
  }
#endif
  return GlobalObject::createBlankPrototype(cx, cx->global(),
                                            &WasmModuleObject::protoClass_);
}

const ClassSpec WasmModuleObject::classSpec_ = {
    CreateWasmConstructor<WasmModuleObject, WasmModuleName>,
    CreateWasmModulePrototype,
    WasmModuleObject::static_methods,
    nullptr,
    WasmModuleObject::methods,
    WasmModuleObject::properties,
    nullptr,
    ClassSpec::DontDefineConstructor,
};

const JSPropertySpec WasmModuleObject::properties[] = {
    JS_STRING_SYM_PS(toStringTag, "WebAssembly.Module", JSPROP_READONLY),
    JS_PS_END,
};

const JSFunctionSpec WasmModuleObject::methods[] = {
    JS_FS_END,
};

const JSFunctionSpec WasmModuleObject::static_methods[] = {
    JS_FN("imports", WasmModuleObject::imports, 1, JSPROP_ENUMERATE),
    JS_FN("exports", WasmModuleObject::exports, 1, JSPROP_ENUMERATE),
    JS_FN("customSections", WasmModuleObject::customSections, 2,
          JSPROP_ENUMERATE),
    JS_FS_END,
};

/* static */
void WasmModuleObject::finalize(JS::GCContext* gcx, JSObject* obj) {
  const Module& module = obj->as<WasmModuleObject>().module();
  size_t codeMemory = module.tier1CodeMemoryUsed();
  if (codeMemory) {
    obj->zone()->decJitMemory(codeMemory);
  }
  gcx->release(obj, &module, module.gcMallocBytesExcludingCode(),
               MemoryUse::WasmModule);
}

struct KindNames {
  Rooted<PropertyName*> kind;
  Rooted<PropertyName*> table;
  Rooted<PropertyName*> memory;
  Rooted<PropertyName*> tag;
  Rooted<PropertyName*> type;

  explicit KindNames(JSContext* cx)
      : kind(cx), table(cx), memory(cx), tag(cx), type(cx) {}
};

static bool InitKindNames(JSContext* cx, KindNames* names) {
  JSAtom* kind = Atomize(cx, "kind", strlen("kind"));
  if (!kind) {
    return false;
  }
  names->kind = kind->asPropertyName();

  JSAtom* table = Atomize(cx, "table", strlen("table"));
  if (!table) {
    return false;
  }
  names->table = table->asPropertyName();

  JSAtom* memory = Atomize(cx, "memory", strlen("memory"));
  if (!memory) {
    return false;
  }
  names->memory = memory->asPropertyName();

  JSAtom* tag = Atomize(cx, "tag", strlen("tag"));
  if (!tag) {
    return false;
  }
  names->tag = tag->asPropertyName();

  JSAtom* type = Atomize(cx, "type", strlen("type"));
  if (!type) {
    return false;
  }
  names->type = type->asPropertyName();

  return true;
}

static JSString* KindToString(JSContext* cx, const KindNames& names,
                              DefinitionKind kind) {
  switch (kind) {
    case DefinitionKind::Function:
      return cx->names().function;
    case DefinitionKind::Table:
      return names.table;
    case DefinitionKind::Memory:
      return names.memory;
    case DefinitionKind::Global:
      return cx->names().global;
    case DefinitionKind::Tag:
      return names.tag;
  }

  MOZ_CRASH("invalid kind");
}

/* static */
bool WasmModuleObject::imports(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);

  if (!args.requireAtLeast(cx, "WebAssembly.Module.imports", 1)) {
    return false;
  }

  if (!args[0].isObject()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_MOD_ARG);
    return false;
  }

  Rooted<WasmModuleObject*> moduleObj(
      cx, args[0].toObject().maybeUnwrapIf<WasmModuleObject>());
  if (!moduleObj) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_MOD_ARG);
    return false;
  }

  KindNames names(cx);
  if (!InitKindNames(cx, &names)) {
    return false;
  }

  const ModuleMetadata& moduleMeta = moduleObj->module().moduleMeta();

  RootedValueVector elems(cx);
  if (!elems.reserve(moduleMeta.imports.length())) {
    return false;
  }

  const CodeMetadata& codeMeta = moduleObj->module().codeMeta();
#if defined(ENABLE_WASM_TYPE_REFLECTIONS)
  size_t numFuncImport = 0;
  size_t numMemoryImport = 0;
  size_t numGlobalImport = 0;
  size_t numTableImport = 0;
  size_t numTagImport = 0;
#endif  // ENABLE_WASM_TYPE_REFLECTIONS

  for (const Import& import : moduleMeta.imports) {
    Maybe<BuiltinModuleId> builtinModule =
        ImportMatchesBuiltinModule(import, codeMeta.features().builtinModules);
    if (builtinModule) {
#ifdef ENABLE_WASM_TYPE_REFLECTIONS
      switch (import.kind) {
        case DefinitionKind::Function:
          numFuncImport++;
          break;
        case DefinitionKind::Table:
          numTableImport++;
          break;
        case DefinitionKind::Memory:
          numMemoryImport++;
          break;
        case DefinitionKind::Global:
          numGlobalImport++;
          break;
        case DefinitionKind::Tag:
          numTagImport++;
          break;
      }
#endif  // ENABLE_WASM_TYPE_REFLECTIONS
      continue;
    }

    Rooted<IdValueVector> props(cx, IdValueVector(cx));
    if (!props.reserve(3)) {
      return false;
    }

    JSString* moduleStr = import.module.toAtom(cx);
    if (!moduleStr) {
      return false;
    }
    props.infallibleAppend(
        IdValuePair(NameToId(cx->names().module), StringValue(moduleStr)));

    JSString* nameStr = import.field.toAtom(cx);
    if (!nameStr) {
      return false;
    }
    props.infallibleAppend(
        IdValuePair(NameToId(cx->names().name), StringValue(nameStr)));

    JSString* kindStr = KindToString(cx, names, import.kind);
    if (!kindStr) {
      return false;
    }
    props.infallibleAppend(
        IdValuePair(NameToId(names.kind), StringValue(kindStr)));

#ifdef ENABLE_WASM_TYPE_REFLECTIONS
    RootedObject typeObj(cx);
    switch (import.kind) {
      case DefinitionKind::Function: {
        size_t funcIndex = numFuncImport++;
        const FuncType& funcType = codeMeta.getFuncType(funcIndex);
        typeObj = FuncTypeToObject(cx, funcType);
        break;
      }
      case DefinitionKind::Table: {
        size_t tableIndex = numTableImport++;
        const TableDesc& table = codeMeta.tables[tableIndex];
        typeObj =
            TableTypeToObject(cx, table.addressType(), table.elemType(),
                              table.initialLength(), table.maximumLength());
        break;
      }
      case DefinitionKind::Memory: {
        size_t memoryIndex = numMemoryImport++;
        const MemoryDesc& memory = codeMeta.memories[memoryIndex];
        typeObj =
            MemoryTypeToObject(cx, memory.isShared(), memory.addressType(),
                               memory.initialPages(), memory.maximumPages());
        break;
      }
      case DefinitionKind::Global: {
        size_t globalIndex = numGlobalImport++;
        const GlobalDesc& global = codeMeta.globals[globalIndex];
        typeObj = GlobalTypeToObject(cx, global.type(), global.isMutable());
        break;
      }
      case DefinitionKind::Tag: {
        size_t tagIndex = numTagImport++;
        const TagDesc& tag = codeMeta.tags[tagIndex];
        typeObj = TagTypeToObject(cx, tag.type->argTypes());
        break;
      }
    }

    if (!typeObj || !props.append(IdValuePair(NameToId(names.type),
                                              ObjectValue(*typeObj)))) {
      ReportOutOfMemory(cx);
      return false;
    }
#endif  // ENABLE_WASM_TYPE_REFLECTIONS

    JSObject* obj = NewPlainObjectWithUniqueNames(cx, props);
    if (!obj) {
      return false;
    }

    elems.infallibleAppend(ObjectValue(*obj));
  }

  JSObject* arr = NewDenseCopiedArray(cx, elems.length(), elems.begin());
  if (!arr) {
    return false;
  }

  args.rval().setObject(*arr);
  return true;
}

/* static */
bool WasmModuleObject::exports(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);

  if (!args.requireAtLeast(cx, "WebAssembly.Module.exports", 1)) {
    return false;
  }

  if (!args[0].isObject()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_MOD_ARG);
    return false;
  }

  Rooted<WasmModuleObject*> moduleObj(
      cx, args[0].toObject().maybeUnwrapIf<WasmModuleObject>());
  if (!moduleObj) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_MOD_ARG);
    return false;
  }

  KindNames names(cx);
  if (!InitKindNames(cx, &names)) {
    return false;
  }

  const ModuleMetadata& moduleMeta = moduleObj->module().moduleMeta();

  RootedValueVector elems(cx);
  if (!elems.reserve(moduleMeta.exports.length())) {
    return false;
  }

#ifdef ENABLE_WASM_TYPE_REFLECTIONS
  const CodeMetadata& codeMeta = moduleObj->module().codeMeta();
#endif  // ENABLE_WASM_TYPE_REFLECTIONS

  for (const Export& exp : moduleMeta.exports) {
    Rooted<IdValueVector> props(cx, IdValueVector(cx));
    if (!props.reserve(2)) {
      return false;
    }

    JSString* nameStr = exp.fieldName().toAtom(cx);
    if (!nameStr) {
      return false;
    }
    props.infallibleAppend(
        IdValuePair(NameToId(cx->names().name), StringValue(nameStr)));

    JSString* kindStr = KindToString(cx, names, exp.kind());
    if (!kindStr) {
      return false;
    }
    props.infallibleAppend(
        IdValuePair(NameToId(names.kind), StringValue(kindStr)));

#ifdef ENABLE_WASM_TYPE_REFLECTIONS
    RootedObject typeObj(cx);
    switch (exp.kind()) {
      case DefinitionKind::Function: {
        const FuncType& funcType = codeMeta.getFuncType(exp.funcIndex());
        typeObj = FuncTypeToObject(cx, funcType);
        break;
      }
      case DefinitionKind::Table: {
        const TableDesc& table = codeMeta.tables[exp.tableIndex()];
        typeObj =
            TableTypeToObject(cx, table.addressType(), table.elemType(),
                              table.initialLength(), table.maximumLength());
        break;
      }
      case DefinitionKind::Memory: {
        const MemoryDesc& memory = codeMeta.memories[exp.memoryIndex()];
        typeObj =
            MemoryTypeToObject(cx, memory.isShared(), memory.addressType(),
                               memory.initialPages(), memory.maximumPages());
        break;
      }
      case DefinitionKind::Global: {
        const GlobalDesc& global = codeMeta.globals[exp.globalIndex()];
        typeObj = GlobalTypeToObject(cx, global.type(), global.isMutable());
        break;
      }
      case DefinitionKind::Tag: {
        const TagDesc& tag = codeMeta.tags[exp.tagIndex()];
        typeObj = TagTypeToObject(cx, tag.type->argTypes());
        break;
      }
    }

    if (!typeObj || !props.append(IdValuePair(NameToId(names.type),
                                              ObjectValue(*typeObj)))) {
      ReportOutOfMemory(cx);
      return false;
    }
#endif  // ENABLE_WASM_TYPE_REFLECTIONS

    JSObject* obj = NewPlainObjectWithUniqueNames(cx, props);
    if (!obj) {
      return false;
    }

    elems.infallibleAppend(ObjectValue(*obj));
  }

  JSObject* arr = NewDenseCopiedArray(cx, elems.length(), elems.begin());
  if (!arr) {
    return false;
  }

  args.rval().setObject(*arr);
  return true;
}

/* static */
bool WasmModuleObject::customSections(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);

  if (!args.requireAtLeast(cx, "WebAssembly.Module.customSections", 2)) {
    return false;
  }

  if (!args[0].isObject()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_MOD_ARG);
    return false;
  }

  Rooted<WasmModuleObject*> moduleObj(
      cx, args[0].toObject().maybeUnwrapIf<WasmModuleObject>());
  if (!moduleObj) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_MOD_ARG);
    return false;
  }

  Vector<char, 8> name(cx);
  {
    RootedString str(cx, ToString(cx, args.get(1)));
    if (!str) {
      return false;
    }

    Rooted<JSLinearString*> linear(cx, str->ensureLinear(cx));
    if (!linear) {
      return false;
    }

    if (!name.initLengthUninitialized(
            JS::GetDeflatedUTF8StringLength(linear))) {
      return false;
    }

    (void)JS::DeflateStringToUTF8Buffer(linear,
                                        Span(name.begin(), name.length()));
  }

  RootedValueVector elems(cx);
  Rooted<ArrayBufferObject*> buf(cx);
  for (const CustomSection& cs :
       moduleObj->module().moduleMeta().customSections) {
    if (name.length() != cs.name.length()) {
      continue;
    }
    if (memcmp(name.begin(), cs.name.begin(), name.length()) != 0) {
      continue;
    }

    buf = ArrayBufferObject::createZeroed(cx, cs.payload->length());
    if (!buf) {
      return false;
    }

    memcpy(buf->dataPointer(), cs.payload->begin(), cs.payload->length());
    if (!elems.append(ObjectValue(*buf))) {
      return false;
    }
  }

  JSObject* arr = NewDenseCopiedArray(cx, elems.length(), elems.begin());
  if (!arr) {
    return false;
  }

  args.rval().setObject(*arr);
  return true;
}

/* static */
WasmModuleObject* WasmModuleObject::create(JSContext* cx, const Module& module,
                                           HandleObject proto) {
  AutoSetNewObjectMetadata metadata(cx);
  auto* obj = NewObjectWithGivenProto<WasmModuleObject>(cx, proto);
  if (!obj) {
    return nullptr;
  }

  // The pipeline state on some architectures may retain stale instructions
  // even after we invalidate the instruction cache. There is no generally
  // available method to broadcast this pipeline flush to all threads after
  // we've compiled new code, so conservatively perform one here when we're
  // receiving a module that may have been compiled from another thread.
  //
  // The cost of this flush is expected to minimal enough to not be worth
  // optimizing away in the case the module was compiled on this thread.
  jit::FlushExecutionContext();

  // This accounts for module allocation size (excluding code which is handled
  // separately - see below). This assumes that the size of associated data
  // doesn't change for the life of the WasmModuleObject. The size is counted
  // once per WasmModuleObject referencing a Module.
  InitReservedSlot(obj, MODULE_SLOT, const_cast<Module*>(&module),
                   module.gcMallocBytesExcludingCode(), MemoryUse::WasmModule);
  module.AddRef();

  // Bug 1569888: We account for the first tier here; the second tier, if
  // different, also needs to be accounted for.
  size_t codeMemory = module.tier1CodeMemoryUsed();
  if (codeMemory) {
    cx->zone()->incJitMemory(codeMemory);
  }
  return obj;
}

struct MOZ_STACK_CLASS AutoPinBufferSourceLength {
  explicit AutoPinBufferSourceLength(JSContext* cx, JSObject* bufferSource)
      : bufferSource_(cx, bufferSource),
        wasPinned_(!JS::PinArrayBufferOrViewLength(bufferSource_, true)) {}
  ~AutoPinBufferSourceLength() {
    if (!wasPinned_) {
      JS::PinArrayBufferOrViewLength(bufferSource_, false);
    }
  }

 private:
  Rooted<JSObject*> bufferSource_;
  bool wasPinned_;
};

// Checks if the `obj` is a buffer source (according to WebIDL rules) and
// returns a view to the underlying memory. Callers should be sure to use
// AutoPinBufferSourceLength for the resulting lifetime of the bytecode
// source.
static bool GetBytecodeSource(JSContext* cx, Handle<JSObject*> obj,
                              unsigned errorNumber, BytecodeSource* bytecode,
                              bool* isShared) {
  JSObject* unwrapped = CheckedUnwrapStatic(obj);

  SharedMem<uint8_t*> dataPointer;
  size_t byteLength;
  if (!unwrapped || !IsBufferSource(cx, unwrapped, /*allowShared*/ true,
                                    /*allowResizable*/ true, &dataPointer,
                                    &byteLength, isShared)) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr, errorNumber);
    return false;
  }

  *bytecode = BytecodeSource(dataPointer.unwrap(), byteLength);
  return true;
}

// The same as `GetBytecodeSource`, but instead returns an owned bytecode
// buffer copy. Callers don't need to use AutoPinBufferSourceLength because
// they own the resulting memory.
static bool GetBytecodeBuffer(JSContext* cx, Handle<JSObject*> obj,
                              unsigned errorNumber, BytecodeBuffer* bytecode) {
  BytecodeSource source;
  bool isShared;
  if (!GetBytecodeSource(cx, obj, errorNumber, &source, &isShared)) {
    return false;
  }
  AutoPinBufferSourceLength pin(cx, obj);

  if (!BytecodeBuffer::fromSource(source, bytecode)) {
    ReportOutOfMemory(cx);
    return false;
  }
  return true;
}

// The same as `GetBytecodeSource`, but instead returns an owned bytecode
// buffer if the buffer source is shared.
static bool GetBytecodeBufferOrSource(JSContext* cx, Handle<JSObject*> obj,
                                      unsigned errorNumber,
                                      BytecodeBufferOrSource* bytecode) {
  BytecodeSource source;
  bool isShared;
  if (!GetBytecodeSource(cx, obj, errorNumber, &source, &isShared)) {
    return false;
  }

  if (!isShared) {
    *bytecode = BytecodeBufferOrSource(source);
    return true;
  }

  // The buffer source is shared, we need to make a copy to ensure it we have an
  // immutable copy.
  AutoPinBufferSourceLength pin(cx, obj);
  BytecodeBuffer buffer;
  if (!BytecodeBuffer::fromSource(source, &buffer)) {
    ReportOutOfMemory(cx);
    return false;
  }

  *bytecode = BytecodeBufferOrSource(std::move(buffer));
  return true;
}

#if defined(__EMSCRIPTEN__)
// Host-passthrough wasm: bridge guest WebAssembly to the host engine (the JIT is
// disabled here, so there is no in-process compiler). These resolve to the
// wasm-host-bridge.js js-library at link time; wasmhost_invoke_import is defined
// in this file and exported. Helpers are defined below, near WebAssembly_instantiate.
extern "C" {
int wasmhost_compile(const void* bytes, int len);
int wasmhost_import_count(int handle);
int wasmhost_import_kind(int handle, int index);
int wasmhost_import_module(int handle, int index, char* buf, int buflen);
int wasmhost_import_name(int handle, int index, char* buf, int buflen);
int wasmhost_instantiate(int handle, const int* callbackIds, int importCount);
int wasmhost_export_count(int handle);
int wasmhost_export_kind(int handle, int index);
int wasmhost_export_name(int handle, int index, char* buf, int buflen);
int wasmhost_export_register_mem(int handle, int index);
int wasmhost_mem_new(int initialPages, int maxPages, int shared);
int wasmhost_table_new(int initial, int maxN, int isExternref);
int wasmhost_global_new(double val, int kind, int mut);
int wasmhost_mem_bytelength(int objId);
int wasmhost_mem_is_shared(int objId);
void wasmhost_obj_set_mirror(int objId, void* ptr, int len);
double wasmhost_call(int handle, int index, const double* args, int argc);
int wasmhost_guest_mem_objid();
int wasmhost_guest_mem_shared();
int wasmhost_jit_table();
int wasmhost_jit_table_set(int handle, int idx);
}

// ===========================================================================
// JS -> WebAssembly JIT.
//
// SpiderMonkey's Baseline/Ion JITs emit native machine code, which can't run in
// the wasm sandbox (hence --disable-jit here). This tier instead lowers a hot JS
// function's bytecode to a small WebAssembly module, instantiates it on the HOST
// engine via the wasmhost_* bridge above (a real native JIT), and calls it -- so
// the whole function body runs as host-JITed wasm and the guest<->host boundary
// is paid once per call. SpiderMonkey bytecode is a stack machine, like wasm, so
// the lowering is close to 1:1.
//
// Calling convention (the "full JIT" foundation): values are NaN-boxed JS::Value
// bits (i64), passed through a fixed scratch buffer in the GUEST heap (gWJScratch)
// rather than as f64 args -- f64 args would route through JS numbers in the bridge
// and canonicalize any object/string-tagged NaN, corrupting the box. The host
// module IMPORTS the guest's own linear memory (emscripten's wasmMemory), so a
// JSObject* (a wasm32 offset) can be dereferenced inline: shape guards and slot
// loads read the guest heap directly with no boundary crossing. The wasm entry is
// `(f64 scratchPtr) -> f64 deopt`: it unboxes args from the scratch buffer (type-
// guarding, returning deopt=1 on a type mismatch so the interpreter re-runs the
// call), runs the body on f64 number locals, and boxes the result back into the
// scratch buffer's result slot. Numeric arithmetic stays pure-f64 internally.
// ===========================================================================
namespace {

// Argument/result transfer buffer in the guest heap (a fixed global => stable
// wasm32 address, readable/writable by the imported-memory host module). Args at
// [0..63], result at [kWJResultSlot]. Sized for the nargs<=32 cap with slack.
static constexpr uint32_t kWJResultSlot = 64;
static constexpr uint32_t kWJResultOff = kWJResultSlot * 8;  // bytes
static constexpr uint32_t kWJThisSlot = 65;                 // receiver (`this`)
static constexpr uint32_t kWJThisOff = kWJThisSlot * 8;
// PHASE F (deopt resume): a Mode VS body that bails mid-execution writes the bytecode
// offset to resume at into gWJScratch[kWJResumePcSlot] and returns deopt code 3.0.
static constexpr uint32_t kWJResumePcSlot = 66;
static constexpr uint32_t kWJResumePcOff = kWJResumePcSlot * 8;
// PHASE F: resume locals are staged into the GC-traced scratch range [32,64) (args use
// [0,32); result/this/resumePc are 64/65/66) so they survive a GC during JitScript creation.
static constexpr uint32_t kWJResumeLocalsMax = 32;  // resume nfixed cap (frame-read)
static uint64_t gWJScratch[72];

// Canonical quiet-NaN bits, used to box a NaN f64 result (a non-canonical NaN
// would alias a NUNBOX32 tag and be misread as a pointer-bearing Value).
static constexpr uint64_t kWJCanonNaN = 0x7FF8000000000000ULL;
// NUNBOX32: tag in the high 32 bits. A value is a number (double or int32) iff
// its tag <= JSVAL_TAG_INT32; int32 specifically has tag == JSVAL_TAG_INT32.
static constexpr uint64_t kWJTagInt32 = 0xFFFFFF81ULL;
static constexpr uint64_t kWJTagObject = 0xFFFFFF8CULL;
static constexpr uint32_t kWJTagClear = 0xFFFFFF80u;      // <= this tag => a double
static constexpr uint32_t kWJTagBoolean = 0xFFFFFF82u;
static constexpr uint32_t kWJTagUndefined = 0xFFFFFF83u;
static constexpr uint32_t kWJTagNull = 0xFFFFFF84u;
static constexpr uint32_t kWJTagString = 0xFFFFFF86u;
static constexpr uint32_t kWJTagSymbol = 0xFFFFFF87u;
static constexpr uint32_t kWJTagBigInt = 0xFFFFFF89u;

// Inline-cache state for GetProp sites (Milestone B). Because the whole GC heap
// is in the guest linear memory shared with the host JIT module, a monomorphic
// data-property read is a shape guard + a slot load done INLINE in wasm. Each
// site has an entry {shape, byteOffset} in gWJICTable, filled lazily by C++ on
// the first shape-guard miss (the wasm records the missing object + site, then
// deopts; WasmJitRunCall fills the entry, so the site misses once then hits).
static constexpr uint32_t kWJMaxSites = 4096;
static constexpr uint32_t kWJNoMiss = 0xFFFFFFFFu;
static uint32_t gWJICTable[2 * kWJMaxSites];  // [2*site]=shape, [2*site+1]=offset
static uint32_t gWJInitToShape[kWJMaxSites];  // INLINEALLOC InitProp IC: target shape after the add
static uint64_t gWJInitHelperCalls = 0;       // InitProp helper invocations (IC misses); diagnostic
static uint32_t gWJMissSite = kWJNoMiss;      // site id of the last guard miss
static uint64_t gWJMissObj = 0;               // Value bits of the missed object
// Gate for the reused-Ion builder's bail/diagnostic logging: spamming stderr for
// every compile attempt across a big bench (deltablue) is both noise and a real
// slowdown, so default-OFF; GECKO_WJVS_IONINT_LOG enables it.
static bool WJIonLog() {
  static bool v = getenv("GECKO_WJVS_IONINT_LOG") != nullptr;
  return v;
}
// Is `p` a plausible guest-heap pointer to deref? Raw JSFunction*/JSObject* values
// recorded in gWJInlineCallee/gWJMethodPoly/gWJShapeSample can go stale across GC,
// or come from a desynced CacheIR read -- derefing an out-of-range one is a hard
// wasm "memory access out of bounds" trap that aborts the engine. Reject null/small
// values and anything past the wasm linear-memory size before any deref.
static bool WJGuestPtrOk(uint32_t p) {
  if (p < 0x10000) return false;
  uint64_t memBytes = (uint64_t)__builtin_wasm_memory_size(0) << 16;
  return (uint64_t)p + 256 <= memBytes;  // room for the object header we read
}
struct WJSite {
  JSScript* script = nullptr;
  uint32_t pcOff = 0;
};
static WJSite gWJSites[kWJMaxSites];
static uint32_t gWJSiteCount = 0;

// Per-call-site cache for JSOp::Call (Milestone E). A call site guards the
// callee against gWJCallFn (a JSFunction* low32, 0 = unfilled) and, on a hit,
// invokes the cached compiled callee via the wasmjit_invoke import.
// Math intrinsic inlining: a 1-arg (or 2-arg min/max) Call site whose callee is a recognized
// Math.* native is emitted as the corresponding wasm f64 op (sqrt/floor/ceil/abs/trunc/min/max),
// guarded by callee identity, with the generic call as fallback. gWJMathOp[site]: 0=none.
enum WJMathOp { WJM_NONE = 0, WJM_SQRT, WJM_FLOOR, WJM_CEIL, WJM_ABS, WJM_TRUNC, WJM_MIN, WJM_MAX };
static uint64_t gWJMathInline = 0;            // intrinsic call sites emitted inline (diagnostic)
static uint32_t gWJCallFn[kWJMaxSites];      // cached callee JSFunction* (low32)
static int32_t gWJCallHandle[kWJMaxSites];   // callee wasm instance handle
static uint32_t gWJCallNargs[kWJMaxSites];   // callee formal arg count
static uint32_t gWJCallArgc[kWJMaxSites];    // arg count passed at this site

// POLYMORPHIC call dispatch: a call site that sees several distinct callees of
// the same arity (deltablue's constraint subclasses dispatch the same method on
// many shapes) used to evict + miss to the generic WHJ_CALL helper on every
// callee flip. An N-way call IC caches up to kWJCallWays compiled callees and
// dispatches via an unrolled (low32(callee)==fn_w) ? handle_w chain to a single
// call_indirect. Way 0 reuses gWJCallFn/gWJCallHandle above; ways 1..N-1 live
// here at [(way-1)*kWJMaxSites + site]. gWJCallWaysFilled[site] = filled ways.
static constexpr uint32_t kWJCallWays = 4;
static uint32_t gWJCallFnX[(kWJCallWays - 1) * kWJMaxSites];
static int32_t gWJCallHandleX[(kWJCallWays - 1) * kWJMaxSites];
static uint8_t gWJCallWaysFilled[kWJMaxSites];
static uint64_t gWJCallPolyFills = 0;  // call-way fills beyond way 0 (poly growth)
static uint64_t gWJCallMegaMiss = 0;   // call misses with all kWJCallWays full

// GetGName sites: cache the resolved global holder object so the global var/
// function is read inline (shape guard + slot load). The slot byte offset is in
// gWJICTable[2*site+1]; its high bit means "dynamic slot" (load via slots_).
static uint32_t gWJGNameHolder[kWJMaxSites];  // resolved global object address
static constexpr uint32_t kWJDynSlot = 0x80000000u;

// Prototype-chain GetProp: when a property is found on the receiver's prototype
// (not own -- e.g. a method on Foo.prototype), cache the holder (prototype)
// object + its shape; the receiver's shape guard fixes the proto, the holder's
// shape guard fixes the slot. gWJProtoHolder[site]==0 means an own-property site.
static uint32_t gWJProtoHolder[kWJMaxSites];       // holder (prototype) address
static uint32_t gWJProtoHolderShape[kWJMaxSites];  // holder's cached shape

// Extra IC ways for POLYMORPHIC Mode V GetProp sites. A GetProp that sees several
// receiver shapes (e.g. a method on a class hierarchy -- deltablue's constraint
// subclasses) used to evict + deopt on every shape flip; an N-way IC inlines all
// N. Way 0 reuses the arrays above (shared with SetProp / GetElem / Mode VS
// GetProp); ways 1..kWJICWays-1 use these "extra" arrays at extra index x=way-1,
// laid out [x*kWJMaxSites + site]. gWJSitePoly[site] marks a site as N-way (only
// Mode V GetProp emits the extra ways + gets the N-way fill policy in WJFillIC).
static constexpr uint32_t kWJICWays = 4;
static uint32_t gWJICTableX[(kWJICWays - 1) * 2 * kWJMaxSites];      // shape,off
static uint32_t gWJProtoHolderX[(kWJICWays - 1) * kWJMaxSites];      // holder(0=own)
static uint32_t gWJProtoHolderShapeX[(kWJICWays - 1) * kWJMaxSites]; // holder shape
static bool gWJSitePoly[kWJMaxSites];               // site uses the N-way GetProp IC
// TYPED ARRAYS: per GetElem/SetElem site, the cached backing-array kind. 0 = dense
// JS Array (the existing Value-element path); 1 = Float64Array; 2 = Int32Array.
// Set in WJFillIC on a typed-array receiver (with the shape cached in gWJICTable).
// Typed-array elements are RAW (no NaN-boxing, no write barrier) -> inline access is
// a plain f64/i32 load/store, the structural win for typed-array numeric loops.
static uint8_t gWJElemKind[kWJMaxSites] = {0};
// Element-kind codes (0 = dense JS Array). The numeric value also carries no
// meaning beyond the per-code load/store sequence emitted in WJVSGetElem/SetElem.
static constexpr uint8_t kWJElemF64 = 1, kWJElemI32 = 2, kWJElemU8 = 3,
                         kWJElemI8 = 4, kWJElemU16 = 5, kWJElemI16 = 6,
                         kWJElemU32 = 7, kWJElemF32 = 8;
// A GetProp site reading `.length`: dense-array length is a CUSTOM data property
// (isDataProperty()==false) so the generic IC bails and the site deopts forever
// (octane-deltablue's OrderedCollection reads `this.elms.length` in a hot loop).
// A length site instead shape-guards the receiver (cached only for ArrayObjects in
// WJFillIC) and loads the length inline from the elements header (elements_ - 4).
static bool gWJSiteLen[kWJMaxSites];

// ---- Mode VS (Value-Safe): no-restart JIT for MUTATING functions ------------
// Functions that write the heap (SetProp/SetElem/...) cannot use the cheap
// deopt-by-restart model (re-running the whole function double-executes the
// writes). Mode VS instead NEVER restarts: a guard miss / type miss calls a C++
// helper (`wjhelp`) that completes the operation in place and the wasm continues.
// Because those helpers run engine code that can allocate + trigger GC -- and GC
// cannot see object pointers held in host-wasm frames -- Mode VS keeps the whole
// frame (args, locals, rval, operand stack) in a GC-TRACED guest-memory frame
// stack (gWJFrameMem), so a moving GC during a helper updates them in place.
static constexpr uint32_t kWJFrameSlots = 8192;  // i64 slots (shared frame stack)
static uint64_t gWJFrameMem[kWJFrameSlots];
static uint32_t gWJFrameSP = 0;  // next free slot (top of the frame stack)
// Helper operand scratch (read by wjhelp; written by the wasm before the call).
static uint64_t gWJHelpA = 0, gWJHelpB = 0, gWJHelpC = 0;
// Operation kinds dispatched by wjhelp.
enum WJHelperKind {
  WJH_ADD = 0, WJH_SUB, WJH_MUL, WJH_DIV, WJH_MOD,
  WJH_NEG, WJH_INC, WJH_DEC,
  WJH_LT, WJH_LE, WJH_GT, WJH_GE, WJH_EQ, WJH_NE, WJH_STRICTEQ, WJH_STRICTNE,
  WJH_GETPROP, WJH_SETPROP, WJH_GETELEM, WJH_SETELEM, WJH_GETGNAME, WJH_CALL,
  WJH_FUNCTIONTHIS,
  WJH_BITOR, WJH_BITAND, WJH_BITXOR, WJH_LSH, WJH_RSH, WJH_URSH, WJH_BITNOT,
  WJH_TONUMBER,  // UNBOX: ToNumber(a) -> number (slow path of ensureF64)
  WJH_GETALIASED,  // read a closed-over var via gWJCurEnv + EnvironmentCoordinate
  WJH_POSTBARRIER,  // GC post-write barrier for an object-valued inline SetProp store
  WJH_RESUME,  // PHASE F: self-resume the bailing fn in PBL from gWJHelpA=script,gWJHelpB=basesp
  WJH_NEWOBJECT,  // ALLOC: create the object-literal/template object for this NewObject/NewInit site
  WJH_INITPROP,   // ALLOC: define a data property (constructor `this.x=v` / object literal); leaves obj
  WJH_IONCALL,   // reused-Ion non-inlined call: a=callee fn, gWJHelpB=argc, this/args in gWJScratch
  WJH_METHCALL,  // dynamic method call: gWJHelpA=PropertyName*, gWJHelpB=argc, recv in
                 // gWJScratch[kWJThisSlot], args in gWJScratch[0..argc). Resolves
                 // recv[name] + JS::Call. The megamorphic-dispatch fallback.
  WJH_CONSTRUCT  // `new callee(args)`: a=callee fn (gWJHelpA), gWJHelpB=argc, args in
                 // gWJScratch[0..argc). JS::Construct -> new object in result slot.
};

struct WasmJitEntry {
  enum class State : uint8_t { Cold, Failed, Compiled };
  int handle = -1;       // wasmhost instance handle
  int tableIdx = -1;     // slot in the shared JIT funcref table (-1 = not added)
  uint32_t nargs = 0;    // formal arg count
  uint32_t bcLen = 0;    // script bytecode length: detects JSScript* ABA reuse
  uint32_t runs = 0;     // successful wasm runs (deopt-guard)
  uint32_t deopts = 0;   // deopts (a chronically-deopting fn self-disables)
  uint32_t consecDeopts = 0;  // consecutive deopts (reset on success): polymorphism
  uint64_t helperCalls = 0;   // Mode VS wjhelp calls attributed to this fn (boundary tax)
  bool modeVS = false;   // compiled with the no-restart mutating emitter
  bool vsCapable = false;  // every op is Mode-VS-emittable (so a recompile can use it)
  bool forceVS = false;    // adaptive: recompile as Mode VS (chronic deopt -> no-restart)
  bool wantInline = false;     // recompile with leaf-method inlining (ICs warm)
  bool wantBake = false;       // LEAN EMISSION: recompile baking shape/off constants (ICs warm)
  bool triggeredInline = false;  // one-shot: don't re-trigger the inline recompile
  bool isIon = false;          // installed via the reused-Ion pipeline (IONINT)
  bool ionWarmTried = false;   // one-shot: retried Ion after ICs warmed
  uint32_t ionBackoff = 0;     // PBL->Ion: calls to wait before retrying Ion compile
  uint8_t ionFails = 0;        // PBL->Ion: failed Ion-compile attempts (retry cap)
  bool hasCall = false;        // contains JSOp::Call (an inline-recompile candidate)
  bool hasMathCall = false;    // observed a Math.* intrinsic call (triggers a recompile to emit it)
  bool usesAliased = false;    // contains JSOp::GetAliasedVar -> never a fast-call target
  State state = State::Cold;
};

// Next free slot in the shared JIT funcref table (every compiled function is
// added so it can be a call_indirect target). Capped at the table's size.
static constexpr uint32_t kWJTableSize = 4096;
static uint32_t gWJTableCount = 0;

using WasmJitMap = std::unordered_map<JSScript*, WasmJitEntry>;
static WasmJitMap* gWasmJitMap = nullptr;

// Cumulative counters for the GECKO_DEBUG_JIT periodic deopt log.
static uint64_t gWJTotalDeopts = 0;
static uint64_t gWJTotalRuns = 0;
static uint64_t gWJMegaMiss = 0;  // poly GetProp misses with all N ways full (>N shapes)
static uint64_t gWJVSStrOps = 0;  // Mode VS helper calls with a STRING operand (JIT'd string work)
static uint64_t gWJForceVS = 0;   // adaptive Mode V -> Mode VS recompiles triggered
static uint64_t gWJHelpCalls = 0; // total wjhelp (Mode VS helper) invocations -- the VS boundary tax
static uint64_t gWJHelpKind[32] = {0};  // wjhelp invocations by WJHelperKind
static WasmJitEntry* gWJCurEntry = nullptr;  // the Mode VS fn currently running (helper attribution)
// Environment chain of the Mode VS fn currently entered via WasmJitRunCall. Read by
// WJH_GETALIASED to resolve closed-over (aliased) variable reads. GC-traced. Saved/
// restored around every entry (like gWJCurEntry) so nested calls keep it correct.
// Functions that use JSOp::GetAliasedVar are excluded from the fast call_indirect
// table (tableIdx stays -1) so they ALWAYS enter through here with the right env.
static JSObject* gWJCurEnv = nullptr;
static uint64_t gWJRunsV = 0;   // successful runs in fast Mode N/V (wasm-locals)
static uint64_t gWJRunsVS = 0;  // successful runs in slow Mode VS (frame-memory)
// Diagnostic: why a GetProp IC fill bailed (so the site deopts forever). Indices:
// 0 receiver not-native, 1 own accessor, 2 proto accessor, 3 proto non-native,
// 4 not found in 8 proto levels.
static uint64_t gWJResolveFail[6] = {0};  // [5] = own-accessor bails that are `.length`
// JIT re-entry recursion depth. Deep recursion re-enters the JIT (wasm -> helper
// -> js::Call -> interpreter -> WasmJitRunCall -> wasm ...), stacking many native
// frames per logical level. Past a cap, run the call in the interpreter (which has
// its own stack checks) to avoid a native stack overflow (a hard renderer crash).
// An uncaught wasm trap aborts the process, so the counter does not leak-and-run.
static int gWJCallDepth = 0;
static constexpr int kWJMaxCallDepth = 24;
// GECKO_DEBUG_JIT diagnostic: for each MUTATING function that fails to compile in
// Mode VS, count the first op Mode VS can't handle -> shows what to add next.
static uint32_t gWJVSBlock[256] = {0};
// The op a Mode N/V emitter last bailed on (an unsupported op), + a histogram of
// those across NON-mutating functions that fail to compile.
static JSOp gWJBailOp = JSOp::Nop;
static uint32_t gWJFailOp[256] = {0};
// Histogram of what op's site DEOPTED (IC miss); plus a count of type-guard deopts
// (non-number arith/arg, no recorded site).
static uint32_t gWJDeoptOp[256] = {0};
static uint64_t gWJDeoptType = 0;
// Per-site deopt counter (which exact GetProp/Call site keeps missing) for the
// GECKO_DEBUG_JIT top-offenders dump.
static uint32_t gWJSiteDeopt[kWJMaxSites] = {0};

static bool WJConst(Encoder& e, double v) {
  return e.writeOp(Op::F64Const) && e.writeFixedF64(v);
}

// Emit one non-control-flow JS op as wasm (operand-stack 1:1; every JS arg/local
// is an f64 wasm local). Returns false on an unsupported op. Control-flow ops
// (jumps, returns) and comparisons are handled by WJEmitBodyCF.
// f64 top-of-stack -> i32 with JS ToInt32 semantics (NaN/Inf -> 0/saturate, then
// wrap to low 32 bits). i64.trunc_sat avoids the trap that i32.trunc_f64_s would
// take on NaN/out-of-range.
static bool WJToInt32(Encoder& e) {
  return e.writeOp(MiscOp::I64TruncSatF64S) && e.writeOp(Op::I32WrapI64);
}

static bool WJEmitOp(Encoder& e, jsbytecode* pc, uint32_t nargs,
                     uint32_t argBase, uint32_t rvalLocal,
                     uint32_t scratchLocal) {
  switch (JSOp(*pc)) {
    case JSOp::GetArg:
      return e.writeOp(Op::LocalGet) && e.writeVarU32(argBase + GET_ARGNO(pc));
    case JSOp::SetArg:
      return e.writeOp(Op::LocalTee) && e.writeVarU32(argBase + GET_ARGNO(pc));
    case JSOp::GetLocal:
      return e.writeOp(Op::LocalGet) &&
             e.writeVarU32(argBase + nargs + GET_LOCALNO(pc));
    case JSOp::SetLocal:
      return e.writeOp(Op::LocalTee) &&
             e.writeVarU32(argBase + nargs + GET_LOCALNO(pc));
    case JSOp::Zero:
      return WJConst(e, 0.0);
    case JSOp::One:
      return WJConst(e, 1.0);
    case JSOp::Int8:
      return WJConst(e, double(GET_INT8(pc)));
    case JSOp::Int32:
      return WJConst(e, double(GET_INT32(pc)));
    case JSOp::Uint16:
      return WJConst(e, double(GET_UINT16(pc)));
    case JSOp::Uint24:
      return WJConst(e, double(GET_UINT24(pc)));
    case JSOp::Double: {
      double d;
      memcpy(&d, pc + 1, sizeof(double));
      return WJConst(e, d);
    }
    case JSOp::Add:
      return e.writeOp(Op::F64Add);
    case JSOp::Sub:
      return e.writeOp(Op::F64Sub);
    case JSOp::Mul:
      return e.writeOp(Op::F64Mul);
    case JSOp::Div:
      return e.writeOp(Op::F64Div);
    case JSOp::Neg:
      return e.writeOp(Op::F64Neg);
    case JSOp::BitOr:
    case JSOp::BitAnd:
    case JSOp::BitXor:
    case JSOp::Lsh:
    case JSOp::Rsh:
    case JSOp::Ursh: {
      // a,b on the f64 stack -> ToInt32 both -> i32 op -> back to f64.
      Op iop;
      bool unsignedResult = false;
      switch (JSOp(*pc)) {
        case JSOp::BitOr: iop = Op::I32Or; break;
        case JSOp::BitAnd: iop = Op::I32And; break;
        case JSOp::BitXor: iop = Op::I32Xor; break;
        case JSOp::Lsh: iop = Op::I32Shl; break;
        case JSOp::Rsh: iop = Op::I32ShrS; break;
        default: iop = Op::I32ShrU; unsignedResult = true; break;  // Ursh
      }
      return e.writeOp(Op::LocalSet) && e.writeVarU32(scratchLocal) &&  // tmp=b
             WJToInt32(e) &&                                           // ToInt32(a)
             e.writeOp(Op::LocalGet) && e.writeVarU32(scratchLocal) &&  // push b
             WJToInt32(e) &&                                           // ToInt32(b)
             e.writeOp(iop) &&
             e.writeOp(unsignedResult ? Op::F64ConvertI32U
                                      : Op::F64ConvertI32S);
    }
    case JSOp::BitNot:
      return WJToInt32(e) && e.writeOp(Op::I32Const) && e.writeVarS32(-1) &&
             e.writeOp(Op::I32Xor) && e.writeOp(Op::F64ConvertI32S);
    case JSOp::Inc:
      return WJConst(e, 1.0) && e.writeOp(Op::F64Add);
    case JSOp::Dec:
      return WJConst(e, 1.0) && e.writeOp(Op::F64Sub);
    case JSOp::Pos:
    case JSOp::ToNumeric:
      return true;  // identity on a number
    case JSOp::Pop:
      return e.writeOp(Op::Drop);
    case JSOp::SetRval:
      return e.writeOp(Op::LocalSet) && e.writeVarU32(rvalLocal);
    case JSOp::GetRval:
      return e.writeOp(Op::LocalGet) && e.writeVarU32(rvalLocal);
    case JSOp::JumpTarget:
    case JSOp::LoopHead:
    case JSOp::Nop:
    case JSOp::NopIsAssignOp:
    case JSOp::NopDestructuring:
      return true;  // markers / no-ops
    default:
      gWJBailOp = JSOp(*pc);  // diagnostic: first unsupported Mode N op
      return false;
  }
}

static bool WJIsCmp(JSOp op) {
  return op == JSOp::Lt || op == JSOp::Le || op == JSOp::Gt || op == JSOp::Ge ||
         op == JSOp::Eq || op == JSOp::Ne || op == JSOp::StrictEq ||
         op == JSOp::StrictNe;
}

// The relooper dispatcher emits each basic block assuming the JS operand stack is
// EMPTY at block boundaries (everything lives in locals), and that a conditional
// branch consumes an i32 produced by an immediately-preceding comparison. Code
// that violates these (a value crossing a block boundary -- e.g. a ternary's
// merge point; a truthy branch `if (x)` on a non-comparison Value; or any op that
// would underflow) would compile to INVALID wasm. This validates the assumptions
// up front via a linear stack-depth walk and returns false (bail to the
// interpreter) for anything unsafe, so the emitters never produce invalid wasm.
static bool WJStackSafe(JSScript* script, const std::vector<bool>& isStart) {
  jsbytecode* const start = script->code();
  jsbytecode* const end = script->codeEnd();
  int depth = 0;
  bool prevWasCmp = false;
  for (jsbytecode* pc = start; pc < end; pc += GetBytecodeLength(pc)) {
    uint32_t off = uint32_t(pc - start);
    JSOp op = JSOp(*pc);
    if (off != 0 && isStart[off] && depth != 0) return false;  // value crosses block
    if ((op == JSOp::JumpIfFalse || op == JSOp::JumpIfTrue) && !prevWasCmp) {
      return false;  // branch condition is a Value, not an i32 comparison result
    }
    uint32_t uses = js::StackUses(op, pc);
    if (depth < int(uses)) return false;  // operand-stack underflow
    depth -= int(uses);
    depth += int(js::StackDefs(op));
    prevWasCmp = WJIsCmp(op);
    if (op == JSOp::Goto || op == JSOp::Return || op == JSOp::RetRval) {
      depth = 0;  // unconditional terminator: next block enters fresh
    }
  }
  return true;
}

// Lower `script`'s bytecode (with control flow) into one wasm function body,
// INCLUDING the local declarations and the closing `end`. Reducible JS control
// flow is emitted as a relooper-style dispatcher: a `loop` whose body is a chain
// of `if (pc == i)` blocks (one per basic block); each block runs its straight-
// line ops, computes its successor into the i32 `pc` local, and `br`s back to the
// loop (or `br`s to the outer `exit` block on return). Comparisons are only
// emitted when immediately followed by a conditional branch (so the i32 result is
// consumed there and the rest of the body stays pure-f64). Anything outside the
// supported subset -> false (the caller leaves the function to the interpreter).
// Malformed output is caught by the host's WebAssembly.Module validation.
static bool WJEmitBodyCF(JSScript* script, Encoder& e, uint32_t nargs,
                         uint32_t nfixed) {
  jsbytecode* const start = script->code();
  jsbytecode* const end = script->codeEnd();
  const uint32_t len = uint32_t(end - start);
  const uint32_t argBase = 1;  // local 0 is the f64 scratch pointer
  const uint32_t rvalLocal = argBase + nargs + nfixed;
  const uint32_t scratchLocal = argBase + nargs + nfixed + 1;  // bitwise temp
  const uint32_t vbitsLocal = argBase + nargs + nfixed + 2;     // i64 unbox temp
  const uint32_t pcLocal = argBase + nargs + nfixed + 3;        // i32
  const uint8_t kF64 = uint8_t(TypeCode::F64);
  const uint8_t kI64 = uint8_t(TypeCode::I64);
  const uint8_t kI32 = uint8_t(TypeCode::I32);
  const uint8_t kVoid = 0x40;

  // Pass 1: discover basic-block starts (offset 0, every jump target, and the
  // offset after every branch/return).
  std::vector<bool> isStart(len + 1, false);
  isStart[0] = true;
  for (jsbytecode* pc = start; pc < end;) {
    JSOp op = JSOp(*pc);
    uint32_t ol = GetBytecodeLength(pc);
    uint32_t cur = uint32_t(pc - start);
    if (IsJumpOpcode(op)) {
      if (op == JSOp::And || op == JSOp::Or || op == JSOp::Coalesce) {
        gWJBailOp = op;
        return false;  // short-circuit branches: unsupported
      }
      int64_t tgt = int64_t(cur) + GET_JUMP_OFFSET(pc);
      if (tgt < 0 || tgt > len) return false;
      isStart[tgt] = true;
      if (cur + ol <= len) isStart[cur + ol] = true;
    } else if (op == JSOp::Return || op == JSOp::RetRval) {
      if (cur + ol <= len) isStart[cur + ol] = true;
    }
    pc += ol;
  }

  // Assign block ids in offset order.
  std::vector<int32_t> ofId(len + 1, -1);
  std::vector<uint32_t> blockOff;
  for (uint32_t o = 0; o <= len; o++) {
    if (isStart[o]) {
      ofId[o] = int32_t(blockOff.size());
      blockOff.push_back(o);
    }
  }
  uint32_t K = uint32_t(blockOff.size());
  if (K == 0 || K > 1024) return false;
  if (!WJStackSafe(script, isStart)) return false;

  // Locals (local 0 is the f64 scratch-ptr param, not declared here):
  //   (nargs args + nfixed JS locals + rval + bitwise scratch) f64,
  //   1 i64 unbox temp, 1 i32 `pc`.
  if (!e.writeVarU32(3)) return false;
  if (!e.writeVarU32(nargs + nfixed + 2) || !e.writeFixedU8(kF64)) return false;
  if (!e.writeVarU32(1) || !e.writeFixedU8(kI64)) return false;
  if (!e.writeVarU32(1) || !e.writeFixedU8(kI32)) return false;

  // Prologue: unbox each formal arg from gWJScratch[i] into its f64 local,
  // type-guarding isNumber (deopt=1.0 otherwise so the interpreter re-runs).
  for (uint32_t i = 0; i < nargs; i++) {
    // vbits = i64.load[trunc(ptr) + i*8]
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(0)) return false;
    if (!e.writeOp(MiscOp::I32TruncSatF64U)) return false;
    if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(i * 8)) {
      return false;
    }
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(vbitsLocal)) return false;
    // if ((vbits >> 32) > JSVAL_TAG_INT32) return 1.0;  (not a number)
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(vbitsLocal)) return false;
    if (!e.writeOp(Op::I64Const) || !e.writeVarU64(32)) return false;
    if (!e.writeOp(Op::I64ShrU)) return false;
    if (!e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(kWJTagInt32))) {
      return false;
    }
    if (!e.writeOp(Op::I64LeU)) return false;
    if (!e.writeOp(Op::I32Eqz)) return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
    if (!WJConst(e, 1.0) || !e.writeOp(Op::Return)) return false;
    if (!e.writeOp(Op::End)) return false;
    // arg = isInt32(vbits) ? f64(i32 payload) : reinterpret_f64(vbits)
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(vbitsLocal)) return false;
    if (!e.writeOp(Op::I64Const) || !e.writeVarU64(32)) return false;
    if (!e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64)) return false;
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(kWJTagInt32))) {
      return false;
    }
    if (!e.writeOp(Op::I32Eq)) return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kF64)) return false;
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(vbitsLocal)) return false;
    if (!e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::F64ConvertI32S)) {
      return false;
    }
    if (!e.writeOp(Op::Else)) return false;
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(vbitsLocal)) return false;
    if (!e.writeOp(Op::F64ReinterpretI64)) return false;
    if (!e.writeOp(Op::End)) return false;
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(argBase + i)) return false;
  }

  // Initialize JS `var` locals to NaN: an uninitialized local read as a number
  // is undefined->ToNumber==NaN, so NaN-init matches JS for the numeric body
  // (the wasm default of 0.0 would be wrong). rval also starts NaN.
  {
    double nan;
    uint64_t nanBits = kWJCanonNaN;
    memcpy(&nan, &nanBits, sizeof(double));
    for (uint32_t j = 0; j <= nfixed; j++) {  // nfixed locals + rval
      if (!WJConst(e, nan)) return false;
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(argBase + nargs + j)) {
        return false;
      }
    }
  }

  // block $exit { loop $loop { <dispatch> } }   (br depth 1 = loop, 2 = exit)
  if (!e.writeOp(Op::Block) || !e.writeFixedU8(kVoid)) return false;
  if (!e.writeOp(Op::Loop) || !e.writeFixedU8(kVoid)) return false;

  for (uint32_t i = 0; i < K; i++) {
    // if (pc == i) { ... block i ... }
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(pcLocal)) return false;
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(i))) return false;
    if (!e.writeOp(Op::I32Eq)) return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;

    jsbytecode* pc = start + blockOff[i];
    bool terminated = false;
    while (pc < end && !terminated) {
      JSOp op = JSOp(*pc);
      uint32_t ol = GetBytecodeLength(pc);
      uint32_t cur = uint32_t(pc - start);

      if (IsJumpOpcode(op)) {
        uint32_t fall = cur + ol;
        int32_t tgtId = ofId[uint32_t(int64_t(cur) + GET_JUMP_OFFSET(pc))];
        int32_t fallId = (fall <= len) ? ofId[fall] : -1;
        if (op == JSOp::Goto) {
          if (tgtId < 0) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(tgtId)) return false;
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(pcLocal)) return false;
          if (!e.writeOp(Op::Br) || !e.writeVarU32(1)) return false;
        } else {  // JumpIfFalse / JumpIfTrue: i32 cond already on the stack
          if (tgtId < 0 || fallId < 0) return false;
          int32_t thenId = (op == JSOp::JumpIfTrue) ? tgtId : fallId;
          int32_t elseId = (op == JSOp::JumpIfTrue) ? fallId : tgtId;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(thenId)) return false;
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(pcLocal)) return false;
          if (!e.writeOp(Op::Else)) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(elseId)) return false;
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(pcLocal)) return false;
          if (!e.writeOp(Op::End)) return false;
          if (!e.writeOp(Op::Br) || !e.writeVarU32(1)) return false;
        }
        terminated = true;
      } else if (op == JSOp::Return) {
        if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(rvalLocal)) return false;
        if (!e.writeOp(Op::Br) || !e.writeVarU32(2)) return false;
        terminated = true;
      } else if (op == JSOp::RetRval) {
        if (!e.writeOp(Op::Br) || !e.writeVarU32(2)) return false;
        terminated = true;
      } else if (WJIsCmp(op)) {
        jsbytecode* nx = pc + ol;
        if (nx >= end ||
            (JSOp(*nx) != JSOp::JumpIfFalse && JSOp(*nx) != JSOp::JumpIfTrue)) {
          return false;  // comparison result used as a value: unsupported
        }
        Op cop;
        switch (op) {
          case JSOp::Lt: cop = Op::F64Lt; break;
          case JSOp::Gt: cop = Op::F64Gt; break;
          case JSOp::Le: cop = Op::F64Le; break;
          case JSOp::Ge: cop = Op::F64Ge; break;
          case JSOp::Eq:
          case JSOp::StrictEq: cop = Op::F64Eq; break;
          default: cop = Op::F64Ne; break;  // Ne / StrictNe
        }
        if (!e.writeOp(cop)) return false;
      } else {
        if (!WJEmitOp(e, pc, nargs, argBase, rvalLocal, scratchLocal)) {
          return false;
        }
        uint32_t nextOff = cur + ol;
        if (nextOff <= len && isStart[nextOff] && nextOff != blockOff[i]) {
          // fall through into the next basic block
          int32_t nid = ofId[nextOff];
          if (nid < 0) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(nid)) return false;
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(pcLocal)) return false;
          if (!e.writeOp(Op::Br) || !e.writeVarU32(1)) return false;
          terminated = true;
        }
      }
      pc += ol;
    }
    if (!terminated) {
      if (!e.writeOp(Op::Br) || !e.writeVarU32(2)) return false;  // -> exit
    }
    if (!e.writeOp(Op::End)) return false;  // end if (pc == i)
  }

  if (!e.writeOp(Op::End)) return false;  // end loop
  if (!e.writeOp(Op::End)) return false;  // end block $exit

  // Epilogue: box the f64 rval into gWJScratch[kWJResultSlot] as a Value, then
  // return deopt=0.0. A NaN rval is stored as the canonical quiet NaN (a non-
  // canonical NaN would alias a NUNBOX32 tag). store needs [addr, value]:
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(0)) return false;  // ptr
  if (!e.writeOp(MiscOp::I32TruncSatF64U)) return false;            // addr
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(rvalLocal)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(rvalLocal)) return false;
  if (!e.writeOp(Op::F64Ne)) return false;  // rval != rval  => isNaN
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI64)) return false;
  if (!e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(kWJCanonNaN))) {
    return false;
  }
  if (!e.writeOp(Op::Else)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(rvalLocal)) return false;
  if (!e.writeOp(Op::I64ReinterpretF64)) return false;
  if (!e.writeOp(Op::End)) return false;
  if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) ||
      !e.writeVarU32(kWJResultOff)) {
    return false;
  }
  if (!WJConst(e, 0.0)) return false;  // deopt flag: ok
  return e.writeOp(Op::End);           // end function body
}

// ---- Mode V (Value-typed) emitter: straight-line functions with objects -----
// The operand stack carries NaN-boxed Values (i64). Arithmetic unboxes operands
// to f64 (guarding isNumber, deopting otherwise) and reboxes; GetProp reads a
// property inline via a shape-guarded slot load against the shared guest heap.

// Push f64 = number value of the i64 Value in local `L`, guarding isNumber
// (else `f64.const 1; return` => deopt). Leaves one f64 on the stack.
static bool WJVUnbox(Encoder& e, uint32_t L) {
  const uint8_t kVoid = 0x40, kF64 = uint8_t(TypeCode::F64);
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(L)) return false;
  if (!e.writeOp(Op::I64Const) || !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU)) {
    return false;
  }
  if (!e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(kWJTagInt32))) {
    return false;
  }
  if (!e.writeOp(Op::I64LeU) || !e.writeOp(Op::I32Eqz)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return) || !e.writeOp(Op::End)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(L)) return false;
  if (!e.writeOp(Op::I64Const) || !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) ||
      !e.writeOp(Op::I32WrapI64)) {
    return false;
  }
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(kWJTagInt32))) {
    return false;
  }
  if (!e.writeOp(Op::I32Eq)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kF64)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(L) ||
      !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::F64ConvertI32S)) {
    return false;
  }
  if (!e.writeOp(Op::Else)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(L) ||
      !e.writeOp(Op::F64ReinterpretI64)) {
    return false;
  }
  return e.writeOp(Op::End);
}

// Consume the f64 on the stack, push it reboxed as a Value (i64). A NaN is
// stored as the canonical quiet NaN. Uses f64 scratch local `tmpF`.
static bool WJVRebox(Encoder& e, uint32_t tmpF) {
  const uint8_t kI64 = uint8_t(TypeCode::I64);
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpF)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpF)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpF)) return false;
  if (!e.writeOp(Op::F64Ne)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI64)) return false;
  if (!e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(kWJCanonNaN))) {
    return false;
  }
  if (!e.writeOp(Op::Else)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpF) ||
      !e.writeOp(Op::I64ReinterpretF64)) {
    return false;
  }
  return e.writeOp(Op::End);
}

static bool WJVPushBits(Encoder& e, uint64_t bits) {
  return e.writeOp(Op::I64Const) && e.writeVarS64(int64_t(bits));
}

// store a fixed u32/u64 from a guest global address (baked as i32.const).
static bool WJVStoreAddr32(Encoder& e, uint32_t addr, uint32_t val) {
  return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(addr)) &&
         e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(val)) &&
         e.writeOp(Op::I32Store) && e.writeVarU32(2) && e.writeVarU32(0);
}

// Emit an inline monomorphic GetProp IC. Consumes the object Value (i64), guards
// the receiver's shape, then loads the property -- either an OWN fixed slot (load
// from the receiver) or, if it lives on the receiver's PROTOTYPE (e.g. a method),
// from the cached holder after a holder-shape guard. On any miss records {site,
// object} and deopts (C++ fills the cache: own or proto).
static bool WJEmitGetPropV(Encoder& e, JSScript* script, jsbytecode* pc,
                           jsbytecode* start, uint32_t tmpA, uint32_t tmpAddr,
                           uint32_t tmpIdx) {
  const uint8_t kVoid = 0x40, kI64 = uint8_t(TypeCode::I64);
  if (gWJSiteCount >= kWJMaxSites) return false;
  uint32_t site = gWJSiteCount++;
  gWJSites[site].script = script;
  gWJSites[site].pcOff = uint32_t(pc - start);
  gWJSitePoly[site] = true;  // Mode V GetProp uses the N-way IC
  // Detect a `.length` read: shape-guarded inline array-length load (see gWJSiteLen).
  {
    JSContext* cxn = js::TlsContext.get();
    js::PropertyName* nm = script->getName(pc);
    if (cxn && nm == cxn->names().length) gWJSiteLen[site] = true;
  }
  auto wayIC = [&](uint32_t w) -> uint32_t {  // addr of way w's {shape,off} pair
    if (w == 0) return uint32_t(uintptr_t(&gWJICTable[2 * site]));
    uint32_t x = w - 1;
    return uint32_t(uintptr_t(&gWJICTableX[(x * kWJMaxSites + site) * 2]));
  };
  auto wayHolder = [&](uint32_t w) -> uint32_t {  // addr of way w's holder cell
    if (w == 0) return uint32_t(uintptr_t(&gWJProtoHolder[site]));
    uint32_t x = w - 1;
    return uint32_t(uintptr_t(&gWJProtoHolderX[x * kWJMaxSites + site]));
  };
  auto wayShape = [&](uint32_t w) -> uint32_t {  // addr of way w's holder-shape cell
    if (w == 0) return uint32_t(uintptr_t(&gWJProtoHolderShape[site]));
    uint32_t x = w - 1;
    return uint32_t(uintptr_t(&gWJProtoHolderShapeX[x * kWJMaxSites + site]));
  };
  uint32_t missSiteAddr = uint32_t(uintptr_t(&gWJMissSite));
  uint32_t missObjAddr = uint32_t(uintptr_t(&gWJMissObj));
  const uint8_t kI64b = uint8_t(TypeCode::I64);
  auto emitMiss = [&]() -> bool {
    return WJVStoreAddr32(e, missSiteAddr, site) && e.writeOp(Op::I32Const) &&
           e.writeVarS32(int32_t(missObjAddr)) && e.writeOp(Op::LocalGet) &&
           e.writeVarU32(tmpA) && e.writeOp(Op::I64Store) && e.writeVarU32(3) &&
           e.writeVarU32(0) && WJConst(e, 1.0) && e.writeOp(Op::Return);
  };
  // load the property value from `baseLocal`. off = i32.load[icAddr+4]; if its
  // high bit (kWJDynSlot) is set it is a DYNAMIC slot: load via slots_(@base+8)
  // + (off & ~bit); else a FIXED slot at base + off. `off` is re-loaded from the
  // IC cell each use (cheap) to avoid needing an i32 scratch local.
  auto emitOff = [&](uint32_t icAddr) -> bool {  // push i32 off = ic[2*site+1]
    return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(icAddr)) &&
           e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(4);
  };
  auto emitSlotLoad = [&](uint32_t baseLocal, uint32_t icAddr) -> bool {
    return emitOff(icAddr) && e.writeOp(Op::I32Const) &&
           e.writeVarS32(int32_t(kWJDynSlot)) && e.writeOp(Op::I32And) &&
           e.writeOp(Op::If) && e.writeFixedU8(kI64b) &&
           // dynamic: i64.load[ i32.load[base+8] + (off & ~kWJDynSlot) ]
           e.writeOp(Op::LocalGet) && e.writeVarU32(baseLocal) &&
           e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(8) &&
           emitOff(icAddr) && e.writeOp(Op::I32Const) &&
           e.writeVarS32(int32_t(~kWJDynSlot)) && e.writeOp(Op::I32And) &&
           e.writeOp(Op::I32Add) && e.writeOp(Op::I64Load) && e.writeVarU32(3) &&
           e.writeVarU32(0) && e.writeOp(Op::Else) &&
           // fixed: i64.load[base + off]
           e.writeOp(Op::LocalGet) && e.writeVarU32(baseLocal) && emitOff(icAddr) &&
           e.writeOp(Op::I32Add) && e.writeOp(Op::I64Load) && e.writeVarU32(3) &&
           e.writeVarU32(0) && e.writeOp(Op::End);
  };
  // Emit a dense-array length load (shape already matched -> known ArrayObject).
  // len = i32.load[ i32.load[obj+12]/*elements_*/ - 4 ]; box as INT32. A length
  // that doesn't fit a signed int32 (>= 2^31, would be a double Value) deopts.
  auto emitLenLoad = [&]() -> bool {
    return e.writeOp(Op::LocalGet) && e.writeVarU32(tmpAddr) &&
           e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(12) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(4) && e.writeOp(Op::I32Sub) &&
           e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0) &&
           e.writeOp(Op::LocalTee) && e.writeVarU32(tmpIdx) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(0) && e.writeOp(Op::I32LtS) &&
           e.writeOp(Op::If) && e.writeFixedU8(kI64b) && emitMiss() &&
           e.writeOp(Op::Else) && e.writeOp(Op::LocalGet) && e.writeVarU32(tmpIdx) &&
           e.writeOp(Op::I64ExtendI32U) && e.writeOp(Op::I64Const) &&
           e.writeVarS64(int64_t(kWJTagInt32 << 32)) && e.writeOp(Op::I64Or) &&
           e.writeOp(Op::End);
  };
  // Emit one IC way's own/proto slot load (its recv shape already matched). Leaves
  // an i64 value on the stack, or returns via emitMiss on a holder-shape miss.
  auto emitEntryLoad = [&](uint32_t icAddr, uint32_t protoHolderAddr,
                           uint32_t protoShapeAddr) -> bool {
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(protoHolderAddr)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpIdx)) {
      return false;
    }
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpIdx) ||
        !e.writeOp(Op::I32Eqz)) {
      return false;
    }
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kI64)) return false;
    if (!emitSlotLoad(tmpAddr, icAddr)) return false;  // own: load from receiver
    if (!e.writeOp(Op::Else)) return false;
    // proto: guard holder shape, then load from the holder
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpIdx) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
      return false;
    }
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(protoShapeAddr)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
      return false;
    }
    if (!e.writeOp(Op::I32Eq)) return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kI64)) return false;
    if (!emitSlotLoad(tmpIdx, icAddr)) return false;  // proto: load from holder
    if (!e.writeOp(Op::Else)) return false;
    if (!emitMiss()) return false;            // holder shape miss
    if (!e.writeOp(Op::End)) return false;
    return e.writeOp(Op::End);                // end own/proto if
  };
  // "if (shape@obj == ic[way].shape) {" -- opens an If(result i64).
  auto emitShapeGuard = [&](uint32_t icAddr) -> bool {
    return e.writeOp(Op::LocalGet) && e.writeVarU32(tmpAddr) &&
           e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(icAddr)) &&
           e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0) &&
           e.writeOp(Op::I32Eq) && e.writeOp(Op::If) && e.writeFixedU8(kI64);
  };
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpA)) return false;  // obj
  // For a `.length` site, dispatch on STRING first: a string's length is in the
  // JSString header (immovable field; no allocation), so it's read inline. Other
  // receivers fall to the array/object path below. (offset asserted at build time)
  if (gWJSiteLen[site]) {
    static_assert(JSString::offsetOfLength() == 4, "wasm32 JSString length offset");
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
        !e.writeOp(Op::I64Const) || !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) ||
        !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(kWJTagString)) || !e.writeOp(Op::I32Eq)) {
      return false;
    }
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kI64)) return false;
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
        !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
        !e.writeVarU32(4) || !e.writeOp(Op::I64ExtendI32U) ||
        !e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(kWJTagInt32 << 32)) ||
        !e.writeOp(Op::I64Or)) {
      return false;
    }
    if (!e.writeOp(Op::Else)) return false;
  }
  // guard isObject else deopt (plain type deopt: no IC miss recorded)
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA)) return false;
  if (!e.writeOp(Op::I64Const) || !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU)) {
    return false;
  }
  if (!e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(kWJTagObject)) ||
      !e.writeOp(Op::I64Ne)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return) || !e.writeOp(Op::End)) {
    return false;
  }
  // tmpAddr = low32(obj)
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
      !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(tmpAddr)) {
    return false;
  }
  // N-way chain: for each way, "if shape matches { load } else { <next way> }";
  // the innermost else is the miss/deopt. Each If is i64-typed, so every else
  // branch produces the result i64 (the miss path Returns -> stack-polymorphic).
  for (uint32_t w = 0; w < kWJICWays; w++) {
    if (!emitShapeGuard(wayIC(w))) return false;
    if (gWJSiteLen[site]) {
      if (!emitLenLoad()) return false;
    } else if (!emitEntryLoad(wayIC(w), wayHolder(w), wayShape(w))) {
      return false;
    }
    if (!e.writeOp(Op::Else)) return false;
  }
  if (!emitMiss()) return false;            // all ways missed
  for (uint32_t w = 0; w < kWJICWays; w++) {
    if (!e.writeOp(Op::End)) return false;  // close each way's If
  }
  if (gWJSiteLen[site] && !e.writeOp(Op::End)) return false;  // close STRING-vs-obj
  return true;
}

// Emit the shape-guard + miss-record common to the element/property ICs. After
// this runs, on the stack is nothing new; control either continues (shape hit,
// inside an `if (result i64)` the caller opened) or has returned (miss/non-obj).
// Returns the opened `if (result i64)` that the caller must close with `Else`
// (miss path) ... `End`. Here we inline it per-helper instead for clarity.

// Dense GetElem IC: stack [obj, index] -> obj[index]. Shape-guards the object
// (a stable shape => same class/dense layout), bounds-checks index against the
// dense initializedLength, and loads the element inline. Miss/oob/non-number
// index -> deopt.
static bool WJEmitGetElemV(Encoder& e, JSScript* script, jsbytecode* pc,
                           jsbytecode* start, uint32_t tmpA, uint32_t tmpB,
                           uint32_t tmpAddr, uint32_t tmpIdx) {
  const uint8_t kVoid = 0x40, kI64 = uint8_t(TypeCode::I64);
  if (gWJSiteCount >= kWJMaxSites) return false;
  uint32_t site = gWJSiteCount++;
  gWJSites[site].script = script;
  gWJSites[site].pcOff = uint32_t(pc - start);
  uint32_t icAddr = uint32_t(uintptr_t(&gWJICTable[2 * site]));
  uint32_t missSiteAddr = uint32_t(uintptr_t(&gWJMissSite));
  uint32_t missObjAddr = uint32_t(uintptr_t(&gWJMissObj));
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpB)) return false;  // index
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpA)) return false;  // obj
  // isObject guard
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA)) return false;
  if (!e.writeOp(Op::I64Const) || !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) ||
      !e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(kWJTagObject)) ||
      !e.writeOp(Op::I64Ne)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return) || !e.writeOp(Op::End)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
      !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(tmpAddr)) {
    return false;
  }
  // shape guard
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpAddr) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(icAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::I32Eq)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI64)) return false;
  // tmpIdx = trunc_s(unbox(index))
  if (!WJVUnbox(e, tmpB) || !e.writeOp(MiscOp::I32TruncSatF64S) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpIdx)) {
    return false;
  }
  // tmpAddr = elements_ = i32.load[obj + 12]
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpAddr) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(12) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpAddr)) {
    return false;
  }
  // if (index u< initializedLength@(elements-12))
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpIdx)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpAddr) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(12) || !e.writeOp(Op::I32Sub) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::I32LtU)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI64)) return false;
  // value = i64.load[elements + index*8]
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpAddr)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpIdx) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) ||
      !e.writeOp(Op::I32Add)) {
    return false;
  }
  if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::Else)) return false;  // out of bounds -> deopt
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return)) return false;
  if (!e.writeOp(Op::End)) return false;  // end bounds if
  if (!e.writeOp(Op::Else)) return false;  // shape miss -> record + deopt
  if (!WJVStoreAddr32(e, missSiteAddr, site)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(missObjAddr)) ||
      !e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
      !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
    return false;
  }
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return)) return false;
  return e.writeOp(Op::End);  // end shape if
}

// Dense SetElem IC: stack [obj, index, value] -> value. Inline-stores only when
// value is a number (no GC post-barrier needed) and index is in bounds; any
// other case (object value, oob/append, shape miss) deopts.
static bool WJEmitSetElemV(Encoder& e, JSScript* script, jsbytecode* pc,
                           jsbytecode* start, uint32_t tmpA, uint32_t tmpB,
                           uint32_t tmpC, uint32_t tmpAddr, uint32_t tmpIdx) {
  const uint8_t kVoid = 0x40, kI64 = uint8_t(TypeCode::I64);
  if (gWJSiteCount >= kWJMaxSites) return false;
  uint32_t site = gWJSiteCount++;
  gWJSites[site].script = script;
  gWJSites[site].pcOff = uint32_t(pc - start);
  uint32_t icAddr = uint32_t(uintptr_t(&gWJICTable[2 * site]));
  uint32_t missSiteAddr = uint32_t(uintptr_t(&gWJMissSite));
  uint32_t missObjAddr = uint32_t(uintptr_t(&gWJMissObj));
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpC)) return false;  // value
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpB)) return false;  // index
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpA)) return false;  // obj
  // isObject guard
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA)) return false;
  if (!e.writeOp(Op::I64Const) || !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) ||
      !e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(kWJTagObject)) ||
      !e.writeOp(Op::I64Ne)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return) || !e.writeOp(Op::End)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
      !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(tmpAddr)) {
    return false;
  }
  // shape guard
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpAddr) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(icAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::I32Eq)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI64)) return false;
  // value must be a number (barrier-free store) else deopt
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpC) ||
      !e.writeOp(Op::I64Const) || !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) ||
      !e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(kWJTagInt32)) ||
      !e.writeOp(Op::I64LeU) || !e.writeOp(Op::I32Eqz)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return) || !e.writeOp(Op::End)) {
    return false;
  }
  // tmpIdx = trunc_s(unbox(index)); tmpAddr = elements_
  if (!WJVUnbox(e, tmpB) || !e.writeOp(MiscOp::I32TruncSatF64S) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpIdx)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpAddr) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(12) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpAddr)) {
    return false;
  }
  // if (index u< initializedLength) { store } else { deopt }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpIdx)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpAddr) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(12) || !e.writeOp(Op::I32Sub) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::I32LtU)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  // i64.store[elements + index*8] = value
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpAddr)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpIdx) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) ||
      !e.writeOp(Op::I32Add)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpC)) return false;
  if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::Else)) return false;  // oob -> deopt
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return)) return false;
  if (!e.writeOp(Op::End)) return false;  // end bounds if
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpC)) return false;  // result
  if (!e.writeOp(Op::Else)) return false;  // shape miss
  if (!WJVStoreAddr32(e, missSiteAddr, site)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(missObjAddr)) ||
      !e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
      !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
    return false;
  }
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return)) return false;
  return e.writeOp(Op::End);  // end shape if
}

// Inline SetProp: stack [obj, value] -> value. Shape-guarded fixed-slot store,
// number values only (barrier-free); other cases deopt. Reuses the GetProp IC
// entry layout {shape, byteOffset}.
static bool WJEmitSetPropV(Encoder& e, JSScript* script, jsbytecode* pc,
                           jsbytecode* start, uint32_t tmpA, uint32_t tmpC,
                           uint32_t tmpAddr) {
  const uint8_t kVoid = 0x40, kI64 = uint8_t(TypeCode::I64);
  if (gWJSiteCount >= kWJMaxSites) return false;
  uint32_t site = gWJSiteCount++;
  gWJSites[site].script = script;
  gWJSites[site].pcOff = uint32_t(pc - start);
  uint32_t icAddr = uint32_t(uintptr_t(&gWJICTable[2 * site]));
  uint32_t missSiteAddr = uint32_t(uintptr_t(&gWJMissSite));
  uint32_t missObjAddr = uint32_t(uintptr_t(&gWJMissObj));
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpC)) return false;  // value
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpA)) return false;  // obj
  // isObject guard
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA)) return false;
  if (!e.writeOp(Op::I64Const) || !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) ||
      !e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(kWJTagObject)) ||
      !e.writeOp(Op::I64Ne)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return) || !e.writeOp(Op::End)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
      !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(tmpAddr)) {
    return false;
  }
  // shape guard
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpAddr) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(icAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::I32Eq)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI64)) return false;
  // value must be a number else deopt
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpC) ||
      !e.writeOp(Op::I64Const) || !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) ||
      !e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(kWJTagInt32)) ||
      !e.writeOp(Op::I64LeU) || !e.writeOp(Op::I32Eqz)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return) || !e.writeOp(Op::End)) {
    return false;
  }
  // i64.store[obj + offset@(ic+4)] = value
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpAddr)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(icAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(4) ||
      !e.writeOp(Op::I32Add)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpC)) return false;
  if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpC)) return false;  // result
  if (!e.writeOp(Op::Else)) return false;  // shape miss
  if (!WJVStoreAddr32(e, missSiteAddr, site)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(missObjAddr)) ||
      !e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
      !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
    return false;
  }
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return)) return false;
  return e.writeOp(Op::End);  // end shape if
}

// Inline call: stack [callee, this, arg0..arg(argc-1)] -> result. Marshals the
// args into gWJScratch (which is free here: the prologue already copied this
// frame's args into locals, so nested/recursive calls can reuse it), guards the
// callee against the site's cached JSFunction, and invokes the cached compiled
// callee DIRECTLY via call_indirect on the shared funcref table -- a native wasm
// call, no JS/C++ bridge hop (fast; works for recursion). Miss / callee-deopt ->
// deopt. `this` is dropped (a `this`-using callee won't compile, so it never
// fills here). gWJCallHandle[site] holds the callee's shared-table index.
static bool WJEmitCallV(Encoder& e, JSScript* script, jsbytecode* pc,
                        jsbytecode* start, uint32_t tmpA) {
  const uint8_t kVoid = 0x40, kI64 = uint8_t(TypeCode::I64);
  uint32_t argc = GET_ARGC(pc);
  if (argc > 32) return false;
  if (gWJSiteCount >= kWJMaxSites) return false;
  uint32_t site = gWJSiteCount++;
  gWJSites[site].script = script;
  gWJSites[site].pcOff = uint32_t(pc - start);
  gWJCallArgc[site] = argc;
  uint32_t scratchBase = uint32_t(uintptr_t(static_cast<void*>(gWJScratch)));
  uint32_t callFnAddr = uint32_t(uintptr_t(&gWJCallFn[site]));
  uint32_t callIdxAddr = uint32_t(uintptr_t(&gWJCallHandle[site]));
  uint32_t missSiteAddr = uint32_t(uintptr_t(&gWJMissSite));
  uint32_t missObjAddr = uint32_t(uintptr_t(&gWJMissObj));
  // Marshal args in reverse (top of stack is the last arg) into gWJScratch[i].
  for (int32_t i = int32_t(argc) - 1; i >= 0; i--) {
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpA)) return false;
    if (!e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(scratchBase + uint32_t(i) * 8))) {
      return false;
    }
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA)) return false;
    if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
      return false;
    }
  }
  // Marshal `this` (the receiver) into gWJScratch[this] so a method callee's
  // FunctionThis can read it (plain calls pass undefined here, harmlessly).
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpA)) return false;  // this
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(scratchBase + kWJThisOff)) ||
      !e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
      !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpA)) return false;  // callee
  // callee guard: low32(callee) == cached JSFunction*
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
      !e.writeOp(Op::I32WrapI64)) {
    return false;
  }
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(callFnAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::I32Eq)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI64)) return false;
  // invoke: call_indirect table[callee idx] (signature type 0 = (f64)->f64),
  // passing the scratch pointer; -> f64 deopt.
  if (!WJConst(e, double(scratchBase))) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(callIdxAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::CallIndirect) || !e.writeVarU32(0) || !e.writeVarU32(0)) {
    return false;  // typeidx 0, tableidx 0
  }
  // propagate a callee deopt to the caller
  if (!WJConst(e, 0.0) || !e.writeOp(Op::F64Ne)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return) || !e.writeOp(Op::End)) {
    return false;
  }
  // result = gWJScratch[result slot]
  if (!e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(scratchBase + kWJResultOff)) ||
      !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::Else)) return false;  // callee miss -> record + deopt
  if (!WJVStoreAddr32(e, missSiteAddr, site)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(missObjAddr)) ||
      !e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
      !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
    return false;
  }
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return)) return false;
  return e.writeOp(Op::End);
}

// Inline GetGName: read a global var/function by name. The resolved holder
// (global object) address lives in a mutable IC cell (so a GC move self-heals
// via refill); guard its shape, then load the cached slot (fixed or dynamic).
// On miss / unfilled (holder==0) -> record site + deopt (C++ resolves + fills).
static bool WJEmitGetGNameV(Encoder& e, JSScript* script, jsbytecode* pc,
                            jsbytecode* start, uint32_t tmpAddr,
                            uint32_t tmpIdx) {
  const uint8_t kVoid = 0x40, kI64 = uint8_t(TypeCode::I64);
  if (gWJSiteCount >= kWJMaxSites) return false;
  uint32_t site = gWJSiteCount++;
  gWJSites[site].script = script;
  gWJSites[site].pcOff = uint32_t(pc - start);
  uint32_t holderCell = uint32_t(uintptr_t(&gWJGNameHolder[site]));
  uint32_t icAddr = uint32_t(uintptr_t(&gWJICTable[2 * site]));
  uint32_t missSiteAddr = uint32_t(uintptr_t(&gWJMissSite));
  // tmpAddr = holder = i32.load[holderCell]
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(holderCell)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpAddr)) {
    return false;
  }
  // if holder==0 (unfilled) -> miss
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpAddr) ||
      !e.writeOp(Op::I32Eqz)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!WJVStoreAddr32(e, missSiteAddr, site)) return false;
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return) || !e.writeOp(Op::End)) {
    return false;
  }
  // shape guard
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpAddr) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(icAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::I32Eq)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI64)) return false;
  // off = i32.load[icAddr+4]
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(icAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(4) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpIdx)) {
    return false;
  }
  // if (off & kWJDynSlot) dynamic else fixed
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpIdx) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(kWJDynSlot)) ||
      !e.writeOp(Op::I32And)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI64)) return false;
  // dynamic: slots_ = i32.load[holder+8]; value = i64.load[slots_ + (off & ~dyn)]
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpAddr) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(8)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpIdx) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(~kWJDynSlot)) ||
      !e.writeOp(Op::I32And) || !e.writeOp(Op::I32Add)) {
    return false;
  }
  if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::Else)) return false;
  // fixed: value = i64.load[holder + off]
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpAddr) ||
      !e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpIdx) ||
      !e.writeOp(Op::I32Add)) {
    return false;
  }
  if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::End)) return false;  // end fixed/dynamic
  if (!e.writeOp(Op::Else)) return false;  // shape miss
  if (!WJVStoreAddr32(e, missSiteAddr, site)) return false;
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return)) return false;
  return e.writeOp(Op::End);  // end shape if
}

// One non-control-flow Value-typed op (operand stack carries i64 Values).
static bool WJVToBool(Encoder& e, uint32_t srcLocal, uint32_t tmpF,
                      uint32_t tmpTag);

// StrictConstantEq/Ne: pop a Value, strictly (===/!==) compare it against the
// encoded constant (null / undefined / boolean / int8), push the boolean. No
// coercion, no throw. int32 constant matches an INT32 payload OR an equal double
// (5 === 5.0). tmpTag is i32 scratch. Leaves a boolean Value on the stack.
static bool WJVStrictConst(Encoder& e, jsbytecode* pc, uint32_t tmpA,
                           uint32_t tmpTag, bool isEq) {
  using ET = js::ConstantCompareOperand::EncodedType;
  js::ConstantCompareOperand cco =
      js::ConstantCompareOperand::fromRawValue(GET_UINT16(pc));
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpA)) return false;  // pop val
  ET t = cco.type();
  if (t == ET::Null || t == ET::Undefined) {
    uint32_t tag = (t == ET::Null) ? kWJTagNull : kWJTagUndefined;
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
        !e.writeOp(Op::I64Const) || !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) ||
        !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(tag)) || !e.writeOp(Op::I32Eq)) {
      return false;
    }
  } else if (t == ET::Boolean) {
    int32_t b = cco.toBoolean() ? 1 : 0;
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
        !e.writeOp(Op::I64Const) || !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) ||
        !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(kWJTagBoolean)) || !e.writeOp(Op::I32Eq) ||
        !e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
        !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(b) || !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32And)) {
      return false;
    }
  } else if (t == ET::Int32) {
    int32_t k = cco.toInt32();
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
        !e.writeOp(Op::I64Const) || !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) ||
        !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::LocalTee) ||
        !e.writeVarU32(tmpTag) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(kWJTagInt32)) || !e.writeOp(Op::I32Eq) ||
        !e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
        !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(k) || !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32And)) {
      return false;  // (tag==INT32) & (low32==k)
    }
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpTag) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(kWJTagClear)) ||
        !e.writeOp(Op::I32LeU) || !e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
        !e.writeOp(Op::F64ReinterpretI64) || !e.writeOp(Op::F64Const) ||
        !e.writeFixedF64(double(k)) || !e.writeOp(Op::F64Eq) ||
        !e.writeOp(Op::I32And) || !e.writeOp(Op::I32Or)) {
      return false;  // | (isDouble & d==k)
    }
  } else {
    return false;  // unknown encoded type
  }
  if (!isEq && !e.writeOp(Op::I32Eqz)) return false;  // !== inverts
  return e.writeOp(Op::I64ExtendI32U) && e.writeOp(Op::I64Const) &&
         e.writeVarS64(int64_t(uint64_t(kWJTagBoolean) << 32)) &&
         e.writeOp(Op::I64Or);
}

// TypeofEq: pop a Value, push (typeof val ==/!= "type"). The constant type is in
// the operand. Tag-only types (undefined/string/number/boolean/symbol/bigint) are
// pure tag checks. object/function need a callability probe -> BAIL (the whole fn
// stays interpreted; no perpetual deopt). tmpTag is i32 scratch.
static bool WJVTypeofEq(Encoder& e, jsbytecode* pc, uint32_t tmpA,
                        uint32_t tmpTag) {
  js::TypeofEqOperand op = js::TypeofEqOperand::fromRawValue(GET_UINT8(pc));
  JSType ty = op.type();
  if (ty == JSTYPE_OBJECT || ty == JSTYPE_FUNCTION) return false;  // callability
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpA)) return false;  // pop val
  // tag = i32.wrap(val >> 32)
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpA) ||
      !e.writeOp(Op::I64Const) || !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) ||
      !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(tmpTag)) {
    return false;
  }
  auto tagEq = [&](uint32_t tag) -> bool {
    return e.writeOp(Op::LocalGet) && e.writeVarU32(tmpTag) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(tag)) &&
           e.writeOp(Op::I32Eq);
  };
  switch (ty) {
    case JSTYPE_UNDEFINED: if (!tagEq(kWJTagUndefined)) return false; break;
    case JSTYPE_STRING:    if (!tagEq(kWJTagString)) return false; break;
    case JSTYPE_BOOLEAN:   if (!tagEq(kWJTagBoolean)) return false; break;
    case JSTYPE_SYMBOL:    if (!tagEq(kWJTagSymbol)) return false; break;
    case JSTYPE_BIGINT:    if (!tagEq(kWJTagBigInt)) return false; break;
    case JSTYPE_NUMBER:  // int32 OR a double (tag u<= JSVAL_TAG_CLEAR)
      if (!tagEq(kWJTagInt32) || !e.writeOp(Op::LocalGet) ||
          !e.writeVarU32(tmpTag) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(kWJTagClear)) || !e.writeOp(Op::I32LeU) ||
          !e.writeOp(Op::I32Or)) {
        return false;
      }
      break;
    default:
      return false;
  }
  if (op.compareOp() == JSOp::Ne && !e.writeOp(Op::I32Eqz)) return false;
  return e.writeOp(Op::I64ExtendI32U) && e.writeOp(Op::I64Const) &&
         e.writeVarS64(int64_t(uint64_t(kWJTagBoolean) << 32)) &&
         e.writeOp(Op::I64Or);
}

// Integer modulo: both operands INT32 -> i32.rem_s -> box INT32. A non-int32
// operand or a zero divisor (JS x%0 = NaN, no i32 equivalent) DEOPTS; a hot
// double-% function self-disables via the deopt-count guard. Sign matches JS
// (sign of the dividend), as does i32.rem_s; INT32_MIN%-1 = 0 (rem_s, no trap).
static bool WJVModInt(Encoder& e, uint32_t tmpA, uint32_t tmpB) {
  auto deoptIf = [&]() -> bool {  // stack has an i32 cond; if true, deopt (return 1.0)
    return e.writeOp(Op::If) && e.writeFixedU8(0x40) && WJConst(e, 1.0) &&
           e.writeOp(Op::Return) && e.writeOp(Op::End);
  };
  auto notInt32 = [&](uint32_t L) -> bool {  // (tag(L) != INT32) cond
    return e.writeOp(Op::LocalGet) && e.writeVarU32(L) && e.writeOp(Op::I64Const) &&
           e.writeVarU64(32) && e.writeOp(Op::I64ShrU) && e.writeOp(Op::I64Const) &&
           e.writeVarS64(int64_t(kWJTagInt32)) && e.writeOp(Op::I64Ne);
  };
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpB) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpA)) {
    return false;
  }
  if (!notInt32(tmpA) || !deoptIf()) return false;
  if (!notInt32(tmpB) || !deoptIf()) return false;
  // divisor 0 -> deopt
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpB) ||
      !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Eqz) || !deoptIf()) {
    return false;
  }
  return e.writeOp(Op::LocalGet) && e.writeVarU32(tmpA) &&
         e.writeOp(Op::I32WrapI64) && e.writeOp(Op::LocalGet) &&
         e.writeVarU32(tmpB) && e.writeOp(Op::I32WrapI64) &&
         e.writeOp(Op::I32RemS) && e.writeOp(Op::I64ExtendI32U) &&
         e.writeOp(Op::I64Const) && e.writeVarS64(int64_t(kWJTagInt32 << 32)) &&
         e.writeOp(Op::I64Or);
}

static bool WJEmitOpV(Encoder& e, jsbytecode* pc, JSScript* script,
                      jsbytecode* start, uint32_t nargs, uint32_t argBase,
                      uint32_t rvalLocal, uint32_t tmpA, uint32_t tmpB,
                      uint32_t tmpC, uint32_t tmpF, uint32_t tmpAddr,
                      uint32_t tmpIdx) {
  switch (JSOp(*pc)) {
    case JSOp::GetArg:
      return e.writeOp(Op::LocalGet) && e.writeVarU32(argBase + GET_ARGNO(pc));
    case JSOp::SetArg:
      return e.writeOp(Op::LocalTee) && e.writeVarU32(argBase + GET_ARGNO(pc));
    case JSOp::GetLocal:
      return e.writeOp(Op::LocalGet) &&
             e.writeVarU32(argBase + nargs + GET_LOCALNO(pc));
    case JSOp::SetLocal:
      return e.writeOp(Op::LocalTee) &&
             e.writeVarU32(argBase + nargs + GET_LOCALNO(pc));
    case JSOp::Zero:
      return WJVPushBits(e, (kWJTagInt32 << 32));
    case JSOp::One:
      return WJVPushBits(e, (kWJTagInt32 << 32) | 1);
    case JSOp::Int8:
      return WJVPushBits(e, (kWJTagInt32 << 32) | uint32_t(GET_INT8(pc)));
    case JSOp::Int32:
      return WJVPushBits(e, (kWJTagInt32 << 32) | uint32_t(GET_INT32(pc)));
    case JSOp::Uint16:
      return WJVPushBits(e, (kWJTagInt32 << 32) | uint32_t(GET_UINT16(pc)));
    case JSOp::Uint24:
      return WJVPushBits(e, (kWJTagInt32 << 32) | uint32_t(GET_UINT24(pc)));
    case JSOp::Undefined:
      return WJVPushBits(e, 0xFFFFFF83ULL << 32);
    case JSOp::Null:
      return WJVPushBits(e, 0xFFFFFF84ULL << 32);
    case JSOp::True:
      return WJVPushBits(e, (0xFFFFFF82ULL << 32) | 1);
    case JSOp::False:
      return WJVPushBits(e, (0xFFFFFF82ULL << 32));
    case JSOp::Double: {
      double d;
      memcpy(&d, pc + 1, sizeof(double));
      uint64_t b;
      memcpy(&b, &d, sizeof(double));
      if (d != d) b = kWJCanonNaN;
      return WJVPushBits(e, b);
    }
    case JSOp::String: {
      // Push a string LITERAL. Only atoms (interned, immovable, kept alive by the
      // script) can be safely baked as a constant pointer -- a non-atom could move
      // under a compacting GC, and the untraced wasm local would dangle. Bail then.
      JSString* s = script->getString(pc);
      if (!s->isAtom()) return false;
      return WJVPushBits(
          e, (uint64_t(kWJTagString) << 32) | uint32_t(uintptr_t(s)));
    }
    case JSOp::Add:
    case JSOp::Sub:
    case JSOp::Mul:
    case JSOp::Div: {
      Op fop = JSOp(*pc) == JSOp::Add   ? Op::F64Add
               : JSOp(*pc) == JSOp::Sub ? Op::F64Sub
               : JSOp(*pc) == JSOp::Mul ? Op::F64Mul
                                        : Op::F64Div;
      return e.writeOp(Op::LocalSet) && e.writeVarU32(tmpB) &&
             e.writeOp(Op::LocalSet) && e.writeVarU32(tmpA) &&
             WJVUnbox(e, tmpA) && WJVUnbox(e, tmpB) && e.writeOp(fop) &&
             WJVRebox(e, tmpF);
    }
    case JSOp::Neg:
      return e.writeOp(Op::LocalSet) && e.writeVarU32(tmpA) &&
             WJVUnbox(e, tmpA) && e.writeOp(Op::F64Neg) && WJVRebox(e, tmpF);
    case JSOp::Inc:
    case JSOp::Dec:
      return e.writeOp(Op::LocalSet) && e.writeVarU32(tmpA) &&
             WJVUnbox(e, tmpA) && WJConst(e, 1.0) &&
             e.writeOp(JSOp(*pc) == JSOp::Inc ? Op::F64Add : Op::F64Sub) &&
             WJVRebox(e, tmpF);
    case JSOp::BitOr:
    case JSOp::BitAnd:
    case JSOp::BitXor:
    case JSOp::Lsh:
    case JSOp::Rsh: {  // signed bitwise: ToInt32 both -> i32 op -> box INT32
      Op iop = JSOp(*pc) == JSOp::BitOr    ? Op::I32Or
               : JSOp(*pc) == JSOp::BitAnd ? Op::I32And
               : JSOp(*pc) == JSOp::BitXor ? Op::I32Xor
               : JSOp(*pc) == JSOp::Lsh    ? Op::I32Shl
                                           : Op::I32ShrS;
      return e.writeOp(Op::LocalSet) && e.writeVarU32(tmpB) &&
             e.writeOp(Op::LocalSet) && e.writeVarU32(tmpA) &&
             WJVUnbox(e, tmpA) && WJToInt32(e) && WJVUnbox(e, tmpB) &&
             WJToInt32(e) && e.writeOp(iop) && e.writeOp(Op::I64ExtendI32U) &&
             e.writeOp(Op::I64Const) &&
             e.writeVarS64(int64_t(kWJTagInt32 << 32)) && e.writeOp(Op::I64Or);
    }
    case JSOp::BitNot:  // ~x -> ToInt32(x) ^ -1 -> box INT32
      return e.writeOp(Op::LocalSet) && e.writeVarU32(tmpA) && WJVUnbox(e, tmpA) &&
             WJToInt32(e) && e.writeOp(Op::I32Const) && e.writeVarS32(-1) &&
             e.writeOp(Op::I32Xor) && e.writeOp(Op::I64ExtendI32U) &&
             e.writeOp(Op::I64Const) && e.writeVarS64(int64_t(kWJTagInt32 << 32)) &&
             e.writeOp(Op::I64Or);
    case JSOp::GetProp:
      return WJEmitGetPropV(e, script, pc, start, tmpA, tmpAddr, tmpIdx);
    case JSOp::GetGName:
      return WJEmitGetGNameV(e, script, pc, start, tmpAddr, tmpIdx);
    case JSOp::SetProp:
    case JSOp::StrictSetProp:
      return WJEmitSetPropV(e, script, pc, start, tmpA, tmpC, tmpAddr);
    case JSOp::GetElem:
      return WJEmitGetElemV(e, script, pc, start, tmpA, tmpB, tmpAddr, tmpIdx);
    case JSOp::SetElem:
    case JSOp::StrictSetElem:
      return WJEmitSetElemV(e, script, pc, start, tmpA, tmpB, tmpC, tmpAddr,
                            tmpIdx);
    case JSOp::Call:
    case JSOp::CallContent:
    case JSOp::CallIgnoresRv:
      return WJEmitCallV(e, script, pc, start, tmpA);
    case JSOp::FunctionThis: {
      // Push the marshalled receiver (gWJScratch[this]); guard it is an object
      // (a method's `this`). undefined/primitive `this` needs sloppy-mode boxing
      // -> deopt to the interpreter.
      uint32_t thisAddr = uint32_t(uintptr_t(static_cast<void*>(gWJScratch))) +
                          kWJThisOff;
      return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(thisAddr)) &&
             e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0) &&
             e.writeOp(Op::LocalSet) && e.writeVarU32(tmpA) &&
             e.writeOp(Op::LocalGet) && e.writeVarU32(tmpA) &&
             e.writeOp(Op::I64Const) && e.writeVarU64(32) &&
             e.writeOp(Op::I64ShrU) && e.writeOp(Op::I64Const) &&
             e.writeVarS64(int64_t(kWJTagObject)) && e.writeOp(Op::I64Ne) &&
             e.writeOp(Op::If) && e.writeFixedU8(0x40) && WJConst(e, 1.0) &&
             e.writeOp(Op::Return) && e.writeOp(Op::End) &&
             e.writeOp(Op::LocalGet) && e.writeVarU32(tmpA);
    }
    case JSOp::Dup:  // [v] -> [v, v]
      return e.writeOp(Op::LocalTee) && e.writeVarU32(tmpA) &&
             e.writeOp(Op::LocalGet) && e.writeVarU32(tmpA);
    case JSOp::Swap:  // [a, b] -> [b, a]
      return e.writeOp(Op::LocalSet) && e.writeVarU32(tmpA) &&
             e.writeOp(Op::LocalSet) && e.writeVarU32(tmpB) &&
             e.writeOp(Op::LocalGet) && e.writeVarU32(tmpA) &&
             e.writeOp(Op::LocalGet) && e.writeVarU32(tmpB);
    case JSOp::Pop:
      return e.writeOp(Op::Drop);
    case JSOp::SetRval:
      return e.writeOp(Op::LocalSet) && e.writeVarU32(rvalLocal);
    case JSOp::GetRval:
      return e.writeOp(Op::LocalGet) && e.writeVarU32(rvalLocal);
    case JSOp::Not:  // !x -> boolean (ToBoolean then invert + box)
      return e.writeOp(Op::LocalSet) && e.writeVarU32(tmpA) &&
             WJVToBool(e, tmpA, tmpF, tmpIdx) && e.writeOp(Op::I32Eqz) &&
             e.writeOp(Op::I64ExtendI32U) && e.writeOp(Op::I64Const) &&
             e.writeVarS64(int64_t(uint64_t(kWJTagBoolean) << 32)) &&
             e.writeOp(Op::I64Or);
    case JSOp::StrictConstantEq:
      return WJVStrictConst(e, pc, tmpA, tmpIdx, true);
    case JSOp::StrictConstantNe:
      return WJVStrictConst(e, pc, tmpA, tmpIdx, false);
    case JSOp::TypeofEq:
      return WJVTypeofEq(e, pc, tmpA, tmpIdx);
    case JSOp::Mod:
      return WJVModInt(e, tmpA, tmpB);
    case JSOp::Pos:
    case JSOp::ToNumeric:
    case JSOp::Nop:
    case JSOp::NopIsAssignOp:
    case JSOp::NopDestructuring:
    case JSOp::JumpTarget:
    case JSOp::LoopHead:
      return true;  // markers / no-ops
    default:
      gWJBailOp = JSOp(*pc);  // diagnostic: first unsupported Mode V op
      return false;
  }
}

// Value-typed comparison: pop 2 i64 -> unbox both (guard isNumber) -> f64 cmp;
// the i32 result is consumed by the following JumpIfFalse/JumpIfTrue.
static bool WJVCmp(Encoder& e, JSOp op, uint32_t tmpA, uint32_t tmpB) {
  Op cop;
  switch (op) {
    case JSOp::Lt: cop = Op::F64Lt; break;
    case JSOp::Gt: cop = Op::F64Gt; break;
    case JSOp::Le: cop = Op::F64Le; break;
    case JSOp::Ge: cop = Op::F64Ge; break;
    case JSOp::Eq:
    case JSOp::StrictEq: cop = Op::F64Eq; break;
    default: cop = Op::F64Ne; break;  // Ne / StrictNe
  }
  return e.writeOp(Op::LocalSet) && e.writeVarU32(tmpB) &&
         e.writeOp(Op::LocalSet) && e.writeVarU32(tmpA) && WJVUnbox(e, tmpA) &&
         WJVUnbox(e, tmpB) && e.writeOp(cop);
}

// ToBoolean(value in srcLocal) -> i32 (1 truthy / 0 falsy) on the stack. Handles
// double / int32 / boolean / object / undefined / null inline; string/symbol/
// bigint/magic DEOPT (return 1.0 -- safe: Mode V is non-mutating, restart re-runs).
// tmpF (f64) + tmpTag (i32) are scratch.
static bool WJVToBool(Encoder& e, uint32_t srcLocal, uint32_t tmpF,
                      uint32_t tmpTag) {
  const uint8_t kI32 = uint8_t(TypeCode::I32);
  // tag = i32.wrap(src >> 32); isDouble = tag u<= JSVAL_TAG_CLEAR
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(srcLocal) ||
      !e.writeOp(Op::I64Const) || !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) ||
      !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::LocalTee) ||
      !e.writeVarU32(tmpTag) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(kWJTagClear)) || !e.writeOp(Op::I32LeU)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;
  // double: (d == d) & (d != 0)
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(srcLocal) ||
      !e.writeOp(Op::F64ReinterpretI64) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(tmpF)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpF) || !e.writeOp(Op::LocalGet) ||
      !e.writeVarU32(tmpF) || !e.writeOp(Op::F64Eq)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpF) || !e.writeOp(Op::F64Const) ||
      !e.writeFixedF64(0.0) || !e.writeOp(Op::F64Ne) || !e.writeOp(Op::I32And)) {
    return false;
  }
  if (!e.writeOp(Op::Else)) return false;
  // tag == OBJECT -> 1
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpTag) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(kWJTagObject)) ||
      !e.writeOp(Op::I32Eq)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(1)) return false;
  if (!e.writeOp(Op::Else)) return false;
  // tag == INT32 || tag == BOOLEAN -> low32 != 0
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpTag) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(kWJTagInt32)) ||
      !e.writeOp(Op::I32Eq) || !e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpTag) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(kWJTagBoolean)) ||
      !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32Or)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(srcLocal) ||
      !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) || !e.writeVarS32(0) ||
      !e.writeOp(Op::I32Ne)) {
    return false;
  }
  if (!e.writeOp(Op::Else)) return false;
  // tag == UNDEFINED || NULL -> 0; else (string/symbol/bigint/magic) -> deopt
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpTag) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(kWJTagUndefined)) ||
      !e.writeOp(Op::I32Eq) || !e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpTag) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(kWJTagNull)) ||
      !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32Or)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) return false;
  if (!e.writeOp(Op::Else)) return false;
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return)) return false;  // deopt
  if (!e.writeOp(Op::End)) return false;  // end undef/null
  if (!e.writeOp(Op::End)) return false;  // end int32/bool
  if (!e.writeOp(Op::End)) return false;  // end object
  return e.writeOp(Op::End);              // end double
}

// isNullish(value in srcLocal) -> i32 (1 if null or undefined). tmpTag scratch.
static bool WJVIsNullish(Encoder& e, uint32_t srcLocal, uint32_t tmpTag) {
  return e.writeOp(Op::LocalGet) && e.writeVarU32(srcLocal) &&
         e.writeOp(Op::I64Const) && e.writeVarU64(32) && e.writeOp(Op::I64ShrU) &&
         e.writeOp(Op::I32WrapI64) && e.writeOp(Op::LocalTee) &&
         e.writeVarU32(tmpTag) && e.writeOp(Op::I32Const) &&
         e.writeVarS32(int32_t(kWJTagUndefined)) && e.writeOp(Op::I32Eq) &&
         e.writeOp(Op::LocalGet) && e.writeVarU32(tmpTag) && e.writeOp(Op::I32Const) &&
         e.writeVarS32(int32_t(kWJTagNull)) && e.writeOp(Op::I32Eq) &&
         e.writeOp(Op::I32Or);
}

// Forward dataflow: the operand-stack DEPTH at each basic-block entry. Supports
// And/Or/Coalesce (which leave a value across the branch -> a depth-1 boundary).
// Bails (false) on inconsistent join depths or depth > 1 (one phi local handles
// depth 1; deeper is unsupported). Replaces the depth-must-be-0 WJStackSafe for
// the Mode V control-flow emitter.
static bool WJComputeEntryDepth(JSScript* script, const std::vector<bool>& isStart,
                                uint32_t len, std::vector<int>& entryDepth) {
  jsbytecode* const start = script->code();
  entryDepth.assign(len + 1, -1);
  std::vector<uint32_t> work;
  entryDepth[0] = 0;
  work.push_back(0);
  auto push = [&](uint32_t off, int d) -> bool {
    if (d < 0 || d > 1) return false;
    if (entryDepth[off] == -1) {
      entryDepth[off] = d;
      work.push_back(off);
    } else if (entryDepth[off] != d) {
      return false;  // inconsistent join depth
    }
    return true;
  };
  while (!work.empty()) {
    uint32_t b = work.back();
    work.pop_back();
    int d = entryDepth[b];
    for (jsbytecode* pc = start + b; pc < start + len;) {
      JSOp op = JSOp(*pc);
      uint32_t ol = GetBytecodeLength(pc);
      uint32_t cur = uint32_t(pc - start);
      int uses = int(js::StackUses(op, pc));
      if (d < uses) return false;  // underflow
      d -= uses;
      d += int(js::StackDefs(op));
      if (op == JSOp::Return || op == JSOp::RetRval) break;  // no successor
      if (IsJumpOpcode(op)) {
        // JumpIfFalse/JumpIfTrue CONSUME the condition; require depth 0 after, so
        // no value lurks beneath it (a `br` would silently discard it -> a stale
        // tmpPhi). And/Or/Coalesce PRESERVE the value (depth may be 1 = the phi).
        if ((op == JSOp::JumpIfFalse || op == JSOp::JumpIfTrue) && d != 0) {
          return false;
        }
        uint32_t fall = cur + ol;
        int64_t tgt = int64_t(cur) + GET_JUMP_OFFSET(pc);
        if (tgt < 0 || uint32_t(tgt) > len) return false;
        if (!push(uint32_t(tgt), d)) return false;
        if (op != JSOp::Goto) {  // conditional: fall-through too
          if (fall > len || !push(fall, d)) return false;
        }
        break;
      }
      uint32_t nextOff = cur + ol;
      if (nextOff <= len && isStart[nextOff]) {  // fall into the next block
        if (!push(nextOff, d)) return false;
        break;
      }
      pc += ol;
    }
  }
  return true;
}

// Value-typed body emitter WITH control flow: same relooper-style dispatcher as
// WJEmitBodyCF, but the operand stack carries i64 Values and supports GetProp.
// This is where a property/arithmetic LOOP runs entirely in one wasm call, so
// the guest<->host boundary is amortized over the whole loop. And/Or/Coalesce
// leave a value across the branch -> a single i64 phi local (tmpPhi) carries it:
// a block entered at depth 1 reloads tmpPhi; an edge into such a block spills it.
static bool WJEmitBodyVCF(JSScript* script, Encoder& e, uint32_t nargs,
                          uint32_t nfixed) {
  jsbytecode* const start = script->code();
  jsbytecode* const end = script->codeEnd();
  const uint32_t len = uint32_t(end - start);
  const uint32_t argBase = 1;  // local 0 = f64 scratch ptr
  const uint32_t rvalLocal = argBase + nargs + nfixed;
  const uint32_t tmpA = rvalLocal + 1;     // i64
  const uint32_t tmpB = rvalLocal + 2;     // i64
  const uint32_t tmpC = rvalLocal + 3;     // i64 (3rd operand: SetElem value)
  const uint32_t tmpF = rvalLocal + 4;     // f64
  const uint32_t tmpAddr = rvalLocal + 5;  // i32
  const uint32_t pcLocal = rvalLocal + 6;  // i32
  const uint32_t tmpIdx = rvalLocal + 7;   // i32 (element index)
  const uint32_t tmpPhi = rvalLocal + 8;   // i64 (And/Or/Coalesce boundary value)
  const uint8_t kF64 = uint8_t(TypeCode::F64);
  const uint8_t kI64 = uint8_t(TypeCode::I64);
  const uint8_t kI32 = uint8_t(TypeCode::I32);
  const uint8_t kVoid = 0x40;

  // Pass 1: basic-block starts. And/Or/Coalesce are conditional jumps (they leave
  // a value across the branch -- handled via tmpPhi + the entry-depth analysis).
  std::vector<bool> isStart(len + 1, false);
  isStart[0] = true;
  for (jsbytecode* pc = start; pc < end;) {
    JSOp op = JSOp(*pc);
    uint32_t ol = GetBytecodeLength(pc);
    uint32_t cur = uint32_t(pc - start);
    if (IsJumpOpcode(op)) {
      int64_t tgt = int64_t(cur) + GET_JUMP_OFFSET(pc);
      if (tgt < 0 || tgt > len) return false;
      isStart[tgt] = true;
      if (cur + ol <= len) isStart[cur + ol] = true;
    } else if (op == JSOp::Return || op == JSOp::RetRval) {
      if (cur + ol <= len) isStart[cur + ol] = true;
    }
    pc += ol;
  }
  std::vector<int32_t> ofId(len + 1, -1);
  std::vector<uint32_t> blockOff;
  for (uint32_t o = 0; o <= len; o++) {
    if (isStart[o]) {
      ofId[o] = int32_t(blockOff.size());
      blockOff.push_back(o);
    }
  }
  uint32_t K = uint32_t(blockOff.size());
  if (K == 0 || K > 1024) return false;
  std::vector<int> entryDepth;
  if (!WJComputeEntryDepth(script, isStart, len, entryDepth)) return false;

  // Locals: (nargs + nfixed + rval + tmpA + tmpB + tmpC) i64, 1 f64 tmpF,
  // 3 i32 (tmpAddr, pc, tmpIdx), 1 i64 tmpPhi.
  if (!e.writeVarU32(4)) return false;
  if (!e.writeVarU32(nargs + nfixed + 4) || !e.writeFixedU8(kI64)) return false;
  if (!e.writeVarU32(1) || !e.writeFixedU8(kF64)) return false;
  if (!e.writeVarU32(3) || !e.writeFixedU8(kI32)) return false;
  if (!e.writeVarU32(1) || !e.writeFixedU8(kI64)) return false;

  // Prologue: copy each arg's raw Value bits (objects pass through unguarded).
  for (uint32_t i = 0; i < nargs; i++) {
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(0)) return false;
    if (!e.writeOp(MiscOp::I32TruncSatF64U)) return false;
    if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(i * 8)) {
      return false;
    }
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(argBase + i)) return false;
  }
  // Initialize JS `var` locals (+ rval) to undefined (the correct JS value for an
  // unread local). Reading one in arithmetic deopts (undefined is not a number),
  // which is correct; the wasm default of 0 would be a wrong number.
  for (uint32_t j = 0; j <= nfixed; j++) {  // nfixed locals + rval
    if (!e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(0xFFFFFF83ULL << 32))) {
      return false;
    }
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(argBase + nargs + j)) {
      return false;
    }
  }

  if (!e.writeOp(Op::Block) || !e.writeFixedU8(kVoid)) return false;
  if (!e.writeOp(Op::Loop) || !e.writeFixedU8(kVoid)) return false;

  // spill the operand-stack top to tmpPhi before branching to a depth-1 block.
  auto spillIfPhi = [&](uint32_t targetOff) -> bool {
    if (targetOff <= len && entryDepth[targetOff] == 1) {
      return e.writeOp(Op::LocalSet) && e.writeVarU32(tmpPhi);
    }
    return true;
  };
  for (uint32_t i = 0; i < K; i++) {
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(pcLocal)) return false;
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(i))) return false;
    if (!e.writeOp(Op::I32Eq)) return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
    // A block entered at depth 1 (And/Or/Coalesce boundary) reloads the value.
    if (entryDepth[blockOff[i]] == 1) {
      if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(tmpPhi)) return false;
    }

    jsbytecode* pc = start + blockOff[i];
    bool terminated = false;
    bool prevCmp = false;
    while (pc < end && !terminated) {
      JSOp op = JSOp(*pc);
      uint32_t ol = GetBytecodeLength(pc);
      uint32_t cur = uint32_t(pc - start);

      if (IsJumpOpcode(op)) {
        uint32_t fall = cur + ol;
        uint32_t tgtOff = uint32_t(int64_t(cur) + GET_JUMP_OFFSET(pc));
        int32_t tgtId = ofId[tgtOff];
        int32_t fallId = (fall <= len) ? ofId[fall] : -1;
        if (op == JSOp::Goto) {
          if (tgtId < 0) return false;
          if (!spillIfPhi(tgtOff)) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(tgtId)) return false;
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(pcLocal)) return false;
          if (!e.writeOp(Op::Br) || !e.writeVarU32(1)) return false;
        } else if (op == JSOp::And || op == JSOp::Or || op == JSOp::Coalesce) {
          // value v on stack -> tmpPhi (both successors reload it); branch on the
          // truthiness (And/Or) or nullishness (Coalesce) of v.
          if (tgtId < 0 || fallId < 0) return false;
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpPhi)) return false;
          if (op == JSOp::Coalesce) {
            if (!WJVIsNullish(e, tmpPhi, tmpIdx)) return false;
          } else if (!WJVToBool(e, tmpPhi, tmpF, tmpIdx)) {
            return false;
          }
          // if (cond) -> A else -> B. And: jump-to-L when falsy; Or: when truthy;
          // Coalesce(cond=nullish): jump-to-L when NOT nullish.
          int32_t aId = (op == JSOp::Or) ? tgtId : fallId;
          int32_t bId = (op == JSOp::Or) ? fallId : tgtId;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(aId)) return false;
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(pcLocal)) return false;
          if (!e.writeOp(Op::Else)) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(bId)) return false;
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(pcLocal)) return false;
          if (!e.writeOp(Op::End)) return false;
          if (!e.writeOp(Op::Br) || !e.writeVarU32(1)) return false;
        } else {  // JumpIfFalse / JumpIfTrue
          if (tgtId < 0 || fallId < 0) return false;
          if (!prevCmp) {  // condition is a Value (not a cmp result): ToBoolean
            if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(tmpA)) return false;
            if (!WJVToBool(e, tmpA, tmpF, tmpIdx)) return false;
          }
          int32_t thenId = (op == JSOp::JumpIfTrue) ? tgtId : fallId;
          int32_t elseId = (op == JSOp::JumpIfTrue) ? fallId : tgtId;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(thenId)) return false;
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(pcLocal)) return false;
          if (!e.writeOp(Op::Else)) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(elseId)) return false;
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(pcLocal)) return false;
          if (!e.writeOp(Op::End)) return false;
          if (!e.writeOp(Op::Br) || !e.writeVarU32(1)) return false;
        }
        terminated = true;
      } else if (op == JSOp::Return) {
        if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(rvalLocal)) return false;
        if (!e.writeOp(Op::Br) || !e.writeVarU32(2)) return false;
        terminated = true;
      } else if (op == JSOp::RetRval) {
        if (!e.writeOp(Op::Br) || !e.writeVarU32(2)) return false;
        terminated = true;
      } else if (WJIsCmp(op)) {
        jsbytecode* nx = pc + ol;
        if (nx >= end ||
            (JSOp(*nx) != JSOp::JumpIfFalse && JSOp(*nx) != JSOp::JumpIfTrue)) {
          return false;
        }
        if (!WJVCmp(e, op, tmpA, tmpB)) return false;
        prevCmp = true;
        pc += ol;
        continue;
      } else {
        if (!WJEmitOpV(e, pc, script, start, nargs, argBase, rvalLocal, tmpA,
                       tmpB, tmpC, tmpF, tmpAddr, tmpIdx)) {
          return false;
        }
        uint32_t nextOff = cur + ol;
        if (nextOff <= len && isStart[nextOff] && nextOff != blockOff[i]) {
          int32_t nid = ofId[nextOff];
          if (nid < 0) return false;
          if (!spillIfPhi(nextOff)) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(nid)) return false;
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(pcLocal)) return false;
          if (!e.writeOp(Op::Br) || !e.writeVarU32(1)) return false;
          terminated = true;
        }
      }
      prevCmp = false;
      pc += ol;
    }
    if (!terminated) {
      if (!e.writeOp(Op::Br) || !e.writeVarU32(2)) return false;
    }
    if (!e.writeOp(Op::End)) return false;  // end if (pc == i)
  }

  if (!e.writeOp(Op::End)) return false;  // end loop
  if (!e.writeOp(Op::End)) return false;  // end block $exit
  // Epilogue: store rval (a Value) to gWJScratch[result], return deopt=0.
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(0)) return false;
  if (!e.writeOp(MiscOp::I32TruncSatF64U)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(rvalLocal)) return false;
  if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) ||
      !e.writeVarU32(kWJResultOff)) {
    return false;
  }
  if (!WJConst(e, 0.0)) return false;
  return e.writeOp(Op::End);
}

// ===================== Mode VS (no-restart, GC-traced frame) =================
// For MUTATING functions. The JS operand stack + args + locals + rval live in the
// GC-traced gWJFrameMem (slot i of this frame at byte fb + i*8). A guard miss
// calls wjhelp (import function 0), which completes the op in place so the
// function NEVER restarts -> heap writes happen exactly once. Operand-stack depth
// is statically tracked (WJStackSafe guarantees an empty stack at block bounds).
static constexpr uint32_t kWJVSHelpIdx = 0;   // wjhelp import function index
static constexpr uint32_t kWJVSMaxStack = 48;
// frame byte base local + scratch locals (see WJEmitBodyVS local decls):
//   0=f64 param; 1,2=i64 t0,t1; 3=f64 tf; 4,5,6,7,8=i32 fb,pc,basesp,ti,ti2;
//   9..9+kWJVSMaxStack = i64 operand-stack registers s[0..kWJVSMaxStack).
static constexpr uint32_t kVSt0 = 1, kVSt1 = 2, kVStf = 3;
static constexpr uint32_t kVSfb = 4, kVSpc = 5, kVSbasesp = 6, kVSti = 7,
                          kVSti2 = 8, kVSpc2 = 9;  // kVSpc2: inlined-callee dispatch
static constexpr uint32_t kVSsBase = 10;  // i64 local for operand-stack depth 0
// UNBOX: a parallel block of f64 operand-stack locals. When an operand-stack
// entry's repr is F64 (GECKO_WJVS_UNBOX), its UNBOXED double lives in
// kVSsBaseF+depth instead of the NaN-boxed i64 in kVSsBase+depth. Numeric ops
// then flow f64->f64 with no per-op box/unbox/type-guard (Mode V speed) inside
// the slow no-restart Mode VS. Box/unbox happens only at boundaries (calls,
// property stores, branches) via materialize().
static constexpr uint32_t kVSsBaseF = kVSsBase + kWJVSMaxStack;  // f64 stack locals
// UNBOX typed locals: a numeric-only arg/local slot (proven by WJAnalyzeNumericSlots)
// lives UNBOXED as an f64 in lf[slot]=kVSsBaseLF+slot, never boxed in the frame
// (a number needs no GC tracing). GetLocal/GetArg of such a slot pushes F64;
// SetLocal/SetArg stores f64 (coerced). Frame slot index: arg k -> k; local k ->
// nargs+k. Capped at kWJVSMaxTLocals slots (bit s of the numericMask).
static constexpr uint32_t kWJVSMaxTLocals = 48;
static constexpr uint32_t kVSsBaseLF = kVSsBaseF + kWJVSMaxStack;  // f64 typed-local regs
static inline uint32_t WJFLoc(uint32_t d) { return kVSsBaseF + d; }  // f64 stack local for depth d
// SHORT-CIRCUIT: a single i64 local holding the boxed phi value carried across an
// And/Or/Coalesce branch (the value the merge block reloads). Declared last so the
// operand-stack/typed-local register indices above are unchanged.
static constexpr uint32_t kVSphi = kVSsBaseLF + kWJVSMaxTLocals;
// PHASE 2b (GECKO_WJVS_CSE): one i64 local caching the most-recent GetProp result for
// within-block load-CSE (a repeated `this.field` read reuses it instead of re-emitting
// the shape-guard + slot-load). Declared after kVSphi so all other indices are unchanged.
static constexpr uint32_t kVScse = kVSphi + 1;
// PTRUNBOX (GECKO_WJVS_PTRUNBOX): a parallel block of i32 operand-stack locals holding
// UNBOXED object pointers. When an operand entry's repr is PTR (==2), its raw JSObject*
// (a wasm32 i32 offset) lives in kVSsBaseP+depth instead of the NaN-boxed i64 in
// kVSsBase+depth. A GetProp/SetProp/Call receiver that is already a known object pointer
// then skips the per-access load-from-frame + isObject tag-check + i32.wrap. Boxed back
// only at materialize (branches/calls/stores of the value itself). Declared LAST so every
// index above is unchanged (Phase A parity preserved when the array is unused).
static constexpr uint32_t kVSsBaseP = kVScse + 1;  // i32 obj-ptr stack regs [0..kWJVSMaxStack)
static inline uint32_t WJPLoc(uint32_t d) { return kVSsBaseP + d; }  // ptr stack local for depth d
// PTRUNBOX guard-hoisting: one i32 local caching `this`'s shape pointer, loaded once per straight-line
// region and reused by every this.field shape guard (skips the repeated i32.load[this+0] memory read).
// Invalidated at block boundaries and after any Call (a callee could add a property -> shape change).
static constexpr uint32_t kVSthisShape = kVSsBaseP + kWJVSMaxStack;
// FIELDPROMO (GECKO_WJVS_FIELDPROMO): scalar replacement of object fields. A block of i64 locals
// caches GetProp results keyed by (receiver-source-slot, field); repeated reads reuse the local
// (skipping shape-guard + slot-load), writes forward into it. The cache persists across blocks
// within a region and is invalidated at any GC-safepoint / possibly-aliasing store. The win lands
// when the hot tree is inlined call-free, so an object's fields stay register-resident across the
// whole iteration -- making the boxed object model irrelevant inside the loop.
static constexpr uint32_t kWJFieldPromoN = 8;
static constexpr uint32_t kVSfcBase = kVSthisShape + 1;  // kWJFieldPromoN i64 cache value locals (boxed)
static constexpr uint32_t kVSfcBaseF = kVSfcBase + kWJFieldPromoN;  // kWJFieldPromoN f64 cache locals (numeric)
static constexpr int32_t kWJThisRecvSentinel = 1 << 20;  // field-cache receiver key for `this` (non-inlined)
static constexpr int32_t kWJFieldResultKeyBase = 2 << 20;  // receiver key for a cached GetProp RESULT (+slot)
// INTUNBOX (GECKO_WJVS_INTUNBOX): a parallel block of i32 operand-stack locals holding UNBOXED int32
// values. repr==3 -> the raw i32 lives in kVSsBaseI32+depth; int ops (bitand/or/xor/shift/cmp/inc/dec)
// flow i32->i32 with NO NaN-box tag manipulation -- the lean per-op codegen that lets the inlined
// monolith do less work than the interpreter on richards' integer-flag dispatch. Box only at boundaries.
static constexpr uint32_t kVSsBaseI32 = kVSfcBaseF + kWJFieldPromoN;  // kWJVSMaxStack i32 unboxed-int regs
static inline uint32_t WJILoc(uint32_t d) { return kVSsBaseI32 + d; }  // i32 stack local for depth d
static bool WJIntUnbox() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_INTUNBOX") ? 1 : 0;
  return v != 0;
}
// NULLCMP: inline Eq/Ne/StrictEq/StrictNe for object/null/undefined operands (no helper).
// Default ON (GECKO_WJVS_NONULLCMP=1 disables). Sound: only fires when both operands are
// object/null/undefined; everything else (numbers handled separately, strings/mixed) helpers.
static bool WJNullCmp() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_NONULLCMP") ? 0 : 1;
  return v != 0;
}
static bool WJFieldPromo() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_FIELDPROMO") ? 1 : 0;
  return v != 0;
}
static bool WJPtrUnbox() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_PTRUNBOX") ? 1 : 0;
  return v != 0;
}

// MODE_VS_REGALLOC Phase 1: the operand stack lives in wasm LOCALS (registers)
// between safepoints, GC-spilled to the traced frame at the only allocation sites
// (wjhelp + call_indirect). Args/locals/rval stay in the frame. GECKO_WJVS_FRAME=1
// reverts to the all-frame-memory operand stack (A/B + rollback).
static bool WJVSUseLocals() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_FRAME") ? 0 : 1;
  return v != 0;
}
// Call-heavy chronically-deopting functions recompile to Mode VS so the hot OO
// call chain stays in wasm (call_indirect between VS fns) instead of re-entering
// the interpreter every call -- the main octane OO win (DeltaBlue ~1.35x). Enabled
// by default now that the TraceJitFrames re-entry crash is fixed (the PBL push-order
// fix in PortableBaselineInterpret.cpp + minor-GC frame tracing). GECKO_WJVS_NOHASCALL=1
// reverts to call-free-only recompile.
static bool WJVSHasCallRecompile() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_NOHASCALL") ? 0 : 1;
  return v != 0;
}

struct WJVSCtx {
  JSScript* script;
  jsbytecode* start;
  uint32_t nargs, nfixed;
  uint32_t localBaseS, rvalS, stackBaseS;
  uint32_t depth;
  // METHOD_INLINING: when emitting an inlined leaf callee's body, GetArg/FunctionThis
  // read the CALLER's marshalled arg/this operand slots (no callee frame), and the
  // callee's operand stack continues in the caller's s[] locals.
  bool inlined = false;
  uint8_t inlineDepth = 0;     // 0 = top-level; N = inlined N levels deep (recursion cap)
  uint32_t inlineArgBase = 0;  // caller absolute slot of inlined callee's arg 0
  uint32_t inlineThis = 0;     // caller absolute slot of inlined callee's `this`
  // UNBOX: per-operand-stack-depth representation. repr[d]==1 -> the entry at
  // depth d is an UNBOXED f64 in local kVSsBaseF+d; ==0 -> the NaN-boxed i64 in
  // kVSsBase+d (or frame). Maintained only when `unbox` is set; empty at block
  // boundaries (WJStackSafe guarantees depth==0 there).
  bool unbox = false;
  uint8_t repr[kWJVSMaxStack] = {0};
  // UNBOX typed locals: bit s set => frame slot s (arg s, or local s-nargs) is a
  // numeric-only slot kept unboxed in lf[s]=kVSsBaseLF+s. 0 when typed locals off.
  uint64_t numMask = 0;
  // PHASE 2b CSE: a single-entry within-block GetProp cache. cseLastSlot = the frame
  // slot the operand on top was loaded from IFF the immediately-preceding op was
  // GetArg/GetLocal (else -1) -- so a GetProp receiver's source slot is known only for
  // the `GetLocal x; GetProp f` (i.e. `x.f`) pattern. cseValid => kVScse holds the value
  // of (cseRecvSlot).cseField, loaded earlier this block with no intervening GC/mutation.
  bool useCSE = false;
  bool thisShapeCached = false;  // PTRUNBOX: kVSthisShape holds `this`'s shape ptr, valid this region
  int32_t cseLastSlot = -1;
  bool cseValid = false;
  int32_t cseRecvSlot = -1;
  uint32_t cseField = 0;
  // FIELDPROMO: N-entry field cache. fcRecv[i] = receiver source frame slot (-1 = unused);
  // fcField[i] = field name ptr; the cached boxed value lives in local kVSfcBase+i.
  bool useFieldPromo = false;
  int32_t fcRecv[kWJFieldPromoN] = {-1, -1, -1, -1, -1, -1, -1, -1};
  uint32_t fcField[kWJFieldPromoN] = {0};
  uint8_t fcRepr[kWJFieldPromoN] = {0};  // 0 = boxed i64 in kVSfcBase+i; 1 = f64 in kVSfcBaseF+i
  // PHASE B GVN: base frame slot of the kWJGvnSlots GC-traced load-cache slots (set in
  // WJEmitBodyVS). Slot s of the cache lives at frame index gvnBase + s.
  uint32_t gvnBase = 0;
};
// Is frame slot `fs` a typed (unboxed-f64) numeric local/arg?
static inline bool WJSlotTyped(const WJVSCtx& c, uint32_t fs) {
  return c.unbox && fs < 64 && fs < kWJVSMaxTLocals && (c.numMask & (uint64_t(1) << fs));
}

// GECKO_WJVS_INLINE=1 enables speculative leaf-method inlining (bring-up gate; default
// off so the verified non-inlining build is the default).
static bool WJVSInline() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_INLINE") ? 1 : 0;
  return v != 0;
}
// GECKO_WJVS_LEAFONLY=1 restricts method inlining to LEAF callees (no calls), the
// original behavior. Default: NON-LEAF inlining is allowed -- a callee may contain
// calls; the inner call emits as a normal (non-inlined) call_indirect/helper. This
// collapses one call frame per inline (e.g. richards schedule->TCB.run, the inner
// task.run call remaining). Requires GECKO_WJVS_INLINE. A/B isolation.
static bool WJLeafOnly() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_LEAFONLY") ? 1 : 0;
  return v != 0;
}
// GECKO_WJVS_NOPOLYINLINE=1 restricts inlining to the FIRST recorded callee
// (monomorphic). Default: polymorphic inlining (up to 4 callees as a guarded chain),
// for megamorphic sites like richards `this.task.run`. Requires GECKO_WJVS_INLINE.
static bool WJNoPolyInline() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_NOPOLYINLINE") ? 1 : 0;
  return v != 0;
}
// GECKO_WJVS_INLINEDEPTH=N: max inlining depth (default 2). richards' hot megamorphic
// `this.task.run` is reached one level below the monomorphic `currentTcb.run()`, so it
// needs depth 2 to inline. Capped to avoid code-size blowup / register-budget overflow.
static uint32_t WJMaxInlineDepth() {
  static int v = -1;
  if (v < 0) { const char* s = getenv("GECKO_WJVS_INLINEDEPTH"); v = s ? atoi(s) : 2; }
  return uint32_t(v);
}
// GECKO_WJVS_SHORTCIRCUIT=1 enables Mode VS compilation of && / || / ?? (short-circuit)
// and value-producing comparisons (a cmp whose result is NOT immediately branched on,
// e.g. `return a != 0 || b`). Without it those ops bail the function to the interpreter
// (e.g. richards TaskControlBlock.isHeldOrSuspended, called every scheduler iteration).
//
// RICHARDS-2X FINDING (2026-06-18): enabling this DOES make the hot outer loop
// Scheduler.schedule (richards.js:188) compile -- it has a value-condition
// `if(this.currentTcb.isHeldOrSuspended())` and without short-circuit it EMIT-FAILs and
// is permanently marked Failed (correcting the rewrite plan's "Blocker A: never submitted"
// premise -- it WAS submitted, it just failed to emit). HOWEVER, compiling schedule
// REGRESSES richards (jit/off 0.91x -> 0.84x): Mode VS's NaN-boxed frame + per-op guards
// are slower than the PBL interpreter for dispatch-dense boxed OO (Blocker B). So this is
// kept DEFAULT OFF (non-regressive); enable for the rewrite via GECKO_WJVS_SHORTCIRCUIT=1.
static bool WJShortCircuit() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_SHORTCIRCUIT") ? 1 : 0;
  return v != 0;
}
// GECKO_WJVS_NOPOLYCALL=1 reverts call sites to a single monomorphic way (the
// pre-poly behavior: way 0 rewritten on every miss). Used to A/B the poly call
// IC within one binary.
static bool WJNoPolyCall() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_NOPOLYCALL") ? 1 : 0;
  return v != 0;
}
// GECKO_WJVS_UNBOX=1 enables the unboxed f64 operand stack in Mode VS (numeric
// kernels run at ~Mode V speed inside the no-restart mutating emitter). Bring-up
// gate; default off so the verified boxed Mode VS stays the default.
static bool WJUnbox() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_NOUNBOX") ? 0 : 1;  // default ON (correct as of 2026-06-18)
  return v != 0;
}
// GECKO_WJVS_UNBOXMASK: bisection knob for the typed-op categories in the unbox
// fast path. Default -1 (all on). bit0=literals bit1=sub/mul/div bit2=bitop
// bit3=inc/dec bit4=add bit5=pop. A clear bit makes that category fall through to
// the boxed path. Used to isolate the unbox correctness bug.
static int WJUnboxMask() {
  static int v = -2;
  if (v == -2) { const char* s = getenv("GECKO_WJVS_UNBOXMASK"); v = s ? atoi(s) : -1; }
  return v;
}
// GECKO_WJVS_OBJSET=1 enables inline OBJECT-valued SetProp stores (own fixed-slot data
// property: shape-guarded raw slot store + lean GC post-write barrier helper) instead of
// the generic js::SetProperty helper. Default OFF (experimental): it is CORRECT (validated
// on richards/splay/earley/deltablue/crypto -- no NaN/crash) but measured NEUTRAL on those
// benches (their hot SetProp sites are polymorphic and miss this monomorphic IC, and the
// object-store-heavy ones -- richards/splay -- are bottlenecked elsewhere: the wasm<->C++
// call boundary per dispatch, megamorphic method dispatch, and Packet allocation). Kept
// gated so it carries no generational-GC-barrier risk in the default path until a poly
// SetProp IC makes it actually fire on real OO code. See [[wasm-jit-richards-analysis]].
static bool WJNoObjSet() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_OBJSET") ? 0 : 1;  // default OFF (experimental)
  return v != 0;
}
// GECKO_WJVS_NOALIASED=1 disables Mode VS GetAliasedVar support (closed-over var
// reads fall back to leaving the function uncompiled). Default ON. A/B isolation.
static bool WJNoAliased() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_NOALIASED") ? 1 : 0;
  return v != 0;
}
// GECKO_WJVS_NOPOLYPROP=1 reverts Mode VS GetProp to a single monomorphic way
// (the pre-poly behavior). Used to A/B the poly GetProp IC within one binary.
static bool WJNoPolyProp() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_NOPOLYPROP") ? 1 : 0;
  return v != 0;
}
// GECKO_WJVS_NOLEN=1 disables inline `.length` (string + dense array) in Mode VS
// GetProp -- length reads fall to the helper as before. A/B isolation gate.
static bool WJNoLen() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_NOLEN") ? 1 : 0;
  return v != 0;
}
// (script,pcOff)->up to 4 callee JSFunction* low32s, recorded by WJFillIC at runtime and
// read at RECOMPILE time to decide inlining (per-site IC indices change across recompiles,
// but (script,pcOff) is stable). MULTIPLE callees -> POLYMORPHIC inlining: a megamorphic
// site (e.g. richards `this.task.run`, 4 task types) emits a guarded chain of inline bodies.
// Math intrinsic record per (caller script, call pcOff): the observed Math.* op + the native fn
// low32 to guard against. Populated on first execution (observe), read at recompile emit-time
// (mirrors gWJInlineCallee's keying, since site indices are not stable across compiles).
struct WJMathRec { uint8_t op; uint32_t fnLow; };
static std::unordered_map<uint64_t, WJMathRec> gWJMathRec;
struct WJInlineRec {
  uint32_t fns[4] = {0, 0, 0, 0};
  uint8_t n = 0;
};
static std::unordered_map<uint64_t, WJInlineRec> gWJInlineCallee;
// LEAN EMISSION (specialized recompile): observed OWN-monomorphic shape + byte offset for a
// GetProp/SetProp site, keyed by (script,pcOff) so a recompile can BAKE them as constants --
// emit `i32/i64.load(this + <off>)` after ONE hoisted `this.shape==<shape> else deopt`, instead
// of the per-access IC-table load + shape compare + branch. Recorded at WJFillIC; read at the
// specialized recompile. Only own data props (holder==0), non-poly sites.
struct WJShapeRec { uint32_t shape; uint32_t off; uint8_t vty; };  // vty: 0=double 1=int32 2=other
static std::unordered_map<uint64_t, WJShapeRec> gWJShapeRec;
// Polymorphic method-dispatch record: per method-load GetProp site (script,pcOff),
// up to 4 (receiver shape -> method JSFunction low32) pairs observed at IC-fill.
// The Ion front-end builds a receiver-shape dispatch chain that inlines each task
// type's method body (richards' task.run is megamorphic over 4 task types).
struct WJMethodPolyRec {
  uint32_t shapes[4] = {0, 0, 0, 0};
  uint32_t fns[4] = {0, 0, 0, 0};
  uint8_t n = 0;
};
static std::unordered_map<uint64_t, WJMethodPolyRec> gWJMethodPoly;
// Polymorphic FIELD access: a GetProp/SetProp site whose receiver has several
// shapes (deltablue's constraint/Variable fields). The oracle records up to 4
// (shape, byte-offset, vty) ways; getPropField/SetProp emit a shape-guard chain
// that loads each shape's slot inline, so an off-shape access does NOT deopt-restart
// the whole (stateful) function. Keyed WJInlineKey(script, pcOff).
struct WJFieldPolyRec {
  uint32_t shapes[4] = {0, 0, 0, 0};
  uint32_t offs[4] = {0, 0, 0, 0};
  uint8_t vtys[4] = {0, 0, 0, 0};
  uint8_t n = 0;
};
static std::unordered_map<uint64_t, WJFieldPolyRec> gWJFieldPoly;
// (shape, byte-offset) -> observed field value type (vty 0/1/3), populated by any
// wj-compiled function's runtime IC fill. The Baseline-IC oracle (WJReadBaselineICs)
// has shape+offset from CacheIR but NOT the value type (typeData is unpopulated in
// the portable-baseline build), so it reuses a vty observed by another function
// that accessed the same shape's field. Keyed (shape<<32)|offset.
static std::unordered_map<uint64_t, uint8_t> gWJFieldVty;
// shape low32 -> a live object pointer (low32) of that shape, captured at IC-fill.
// Lets the Ion oracle read the value type of ANY field of an observed shape (even
// fields no function accessed) by walking a representative instance at compile
// time -> unboxed typed loads instead of the boxed/tag-dispatch path. Validated
// (shape word still matches) before use; cleared on GC (addresses can be reused).
static std::unordered_map<uint32_t, uint32_t> gWJShapeSample;
// Dense-array `.length` sites: WJInlineKey(script,pcOff) -> the array shape. The
// oracle records these when it sees CacheOp::LoadInt32ArrayLengthResult, so the FE
// loads the length from the ObjectElements header (elements_-4) behind that shape
// guard. Detecting array-length by the CacheIR op (NOT the "length" property name)
// is essential: a plain `length` data property / string length is a normal slot,
// and treating it as an array header read OOB (regressed raytrace).
static std::unordered_map<uint64_t, uint32_t> gWJLenSite;
// Dense-array element site (GetElem/SetElem): the observed array shape, so the
// Ion FE can shape-guard then load/store elements_[index] directly. Keyed
// (script,pcOff).
static std::unordered_map<uint64_t, uint32_t> gWJElemShape;
// Program-wide own-property record keyed by property NAME (atom low32): the
// (shape, offset, vty) seen for that name, or ambiguous if seen on >1 shape/off.
// Lets a COLD GetProp site (fallback-only IC, e.g. a rarely-taken branch) resolve
// a monomorphic-by-name field via a shape guard, even with no per-site IC data.
struct WJPropRec { uint32_t shape; uint32_t off; uint8_t vty; bool ambig; };
static std::unordered_map<uint32_t, WJPropRec> gWJPropByName;
// Ion-rewrite property-access gate: a sample live receiver objptr (i32) per
// (script,pcOff), captured at IC-fill so the self-contained Ion front-end test
// can pass a real object to its compiled wasm. Test-only; not used in production.
static std::unordered_map<uint64_t, uint32_t> gWJSampleRecv;
static uint64_t gWJInlinedCalls = 0;  // call sites emitted as inline bodies (diagnostic)
static bool gWJEmitInline = false;  // this compile may inline (set in WJCompile)
static bool gWJEmitBake = false;  // LEAN EMISSION: this (specialized) compile bakes shape/off constants
static bool WJBake() {  // GECKO_WJVS_BAKE: enable baked direct-field emission at warm recompile
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_BAKE") ? 1 : 0;
  return v != 0;
}
static uint64_t gWJCseHits = 0;  // PHASE 2b: GetProp loads served from the CSE cache
// PHASE 2b: GECKO_WJVS_CSE=1 enables within-block GetProp load-CSE in Mode VS. Default
// OFF (bring-up gate; correctness-critical -- a stale reuse is a silent miscompile).
static bool WJVSCSE() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_CSE") ? 1 : 0;
  return v != 0;
}
// PHASE A (boxed-oo-middleend-plan.md): GECKO_WJVS_IR=1 routes each straight-line region
// of a Mode VS body through an SSA IR (build a value graph, then lower it back to wasm).
// Phase A's lowerer DELEGATES per node to the same WJEmitOpVS the per-op path calls, so the
// emitted bytes are identical -> parity by construction. The value graph (def/use + a type
// lattice) is the substrate the optimizer phases (B GVN/guard-elim, C LICM, D scalar-repl)
// will consume; Phase A performs no optimization. Default OFF (bring-up gate).
static bool WJVSIR() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_IR") ? 1 : 0;
  return v != 0;
}
// PHASE B (boxed-oo-middleend-plan.md): GECKO_WJVS_GVN=1 enables redundant-load elimination
// over the IR -- a GetProp whose (receiver SSA value, field) was already loaded earlier in
// the block with no intervening clobber reuses the cached result instead of re-emitting the
// shape-guard chain + slot-load + helper fallback. Sound WITHOUT deopt: it never removes a
// guard that could fire differently; it only reuses a value the conservative clobber model
// proves unchanged. The cached value lives in a GC-TRACED frame slot, so a moving GC during
// a later op updates it in place (no dangling pointer). Implies IR routing. Boxed path only
// (requires GECKO_WJVS_NOUNBOX for now), non-inlined, top-level. Default OFF (bring-up gate).
static bool WJVSGvn() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_GVN") ? 1 : 0;
  return v != 0;
}
// INLINEALLOC (plan §8.3): make NewObject/InitProp Mode-VS-eligible so constructors / object
// literals can JIT. First increment routes them to correct helpers (WJH_NEWOBJECT/WJH_INITPROP);
// later increments add the barrier-flag-gated inline add-property IC (§8.3b) + Construct hook.
// Default OFF (bring-up; GC-critical path).
static bool WJVSInlineAlloc() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_INLINEALLOC") ? 1 : 0;
  return v != 0;
}
// Route a NON-mutating function that reads closed-over vars to Mode VS (the only mode that emits
// GetAliasedVar) instead of letting it fall to Mode V and EMIT-FAIL. DEFAULT OFF: measured net
// negative (navier +3% but richards -4% — JIT'ing a call-bound closure loses to the interpreter,
// and navier's real ceiling is the boxed regular-Array access, not the un-JIT'd solver). Opt in.
static bool WJAliasedVS() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_ALIASEDVS") ? 1 : 0;
  return v != 0;
}
// LEANINIT: replace the per-call O(frameSize) prologue store-loop that inits the operand-stack
// region to Undefined with a single memory.fill of 0x00 (= double +0.0, a valid non-GC Value).
// Semantics-preserving (whole frame stays valid + traced). Targets call-heavy benches where the
// per-call frame init is pure overhead. Default OFF (bring-up gate); intended to become default.
static bool WJLeanInit() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_LEANINIT") ? 1 : 0;
  return v != 0;
}
static uint64_t gWJIRRegions = 0;  // straight-line regions lowered through the IR
static uint64_t gWJIRNodes = 0;    // IR nodes built (diagnostic)
static uint64_t gWJGvnHits = 0;    // PHASE B: redundant GetProp loads served from the cache
static constexpr uint32_t kWJGvnSlots = 8;  // GC-traced frame slots reserved for load-CSE
static uint64_t gWJDeoptResumes = 0;  // PHASE F: mid-execution bails that resumed in the interp
// PHASE F (deopt/bailout) bring-up: GECKO_WJVS_FDEOPT=N forces a Mode VS body to bail to the
// interpreter at the TOP of block N (a block boundary -> empty operand stack), exercising the
// resume path (which re-enters PBL at that pc with the wasm's current locals). Default -1 (off).
// This is the test harness for Phase F before a real speculative opt (hoisted guard, promoted
// field) drives the bailout; the bail fires only when control actually reaches block N, so it
// is control-flow-correct. The resume is sound only for no-SetArg bodies (the interpreter's
// formal args stay the original values) and empty-stack boundaries -> both are enforced.
static int WJForceDeopt() {
  static int v = -2;
  if (v == -2) { const char* s = getenv("GECKO_WJVS_FDEOPT"); v = s ? atoi(s) : -1; }
  return v;
}
static uint64_t gWJTypedFieldHits = 0;  // PHASE 2a: GetProp results materialized to the typed f64 stack
// PHASE 2a: GECKO_WJVS_TYPEDFIELD=1 (requires UNBOX) enables type-specialized field reads:
// a GetProp whose result is immediately consumed by a numeric op is converted straight onto
// the typed f64 operand stack (repr=1) instead of leaving a boxed Value for the consumer to
// unbox. SOUND because the consumer would ToNumber it anyway (so the isNum?unbox:ToNumber done
// here is exact); the value never escapes between GetProp and the numeric op. Default OFF.
static bool WJVSTypedField() {
  // DEFAULT ON (opt out with GECKO_WJVS_TYPEDFIELD=0). Measured net win on the JS->wasm JIT
  // under host-V8 TurboFan: crypto +16%, splay +6%, others within noise. Only active on the
  // UNBOX path (typed f64 operand stack); inert when boxed.
  static int v = -1;
  if (v < 0) { const char* s = getenv("GECKO_WJVS_TYPEDFIELD"); v = (s && s[0] == '0') ? 0 : 1; }
  return v != 0;
}
// PHASE 2a: is `op` a numeric op that ToNumber-coerces the operand-stack top (so a value
// produced just before it is consumed purely numerically)?
static bool WJIsNumericConsumer(JSOp op) {
  switch (op) {
    case JSOp::Add: case JSOp::Sub: case JSOp::Mul: case JSOp::Div:
    case JSOp::Inc: case JSOp::Dec:
    case JSOp::BitOr: case JSOp::BitAnd: case JSOp::BitXor:
    case JSOp::Lsh: case JSOp::Rsh: case JSOp::Ursh: case JSOp::BitNot:
      return true;
    default:
      return false;
  }
}
// PHASE 2a/D: is `op` a numeric BINARY op (consumes top 2, coerces both)?
static bool WJIsNumBinop(JSOp op) {
  switch (op) {
    case JSOp::Add: case JSOp::Sub: case JSOp::Mul: case JSOp::Div: case JSOp::Mod:
    case JSOp::BitOr: case JSOp::BitAnd: case JSOp::BitXor:
    case JSOp::Lsh: case JSOp::Rsh: case JSOp::Ursh:
      return true;
    default: return false;
  }
}
static bool WJIsNumUnop(JSOp op) {
  switch (op) {
    case JSOp::Inc: case JSOp::Dec: case JSOp::BitNot: case JSOp::Neg: case JSOp::Pos:
      return true;
    default: return false;
  }
}
// Pure stack pushes that don't consume our tracked value (they pile a sibling operand on top).
static bool WJIsPurePush(JSOp op) {
  switch (op) {
    case JSOp::GetLocal: case JSOp::GetArg: case JSOp::Zero: case JSOp::One:
    case JSOp::Int8: case JSOp::Int32: case JSOp::Uint16: case JSOp::Uint24:
    case JSOp::Double: case JSOp::True: case JSOp::False:
      return true;
    default: return false;
  }
}
// PHASE D (field unboxing): does the value just produced by `pc` (a GetProp, currently on top
// of the operand stack) get consumed by a NUMERIC op before any non-numeric use? A bounded,
// conservative forward scan: tracks the value's distance from the stack top as sibling operands
// are pushed (GetLocal/GetProp/const) and binops collapse the stack, returning true the moment a
// numeric op consumes it. Generalizes the immediate-next-op check so `this.x * w.x` (where the
// sibling load sits between the GetProp and the *) types BOTH field reads onto the f64 stack.
static bool WJFieldNumConsumed(const WJVSCtx& c, jsbytecode* pc) {
  jsbytecode* end = c.script->codeEnd();
  jsbytecode* p = pc + GetBytecodeLength(pc);
  int posFromTop = 0;  // 0 = our value is on top
  for (int steps = 0; steps < 24 && p < end; steps++) {
    JSOp op = JSOp(*p);
    if (WJIsNumBinop(op)) {
      if (posFromTop <= 1) return true;  // our value is one of the two operands
      posFromTop -= 1;                   // net pop: our value rises one slot
    } else if (WJIsNumUnop(op)) {
      if (posFromTop == 0) return true;
      // net 0: posFromTop unchanged
    } else if (WJIsPurePush(op)) {
      posFromTop += 1;
    } else if (op == JSOp::GetProp) {
      if (posFromTop == 0) return false;  // our value used as a GetProp receiver (object use)
      // net 0: pops the receiver above us, pushes its field
    } else {
      return false;  // Call/SetProp/Pop/Dup/Swap/compare/branch/... -> conservative stop
    }
    p += GetBytecodeLength(p);
  }
  return false;
}
static inline uint64_t WJInlineKey(JSScript* s, uint32_t pcOff) {
  return (uint64_t(uint32_t(uintptr_t(s))) << 32) | pcOff;
}
// If `fun` is a recognized Math.* native (sqrt/floor/ceil/abs/trunc/min/max), return its WJMathOp.
// The Math method JSFunction pointers are looked up once from the active global (octane: single
// global) and cached; Math is a plain object with data-property methods, so the lookup is safe.
static int WJMathIntrinsic(JSFunction* fun) {
  static bool sInit = false;
  static uint32_t sFns[8] = {0};  // [WJMathOp] -> Math method fn low32
  if (!sInit) {
    JSContext* cx = js::TlsContext.get();
    if (!cx) return 0;
    JS::Rooted<JSObject*> g(cx, cx->global());
    JS::RootedValue mv(cx);
    if (g && JS_GetProperty(cx, g, "Math", &mv) && mv.isObject()) {
      JS::Rooted<JSObject*> mo(cx, &mv.toObject());
      const struct { const char* n; int op; } tab[] = {
          {"sqrt", WJM_SQRT}, {"floor", WJM_FLOOR}, {"ceil", WJM_CEIL}, {"abs", WJM_ABS},
          {"trunc", WJM_TRUNC}, {"min", WJM_MIN}, {"max", WJM_MAX}};
      for (const auto& t : tab) {
        JS::RootedValue fv(cx);
        if (JS_GetProperty(cx, mo, t.n, &fv) && fv.isObject() &&
            fv.toObject().is<JSFunction>()) {
          sFns[t.op] = uint32_t(uintptr_t(&fv.toObject()));
        }
      }
    }
    sInit = true;
  }
  uint32_t f = uint32_t(uintptr_t(fun));
  for (int op = WJM_SQRT; op <= WJM_MAX; op++) {
    if (sFns[op] && sFns[op] == f) return op;
  }
  return 0;
}
static bool WJMathInlineEnabled() {
  static int v = -1;
  if (v < 0) { const char* s = getenv("GECKO_WJVS_NOMATH"); v = (s && s[0] == '1') ? 0 : 1; }
  return v != 0;
}
// Record a callee for an inline site (dedup; cap 4). Builds the polymorphic callee set.
static void WJRecordInlineCallee(JSScript* s, uint32_t pcOff, uint32_t fnLow) {
  WJInlineRec& r = gWJInlineCallee[WJInlineKey(s, pcOff)];
  for (uint8_t i = 0; i < r.n; i++) {
    if (r.fns[i] == fnLow) return;
  }
  if (r.n < 4) r.fns[r.n++] = fnLow;
}
static bool WJCallInlinable(JSScript* cs, bool* hasCF = nullptr);  // near WJFillIC
static bool WJEmitOpVS(Encoder& e, jsbytecode* pc, WJVSCtx& c);  // for inline body
static bool WJEmitOpVSInner(Encoder& e, jsbytecode* pc, WJVSCtx& c);  // PHASE 2b: CSE wraps it
static bool WJMaterializeAll(Encoder& e, WJVSCtx& c);  // UNBOX: box live f64 stack
// PHASE 2b: an op is CSE-transparent if it neither mutates the heap, reassigns a local/
// arg, nor can trigger GC/allocation (so a cached object pointer in kVScse can't move or
// go stale across it). Pure stack/const/load ops only. Everything else clears the cache.
static bool WJCseTransparent(JSOp op) {
  switch (op) {
    case JSOp::GetArg: case JSOp::GetLocal: case JSOp::Pop: case JSOp::Dup:
    case JSOp::Zero: case JSOp::One: case JSOp::Int8: case JSOp::Int32:
    case JSOp::Uint16: case JSOp::Uint24: case JSOp::Double:
    case JSOp::Null: case JSOp::Undefined: case JSOp::True: case JSOp::False:
    case JSOp::String: case JSOp::Swap: case JSOp::JumpTarget: case JSOp::LoopHead:
    case JSOp::Nop: case JSOp::NopDestructuring: case JSOp::NopIsAssignOp:
      return true;
    default:
      return false;  // GetProp is handled specially (it is the producer)
  }
}

// Is `slot` an operand-stack slot (vs an arg/local/rval frame slot)?
static bool WJVSIsStack(const WJVSCtx& c, uint32_t slot) {
  return WJVSUseLocals() && slot >= c.stackBaseS;
}
static uint32_t WJVSLocalFor(const WJVSCtx& c, uint32_t slot) {
  return kVSsBase + (slot - c.stackBaseS);
}

static bool WJSAddr(Encoder& e, uint32_t slot) {  // push i32 = fb + slot*8
  return e.writeOp(Op::LocalGet) && e.writeVarU32(kVSfb) &&
         e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(slot) * 8) &&
         e.writeOp(Op::I32Add);
}
static bool WJSStoreEnd(Encoder& e) {  // stack [addr,i64] -> store
  return e.writeOp(Op::I64Store) && e.writeVarU32(3) && e.writeVarU32(0);
}
// Push i64 = the value at `slot`: an operand-stack local.get, or a frame load.
static bool WJSLoad(Encoder& e, const WJVSCtx& c, uint32_t slot) {
  if (WJVSIsStack(c, slot)) {
    return e.writeOp(Op::LocalGet) && e.writeVarU32(WJVSLocalFor(c, slot));
  }
  return WJSAddr(e, slot) && e.writeOp(Op::I64Load) && e.writeVarU32(3) &&
         e.writeVarU32(0);
}
// Store-to-`slot` framing: Pre emits the destination address (frame) or nothing
// (operand-stack local); the caller then emits the i64 value; Post commits it.
static bool WJSStorePre(Encoder& e, const WJVSCtx& c, uint32_t slot) {
  if (WJVSIsStack(c, slot)) return true;
  return WJSAddr(e, slot);
}
static bool WJSStorePost(Encoder& e, const WJVSCtx& c, uint32_t slot) {
  if (WJVSIsStack(c, slot)) {
    return e.writeOp(Op::LocalSet) && e.writeVarU32(WJVSLocalFor(c, slot));
  }
  return e.writeOp(Op::I64Store) && e.writeVarU32(3) && e.writeVarU32(0);
}
// At a safepoint (allocation/GC), the BYSTANDER operand-stack slots s[0..n) must
// be copied to the GC-traced frame so a moving GC updates the pointers; reload
// after so the running code sees the moved pointers. gWJFrameSP already covers the
// whole frame (set in the prologue), so no SP republish is needed.
//
// `n` is the bystander count: depth MINUS the operands this op consumes, because
// those operands are passed to the helper via gWJHelpA/B/C (or gWJScratch for
// calls), which WJTraceRoots ALSO traces -- so the GC updates the moved pointers
// there. The un-spilled operand/free frame slots still get traced, but the frame
// is init'd to Undefined and only ever written with valid Values, so tracing a
// stale-but-valid slot is safe (it can only over-retain, never dangle). Spilling
// only bystanders is the difference between O(depth) and ~O(0) traffic at the
// common statement-level call (where every live slot is a call operand).
static bool WJVSSpillRange(Encoder& e, const WJVSCtx& c, uint32_t n) {
  if (!WJVSUseLocals()) return true;
  for (uint32_t d = 0; d < n; d++) {
    if (!WJSAddr(e, c.stackBaseS + d) || !e.writeOp(Op::LocalGet) ||
        !e.writeVarU32(kVSsBase + d) || !WJSStoreEnd(e)) {
      return false;
    }
  }
  return true;
}
static bool WJVSReloadRange(Encoder& e, const WJVSCtx& c, uint32_t n) {
  if (!WJVSUseLocals()) return true;
  for (uint32_t d = 0; d < n; d++) {
    if (!WJSAddr(e, c.stackBaseS + d) || !e.writeOp(Op::I64Load) ||
        !e.writeVarU32(3) || !e.writeVarU32(0) || !e.writeOp(Op::LocalSet) ||
        !e.writeVarU32(kVSsBase + d)) {
      return false;
    }
  }
  return true;
}
static bool WJSIsNum(Encoder& e, uint32_t L) {  // push i32: is local L a number?
  return e.writeOp(Op::LocalGet) && e.writeVarU32(L) && e.writeOp(Op::I64Const) &&
         e.writeVarU64(32) && e.writeOp(Op::I64ShrU) && e.writeOp(Op::I64Const) &&
         e.writeVarS64(int64_t(kWJTagInt32)) && e.writeOp(Op::I64LeU);
}
static bool WJSIsObj(Encoder& e, uint32_t L) {  // push i32: is local L an object?
  return e.writeOp(Op::LocalGet) && e.writeVarU32(L) && e.writeOp(Op::I64Const) &&
         e.writeVarU64(32) && e.writeOp(Op::I64ShrU) && e.writeOp(Op::I64Const) &&
         e.writeVarS64(int64_t(kWJTagObject)) && e.writeOp(Op::I64Eq);
}
static bool WJVUnboxNG(Encoder& e, uint32_t L) {  // local L (a number) -> f64
  const uint8_t kF64 = uint8_t(TypeCode::F64);
  return e.writeOp(Op::LocalGet) && e.writeVarU32(L) && e.writeOp(Op::I64Const) &&
         e.writeVarU64(32) && e.writeOp(Op::I64ShrU) && e.writeOp(Op::I32WrapI64) &&
         e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(kWJTagInt32)) &&
         e.writeOp(Op::I32Eq) && e.writeOp(Op::If) && e.writeFixedU8(kF64) &&
         e.writeOp(Op::LocalGet) && e.writeVarU32(L) && e.writeOp(Op::I32WrapI64) &&
         e.writeOp(Op::F64ConvertI32S) && e.writeOp(Op::Else) &&
         e.writeOp(Op::LocalGet) && e.writeVarU32(L) &&
         e.writeOp(Op::F64ReinterpretI64) && e.writeOp(Op::End);
}
// After a wjhelp call (status f64 on stack): on a thrown exception, restore the
// frame stack pointer and return EXC (2.0) from the function.
static bool WJVSExcCheck(Encoder& e) {
  uint32_t spAddr = uint32_t(uintptr_t(&gWJFrameSP));
  const uint8_t kVoid = 0x40;
  return WJConst(e, 0.0) && e.writeOp(Op::F64Ne) && e.writeOp(Op::If) &&
         e.writeFixedU8(kVoid) && e.writeOp(Op::I32Const) &&
         e.writeVarS32(int32_t(spAddr)) && e.writeOp(Op::LocalGet) &&
         e.writeVarU32(kVSbasesp) && e.writeOp(Op::I32Store) && e.writeVarU32(2) &&
         e.writeVarU32(0) && WJConst(e, 2.0) && e.writeOp(Op::Return) &&
         e.writeOp(Op::End);
}
static bool WJVSStoreGlobal(Encoder& e, uint32_t addr, uint32_t L) {  // *addr = L
  return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(addr)) &&
         e.writeOp(Op::LocalGet) && e.writeVarU32(L) && WJSStoreEnd(e);
}
// Emit a wjhelp(kind, site) call (operands already in gWJHelp*). Spills the live
// operand stack to the traced frame before the call (the helper may allocate/GC)
// and reloads it after (picking up moved pointers). Leaves nothing; caller reads
// the result from gWJScratch[result] if needed.
// PHASE 4 (GECKO_WJVS_LEANCALL): a helper that provably cannot allocate / run user code /
// GC. js::StrictlyEqual does no coercion (no valueOf), no allocation, and no GC -- so the
// GC-spill of live operand-stack pointers around it is unnecessary. (Loose Eq/Ne use
// LooselyEqual, which CAN call valueOf, so they are NOT GC-safe.)
static bool WJHelperGCSafe(uint32_t kind) {
  return kind == WJH_STRICTEQ || kind == WJH_STRICTNE;
}
static bool WJVSLeanCall() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_LEANCALL") ? 1 : 0;
  return v != 0;
}
static uint64_t gWJLeanCalls = 0;  // PHASE 4: helper calls emitted without the GC spill/reload
static bool WJVSCallHelper(Encoder& e, const WJVSCtx& c, uint32_t kind,
                           uint32_t site, uint32_t spillN) {
  // PHASE 4: GC-safe helpers skip the spill/reload (no GC can move the unspilled pointers).
  if (WJVSLeanCall() && WJHelperGCSafe(kind)) {
    gWJLeanCalls++;
    return WJConst(e, double(kind)) && WJConst(e, double(site)) && e.writeOp(Op::Call) &&
           e.writeVarU32(kWJVSHelpIdx) && WJVSExcCheck(e);
  }
  return WJVSSpillRange(e, c, spillN) && WJConst(e, double(kind)) &&
         WJConst(e, double(site)) && e.writeOp(Op::Call) &&
         e.writeVarU32(kWJVSHelpIdx) && WJVSExcCheck(e) &&
         WJVSReloadRange(e, c, spillN);
}
static bool WJVSPushResult(Encoder& e, const WJVSCtx& c,
                           uint32_t destSlot) {  // dest = scratch[res]
  uint32_t resAddr = uint32_t(uintptr_t(&gWJScratch[kWJResultSlot]));
  return WJSStorePre(e, c, destSlot) && e.writeOp(Op::I32Const) &&
         e.writeVarS32(int32_t(resAddr)) && e.writeOp(Op::I64Load) &&
         e.writeVarU32(3) && e.writeVarU32(0) && WJSStorePost(e, c, destSlot);
}

// Binary arithmetic in Mode VS: numbers -> inline f64; otherwise wjhelp.
static bool WJVSBinArith(Encoder& e, WJVSCtx& c, Op fop, uint32_t kind) {
  const uint8_t kVoid = 0x40;
  uint32_t aS = c.stackBaseS + c.depth - 2, bS = c.stackBaseS + c.depth - 1;
  uint32_t helpA = uint32_t(uintptr_t(&gWJHelpA));
  uint32_t helpB = uint32_t(uintptr_t(&gWJHelpB));
  if (!WJSLoad(e, c, aS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0)) {
    return false;
  }
  if (!WJSLoad(e, c, bS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt1)) {
    return false;
  }
  if (!WJSIsNum(e, kVSt0) || !WJSIsNum(e, kVSt1) || !e.writeOp(Op::I32And)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!WJSStorePre(e, c, aS)) return false;
  if (!WJVUnboxNG(e, kVSt0) || !WJVUnboxNG(e, kVSt1) || !e.writeOp(fop) ||
      !WJVRebox(e, kVStf) || !WJSStorePost(e, c, aS)) {
    return false;
  }
  if (!e.writeOp(Op::Else)) return false;
  if (!WJVSStoreGlobal(e, helpA, kVSt0) || !WJVSStoreGlobal(e, helpB, kVSt1)) {
    return false;
  }
  if (!WJVSCallHelper(e, c, kind, 0, c.depth - 2) || !WJVSPushResult(e, c, aS)) return false;
  if (!e.writeOp(Op::End)) return false;
  c.depth -= 1;
  return true;
}
// Unary inc/dec in Mode VS.
static bool WJVSUnary(Encoder& e, WJVSCtx& c, bool inc) {
  const uint8_t kVoid = 0x40;
  uint32_t aS = c.stackBaseS + c.depth - 1;
  uint32_t helpA = uint32_t(uintptr_t(&gWJHelpA));
  if (!WJSLoad(e, c, aS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0)) {
    return false;
  }
  if (!WJSIsNum(e, kVSt0)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!WJSStorePre(e, c, aS) || !WJVUnboxNG(e, kVSt0) || !WJConst(e, 1.0) ||
      !e.writeOp(inc ? Op::F64Add : Op::F64Sub) || !WJVRebox(e, kVStf) ||
      !WJSStorePost(e, c, aS)) {
    return false;
  }
  if (!e.writeOp(Op::Else)) return false;
  if (!WJVSStoreGlobal(e, helpA, kVSt0)) return false;
  if (!WJVSCallHelper(e, c, inc ? WJH_INC : WJH_DEC, 0, c.depth - 1) || !WJVSPushResult(e, c, aS)) {
    return false;
  }
  return e.writeOp(Op::End);
}
// ToInt32 of the number in local L: unbox -> f64 -> i64 (sat trunc) -> wrap i32
// (matches Mode N; the f64->i64 trunc then i32.wrap gives JS ToInt32's low 32 bits.
// Using f64->i32 trunc directly would leave an i32 and make the i32.wrap_i64 below
// a type error -- that was the bitwise compile-failure bug).
static bool WJSToInt32(Encoder& e, uint32_t L) {
  return WJVUnboxNG(e, L) && e.writeOp(MiscOp::I64TruncSatF64S) &&
         e.writeOp(Op::I32WrapI64);
}
static bool WJEnsureF64(Encoder& e, WJVSCtx& c, uint32_t d);
// INTUNBOX: ensure the entry at depth d is a raw i32 in iLoc[d] (= JS ToInt32 of the
// operand). 3=already i32; otherwise coerce via f64 (WJEnsureF64 handles boxed/ptr/i32
// numerics incl. the ToNumber slow path) then sat-trunc+wrap to i32.
static bool WJEnsureI32(Encoder& e, WJVSCtx& c, uint32_t d) {
  if (c.repr[d] == 3) return true;
  if (c.repr[d] != 1 && !WJEnsureF64(e, c, d)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(WJFLoc(d)) ||
      !e.writeOp(MiscOp::I64TruncSatF64S) || !e.writeOp(Op::I32WrapI64) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(WJILoc(d))) {
    return false;
  }
  c.repr[d] = 3;
  return true;
}
// Binary bitwise/shift in Mode VS: numbers -> inline (ToInt32 each, i32 op,
// convert back to f64, rebox); otherwise wjhelp. `uns` selects unsigned result
// conversion (>>>); the wasm shift ops already mask the count mod 32 (= JS).
static bool WJVSBitOp(Encoder& e, WJVSCtx& c, Op i32op, uint32_t kind, bool uns) {
  const uint8_t kVoid = 0x40;
  // INTUNBOX fast path: when BOTH operands are register-resident numbers (repr 1/3),
  // do the signed bitwise op i32->i32 with NO memory traffic, isNum guard, f64
  // round-trip, or rebox -- the result stays an unboxed i32 (repr=3), so int chains
  // (richards' flag dispatch) flow entirely in registers. Skip for >>> (uns): its
  // result can exceed INT32_MAX and must be a double, not a kept int32.
  if (WJIntUnbox() && !uns && c.depth >= 2) {
    uint32_t dA = c.depth - 2, dB = c.depth - 1;
    bool aReg = c.repr[dA] == 1 || c.repr[dA] == 3;
    bool bReg = c.repr[dB] == 1 || c.repr[dB] == 3;
    if (aReg && bReg) {
      if (!WJEnsureI32(e, c, dA) || !WJEnsureI32(e, c, dB)) return false;
      if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(WJILoc(dA)) ||
          !e.writeOp(Op::LocalGet) || !e.writeVarU32(WJILoc(dB)) ||
          !e.writeOp(i32op) ||
          !e.writeOp(Op::LocalSet) || !e.writeVarU32(WJILoc(dA))) {
        return false;
      }
      c.repr[dA] = 3;
      c.depth -= 1;
      return true;
    }
  }
  uint32_t aS = c.stackBaseS + c.depth - 2, bS = c.stackBaseS + c.depth - 1;
  uint32_t helpA = uint32_t(uintptr_t(&gWJHelpA));
  uint32_t helpB = uint32_t(uintptr_t(&gWJHelpB));
  if (!WJSLoad(e, c, aS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0) ||
      !WJSLoad(e, c, bS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt1)) {
    return false;
  }
  if (!WJSIsNum(e, kVSt0) || !WJSIsNum(e, kVSt1) || !e.writeOp(Op::I32And)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!WJSStorePre(e, c, aS) || !WJSToInt32(e, kVSt0) || !WJSToInt32(e, kVSt1) ||
      !e.writeOp(i32op) ||
      !e.writeOp(uns ? Op::F64ConvertI32U : Op::F64ConvertI32S) ||
      !WJVRebox(e, kVStf) || !WJSStorePost(e, c, aS)) {
    return false;
  }
  if (!e.writeOp(Op::Else)) return false;
  if (!WJVSStoreGlobal(e, helpA, kVSt0) || !WJVSStoreGlobal(e, helpB, kVSt1) ||
      !WJVSCallHelper(e, c, kind, 0, c.depth - 2) || !WJVSPushResult(e, c, aS)) {
    return false;
  }
  if (!e.writeOp(Op::End)) return false;
  c.depth -= 1;
  return true;
}
// Unary BitNot in Mode VS: ~ToInt32(x).
static bool WJVSBitNot(Encoder& e, WJVSCtx& c) {
  const uint8_t kVoid = 0x40;
  uint32_t aS = c.stackBaseS + c.depth - 1;
  uint32_t helpA = uint32_t(uintptr_t(&gWJHelpA));
  if (!WJSLoad(e, c, aS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0)) {
    return false;
  }
  if (!WJSIsNum(e, kVSt0)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!WJSStorePre(e, c, aS) || !WJSToInt32(e, kVSt0) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(-1) || !e.writeOp(Op::I32Xor) ||
      !e.writeOp(Op::F64ConvertI32S) || !WJVRebox(e, kVStf) ||
      !WJSStorePost(e, c, aS)) {
    return false;
  }
  if (!e.writeOp(Op::Else)) return false;
  if (!WJVSStoreGlobal(e, helpA, kVSt0) || !WJVSCallHelper(e, c, WJH_BITNOT, 0, c.depth - 1) ||
      !WJVSPushResult(e, c, aS)) {
    return false;
  }
  if (!e.writeOp(Op::End)) return false;
  return true;
}
// Comparison in Mode VS: leaves an i32 (0/1) on the wasm stack for the following
// JumpIfFalse/JumpIfTrue. Numbers -> inline f64 compare; otherwise wjhelp. When
// asValue (SHORT-CIRCUIT: the result is consumed as a Value, not a branch), box the
// 0/1 into a boolean Value in the result operand slot instead (depth -1, not -2).
static bool WJVSCmp(Encoder& e, WJVSCtx& c, JSOp op, bool asValue = false) {
  const uint8_t kI32 = uint8_t(TypeCode::I32);
  uint32_t aS = c.stackBaseS + c.depth - 2, bS = c.stackBaseS + c.depth - 1;
  uint32_t helpA = uint32_t(uintptr_t(&gWJHelpA));
  uint32_t helpB = uint32_t(uintptr_t(&gWJHelpB));
  uint32_t resAddr = uint32_t(uintptr_t(&gWJScratch[kWJResultSlot]));
  Op fop;
  uint32_t kind;
  switch (op) {
    case JSOp::Lt: fop = Op::F64Lt; kind = WJH_LT; break;
    case JSOp::Le: fop = Op::F64Le; kind = WJH_LE; break;
    case JSOp::Gt: fop = Op::F64Gt; kind = WJH_GT; break;
    case JSOp::Ge: fop = Op::F64Ge; kind = WJH_GE; break;
    case JSOp::Eq: fop = Op::F64Eq; kind = WJH_EQ; break;
    case JSOp::Ne: fop = Op::F64Ne; kind = WJH_NE; break;
    case JSOp::StrictEq: fop = Op::F64Eq; kind = WJH_STRICTEQ; break;
    default: fop = Op::F64Ne; kind = WJH_STRICTNE; break;
  }
  if (!WJSLoad(e, c, aS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0)) {
    return false;
  }
  if (!WJSLoad(e, c, bS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt1)) {
    return false;
  }
  if (!WJSIsNum(e, kVSt0) || !WJSIsNum(e, kVSt1) || !e.writeOp(Op::I32And)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;
  if (!WJVUnboxNG(e, kVSt0) || !WJVUnboxNG(e, kVSt1) || !e.writeOp(fop)) {
    return false;
  }
  if (!e.writeOp(Op::Else)) return false;
  // NULLCMP fast path: when BOTH operands are object/null/undefined (the dominant
  // richards `x != null` / object-identity case), compute (in)equality inline with NO
  // helper call. Sound: numbers handled above; strings/bigints/booleans/mixed fall to
  // the helper. Loose: null==undefined; object==object is identity. Strict: bit-identity.
  // Eliminates the per-op Ne/Eq wasm->C++ boundary crossing (the measured 463K helpers).
  if (!WJNullCmp()) {
    if (!WJVSStoreGlobal(e, helpA, kVSt0) || !WJVSStoreGlobal(e, helpB, kVSt1) ||
        !WJVSCallHelper(e, c, kind, 0, c.depth - 2) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(resAddr)) ||
        !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0) ||
        !e.writeOp(Op::I32WrapI64)) {
      return false;
    }
    if (!e.writeOp(Op::End)) return false;
    if (asValue) goto cmpBox; else { c.depth -= 2; return true; }
  }
  {
    const int32_t OBJ = int32_t(kWJTagObject);          // 0xFFFFFF8C
    const int32_t UNDEF = int32_t(kWJTagUndefined);     // 0xFFFFFF83 (null=+1)
    // tags: ti = tag(a), ti2 = tag(b)
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt0) || !e.writeOp(Op::I64Const) ||
        !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSti) ||
        !e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt1) || !e.writeOp(Op::I64Const) ||
        !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSti2)) {
      return false;
    }
    // objOrNu(L): (L==OBJ) | ((u32)(L-UNDEF) <= 1)
    auto objOrNu = [&](uint32_t L) -> bool {
      return e.writeOp(Op::LocalGet) && e.writeVarU32(L) && e.writeOp(Op::I32Const) &&
             e.writeVarS32(OBJ) && e.writeOp(Op::I32Eq) &&
             e.writeOp(Op::LocalGet) && e.writeVarU32(L) && e.writeOp(Op::I32Const) &&
             e.writeVarS32(UNDEF) && e.writeOp(Op::I32Sub) && e.writeOp(Op::I32Const) &&
             e.writeVarS32(1) && e.writeOp(Op::I32LeU) && e.writeOp(Op::I32Or);
    };
    auto isObj = [&](uint32_t L) -> bool {
      return e.writeOp(Op::LocalGet) && e.writeVarU32(L) && e.writeOp(Op::I32Const) &&
             e.writeVarS32(OBJ) && e.writeOp(Op::I32Eq);
    };
    auto isNu = [&](uint32_t L) -> bool {
      return e.writeOp(Op::LocalGet) && e.writeVarU32(L) && e.writeOp(Op::I32Const) &&
             e.writeVarS32(UNDEF) && e.writeOp(Op::I32Sub) && e.writeOp(Op::I32Const) &&
             e.writeVarS32(1) && e.writeOp(Op::I32LeU);
    };
    // cond = objOrNu(a) & objOrNu(b)
    if (!objOrNu(kVSti) || !objOrNu(kVSti2) || !e.writeOp(Op::I32And)) return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;
    bool strict = (op == JSOp::StrictEq || op == JSOp::StrictNe);
    bool neg = (op == JSOp::Ne || op == JSOp::StrictNe);
    if (strict) {
      // bit identity
      if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt0) || !e.writeOp(Op::LocalGet) ||
          !e.writeVarU32(kVSt1) || !e.writeOp(Op::I64Eq)) {
        return false;
      }
    } else {
      // select(bothObj, bitEq, nu_a & nu_b)
      if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt0) || !e.writeOp(Op::LocalGet) ||
          !e.writeVarU32(kVSt1) || !e.writeOp(Op::I64Eq)) {  // bitEq (val1)
        return false;
      }
      if (!isNu(kVSti) || !isNu(kVSti2) || !e.writeOp(Op::I32And)) return false;  // nu&nu (val2)
      if (!isObj(kVSti) || !isObj(kVSti2) || !e.writeOp(Op::I32And)) return false;  // bothObj (cond)
      if (!e.writeOp(Op::SelectNumeric)) return false;
    }
    if (neg && (!e.writeOp(Op::I32Eqz))) return false;
    if (!e.writeOp(Op::Else)) return false;
    if (!WJVSStoreGlobal(e, helpA, kVSt0) || !WJVSStoreGlobal(e, helpB, kVSt1) ||
        !WJVSCallHelper(e, c, kind, 0, c.depth - 2) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(resAddr)) ||
        !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0) ||
        !e.writeOp(Op::I32WrapI64)) {
      return false;
    }
    if (!e.writeOp(Op::End)) return false;  // inner objOrNu If
  }
  if (!e.writeOp(Op::End)) return false;  // outer number If
cmpBox:;
  if (asValue) {
    // box the 0/1 (i32 on stack) into a boolean Value and store to the result slot
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSti)) return false;  // save i32
    if (!WJSStorePre(e, c, aS) || !e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) ||
        !e.writeOp(Op::I64ExtendI32U) || !e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(uint64_t(kWJTagBoolean) << 32)) || !e.writeOp(Op::I64Or) ||
        !WJSStorePost(e, c, aS)) {
      return false;
    }
    c.depth -= 1;
  } else {
    c.depth -= 2;
  }
  return true;
}
// GetProp in Mode VS: N-WAY polymorphic shape-guarded inline load (own fixed/
// dynamic slot + prototype-chain data property); else wjhelp. DeltaBlue's class-
// hierarchy method/field reads see several receiver shapes -- a monomorphic guard
// thrashed way 0 and fell to the helper on every shape flip (the dominant Mode VS
// boundary tax). Reuses the shared N-way IC arrays + poly fill policy (WJFillIC).
static bool WJVSGetProp(Encoder& e, WJVSCtx& c, jsbytecode* pc, bool recvPtr = false) {
  const uint8_t kVoid = 0x40, kI64 = uint8_t(TypeCode::I64);
  if (gWJSiteCount >= kWJMaxSites) return false;
  uint32_t site = gWJSiteCount++;
  gWJSites[site].script = c.script;
  gWJSites[site].pcOff = uint32_t(pc - c.start);
  bool poly = !WJNoPolyProp();
  if (poly) gWJSitePoly[site] = true;  // Mode VS GetProp uses the N-way IC
  uint32_t nways = poly ? kWJICWays : 1;
  // Detect a `.length` read: STRING length (header field) + dense-array length
  // (elements header) inline, shape-guarded for arrays (see gWJSiteLen). This was
  // the dominant Mode VS GetProp helper tax -- array `.length` is a custom data
  // property (isDataProperty()==false) so the slot IC always bailed to the helper.
  if (!WJNoLen()) {
    JSContext* cxn = js::TlsContext.get();
    js::PropertyName* nm = c.script->getName(pc);
    if (cxn && nm == cxn->names().length) gWJSiteLen[site] = true;
  }
  uint32_t helpA = uint32_t(uintptr_t(&gWJHelpA));
  uint32_t topS = c.stackBaseS + c.depth - 1;
  auto wayIC = [&](uint32_t w) -> uint32_t {  // addr of way w's {shape,off} pair
    return w == 0 ? uint32_t(uintptr_t(&gWJICTable[2 * site]))
                  : uint32_t(uintptr_t(&gWJICTableX[((w - 1) * kWJMaxSites + site) * 2]));
  };
  auto wayHolder = [&](uint32_t w) -> uint32_t {  // addr of way w's holder cell
    return w == 0 ? uint32_t(uintptr_t(&gWJProtoHolder[site]))
                  : uint32_t(uintptr_t(&gWJProtoHolderX[(w - 1) * kWJMaxSites + site]));
  };
  auto wayShape = [&](uint32_t w) -> uint32_t {  // addr of way w's holder-shape cell
    return w == 0 ? uint32_t(uintptr_t(&gWJProtoHolderShape[site]))
                  : uint32_t(uintptr_t(&gWJProtoHolderShapeX[(w - 1) * kWJMaxSites + site]));
  };
  auto helperGet = [&]() -> bool {
    return WJVSStoreGlobal(e, helpA, kVSt0) && WJVSCallHelper(e, c, WJH_GETPROP, site, c.depth - 1) &&
           WJVSPushResult(e, c, topS);
  };
  // LEAN EMISSION: baked direct-field GetProp. At a specialized recompile, if this site's
  // (script,pcOff) has a recorded OWN FIXED-slot {shape,off}, emit shape+off as CONSTANTS:
  // one shape compare (no IC-table load) then a direct i64.load at the const offset -- skipping
  // the entire N-way IC chain (the ~20-instr-per-access bloat). Helper on shape miss / non-object.
  if (gWJEmitBake && !gWJSiteLen[site]) {
    auto rec = gWJShapeRec.find(WJInlineKey(c.script, uint32_t(pc - c.start)));
    if (rec != gWJShapeRec.end() && rec->second.shape != 0 && !(rec->second.off & kWJDynSlot)) {
      uint32_t bShape = rec->second.shape, bOff = rec->second.off;
      if (!WJSLoad(e, c, topS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0)) return false;
      if (!WJSIsObj(e, kVSt0) || !e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
      if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt0) || !e.writeOp(Op::I32WrapI64) ||
          !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSti)) {
        return false;
      }
      if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) ||
          !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(bShape)) || !e.writeOp(Op::I32Eq) ||
          !e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) {
        return false;
      }
      if (!WJSStorePre(e, c, topS) || !e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) ||
          !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(bOff) ||
          !WJSStorePost(e, c, topS)) {
        return false;
      }
      if (!e.writeOp(Op::Else) || !helperGet() || !e.writeOp(Op::End)) return false;  // shape miss
      if (!e.writeOp(Op::Else) || !helperGet() || !e.writeOp(Op::End)) return false;  // not object
      return true;
    }
  }
  auto emitOff = [&](uint32_t icAddr) -> bool {  // push i32 off = ic[icAddr+4]
    return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(icAddr)) &&
           e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(4);
  };
  // push i64 = property value at byte offset `off` from `base` (fixed slot:
  // base+off; dynamic slot (kWJDynSlot bit): slots_(@base+8) + (off & ~bit)).
  auto emitSlotLoad = [&](uint32_t base, uint32_t icAddr) -> bool {
    return emitOff(icAddr) && e.writeOp(Op::I32Const) &&
           e.writeVarS32(int32_t(kWJDynSlot)) && e.writeOp(Op::I32And) &&
           e.writeOp(Op::If) && e.writeFixedU8(kI64) && e.writeOp(Op::LocalGet) &&
           e.writeVarU32(base) && e.writeOp(Op::I32Load) && e.writeVarU32(2) &&
           e.writeVarU32(8) && emitOff(icAddr) && e.writeOp(Op::I32Const) &&
           e.writeVarS32(int32_t(~kWJDynSlot)) && e.writeOp(Op::I32And) &&
           e.writeOp(Op::I32Add) && e.writeOp(Op::I64Load) && e.writeVarU32(3) &&
           e.writeVarU32(0) && e.writeOp(Op::Else) && e.writeOp(Op::LocalGet) &&
           e.writeVarU32(base) && emitOff(icAddr) && e.writeOp(Op::I32Add) &&
           e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0) &&
           e.writeOp(Op::End);
  };
  auto emitLoadTo = [&](uint32_t base, uint32_t icAddr) -> bool {  // frame[topS] = slotLoad
    return WJSStorePre(e, c, topS) && emitSlotLoad(base, icAddr) && WJSStorePost(e, c, topS);
  };
  // frame[topS] = dense-array length (receiver shape already matched -> known
  // ArrayObject). len = i32.load[ i32.load[kVSti+12]/*elements_*/ - 4 ]; a length
  // >= 2^31 (would be a double Value) falls to the helper. Uses kVSti2 as scratch.
  auto emitLenTo = [&]() -> bool {
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(12) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(4) || !e.writeOp(Op::I32Sub) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::LocalSet) ||
        !e.writeVarU32(kVSti2)) {
      return false;
    }
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti2) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(0) || !e.writeOp(Op::I32LtS) || !e.writeOp(Op::If) ||
        !e.writeFixedU8(kVoid)) {
      return false;
    }
    if (!helperGet()) return false;  // length doesn't fit int32 -> helper
    if (!e.writeOp(Op::Else)) return false;
    if (!WJSStorePre(e, c, topS) || !e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti2) ||
        !e.writeOp(Op::I64ExtendI32U) || !e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(kWJTagInt32 << 32)) || !e.writeOp(Op::I64Or) ||
        !WJSStorePost(e, c, topS)) {
      return false;
    }
    return e.writeOp(Op::End);
  };
  // Way w's load (its receiver shape already matched): own (holder==0, load from
  // receiver kVSti) or proto (holder-shape guard, load from the holder). An inner
  // miss (holder shape changed) falls to the helper.
  auto loadWay = [&](uint32_t w) -> bool {
    uint32_t ic = wayIC(w);
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(wayHolder(w))) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSti2)) {
      return false;
    }
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti2) || !e.writeOp(Op::I32Eqz)) {
      return false;
    }
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
    if (!emitLoadTo(kVSti, ic)) return false;  // own: load from receiver
    if (!e.writeOp(Op::Else)) return false;
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti2) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(wayShape(w))) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Eq)) {
      return false;
    }
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
    if (!emitLoadTo(kVSti2, ic)) return false;  // proto: load from holder
    if (!e.writeOp(Op::Else) || !helperGet() || !e.writeOp(Op::End)) return false;  // holder miss
    return e.writeOp(Op::End);  // end own/proto if
  };
  // "if (shape@receiver == ic[way].shape) {" -- opens an If(void).
  static int noGuardProbe = -1;  // PERF PROBE (unsound): GECKO_WJVS_NOGUARD assumes way 0 always hits
  if (noGuardProbe < 0) noGuardProbe = getenv("GECKO_WJVS_NOGUARD") ? 1 : 0;
  auto shapeGuard = [&](uint32_t w) -> bool {
    if (noGuardProbe && w == 0) {  // skip the shape load+compare; always take the slot-load path
      return e.writeOp(Op::I32Const) && e.writeVarS32(1) && e.writeOp(Op::If) &&
             e.writeFixedU8(kVoid);
    }
    // PTRUNBOX: reuse the hoisted `this`-shape local instead of reloading i32.load[recv+0].
    bool useCached = recvPtr && c.thisShapeCached;
    if (useCached) {
      if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSthisShape)) return false;
    } else if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) ||
               !e.writeVarU32(2) || !e.writeVarU32(0)) {
      return false;
    }
    return e.writeOp(Op::I32Const) &&
           e.writeVarS32(int32_t(wayIC(w))) && e.writeOp(Op::I32Load) &&
           e.writeVarU32(2) && e.writeVarU32(0) && e.writeOp(Op::I32Eq) &&
           e.writeOp(Op::If) && e.writeFixedU8(kVoid);
  };
  // Object path: ti = receiver low32, then the N-way receiver-shape chain (each
  // way loads its slot, or the array length for a `.length` site); innermost else
  // = helper. Assumes kVSt0 is a known object.
  auto objPath = [&]() -> bool {
    if (recvPtr) {  // PTRUNBOX: receiver ptr already cached unboxed -> skip the i64.wrap
      if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(WJPLoc(c.depth - 1)) ||
          !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSti)) {
        return false;
      }
      if (!c.thisShapeCached) {  // hoist: load this.shape once, reuse across this region's guards
        if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) ||
            !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::LocalSet) ||
            !e.writeVarU32(kVSthisShape)) {
          return false;
        }
        c.thisShapeCached = true;
      }
    } else if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt0) ||
               !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::LocalSet) ||
               !e.writeVarU32(kVSti)) {
      return false;
    }
    for (uint32_t w = 0; w < nways; w++) {
      if (!shapeGuard(w)) return false;
      if (gWJSiteLen[site] ? !emitLenTo() : !loadWay(w)) return false;
      if (!e.writeOp(Op::Else)) return false;
    }
    if (!helperGet()) return false;  // all ways missed
    for (uint32_t w = 0; w < nways; w++) {
      if (!e.writeOp(Op::End)) return false;  // close each way's If
    }
    return true;
  };
  auto objGuarded = [&]() -> bool {  // if (isObject) { objPath } else { helper }
    return WJSIsObj(e, kVSt0) && e.writeOp(Op::If) && e.writeFixedU8(kVoid) &&
           objPath() && e.writeOp(Op::Else) && helperGet() && e.writeOp(Op::End);
  };
  if (!WJSLoad(e, c, topS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0)) {
    return false;
  }
  if (gWJSiteLen[site]) {
    // STRING length: read inline from the JSString header (immovable field).
    static_assert(JSString::offsetOfLength() == 4, "wasm32 JSString length offset");
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt0) || !e.writeOp(Op::I64Const) ||
        !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(kWJTagString)) ||
        !e.writeOp(Op::I32Eq) || !e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) {
      return false;
    }
    if (!WJSStorePre(e, c, topS) || !e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt0) ||
        !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
        !e.writeVarU32(4) || !e.writeOp(Op::I64ExtendI32U) || !e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(kWJTagInt32 << 32)) || !e.writeOp(Op::I64Or) ||
        !WJSStorePost(e, c, topS)) {
      return false;
    }
    if (!e.writeOp(Op::Else) || !objGuarded() || !e.writeOp(Op::End)) return false;
    return true;
  }
  // PTRUNBOX: receiver statically known to be an object -> skip the isObject guard.
  return recvPtr ? objPath() : objGuarded();
}

// Mode VS GetAliasedVar: read a closed-over variable via wjhelp (WJH_GETALIASED
// walks gWJCurEnv by the bytecode's EnvironmentCoordinate). Pushes the boxed
// result. Reached only after WJMaterializeAll (under unbox), so repr[] is clear.
// The enclosing function is excluded from the fast call table so it always enters
// through WasmJitRunCall, which sets gWJCurEnv to its environment chain.
static bool WJVSGetAliased(Encoder& e, WJVSCtx& c, jsbytecode* pc) {
  if (gWJSiteCount >= kWJMaxSites) return false;
  uint32_t site = gWJSiteCount++;
  gWJSites[site].script = c.script;
  gWJSites[site].pcOff = uint32_t(pc - c.start);
  uint32_t top = c.stackBaseS + c.depth;
  if (!WJVSCallHelper(e, c, WJH_GETALIASED, site, c.depth) ||
      !WJVSPushResult(e, c, top)) {
    return false;
  }
  c.depth++;
  return true;
}
// Inline Mode VS GetGName: guard the cached global-holder shape + load the slot
// (the holder/shape/slot are filled by WJH_GETGNAME -> WJFillIC). holder==0
// (unfilled) or a shape miss (or a lexical name, never cached) -> helper. Removes
// the dominant Mode VS boundary tax on global-name reads.
static bool WJVSGetGName(Encoder& e, WJVSCtx& c, jsbytecode* pc) {
  const uint8_t kVoid = 0x40, kI64 = uint8_t(TypeCode::I64);
  if (gWJSiteCount >= kWJMaxSites) return false;
  uint32_t site = gWJSiteCount++;
  gWJSites[site].script = c.script;
  gWJSites[site].pcOff = uint32_t(pc - c.start);
  uint32_t holderCell = uint32_t(uintptr_t(&gWJGNameHolder[site]));
  uint32_t icAddr = uint32_t(uintptr_t(&gWJICTable[2 * site]));
  uint32_t top = c.stackBaseS + c.depth;
  auto helperGet = [&]() -> bool {
    return WJVSCallHelper(e, c, WJH_GETGNAME, site, c.depth) && WJVSPushResult(e, c, top);
  };
  auto emitOff = [&]() -> bool {  // push i32 off = gWJICTable[2*site+1]
    return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(icAddr)) &&
           e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(4);
  };
  auto emitSlotLoad = [&](uint32_t base) -> bool {  // fixed/dynamic slot at `base`
    return emitOff() && e.writeOp(Op::I32Const) &&
           e.writeVarS32(int32_t(kWJDynSlot)) && e.writeOp(Op::I32And) &&
           e.writeOp(Op::If) && e.writeFixedU8(kI64) && e.writeOp(Op::LocalGet) &&
           e.writeVarU32(base) && e.writeOp(Op::I32Load) && e.writeVarU32(2) &&
           e.writeVarU32(8) && emitOff() && e.writeOp(Op::I32Const) &&
           e.writeVarS32(int32_t(~kWJDynSlot)) && e.writeOp(Op::I32And) &&
           e.writeOp(Op::I32Add) && e.writeOp(Op::I64Load) && e.writeVarU32(3) &&
           e.writeVarU32(0) && e.writeOp(Op::Else) && e.writeOp(Op::LocalGet) &&
           e.writeVarU32(base) && emitOff() && e.writeOp(Op::I32Add) &&
           e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0) &&
           e.writeOp(Op::End);
  };
  // ti = holder = i32.load[holderCell]
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(holderCell)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSti)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Eqz)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;  // holder==0
  if (!helperGet()) return false;
  if (!e.writeOp(Op::Else)) return false;
  // shape guard: i32.load[holder+0] == cached shape
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) ||
      !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(icAddr)) || !e.writeOp(Op::I32Load) ||
      !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Eq)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!WJSStorePre(e, c, top) || !emitSlotLoad(kVSti) || !WJSStorePost(e, c, top)) {
    return false;  // hit
  }
  if (!e.writeOp(Op::Else) || !helperGet() || !e.writeOp(Op::End)) return false;  // shape miss
  if (!e.writeOp(Op::End)) return false;  // end holder!=0 if
  c.depth++;
  return true;
}
// SetProp in Mode VS: own fixed-slot number store inline (shape+number guarded);
// else wjhelp. Leaves the assigned value on the operand stack (depth -1 net).
// INLINEALLOC (plan §8.3, helper-based first cut): NewObject/NewInit create the literal/template
// object via WJH_NEWOBJECT (pushes it). A later increment inlines the nursery bump-alloc.
static bool WJVSNewObject(Encoder& e, WJVSCtx& c, jsbytecode* pc) {
  if (gWJSiteCount >= kWJMaxSites) return false;
  uint32_t site = gWJSiteCount++;
  gWJSites[site].script = c.script;
  gWJSites[site].pcOff = uint32_t(pc - c.start);
  uint32_t top = c.stackBaseS + c.depth;
  if (!WJVSCallHelper(e, c, WJH_NEWOBJECT, site, c.depth)) return false;
  if (!WJVSPushResult(e, c, top)) return false;
  c.depth++;
  return true;
}
// INLINEALLOC: InitProp defines obj.<name> = val (constructor field-init / object literal). Stack
// [obj,val] -> [obj] (val popped, obj kept). Helper-based first cut (WJH_INITPROP); a later
// increment adds the barrier-flag-gated inline add-property IC (§8.3b).
// GECKO_WJVS_INITINLINE: inline add-property IC for InitProp (else helper-only = component A).
static bool WJVSInitInline() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_INITINLINE") ? 1 : 0;
  return v != 0;
}
static bool WJVSInitProp(Encoder& e, WJVSCtx& c, jsbytecode* pc) {
  const uint8_t kVoid = 0x40, kI64 = uint8_t(TypeCode::I64);
  if (gWJSiteCount >= kWJMaxSites) return false;
  uint32_t site = gWJSiteCount++;
  gWJSites[site].script = c.script;
  gWJSites[site].pcOff = uint32_t(pc - c.start);
  uint32_t helpA = uint32_t(uintptr_t(&gWJHelpA));
  uint32_t helpB = uint32_t(uintptr_t(&gWJHelpB));
  uint32_t objS = c.stackBaseS + c.depth - 2, valS = c.stackBaseS + c.depth - 1;
  if (!WJSLoad(e, c, objS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0)) return false;
  if (!WJSLoad(e, c, valS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt1)) return false;
  auto helper = [&]() -> bool {  // correct fallback (may GC/move obj -> reload obj from result)
    return WJVSStoreGlobal(e, helpA, kVSt0) && WJVSStoreGlobal(e, helpB, kVSt1) &&
           WJVSCallHelper(e, c, WJH_INITPROP, site, c.depth - 2) && WJVSPushResult(e, c, objS);
  };
  // INLINE add-property IC (§8.3b/e): fast path when (a) no incremental marking (barrier flag clear,
  // so the shape PRE-barrier is unnecessary) and (b) obj's shape == the IC's cached fromShape. Then
  // store val into the cached fixed slot and set the cached toShape -- all inline, no GC. Any miss
  // (marking active, shape mismatch, unfilled IC) -> the correct barriered helper.
  uint32_t barrierAddr = 0;
  if (WJVSInitInline()) {
    JSContext* cx = js::TlsContext.get();
    if (cx && c.script->zone()) {
      barrierAddr = uint32_t(uintptr_t(JS::shadow::Zone::from(c.script->zone()))) + 8;
    }
  }
  if (!barrierAddr) {  // helper-only (component A) or no inline gate
    if (!helper()) return false;
    c.depth -= 1;
    return true;
  }
  uint32_t fromAddr = uint32_t(uintptr_t(&gWJICTable[2 * site]));
  uint32_t offAddr = uint32_t(uintptr_t(&gWJICTable[2 * site + 1]));
  uint32_t toAddr = uint32_t(uintptr_t(&gWJInitToShape[site]));
  // ti = (i32)obj
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt0) || !e.writeOp(Op::I32WrapI64) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSti)) {
    return false;
  }
  // cond = (i32.load[barrierAddr] == 0) & (i32.load[ti+0] == i32.load[fromAddr])
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(barrierAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Eqz)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) ||
      !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(fromAddr)) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
      !e.writeVarU32(0) || !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32And)) {
    return false;
  }
  // AND val is a number: a primitive store never needs the generational POST-barrier (only a
  // tenured-obj -> nursery-GC-thing edge does). Object/string-valued fields fall to the helper.
  if (!WJSIsNum(e, kVSt1) || !e.writeOp(Op::I32And)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  // FAST: i64.store[ti + off] = val ; i32.store[ti+0] = toShape  (obj does not move here)
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(offAddr)) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
      !e.writeVarU32(0) || !e.writeOp(Op::I32Add) || !e.writeOp(Op::LocalGet) ||
      !e.writeVarU32(kVSt1) || !e.writeOp(Op::I64Store) || !e.writeVarU32(3) ||
      !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(toAddr)) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
      !e.writeVarU32(0) || !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::Else)) return false;
  if (!helper()) return false;  // miss -> correct barriered helper (may move obj)
  if (!e.writeOp(Op::End)) return false;
  c.depth -= 1;
  return true;
}
static bool WJVSSetProp(Encoder& e, WJVSCtx& c, jsbytecode* pc) {
  const uint8_t kVoid = 0x40;
  if (gWJSiteCount >= kWJMaxSites) return false;
  uint32_t site = gWJSiteCount++;
  gWJSites[site].script = c.script;
  gWJSites[site].pcOff = uint32_t(pc - c.start);
  uint32_t icAddr = uint32_t(uintptr_t(&gWJICTable[2 * site]));
  uint32_t helpA = uint32_t(uintptr_t(&gWJHelpA));
  uint32_t helpB = uint32_t(uintptr_t(&gWJHelpB));
  uint32_t objS = c.stackBaseS + c.depth - 2, valS = c.stackBaseS + c.depth - 1;
  auto helperSet = [&]() -> bool {
    return WJVSStoreGlobal(e, helpA, kVSt0) && WJVSStoreGlobal(e, helpB, kVSt1) &&
           WJVSCallHelper(e, c, WJH_SETPROP, site, c.depth - 2);
  };
  // LEAN EMISSION: baked direct-field SetProp -- shape+off as constants, direct i64.store at
  // the const offset (+ post-barrier only for object values), skipping the IC chain. Fixed
  // slots only (SetProp records are always own fixed-slot). Helper on shape miss / non-object.
  if (gWJEmitBake) {
    auto rec = gWJShapeRec.find(WJInlineKey(c.script, uint32_t(pc - c.start)));
    if (rec != gWJShapeRec.end() && rec->second.shape != 0 && !(rec->second.off & kWJDynSlot)) {
      uint32_t bShape = rec->second.shape, bOff = rec->second.off;
      if (!WJSLoad(e, c, objS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0) ||
          !WJSLoad(e, c, valS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt1)) {
        return false;
      }
      if (!WJSIsObj(e, kVSt0) || !e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
      if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt0) || !e.writeOp(Op::I32WrapI64) ||
          !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSti)) {
        return false;
      }
      if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) ||
          !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(bShape)) || !e.writeOp(Op::I32Eq) ||
          !e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) {
        return false;
      }
      // i64.store[ti + bOff] = val
      if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::LocalGet) ||
          !e.writeVarU32(kVSt1) || !e.writeOp(Op::I64Store) || !e.writeVarU32(3) ||
          !e.writeVarU32(bOff)) {
        return false;
      }
      // post-barrier only for non-number (potential nursery-ptr) values
      if (!WJSIsNum(e, kVSt1) || !e.writeOp(Op::I32Eqz) || !e.writeOp(Op::If) ||
          !e.writeFixedU8(kVoid)) {
        return false;
      }
      if (!WJVSStoreGlobal(e, helpA, kVSt0) || !WJVSStoreGlobal(e, helpB, kVSt1) ||
          !WJConst(e, double(WJH_POSTBARRIER)) || !WJConst(e, 0.0) ||
          !e.writeOp(Op::Call) || !e.writeVarU32(kWJVSHelpIdx) || !e.writeOp(Op::Drop)) {
        return false;
      }
      if (!e.writeOp(Op::End)) return false;  // barrier if
      if (!e.writeOp(Op::Else) || !helperSet() || !e.writeOp(Op::End)) return false;  // shape miss
      if (!e.writeOp(Op::Else) || !helperSet() || !e.writeOp(Op::End)) return false;  // not object
      if (!WJSStorePre(e, c, objS) || !e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt1) ||
          !WJSStorePost(e, c, objS)) {
        return false;
      }
      c.depth -= 1;
      return true;
    }
  }
  if (!WJSLoad(e, c, objS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0)) {
    return false;
  }
  if (!WJSLoad(e, c, valS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt1)) {
    return false;
  }
  if (!WJSIsObj(e, kVSt0)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt0) ||
      !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(kVSti)) {
    return false;
  }
  // Guard: shape matches the cached own-fixed-slot data property. Without the
  // object-store path, additionally require the value to be a number (barrier-free).
  bool objset = !WJNoObjSet();
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) ||
      !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(icAddr)) || !e.writeOp(Op::I32Load) ||
      !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Eq)) {
    return false;
  }
  if (!objset) {
    if (!WJSIsNum(e, kVSt1) || !e.writeOp(Op::I32And)) return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  // inline store: i64.store[ti + off] = t1 (raw NaN-boxed bits; valid for any Value).
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(icAddr)) || !e.writeOp(Op::I32Load) ||
      !e.writeVarU32(2) || !e.writeVarU32(4) || !e.writeOp(Op::I32Add) ||
      !e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt1) || !WJSStoreEnd(e)) {
    return false;
  }
  if (objset) {
    // Object store: the raw write skipped the GC post-write barrier. If the value
    // is not a number it may be a nursery GC thing -> run the barrier (cheap helper;
    // numbers, the common deltablue case, skip it entirely).
    if (!WJSIsNum(e, kVSt1) || !e.writeOp(Op::I32Eqz) || !e.writeOp(Op::If) ||
        !e.writeFixedU8(kVoid)) {
      return false;
    }
    // LEAN call: the barrier never allocates-and-moves GC cells and never throws,
    // so skip the operand-stack spill/reload + exception check that WJVSCallHelper
    // does. Just stage obj/val and call; drop the (always-0) result.
    if (!WJVSStoreGlobal(e, helpA, kVSt0) || !WJVSStoreGlobal(e, helpB, kVSt1) ||
        !WJConst(e, double(WJH_POSTBARRIER)) || !WJConst(e, 0.0) ||
        !e.writeOp(Op::Call) || !e.writeVarU32(kWJVSHelpIdx) || !e.writeOp(Op::Drop)) {
      return false;
    }
    if (!e.writeOp(Op::End)) return false;
  }
  if (!e.writeOp(Op::Else)) return false;  // shape (or number, when !objset) miss
  if (!helperSet()) return false;
  if (!e.writeOp(Op::End)) return false;
  if (!e.writeOp(Op::Else)) return false;  // not object
  if (!helperSet()) return false;
  if (!e.writeOp(Op::End)) return false;
  // result (the assigned value, t1) -> operand stack[objS]
  if (!WJSStorePre(e, c, objS) || !e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt1) ||
      !WJSStorePost(e, c, objS)) {
    return false;
  }
  c.depth -= 1;
  return true;
}
// Restore the frame stack pointer + return `v` from the function (used for the
// EXC-propagate path of a call_indirect that returned 2.0).
static bool WJVSReturnVal(Encoder& e, double v) {
  uint32_t spAddr = uint32_t(uintptr_t(&gWJFrameSP));
  return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(spAddr)) &&
         e.writeOp(Op::LocalGet) && e.writeVarU32(kVSbasesp) &&
         e.writeOp(Op::I32Store) && e.writeVarU32(2) && e.writeVarU32(0) &&
         WJConst(e, v) && e.writeOp(Op::Return);
}
// UNBOX typed-array element size in bytes for a kind code.
static uint32_t WJElemSize(uint8_t kind) {
  switch (kind) {
    case kWJElemF64: return 8;
    case kWJElemI32: case kWJElemU32: case kWJElemF32: return 4;
    case kWJElemU16: case kWJElemI16: return 2;
    default: return 1;  // U8 / I8
  }
}
// Emit a raw typed-array element LOAD as an f64 (preconditions: kVSti = obj low32,
// kVSti2 = int index, both validated). addr = i32.load[obj+40](data) + index*size.
static bool WJEmitTypedLoad(Encoder& e, uint8_t kind) {
  uint32_t sz = WJElemSize(kind);
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) ||
      !e.writeVarU32(2) || !e.writeVarU32(40) || !e.writeOp(Op::LocalGet) ||
      !e.writeVarU32(kVSti2) || !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(sz)) ||
      !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add)) {
    return false;
  }
  switch (kind) {
    case kWJElemF64:
      return e.writeOp(Op::F64Load) && e.writeVarU32(3) && e.writeVarU32(0);
    case kWJElemF32:
      return e.writeOp(Op::F32Load) && e.writeVarU32(2) && e.writeVarU32(0) &&
             e.writeOp(Op::F64PromoteF32);
    case kWJElemI32:
      return e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0) &&
             e.writeOp(Op::F64ConvertI32S);
    case kWJElemU32:
      return e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0) &&
             e.writeOp(Op::F64ConvertI32U);
    case kWJElemU16:
      return e.writeOp(Op::I32Load16U) && e.writeVarU32(1) && e.writeVarU32(0) &&
             e.writeOp(Op::F64ConvertI32S);
    case kWJElemI16:
      return e.writeOp(Op::I32Load16S) && e.writeVarU32(1) && e.writeVarU32(0) &&
             e.writeOp(Op::F64ConvertI32S);
    case kWJElemU8:
      return e.writeOp(Op::I32Load8U) && e.writeVarU32(0) && e.writeVarU32(0) &&
             e.writeOp(Op::F64ConvertI32S);
    default:  // kWJElemI8
      return e.writeOp(Op::I32Load8S) && e.writeVarU32(0) && e.writeVarU32(0) &&
             e.writeOp(Op::F64ConvertI32S);
  }
}
// Emit a raw typed-array element STORE (preconditions: kVSti=obj, kVSti2=index,
// kVSt0 = the boxed number value). Integer kinds ToInt32 the value (the narrow
// store ops keep the low bits = JS ToUint8/ToInt16/... for in-range values).
static bool WJEmitTypedStore(Encoder& e, uint8_t kind) {
  uint32_t sz = WJElemSize(kind);
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) ||
      !e.writeVarU32(2) || !e.writeVarU32(40) || !e.writeOp(Op::LocalGet) ||
      !e.writeVarU32(kVSti2) || !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(sz)) ||
      !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add)) {
    return false;
  }
  if (kind == kWJElemF64) {
    return WJVUnboxNG(e, kVSt0) && e.writeOp(Op::F64Store) && e.writeVarU32(3) &&
           e.writeVarU32(0);
  }
  if (kind == kWJElemF32) {
    return WJVUnboxNG(e, kVSt0) && e.writeOp(Op::F32DemoteF64) &&
           e.writeOp(Op::F32Store) && e.writeVarU32(2) && e.writeVarU32(0);
  }
  // integer: ToInt32(value), then a width-appropriate store (low bits).
  if (!WJVUnboxNG(e, kVSt0) || !e.writeOp(MiscOp::I64TruncSatF64S) ||
      !e.writeOp(Op::I32WrapI64)) {
    return false;
  }
  switch (kind) {
    case kWJElemI32: case kWJElemU32:
      return e.writeOp(Op::I32Store) && e.writeVarU32(2) && e.writeVarU32(0);
    case kWJElemU16: case kWJElemI16:
      return e.writeOp(Op::I32Store16) && e.writeVarU32(1) && e.writeVarU32(0);
    default:  // U8 / I8
      return e.writeOp(Op::I32Store8) && e.writeVarU32(0) && e.writeVarU32(0);
  }
}
// All typed-array element kinds, in IC-chain order (first 7 are if-arms, last is
// the final else; the runtime kind is always one of these inside the typed path).
static const uint8_t kWJElemCodes[8] = {kWJElemF64, kWJElemI32, kWJElemU8,
                                        kWJElemI8,  kWJElemU16, kWJElemI16,
                                        kWJElemU32, kWJElemF32};
// Dense GetElem in Mode VS: object[int32] inline (shape+bounds guarded); else
// wjhelp. Stack [obj,index] -> value (result at the obj slot; depth -1).
static bool WJVSGetElem(Encoder& e, WJVSCtx& c, jsbytecode* pc) {
  const uint8_t kVoid = 0x40;
  if (gWJSiteCount >= kWJMaxSites) return false;
  uint32_t site = gWJSiteCount++;
  gWJSites[site].script = c.script;
  gWJSites[site].pcOff = uint32_t(pc - c.start);
  uint32_t icAddr = uint32_t(uintptr_t(&gWJICTable[2 * site]));
  uint32_t helpA = uint32_t(uintptr_t(&gWJHelpA));
  uint32_t helpB = uint32_t(uintptr_t(&gWJHelpB));
  uint32_t objS = c.stackBaseS + c.depth - 2, idxS = c.stackBaseS + c.depth - 1;
  auto helperGet = [&]() -> bool {
    return WJVSStoreGlobal(e, helpA, kVSt0) && WJVSStoreGlobal(e, helpB, kVSt1) &&
           WJVSCallHelper(e, c, WJH_GETELEM, site, c.depth - 2) && WJVSPushResult(e, c, objS);
  };
  if (!WJSLoad(e, c, objS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0) ||
      !WJSLoad(e, c, idxS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt1)) {
    return false;
  }
  uint32_t kindAddr = uint32_t(uintptr_t(&gWJElemKind[site]));
  const uint8_t kF64t = uint8_t(TypeCode::F64);
  // Box the loaded f64 into the boxed operand slot objS. The op has three runtime
  // arms (typed-array / dense Array / helper); the dense and helper arms produce a
  // boxed Value, so this typed-array arm must ALSO box -- leaving it unboxed in
  // sf[] (repr=1) while the other arms write s[] is a static/runtime repr mismatch
  // that leaks a stale sf[] slot (NaN) whenever the dense/helper arm runs at runtime.
  auto pushF64 = [&]() -> bool {
    return e.writeOp(Op::LocalSet) && e.writeVarU32(kVStf) && WJSStorePre(e, c, objS) &&
           e.writeOp(Op::LocalGet) && e.writeVarU32(kVStf) && WJVRebox(e, kVStf) &&
           WJSStorePost(e, c, objS);
  };
  if (!WJSIsObj(e, kVSt0)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt0) ||
      !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(kVSti)) {
    return false;
  }
  // TYPED-ARRAY path if the cached element kind is set (Float64/Int32). Elements are
  // raw -> shape guard + bounds (length@24) + raw load from data@40, no boxing.
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(kindAddr)) ||
      !e.writeOp(Op::I32Load8U) || !e.writeVarU32(0) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;  // kind != 0 -> typed
  {
    // shape guard
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(icAddr)) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Eq) ||
        !e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) {
      return false;
    }
    // index is a number?
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt1) || !e.writeOp(Op::I64Const) ||
        !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(kWJTagInt32)) || !e.writeOp(Op::I64LeU) ||
        !e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) {
      return false;
    }
    // ti2 = trunc(idx); bounds: ti2 u< length@24 (element count)
    if (!WJVUnboxNG(e, kVSt1) || !e.writeOp(MiscOp::I32TruncSatF64S) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSti2)) {
      return false;
    }
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti2) || !e.writeOp(Op::LocalGet) ||
        !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
        !e.writeVarU32(24) || !e.writeOp(Op::I32LtU) || !e.writeOp(Op::If) ||
        !e.writeFixedU8(kVoid)) {
      return false;
    }
    // N-way kind chain -> f64 element (kind is always one of kWJElemCodes here).
    for (int ci = 0; ci < 7; ci++) {
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(kindAddr)) ||
          !e.writeOp(Op::I32Load8U) || !e.writeVarU32(0) || !e.writeVarU32(0) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(kWJElemCodes[ci])) ||
          !e.writeOp(Op::I32Eq) || !e.writeOp(Op::If) || !e.writeFixedU8(kF64t)) {
        return false;
      }
      if (!WJEmitTypedLoad(e, kWJElemCodes[ci]) || !e.writeOp(Op::Else)) return false;
    }
    if (!WJEmitTypedLoad(e, kWJElemCodes[7])) return false;  // final else
    for (int ci = 0; ci < 7; ci++) {
      if (!e.writeOp(Op::End)) return false;
    }
    if (!pushF64()) return false;
    if (!e.writeOp(Op::Else) || !helperGet() || !e.writeOp(Op::End)) return false;  // oob
    if (!e.writeOp(Op::Else) || !helperGet() || !e.writeOp(Op::End)) return false;  // non-int
    if (!e.writeOp(Op::Else) || !helperGet() || !e.writeOp(Op::End)) return false;  // shape miss
  }
  if (!e.writeOp(Op::Else)) return false;  // kind == 0 -> DENSE Array path
  {
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(icAddr)) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Eq)) {
      return false;
    }
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt1) || !e.writeOp(Op::I64Const) ||
        !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(kWJTagInt32)) || !e.writeOp(Op::I64LeU)) {
      return false;
    }
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
    if (!WJVUnboxNG(e, kVSt1) || !e.writeOp(MiscOp::I32TruncSatF64S) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSti2)) {
      return false;
    }
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti2) || !e.writeOp(Op::LocalGet) ||
        !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
        !e.writeVarU32(12) || !e.writeOp(Op::I32Const) || !e.writeVarS32(12) ||
        !e.writeOp(Op::I32Sub) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
        !e.writeVarU32(0) || !e.writeOp(Op::I32LtU)) {
      return false;
    }
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
    if (!WJSStorePre(e, c, objS) || !e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(12) ||
        !e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti2) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add) ||
        !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0) ||
        !WJSStorePost(e, c, objS)) {
      return false;
    }
    if (!e.writeOp(Op::Else) || !helperGet() || !e.writeOp(Op::End)) return false;  // oob
    if (!e.writeOp(Op::Else) || !helperGet() || !e.writeOp(Op::End)) return false;  // non-int idx
    if (!e.writeOp(Op::Else) || !helperGet() || !e.writeOp(Op::End)) return false;  // shape miss
  }
  if (!e.writeOp(Op::End)) return false;  // end kind typed/dense
  if (!e.writeOp(Op::Else) || !helperGet() || !e.writeOp(Op::End)) return false;  // not object
  c.depth -= 1;
  return true;
}
// Dense SetElem in Mode VS: object[int32] = number inline (shape+bounds+number
// guarded); else wjhelp. Stack [obj,index,value] -> value (depth -2).
static bool WJVSSetElem(Encoder& e, WJVSCtx& c, jsbytecode* pc) {
  const uint8_t kVoid = 0x40;
  if (gWJSiteCount >= kWJMaxSites) return false;
  uint32_t site = gWJSiteCount++;
  gWJSites[site].script = c.script;
  gWJSites[site].pcOff = uint32_t(pc - c.start);
  uint32_t icAddr = uint32_t(uintptr_t(&gWJICTable[2 * site]));
  uint32_t helpA = uint32_t(uintptr_t(&gWJHelpA));
  uint32_t helpB = uint32_t(uintptr_t(&gWJHelpB));
  uint32_t helpC = uint32_t(uintptr_t(&gWJHelpC));
  uint32_t objS = c.stackBaseS + c.depth - 3, idxS = c.stackBaseS + c.depth - 2,
           valS = c.stackBaseS + c.depth - 1;
  // t0 = obj, t1 = index; the value stays in frame[valS] (loaded where needed).
  auto helperSet = [&]() -> bool {
    return WJVSStoreGlobal(e, helpA, kVSt0) && WJVSStoreGlobal(e, helpB, kVSt1) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(helpC)) &&
           WJSLoad(e, c, valS) && WJSStoreEnd(e) &&
           WJVSCallHelper(e, c, WJH_SETELEM, site, c.depth - 3);
  };
  if (!WJSLoad(e, c, objS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0) ||
      !WJSLoad(e, c, idxS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt1)) {
    return false;
  }
  if (!WJSIsObj(e, kVSt0)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt0) ||
      !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(kVSti)) {
    return false;
  }
  uint32_t kindAddr = uint32_t(uintptr_t(&gWJElemKind[site]));
  // TYPED-ARRAY store path if the cached element kind is set (raw f64/i32 store, no
  // box, no write barrier). val must be a number (else helper -> ToNumber).
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(kindAddr)) ||
      !e.writeOp(Op::I32Load8U) || !e.writeVarU32(0) || !e.writeVarU32(0) ||
      !e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) {
    return false;  // kind != 0 -> typed
  }
  {
    // cond = shape match & idx num & val num
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(icAddr)) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Eq)) {
      return false;
    }
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt1) || !e.writeOp(Op::I64Const) ||
        !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(kWJTagInt32)) || !e.writeOp(Op::I64LeU) ||
        !e.writeOp(Op::I32And)) {
      return false;
    }
    if (!WJSLoad(e, c, valS) || !e.writeOp(Op::I64Const) || !e.writeVarU64(32) ||
        !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(kWJTagInt32)) || !e.writeOp(Op::I64LeU) ||
        !e.writeOp(Op::I32And) || !e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) {
      return false;
    }
    if (!WJVUnboxNG(e, kVSt1) || !e.writeOp(MiscOp::I32TruncSatF64S) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSti2)) {
      return false;
    }
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti2) || !e.writeOp(Op::LocalGet) ||
        !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
        !e.writeVarU32(24) || !e.writeOp(Op::I32LtU) || !e.writeOp(Op::If) ||
        !e.writeFixedU8(kVoid)) {
      return false;
    }
    // val -> kVSt0 (obj no longer needed; no helper on this path), unbox
    if (!WJSLoad(e, c, valS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0)) {
      return false;
    }
    // N-way kind chain -> raw element store (kind is always one of kWJElemCodes).
    for (int ci = 0; ci < 7; ci++) {
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(kindAddr)) ||
          !e.writeOp(Op::I32Load8U) || !e.writeVarU32(0) || !e.writeVarU32(0) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(kWJElemCodes[ci])) ||
          !e.writeOp(Op::I32Eq) || !e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) {
        return false;
      }
      if (!WJEmitTypedStore(e, kWJElemCodes[ci]) || !e.writeOp(Op::Else)) return false;
    }
    if (!WJEmitTypedStore(e, kWJElemCodes[7])) return false;  // final else
    for (int ci = 0; ci < 7; ci++) {
      if (!e.writeOp(Op::End)) return false;
    }
    if (!e.writeOp(Op::Else) || !helperSet() || !e.writeOp(Op::End)) return false;  // oob
    if (!e.writeOp(Op::Else) || !helperSet() || !e.writeOp(Op::End)) return false;  // guard miss
  }
  if (!e.writeOp(Op::Else)) return false;  // kind == 0 -> DENSE Array path
  {
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(icAddr)) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Eq)) {
      return false;
    }
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt1) || !e.writeOp(Op::I64Const) ||
        !e.writeVarU64(32) || !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(kWJTagInt32)) || !e.writeOp(Op::I64LeU) ||
        !e.writeOp(Op::I32And)) {
      return false;
    }
    if (!WJSLoad(e, c, valS) || !e.writeOp(Op::I64Const) || !e.writeVarU64(32) ||
        !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(kWJTagInt32)) || !e.writeOp(Op::I64LeU) ||
        !e.writeOp(Op::I32And)) {
      return false;
    }
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
    if (!WJVUnboxNG(e, kVSt1) || !e.writeOp(MiscOp::I32TruncSatF64S) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSti2)) {
      return false;
    }
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti2) || !e.writeOp(Op::LocalGet) ||
        !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
        !e.writeVarU32(12) || !e.writeOp(Op::I32Const) || !e.writeVarS32(12) ||
        !e.writeOp(Op::I32Sub) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
        !e.writeVarU32(0) || !e.writeOp(Op::I32LtU)) {
      return false;
    }
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(12) || !e.writeOp(Op::LocalGet) ||
        !e.writeVarU32(kVSti2) || !e.writeOp(Op::I32Const) || !e.writeVarS32(8) ||
        !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add) || !WJSLoad(e, c, valS) ||
        !WJSStoreEnd(e)) {
      return false;
    }
    if (!e.writeOp(Op::Else) || !helperSet() || !e.writeOp(Op::End)) return false;  // oob
    if (!e.writeOp(Op::Else) || !helperSet() || !e.writeOp(Op::End)) return false;  // guard miss
  }
  if (!e.writeOp(Op::End)) return false;  // end kind typed/dense
  if (!e.writeOp(Op::Else) || !helperSet() || !e.writeOp(Op::End)) return false;  // not object
  // result (the assigned value) -> operand stack[objS]
  if (!WJSStorePre(e, c, objS) || !WJSLoad(e, c, valS) || !WJSStorePost(e, c, objS)) {
    return false;
  }
  c.depth -= 2;
  return true;
}
// Call in Mode VS. Marshals args/this into gWJScratch + callee into gWJHelpA;
// METHOD_INLINING Phase B: emit an inlined callee that has control flow as its own
// nested relooper (block $inlexit { loop { if(pc2==i){...} ... } }) using a SEPARATE
// dispatch local (kVSpc2) so the caller's kVSpc is untouched. `c` is the callee's
// sub-context (c.script/start = callee; localBaseS/rvalS in the caller's register file;
// c.depth on entry = the callee's operand-empty base). Return/RetRval store the result
// to calleeS and `br $inlexit`. Branch targets mirror the main emitter (br 1 = loop,
// br 2 = $inlexit) since the block/loop/if nesting depth is identical. WJCallInlinable
// has already validated the callee so emission here cannot structurally fail.
static bool WJEmitInlineCFG(Encoder& e, WJVSCtx& c, uint32_t calleeS) {
  const uint8_t kVoid = 0x40;
  const uint32_t c2Base = c.depth;  // operand-empty depth for the inlined callee
  jsbytecode* const start = c.script->code();
  jsbytecode* const end = c.script->codeEnd();
  const uint32_t len = uint32_t(end - start);
  std::vector<bool> isStart(len + 1, false);
  isStart[0] = true;
  for (jsbytecode* pc = start; pc < end; pc += GetBytecodeLength(pc)) {
    JSOp op = JSOp(*pc);
    uint32_t cur = uint32_t(pc - start);
    uint32_t ol = GetBytecodeLength(pc);
    if (IsJumpOpcode(op)) {
      int64_t tgt = int64_t(cur) + GET_JUMP_OFFSET(pc);
      if (tgt < 0 || tgt > len) return false;
      isStart[tgt] = true;
      if (cur + ol <= len) isStart[cur + ol] = true;
    } else if (op == JSOp::Return || op == JSOp::RetRval) {
      if (cur + ol <= len) isStart[cur + ol] = true;
    }
  }
  std::vector<int32_t> ofId(len + 1, -1);
  std::vector<uint32_t> blockOff;
  for (uint32_t o = 0; o <= len; o++) {
    if (isStart[o]) {
      ofId[o] = int32_t(blockOff.size());
      blockOff.push_back(o);
    }
  }
  uint32_t K = uint32_t(blockOff.size());
  if (K == 0 || K > 1024) return false;
  // result = c.rvalS -> calleeS (used by RetRval and the fall-off-end path).
  auto retRval = [&]() -> bool {
    return WJSStorePre(e, c, calleeS) && WJSLoad(e, c, c.rvalS) &&
           WJSStorePost(e, c, calleeS) && e.writeOp(Op::Br) && e.writeVarU32(2);
  };
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(kVSpc2)) {
    return false;
  }
  if (!e.writeOp(Op::Block) || !e.writeFixedU8(kVoid)) return false;  // $inlexit
  if (!e.writeOp(Op::Loop) || !e.writeFixedU8(kVoid)) return false;
  for (uint32_t i = 0; i < K; i++) {
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSpc2) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(i)) || !e.writeOp(Op::I32Eq) || !e.writeOp(Op::If) ||
        !e.writeFixedU8(kVoid)) {
      return false;
    }
    jsbytecode* pc = start + blockOff[i];
    c.depth = c2Base;
    bool terminated = false;
    while (pc < end && !terminated) {
      JSOp op = JSOp(*pc);
      uint32_t ol = GetBytecodeLength(pc);
      uint32_t cur = uint32_t(pc - start);
      if (c.depth >= kWJVSMaxStack) return false;
      if (IsJumpOpcode(op)) {
        uint32_t fall = cur + ol;
        int32_t tgtId = ofId[uint32_t(int64_t(cur) + GET_JUMP_OFFSET(pc))];
        int32_t fallId = (fall <= len) ? ofId[fall] : -1;
        if (op == JSOp::Goto) {
          if (tgtId < 0) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(tgtId) ||
              !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSpc2) ||
              !e.writeOp(Op::Br) || !e.writeVarU32(1)) {
            return false;
          }
        } else {
          if (tgtId < 0 || fallId < 0) return false;
          int32_t thenId = (op == JSOp::JumpIfTrue) ? tgtId : fallId;
          int32_t elseId = (op == JSOp::JumpIfTrue) ? fallId : tgtId;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid) ||
              !e.writeOp(Op::I32Const) || !e.writeVarS32(thenId) ||
              !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSpc2) ||
              !e.writeOp(Op::Else) || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(elseId) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(kVSpc2) || !e.writeOp(Op::End) || !e.writeOp(Op::Br) ||
              !e.writeVarU32(1)) {
            return false;
          }
        }
        terminated = true;
      } else if (op == JSOp::Return) {  // result = operand top -> calleeS; br $inlexit
        if (!WJSStorePre(e, c, calleeS) ||
            !WJSLoad(e, c, c.stackBaseS + c.depth - 1) || !WJSStorePost(e, c, calleeS) ||
            !e.writeOp(Op::Br) || !e.writeVarU32(2)) {
          return false;
        }
        terminated = true;
      } else if (op == JSOp::RetRval) {
        if (!retRval()) return false;
        terminated = true;
      } else if (WJIsCmp(op)) {
        jsbytecode* nx = pc + ol;
        if (nx >= end ||
            (JSOp(*nx) != JSOp::JumpIfFalse && JSOp(*nx) != JSOp::JumpIfTrue)) {
          return false;
        }
        if (c.unbox && !WJMaterializeAll(e, c)) return false;  // box cmp operands
        if (!WJVSCmp(e, c, op)) return false;
      } else {
        if (!WJEmitOpVS(e, pc, c)) return false;
        uint32_t nextOff = cur + ol;
        if (nextOff <= len && isStart[nextOff] && nextOff != blockOff[i]) {
          int32_t nid = ofId[nextOff];
          if (nid < 0) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(nid) ||
              !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSpc2) ||
              !e.writeOp(Op::Br) || !e.writeVarU32(1)) {
            return false;
          }
          terminated = true;
        }
      }
      pc += ol;
    }
    if (!terminated) {  // fell off the end of the bytecode -> result = rval
      if (!retRval()) return false;
    }
    if (!e.writeOp(Op::End)) return false;  // end if (pc2 == i)
  }
  if (!e.writeOp(Op::End) || !e.writeOp(Op::End)) return false;  // loop, $inlexit
  return true;
}

// cached callee runs via call_indirect (a Mode VS callee allocates its frame
// above this one -- the shared frame stack keeps both GC-traced). On a callee-
// guard miss, or a cached callee that could not finish in wasm (deopt 1; it did
// not mutate), the generic call helper runs it in the interpreter. A callee that
// THREW (deopt 2) propagates without re-running. Stack [callee,this,arg..] ->
// result (depth -= argc+1).
static bool WJVSCall(Encoder& e, WJVSCtx& c, jsbytecode* pc) {
  const uint8_t kVoid = 0x40;
  uint32_t argc = GET_ARGC(pc);
  if (argc > 32) return false;
  if (gWJSiteCount >= kWJMaxSites) return false;
  uint32_t site = gWJSiteCount++;
  gWJSites[site].script = c.script;
  gWJSites[site].pcOff = uint32_t(pc - c.start);
  gWJCallArgc[site] = argc;
  uint32_t scratchBase = uint32_t(uintptr_t(static_cast<void*>(gWJScratch)));
  uint32_t helpA = uint32_t(uintptr_t(&gWJHelpA));
  uint32_t callFnAddr = uint32_t(uintptr_t(&gWJCallFn[site]));
  uint32_t callIdxAddr = uint32_t(uintptr_t(&gWJCallHandle[site]));
  uint32_t calleeS = c.stackBaseS + c.depth - argc - 2;
  uint32_t thisS = c.stackBaseS + c.depth - argc - 1;
  uint32_t arg0S = c.stackBaseS + c.depth - argc;
  // Marshal args + this -> gWJScratch, callee -> gWJHelpA. (Re-emit before the
  // generic helper too, since a call_indirect callee may have clobbered scratch.)
  auto marshal = [&]() -> bool {
    for (uint32_t i = 0; i < argc; i++) {
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(scratchBase + i * 8)) || !WJSLoad(e, c, arg0S + i) ||
          !WJSStoreEnd(e)) {
        return false;
      }
    }
    return e.writeOp(Op::I32Const) &&
           e.writeVarS32(int32_t(scratchBase + kWJThisOff)) && WJSLoad(e, c, thisS) &&
           WJSStoreEnd(e) && WJVSStoreGlobal(e, helpA, kVSt0);
  };
  if (!WJSLoad(e, c, calleeS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0)) {
    return false;
  }
  // MATH INTRINSIC: a recorded Math.* native (gWJMathRec, keyed by script+pcOff) of matching
  // arity is emitted as the wasm f64 op, guarded by callee identity + numeric args; the generic
  // helper call is the miss/non-number fallback. Result replaces [callee,this,arg..] at calleeS.
  auto mathIt = WJMathInlineEnabled()
                    ? gWJMathRec.find(WJInlineKey(c.script, uint32_t(pc - c.start)))
                    : gWJMathRec.end();
  if (mathIt != gWJMathRec.end()) {
    uint32_t mop = mathIt->second.op;
    uint32_t mathFnLow = mathIt->second.fnLow;
    bool binary = (mop == WJM_MIN || mop == WJM_MAX);
    if (argc == (binary ? 2u : 1u)) {
      Op fop = mop == WJM_SQRT    ? Op::F64Sqrt
               : mop == WJM_FLOOR ? Op::F64Floor
               : mop == WJM_CEIL  ? Op::F64Ceil
               : mop == WJM_ABS   ? Op::F64Abs
               : mop == WJM_TRUNC ? Op::F64Trunc
               : mop == WJM_MIN   ? Op::F64Min
                                  : Op::F64Max;
      // arg0 -> kVSt0 (overwrites callee; reloaded for the generic fallback), arg1 -> kVSt1.
      if (!WJSLoad(e, c, arg0S) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0)) return false;
      if (binary && (!WJSLoad(e, c, arg0S + 1) || !e.writeOp(Op::LocalSet) ||
                     !e.writeVarU32(kVSt1))) {
        return false;
      }
      // guard: low32(callee) == baked Math native ptr && args are numbers
      if (!WJSLoad(e, c, calleeS) || !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(mathFnLow)) || !e.writeOp(Op::I32Eq)) {
        return false;
      }
      if (!WJSIsNum(e, kVSt0) || !e.writeOp(Op::I32And)) return false;
      if (binary && (!WJSIsNum(e, kVSt1) || !e.writeOp(Op::I32And))) return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
      // FAST: result = box(fop(unbox arg0[, unbox arg1]))
      if (!WJSStorePre(e, c, calleeS) || !WJVUnboxNG(e, kVSt0)) return false;
      if (binary && !WJVUnboxNG(e, kVSt1)) return false;
      if (!e.writeOp(fop) || !WJVRebox(e, kVStf) || !WJSStorePost(e, c, calleeS)) return false;
      if (!e.writeOp(Op::Else)) return false;
      // SLOW: generic call helper (reload callee -> kVSt0 for marshal).
      if (!WJSLoad(e, c, calleeS) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0)) return false;
      if (!marshal() || !WJVSCallHelper(e, c, WJH_CALL, site, c.depth - argc - 2) ||
          !WJVSPushResult(e, c, calleeS)) {
        return false;
      }
      if (!e.writeOp(Op::End)) return false;
      gWJMathInline++;
      if (getenv("GECKO_DEBUG_JIT")) {
        fprintf(stderr, "[math-inline] %s:%u op=%u argc=%u\n",
                c.script->filename() ? c.script->filename() : "?",
                unsigned(c.script->lineno()), mop, argc);
        fflush(stderr);
      }
      c.depth -= argc + 1;
      return true;
    }
  }
  // METHOD_INLINING (Phase A, gated): if this site has a recorded monomorphic callee
  // that is a small straight-line 0-local leaf, inline its body where the call_indirect
  // would be. Guard low32(callee)==the inlined fn (baked) -> HIT runs the inline body
  // (no marshal/frame/call); MISS -> generic WJH_CALL. Reads args/this from the caller's
  // operand slots in place.
  // METHOD_INLINING (gated GECKO_WJVS_INLINE): inline recorded callee(s) guarded by
  // callee identity; the final else is the generic call_indirect/helper. POLYMORPHIC:
  // up to 4 callees -> a guarded if/else-if chain (richards `this.task.run`, 4 task
  // types). NON-LEAF: a body may itself call (the inner call, guarded by !c.inlined,
  // emits as a normal call). !c.inlined keeps inlining to depth 1.
  struct WJCand { JSScript* cs; uint32_t low32; bool cf; };
  WJCand cands[4];
  int nc = 0;
  if (gWJEmitInline && WJVSInline() && c.inlineDepth < WJMaxInlineDepth() &&
      !getenv("GECKO_WJVS_NOEMIT")) {
    auto it = gWJInlineCallee.find(WJInlineKey(c.script, uint32_t(pc - c.start)));
    if (it != gWJInlineCallee.end()) {
      const WJInlineRec& rec = it->second;
      uint8_t maxWays = WJNoPolyInline() ? 1 : 4;
      for (uint8_t i = 0; i < rec.n && nc < int(maxWays); i++) {
        JSFunction* fun = reinterpret_cast<JSFunction*>(uintptr_t(rec.fns[i]));
        if (!fun || !fun->isInterpreted() || !fun->baseScript() ||
            !fun->baseScript()->hasBytecode()) {
          continue;
        }
        JSScript* cs = fun->baseScript()->asJSScript();
        bool cf = false;
        // Budget: caller stack + callee (locals + rval + its own stack) <= register file.
        if (cs != c.script && fun->nargs() == argc &&
            c.depth + cs->nslots() + 1 <= kWJVSMaxStack && WJCallInlinable(cs, &cf) &&
            !(cf && getenv("GECKO_WJVS_NOCF"))) {  // GECKO_WJVS_NOCF: straight-line only
          cands[nc].cs = cs;
          cands[nc].low32 = rec.fns[i];
          cands[nc].cf = cf;
          nc++;
        }
      }
    }
  }
  if (nc > 0) {
    const uint64_t undefBits = 0xFFFFFF83ULL << 32;
    // Emit one callee's body in the caller's register file (no guard). The callee's
    // locals/rval/operand-stack occupy slots above the caller's stack; GetArg/
    // FunctionThis read the caller's marshalled arg/this slots in place; the result is
    // written to calleeS. c.depth is NOT modified (decremented once after the chain).
    auto emitBody = [&](JSScript* inlineScript, bool inlineHasCF) -> bool {
      uint32_t cNfixed = inlineScript->nfixed();
      WJVSCtx c2 = c;
      c2.script = inlineScript;
      c2.start = inlineScript->code();
      c2.nargs = argc;
      c2.nfixed = cNfixed;
      c2.inlined = true;
      c2.inlineDepth = c.inlineDepth + 1;
      c2.inlineArgBase = arg0S;
      c2.inlineThis = thisS;
      c2.localBaseS = c.stackBaseS + c.depth;
      c2.rvalS = c2.localBaseS + cNfixed;
      c2.depth = c.depth + cNfixed + 1;
      for (uint32_t j = 0; j <= cNfixed; j++) {
        if (!WJSStorePre(e, c2, c2.localBaseS + j) || !e.writeOp(Op::I64Const) ||
            !e.writeVarS64(int64_t(undefBits)) ||
            !WJSStorePost(e, c2, c2.localBaseS + j)) {
          return false;
        }
      }
      if (!inlineHasCF) {
        bool emittedReturn = false;
        for (jsbytecode* ip = inlineScript->code(); ip < inlineScript->codeEnd();) {
          JSOp iop = JSOp(*ip);
          uint32_t il = GetBytecodeLength(ip);
          if (c2.depth >= kWJVSMaxStack) return false;
          if (iop == JSOp::Return) {
            if (!WJSStorePre(e, c, calleeS) ||
                !WJSLoad(e, c2, c2.stackBaseS + c2.depth - 1) ||
                !WJSStorePost(e, c, calleeS)) {
              return false;
            }
            c2.depth--;
            emittedReturn = true;
            break;
          } else if (iop == JSOp::RetRval) {
            if (!WJSStorePre(e, c, calleeS) || !WJSLoad(e, c2, c2.rvalS) ||
                !WJSStorePost(e, c, calleeS)) {
              return false;
            }
            emittedReturn = true;
            break;
          } else if (!WJEmitOpVS(e, ip, c2)) {
            return false;
          }
          ip += il;
        }
        if (!emittedReturn) {
          if (!WJSStorePre(e, c, calleeS) || !e.writeOp(Op::I64Const) ||
              !e.writeVarS64(int64_t(undefBits)) || !WJSStorePost(e, c, calleeS)) {
            return false;
          }
        }
        return true;
      }
      return WJEmitInlineCFG(e, c2, calleeS);  // Phase B: callee with control flow
    };
    gWJInlinedCalls++;
    if (getenv("GECKO_WJVS_INLINE_LOG")) {
      fprintf(stderr, "[wj-inline] caller=%s:%u depth=%u argc=%u nc=%d callees=",
              c.script->filename() ? c.script->filename() : "?",
              uint32_t(c.script->lineno()), uint32_t(c.inlineDepth), argc, nc);
      for (int z = 0; z < nc; z++) {
        fprintf(stderr, "%u%s", uint32_t(cands[z].cs->lineno()), z + 1 < nc ? "," : "\n");
      }
      fflush(stderr);
    }
    // Guarded chain: if(callee==fn0){body0} else if(callee==fn1){body1} ... else {call}.
    for (int i = 0; i < nc; i++) {
      if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt0) ||
          !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(cands[i].low32)) || !e.writeOp(Op::I32Eq) ||
          !e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) {
        return false;
      }
      if (!emitBody(cands[i].cs, cands[i].cf)) return false;
      if (!e.writeOp(Op::Else)) return false;
    }
    // FALL THROUGH to the generic call_indirect path below as the innermost `else`,
    // so a compiled-but-not-inlined callee (e.g. HandlerTask.run, too large to inline)
    // dispatches wasm->wasm via call_indirect instead of crossing to the C++ WJH_CALL
    // helper. The nc guard `else` blocks are closed after the call result is pushed.
  }
  if (!marshal()) return false;
  // POLYMORPHIC dispatch: load callee low32 into kVSti2, then build the selected
  // call_indirect table index in kVSti via an unrolled N-way chain:
  //   handle = (low32(callee) == fn_w) ? handle_w : handle   (init -1 = no match)
  // A real table index is >= 0, so -1 cleanly means "no cached way matched".
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSt0) ||
      !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(kVSti2)) {
    return false;
  }
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(-1) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(kVSti)) {
    return false;
  }
  uint32_t nways = WJNoPolyCall() ? 1 : kWJCallWays;
  for (uint32_t w = 0; w < nways; w++) {
    uint32_t fnA = w == 0 ? callFnAddr
                          : uint32_t(uintptr_t(&gWJCallFnX[(w - 1) * kWJMaxSites + site]));
    uint32_t hA = w == 0 ? callIdxAddr
                         : uint32_t(uintptr_t(&gWJCallHandleX[(w - 1) * kWJMaxSites + site]));
    // (a=handle_w) (b=cur) (cond=low32==fn_w) select -> kVSti
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(hA)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) ||
        !e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti2) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(fnA)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::I32Eq) || !e.writeOp(Op::SelectNumeric) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSti)) {
      return false;
    }
  }
  // guard: a cached way matched (selected handle != -1)
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(-1) || !e.writeOp(Op::I32Ne)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  // cached hit: spill the live BYSTANDER stack (args/this/callee are already in
  // gWJScratch/gWJHelpA, which WJTraceRoots also traces), then call_indirect
  // (typeidx 0, tableidx 0) with the selected handle and reload (picks up moved
  // pointers).
  if (!WJVSSpillRange(e, c, c.depth - argc - 2)) return false;
  if (!WJConst(e, double(scratchBase)) || !e.writeOp(Op::LocalGet) ||
      !e.writeVarU32(kVSti) || !e.writeOp(Op::CallIndirect) ||
      !e.writeVarU32(0) || !e.writeVarU32(0) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(kVStf)) {
    return false;
  }
  if (!WJVSReloadRange(e, c, c.depth - argc - 2)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVStf) || !WJConst(e, 0.0) ||
      !e.writeOp(Op::F64Ne)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;  // deopt != 0
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVStf) || !WJConst(e, 2.0) ||
      !e.writeOp(Op::F64Eq)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;  // == 2: propagate
  if (!WJVSReturnVal(e, 2.0) || !e.writeOp(Op::End)) return false;
  // deopt 1: callee did not mutate -> generic call (re-marshal first)
  if (!marshal() || !WJVSCallHelper(e, c, WJH_CALL, site, c.depth - argc - 2)) return false;
  if (!e.writeOp(Op::End)) return false;  // end deopt != 0
  if (!e.writeOp(Op::Else)) return false;  // callee guard miss
  if (!WJVSCallHelper(e, c, WJH_CALL, site, c.depth - argc - 2)) return false;
  if (!e.writeOp(Op::End)) return false;  // end guard if
  if (!WJVSPushResult(e, c, calleeS)) return false;
  // Close the nc inline-guard `else` blocks (0 when not a partially-inlined site).
  for (int i = 0; i < nc; i++) {
    if (!e.writeOp(Op::End)) return false;
  }
  c.depth -= (argc + 1);
  return true;
}
// FunctionThis in Mode VS: push the `this` binding. Strict -> the passed value;
// sloppy object -> as-is; sloppy primitive/undefined -> wjhelp (BoxNonStrictThis).
static bool WJVSFunctionThis(Encoder& e, WJVSCtx& c) {
  const uint8_t kVoid = 0x40;
  uint32_t top = c.stackBaseS + c.depth;
  // Inlined callee: `this` is the caller's receiver slot (a method receiver is an
  // object -> use as-is, like the strict/object-receiver case).
  if (c.inlined) {
    if (!WJSStorePre(e, c, top) || !WJSLoad(e, c, c.inlineThis) ||
        !WJSStorePost(e, c, top)) {
      return false;
    }
    c.depth++;
    return true;
  }
  uint32_t thisAddr = uint32_t(uintptr_t(static_cast<void*>(gWJScratch))) + kWJThisOff;
  uint32_t helpA = uint32_t(uintptr_t(&gWJHelpA));
  // t0 = gWJScratch[this]
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(thisAddr)) ||
      !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0)) {
    return false;
  }
  if (c.script->strict()) {
    if (!WJSStorePre(e, c, top) || !e.writeOp(Op::LocalGet) ||
        !e.writeVarU32(kVSt0) || !WJSStorePost(e, c, top)) {
      return false;
    }
  } else {
    if (!WJSIsObj(e, kVSt0)) return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
    if (!WJSStorePre(e, c, top) || !e.writeOp(Op::LocalGet) ||
        !e.writeVarU32(kVSt0) || !WJSStorePost(e, c, top)) {
      return false;
    }
    if (!e.writeOp(Op::Else)) return false;
    if (!WJVSStoreGlobal(e, helpA, kVSt0) ||
        !WJVSCallHelper(e, c, WJH_FUNCTIONTHIS, 0, c.depth) || !WJVSPushResult(e, c, top)) {
      return false;
    }
    if (!e.writeOp(Op::End)) return false;
    // PTRUNBOX: a sloppy-mode `this` is ALWAYS an object (the spec ToObject's it; the helper
    // returns a wrapper object too). Cache its unboxed i32 ptr in pLoc[top] so the many
    // `this.field` GetProp/SetProp/Call receivers skip the per-access load+isObject+wrap. The
    // boxed Value stays live in s[top], so every other consumer (and materialize) is unaffected.
    if (WJPtrUnbox() && c.unbox) {
      if (!WJSLoad(e, c, top) || !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::LocalSet) ||
          !e.writeVarU32(WJPLoc(c.depth))) {
        return false;
      }
      c.repr[c.depth] = 2;
    }
  }
  c.depth++;
  return true;
}
// ===== UNBOX: typed (f64) operand-stack fast path (GECKO_WJVS_UNBOX) ==========
// When c.unbox, numeric operand-stack entries may live UNBOXED as f64 in the
// parallel locals sf[d]=kVSsBaseF+d (repr[d]==1) instead of NaN-boxed i64 in
// s[d]=kVSsBase+d (repr[d]==0). Numeric ops then flow f64->f64 with no per-op
// box/unbox/type-guard. Box/unbox happens only at boundaries via materialize.
// Requires WJVSUseLocals() (operand stack in registers) and !c.inlined.

// Box the f64 in sf[d] back to s[d] (or frame) and mark Boxed. Clears repr[d].
static bool WJMaterialize(Encoder& e, WJVSCtx& c, uint32_t d) {
  if (d >= kWJVSMaxStack || c.repr[d] == 0) return true;
  if (c.repr[d] == 2) {  // PTRUNBOX: the boxed object Value already lives in s[d]
    c.repr[d] = 0;       // (pLoc[d] is just a cached ptr hint); drop the hint, value is valid
    return true;
  }
  if (c.repr[d] == 3) {  // INTUNBOX: box the raw i32 in iLoc[d] -> int32 Value in s[d]
    uint32_t slot = c.stackBaseS + d;
    if (!WJSStorePre(e, c, slot) || !e.writeOp(Op::LocalGet) ||
        !e.writeVarU32(WJILoc(d)) || !e.writeOp(Op::I64ExtendI32U) ||
        !e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(kWJTagInt32 << 32)) ||
        !e.writeOp(Op::I64Or) || !WJSStorePost(e, c, slot)) {
      return false;
    }
    c.repr[d] = 0;
    return true;
  }
  uint32_t slot = c.stackBaseS + d;
  if (!WJSStorePre(e, c, slot) || !e.writeOp(Op::LocalGet) ||
      !e.writeVarU32(WJFLoc(d)) || !WJVRebox(e, kVStf) || !WJSStorePost(e, c, slot)) {
    return false;
  }
  c.repr[d] = 0;
  return true;
}
// Box every live F64 entry, then clear the whole repr map (dead slots reset to
// Boxed so a later boxed push sees repr==0). Called before any non-typed op.
static bool WJMaterializeAll(Encoder& e, WJVSCtx& c) {
  for (uint32_t d = 0; d < c.depth && d < kWJVSMaxStack; d++) {
    if (!WJMaterialize(e, c, d)) return false;
  }
  for (uint32_t d = 0; d < kWJVSMaxStack; d++) c.repr[d] = 0;
  return true;
}
// Ensure the entry at depth d is an unboxed f64 in sf[d]. If currently boxed:
// sf[d] = isNum(s[d]) ? unbox(s[d]) : ToNumber(s[d]) (the exact operand coercion
// for arithmetic/bitwise ops). The ToNumber slow path spills+reloads the whole
// operand stack (F64 bystanders survive in their wasm locals; boxed ones are
// traced/updated by a moving GC). Leaves repr[d]==1.
static bool WJEnsureF64(Encoder& e, WJVSCtx& c, uint32_t d) {
  const uint8_t kF64 = uint8_t(TypeCode::F64);
  if (c.repr[d] == 1) return true;
  if (c.repr[d] == 3) {  // INTUNBOX: raw i32 in iLoc[d] -> f64 in sf[d]
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(WJILoc(d)) ||
        !e.writeOp(Op::F64ConvertI32S) || !e.writeOp(Op::LocalSet) ||
        !e.writeVarU32(WJFLoc(d))) {
      return false;
    }
    c.repr[d] = 1;
    return true;
  }
  uint32_t sLoc = WJVSLocalFor(c, c.stackBaseS + d);
  uint32_t helpA = uint32_t(uintptr_t(&gWJHelpA));
  uint32_t resAddr = uint32_t(uintptr_t(&gWJScratch[kWJResultSlot]));
  if (!WJSIsNum(e, sLoc) || !e.writeOp(Op::If) || !e.writeFixedU8(kF64)) return false;
  if (!WJVUnboxNG(e, sLoc)) return false;
  if (!e.writeOp(Op::Else)) return false;
  if (!WJVSStoreGlobal(e, helpA, sLoc) ||
      !WJVSCallHelper(e, c, WJH_TONUMBER, 0, c.depth)) {
    return false;
  }
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(resAddr)) ||
      !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0) || !WJVUnboxNG(e, kVSt0)) {
    return false;
  }
  if (!e.writeOp(Op::End) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(WJFLoc(d))) {
    return false;
  }
  c.repr[d] = 1;
  return true;
}
// Push a constant f64 as a new F64 stack entry (numeric literal; no boxing).
static bool WJPushF64Const(WJVSCtx& c, Encoder& e, double v) {
  uint32_t d = c.depth;
  if (d >= kWJVSMaxStack) return false;
  if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(v) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(WJFLoc(d))) {
    return false;
  }
  c.repr[d] = 1;
  c.depth++;
  return true;
}
// Typed binary op with pure-ToNumber operands (Sub/Mul/Div): sf[a] = sf[a] fop sf[b].
static bool WJVSBinArithU(Encoder& e, WJVSCtx& c, Op fop) {
  uint32_t dA = c.depth - 2, dB = c.depth - 1;
  if (!WJEnsureF64(e, c, dA) || !WJEnsureF64(e, c, dB)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(WJFLoc(dA)) ||
      !e.writeOp(Op::LocalGet) || !e.writeVarU32(WJFLoc(dB)) || !e.writeOp(fop) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(WJFLoc(dA))) {
    return false;
  }
  c.repr[dA] = 1;
  c.depth--;
  return true;
}
// Typed bitwise/shift: sf[a] = convert( ToInt32(sf[a]) i32op ToInt32(sf[b]) ).
static bool WJVSBitOpU(Encoder& e, WJVSCtx& c, Op i32op, bool uns) {
  uint32_t dA = c.depth - 2, dB = c.depth - 1;
  // INTUNBOX: signed bitwise stays in the i32 register file -- iLoc[a] = iLoc[a] op
  // iLoc[b], result repr=3 (a valid int32). No f64 round-trip, so chained int ops
  // (richards' flag dispatch) never leave registers. >>> (uns) excluded: result can
  // exceed INT32_MAX and must become a double.
  if (WJIntUnbox() && !uns) {
    if (!WJEnsureI32(e, c, dA) || !WJEnsureI32(e, c, dB)) return false;
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(WJILoc(dA)) ||
        !e.writeOp(Op::LocalGet) || !e.writeVarU32(WJILoc(dB)) ||
        !e.writeOp(i32op) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(WJILoc(dA))) {
      return false;
    }
    c.repr[dA] = 3;
    c.depth--;
    return true;
  }
  if (!WJEnsureF64(e, c, dA) || !WJEnsureF64(e, c, dB)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(WJFLoc(dA)) ||
      !e.writeOp(MiscOp::I64TruncSatF64S) || !e.writeOp(Op::I32WrapI64) ||
      !e.writeOp(Op::LocalGet) || !e.writeVarU32(WJFLoc(dB)) ||
      !e.writeOp(MiscOp::I64TruncSatF64S) || !e.writeOp(Op::I32WrapI64) ||
      !e.writeOp(i32op) ||
      !e.writeOp(uns ? Op::F64ConvertI32U : Op::F64ConvertI32S) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(WJFLoc(dA))) {
    return false;
  }
  c.repr[dA] = 1;
  c.depth--;
  return true;
}
// Typed unary inc/dec.
static bool WJVSUnaryU(Encoder& e, WJVSCtx& c, bool inc) {
  uint32_t d = c.depth - 1;
  if (!WJEnsureF64(e, c, d)) return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(WJFLoc(d)) || !WJConst(e, 1.0) ||
      !e.writeOp(inc ? Op::F64Add : Op::F64Sub) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(WJFLoc(d))) {
    return false;
  }
  c.repr[d] = 1;
  return true;
}

// One non-control-flow op in Mode VS. The operand stack is in wasm locals (or the
// frame, under GECKO_WJVS_FRAME); args/locals/rval are always in the frame.
static bool WJEmitOpVSInner(Encoder& e, jsbytecode* pc, WJVSCtx& c) {
  // PTRUNBOX: capture whether the GetProp receiver is a statically-known object ptr (repr==2,
  // e.g. sloppy `this`) BEFORE any materialize clears the repr flag. pLoc[depth-1] survives
  // MaterializeAll (it only touches repr + f64 regs), so WJVSGetProp can use the cached ptr.
  bool recvPtr = WJPtrUnbox() && c.unbox && JSOp(*pc) == JSOp::GetProp && c.depth > 0 &&
                 c.repr[c.depth - 1] == 2;
  // PTRUNBOX: a non-pure op (call/store/arith/cmp -> possible GC safepoint or this-shape change)
  // invalidates the hoisted this-shape; only consecutive pure reads keep it valid.
  if (JSOp(*pc) != JSOp::GetProp && !WJCseTransparent(JSOp(*pc))) c.thisShapeCached = false;
  // UNBOX dispatch: numeric ops use the typed f64 stack; all other ops first
  // materialize (box) any live F64 entries so the boxed stack is authoritative.
  if (c.unbox) {
    int umask = WJUnboxMask();
    switch (JSOp(*pc)) {
      case JSOp::Pop:  // discard top; no need to box an F64 just to drop it
        if ((umask & 32) && c.depth > 0) { c.repr[c.depth - 1] = 0; c.depth--; return true; }
        break;
      case JSOp::GetArg: {  // typed numeric arg -> push F64 from lf[]
        uint32_t fs = GET_ARGNO(pc);
        if (WJSlotTyped(c, fs)) {
          uint32_t d = c.depth;
          if (d >= kWJVSMaxStack) return false;
          if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSsBaseLF + fs) ||
              !e.writeOp(Op::LocalSet) || !e.writeVarU32(WJFLoc(d))) {
            return false;
          }
          c.repr[d] = 1;
          c.depth++;
          return true;
        }
        break;
      }
      case JSOp::GetLocal: {  // typed numeric local -> push F64 from lf[]
        uint32_t fs = c.localBaseS + GET_LOCALNO(pc);
        if (WJSlotTyped(c, fs)) {
          uint32_t d = c.depth;
          if (d >= kWJVSMaxStack) return false;
          if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSsBaseLF + fs) ||
              !e.writeOp(Op::LocalSet) || !e.writeVarU32(WJFLoc(d))) {
            return false;
          }
          c.repr[d] = 1;
          c.depth++;
          return true;
        }
        break;
      }
      case JSOp::SetLocal: {  // tee: store F64 to lf[], value stays on stack
        uint32_t fs = c.localBaseS + GET_LOCALNO(pc);
        if (WJSlotTyped(c, fs)) {
          uint32_t d = c.depth - 1;
          if (!WJEnsureF64(e, c, d) || !e.writeOp(Op::LocalGet) ||
              !e.writeVarU32(WJFLoc(d)) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(kVSsBaseLF + fs)) {
            return false;
          }
          return true;
        }
        break;
      }
      case JSOp::SetArg: {
        uint32_t fs = GET_ARGNO(pc);
        if (WJSlotTyped(c, fs)) {
          uint32_t d = c.depth - 1;
          if (!WJEnsureF64(e, c, d) || !e.writeOp(Op::LocalGet) ||
              !e.writeVarU32(WJFLoc(d)) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(kVSsBaseLF + fs)) {
            return false;
          }
          return true;
        }
        break;
      }
      case JSOp::Zero: if (umask & 1) return WJPushF64Const(c, e, 0.0); break;
      case JSOp::One: if (umask & 1) return WJPushF64Const(c, e, 1.0); break;
      case JSOp::Int8: if (umask & 1) return WJPushF64Const(c, e, double(int32_t(GET_INT8(pc)))); break;
      case JSOp::Uint16: if (umask & 1) return WJPushF64Const(c, e, double(GET_UINT16(pc))); break;
      case JSOp::Uint24: if (umask & 1) return WJPushF64Const(c, e, double(GET_UINT24(pc))); break;
      case JSOp::Int32: if (umask & 1) return WJPushF64Const(c, e, double(GET_INT32(pc))); break;
      case JSOp::Double: {
        if (umask & 1) {
          double d;
          memcpy(&d, pc + 1, sizeof(double));
          return WJPushF64Const(c, e, d);
        }
        break;
      }
      case JSOp::Sub: if (umask & 2) return WJVSBinArithU(e, c, Op::F64Sub); break;
      case JSOp::Mul: if (umask & 2) return WJVSBinArithU(e, c, Op::F64Mul); break;
      case JSOp::Div: if (umask & 2) return WJVSBinArithU(e, c, Op::F64Div); break;
      case JSOp::BitOr: if (umask & 4) return WJVSBitOpU(e, c, Op::I32Or, false); break;
      case JSOp::BitAnd: if (umask & 4) return WJVSBitOpU(e, c, Op::I32And, false); break;
      case JSOp::BitXor: if (umask & 4) return WJVSBitOpU(e, c, Op::I32Xor, false); break;
      case JSOp::Lsh: if (umask & 4) return WJVSBitOpU(e, c, Op::I32Shl, false); break;
      case JSOp::Rsh: if (umask & 4) return WJVSBitOpU(e, c, Op::I32ShrS, false); break;
      case JSOp::Ursh: if (umask & 4) return WJVSBitOpU(e, c, Op::I32ShrU, true); break;
      case JSOp::Inc: if (umask & 8) return WJVSUnaryU(e, c, true); break;
      case JSOp::Dec: if (umask & 8) return WJVSUnaryU(e, c, false); break;
      case JSOp::Add: {  // numeric add only when both already F64 (no string risk)
        uint32_t dA = c.depth - 2, dB = c.depth - 1;
        if ((umask & 16) && c.repr[dA] == 1 && c.repr[dB] == 1) {
          if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(WJFLoc(dA)) ||
              !e.writeOp(Op::LocalGet) || !e.writeVarU32(WJFLoc(dB)) ||
              !e.writeOp(Op::F64Add) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(WJFLoc(dA))) {
            return false;
          }
          c.repr[dA] = 1;
          c.depth--;
          return true;
        }
        break;  // mixed/boxed -> materialize + boxed Add below
      }
      case JSOp::GetProp: {
        // PHASE 2a: if this field read is immediately consumed by a numeric op, emit the
        // boxed GetProp then convert the result straight onto the typed f64 stack (repr=1)
        // so the consumer finds it unboxed. Sound (consumer ToNumber-coerces it anyway).
        if (WJVSTypedField()) {
          jsbytecode* nx = pc + GetBytecodeLength(pc);
          bool consumed = (nx < c.script->codeEnd() && WJIsNumericConsumer(JSOp(*nx))) ||
                          WJFieldNumConsumed(c, pc);
          if (consumed) {
            if (!WJMaterializeAll(e, c)) return false;  // boxed inputs for GetProp
            if (!WJVSGetProp(e, c, pc, recvPtr)) return false;   // boxed result, repr=0, depth++
            if (!WJEnsureF64(e, c, c.depth - 1)) return false;  // -> f64 typed stack, repr=1
            gWJTypedFieldHits++;
            return true;
          }
        }
        break;  // -> materialize + boxed GetProp below
      }
      case JSOp::GetElem: {
        // TYPEDELEM: same as TYPEDFIELD for array/typed-array element reads consumed numerically
        // (crypto/navier are array+arithmetic dense). Boxed GetElem -> typed f64 stack.
        if (WJVSTypedField()) {
          jsbytecode* nx = pc + GetBytecodeLength(pc);
          bool consumed = (nx < c.script->codeEnd() && WJIsNumericConsumer(JSOp(*nx))) ||
                          WJFieldNumConsumed(c, pc);
          if (consumed) {
            if (!WJMaterializeAll(e, c)) return false;
            if (!WJVSGetElem(e, c, pc)) return false;  // boxed result (depth -1), repr=0
            if (!WJEnsureF64(e, c, c.depth - 1)) return false;  // -> f64 typed stack, repr=1
            gWJTypedFieldHits++;
            return true;
          }
        }
        break;  // -> materialize + boxed GetElem below
      }
      default: break;
    }
    if (!WJMaterializeAll(e, c)) return false;
  }
  uint32_t top = c.stackBaseS + c.depth;  // next free stack slot
  switch (JSOp(*pc)) {
    case JSOp::GetArg: {
      // Inlined callee: arg k is the caller's marshalled operand slot inlineArgBase+k.
      uint32_t argSlot = c.inlined ? c.inlineArgBase + GET_ARGNO(pc) : GET_ARGNO(pc);
      if (!WJSStorePre(e, c, top) || !WJSLoad(e, c, argSlot) ||
          !WJSStorePost(e, c, top)) {
        return false;
      }
      c.depth++;
      return true;
    }
    case JSOp::GetLocal:
      if (!WJSStorePre(e, c, top) || !WJSLoad(e, c, c.localBaseS + GET_LOCALNO(pc)) ||
          !WJSStorePost(e, c, top)) {
        return false;
      }
      c.depth++;
      return true;
    case JSOp::SetLocal: {  // tee: store tos to local, leave it
      // Route through Pre/Post so an inlined callee's register-resident locals
      // are written via local.set, matching GetLocal's register-aware load.
      uint32_t dst = c.localBaseS + GET_LOCALNO(pc);
      return WJSStorePre(e, c, dst) && WJSLoad(e, c, c.stackBaseS + c.depth - 1) &&
             WJSStorePost(e, c, dst);
    }
    case JSOp::SetArg: {
      uint32_t dst = c.inlined ? c.inlineArgBase + GET_ARGNO(pc) : GET_ARGNO(pc);
      return WJSStorePre(e, c, dst) && WJSLoad(e, c, c.stackBaseS + c.depth - 1) &&
             WJSStorePost(e, c, dst);
    }
    case JSOp::Zero:
    case JSOp::One:
    case JSOp::Int8:
    case JSOp::Int32:
    case JSOp::Uint16:
    case JSOp::Uint24: {
      int32_t v = JSOp(*pc) == JSOp::Zero     ? 0
                  : JSOp(*pc) == JSOp::One    ? 1
                  : JSOp(*pc) == JSOp::Int8   ? GET_INT8(pc)
                  : JSOp(*pc) == JSOp::Int32  ? GET_INT32(pc)
                  : JSOp(*pc) == JSOp::Uint16 ? int32_t(GET_UINT16(pc))
                                              : int32_t(GET_UINT24(pc));
      uint64_t bits = (kWJTagInt32 << 32) | uint32_t(v);
      if (!WJSStorePre(e, c, top) || !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(int64_t(bits)) || !WJSStorePost(e, c, top)) {
        return false;
      }
      c.depth++;
      return true;
    }
    case JSOp::Null:
    case JSOp::Undefined:
    case JSOp::True:
    case JSOp::False:
    case JSOp::Double: {
      uint64_t bits;
      if (JSOp(*pc) == JSOp::Null) {
        bits = 0xFFFFFF84ULL << 32;
      } else if (JSOp(*pc) == JSOp::Undefined) {
        bits = 0xFFFFFF83ULL << 32;
      } else if (JSOp(*pc) == JSOp::True) {
        bits = (0xFFFFFF82ULL << 32) | 1;
      } else if (JSOp(*pc) == JSOp::False) {
        bits = (0xFFFFFF82ULL << 32);
      } else {
        double d;
        memcpy(&d, pc + 1, sizeof(double));
        memcpy(&bits, &d, sizeof(double));
        if (d != d) bits = kWJCanonNaN;
      }
      if (!WJSStorePre(e, c, top) || !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(int64_t(bits)) || !WJSStorePost(e, c, top)) {
        return false;
      }
      c.depth++;
      return true;
    }
    case JSOp::String: {  // push a string literal (atom; baked constant must be immovable)
      JSString* s = c.script->getString(pc);
      if (!s->isAtom()) return false;
      uint64_t bits = (uint64_t(kWJTagString) << 32) | uint32_t(uintptr_t(s));
      if (!WJSStorePre(e, c, top) || !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(int64_t(bits)) || !WJSStorePost(e, c, top)) {
        return false;
      }
      c.depth++;
      return true;
    }
    case JSOp::Pop:
      c.depth--;
      return true;
    case JSOp::Dup:
      if (!WJSStorePre(e, c, top) || !WJSLoad(e, c, c.stackBaseS + c.depth - 1) ||
          !WJSStorePost(e, c, top)) {
        return false;
      }
      c.depth++;
      return true;
    case JSOp::Add: return WJVSBinArith(e, c, Op::F64Add, WJH_ADD);
    case JSOp::Sub: return WJVSBinArith(e, c, Op::F64Sub, WJH_SUB);
    case JSOp::Mul: return WJVSBinArith(e, c, Op::F64Mul, WJH_MUL);
    case JSOp::Div: return WJVSBinArith(e, c, Op::F64Div, WJH_DIV);
    case JSOp::Inc: return WJVSUnary(e, c, true);
    case JSOp::Dec: return WJVSUnary(e, c, false);
    case JSOp::BitOr: return WJVSBitOp(e, c, Op::I32Or, WJH_BITOR, false);
    case JSOp::BitAnd: return WJVSBitOp(e, c, Op::I32And, WJH_BITAND, false);
    case JSOp::BitXor: return WJVSBitOp(e, c, Op::I32Xor, WJH_BITXOR, false);
    case JSOp::Lsh: return WJVSBitOp(e, c, Op::I32Shl, WJH_LSH, false);
    case JSOp::Rsh: return WJVSBitOp(e, c, Op::I32ShrS, WJH_RSH, false);
    case JSOp::Ursh: return WJVSBitOp(e, c, Op::I32ShrU, WJH_URSH, true);
    case JSOp::BitNot: return WJVSBitNot(e, c);
    case JSOp::GetProp: return WJVSGetProp(e, c, pc, recvPtr);
    case JSOp::GetGName: return WJVSGetGName(e, c, pc);
    case JSOp::GetAliasedVar: return WJVSGetAliased(e, c, pc);
    case JSOp::NewObject:
    case JSOp::NewInit: return WJVSNewObject(e, c, pc);
    case JSOp::InitProp: return WJVSInitProp(e, c, pc);
    case JSOp::SetProp:
    case JSOp::StrictSetProp: return WJVSSetProp(e, c, pc);
    case JSOp::GetElem: return WJVSGetElem(e, c, pc);
    case JSOp::SetElem:
    case JSOp::StrictSetElem: return WJVSSetElem(e, c, pc);
    case JSOp::Call:
    case JSOp::CallContent:
    case JSOp::CallIgnoresRv: return WJVSCall(e, c, pc);
    case JSOp::FunctionThis: return WJVSFunctionThis(e, c);
    case JSOp::Swap: {  // [..,x,y] -> [..,y,x]
      uint32_t s2 = c.stackBaseS + c.depth - 2, s1 = c.stackBaseS + c.depth - 1;
      return WJSLoad(e, c, s2) && e.writeOp(Op::LocalSet) && e.writeVarU32(kVSt0) &&
             WJSStorePre(e, c, s2) && WJSLoad(e, c, s1) && WJSStorePost(e, c, s2) &&
             WJSStorePre(e, c, s1) && e.writeOp(Op::LocalGet) &&
             e.writeVarU32(kVSt0) && WJSStorePost(e, c, s1);
    }
    case JSOp::SetRval:
      if (!WJSStorePre(e, c, c.rvalS) || !WJSLoad(e, c, c.stackBaseS + c.depth - 1) ||
          !WJSStorePost(e, c, c.rvalS)) {
        return false;
      }
      c.depth--;
      return true;
    case JSOp::GetRval:
      if (!WJSStorePre(e, c, top) || !WJSLoad(e, c, c.rvalS) || !WJSStorePost(e, c, top)) {
        return false;
      }
      c.depth++;
      return true;
    case JSOp::Nop:
    case JSOp::NopIsAssignOp:
    case JSOp::NopDestructuring:
    case JSOp::JumpTarget:
    case JSOp::LoopHead:
    case JSOp::Pos:
    case JSOp::ToNumeric:
      return true;
    default:
      return false;  // unsupported in Mode VS (Stage 1)
  }
}

// PHASE 2b: CSE wrapper around WJEmitOpVSInner. For the `x.f` pattern (GetLocal/GetArg
// then GetProp) within one basic block, a repeated read reuses kVScse instead of
// re-emitting the shape-guard + slot-load. Correctness: the cache is cleared before ANY
// op that is not WJCseTransparent (mutation / reassignment / possible GC), so the cached
// boxed value (possibly an object pointer in kVScse, untraced) is never moved or stale
// between store and reuse. Restricted to the boxed (non-unbox) path and to top-level
// (non-inlined) bodies to avoid f64-repr and sub-context interactions.
// FIELDPROMO: multi-entry field read-cache (scalar replacement). A GetProp on a known receiver
// slot reuses a cached wasm local instead of re-emitting shape-guard+slot-load; the cache is
// invalidated at stores/calls/alloc (which may change a field or run GC). Works under unbox and
// inlined. Cleared at block boundaries (within-block for now; cross-block is the inlined-monolith
// extension). Only boxed (repr 0) results are cached; numeric (repr 1) results stay on the f64 stack.
static bool WJEmitOpVSFieldPromo(Encoder& e, jsbytecode* pc, WJVSCtx& c) {
  JSOp op = JSOp(*pc);
  int32_t recv = c.cseLastSlot;  // receiver source slot iff prev op was GetArg/GetLocal
  uint32_t field = (op == JSOp::GetProp) ? uint32_t(uintptr_t(c.script->getName(pc))) : 0;
  if (op == JSOp::GetProp && c.depth > 0 && recv >= 0) {  // HIT?
    for (uint32_t i = 0; i < kWJFieldPromoN; i++) {
      if (c.fcRecv[i] == recv && c.fcField[i] == field) {
        uint32_t topD = c.depth - 1;  // receiver slot -> result (net-0 depth)
        if (c.fcRepr[i] == 1) {  // numeric: reuse the cached f64 on the typed stack
          if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSfcBaseF + i) ||
              !e.writeOp(Op::LocalSet) || !e.writeVarU32(WJFLoc(topD))) {
            return false;
          }
          c.repr[topD] = 1;
        } else {  // boxed object/value
          uint32_t topSlot = c.stackBaseS + topD;
          if (!WJSStorePre(e, c, topSlot) || !e.writeOp(Op::LocalGet) ||
              !e.writeVarU32(kVSfcBase + i) || !WJSStorePost(e, c, topSlot)) {
            return false;
          }
          if (c.unbox) c.repr[topD] = 0;
        }
        c.cseLastSlot = -1;
        gWJCseHits++;
        return true;
      }
    }
  }
  // Invalidation: a named-property store invalidates only THAT field (the inline fast path does a
  // raw store with no GC; a miss deopts the whole fn, so cached pointers can't go stale on the wasm
  // path). Everything that may GC or change arbitrary fields -- calls, alloc, SetElem (dynamic idx),
  // env/global stores -- clears the whole cache.
  switch (op) {
    case JSOp::SetProp: case JSOp::StrictSetProp: {
      uint32_t wf = uint32_t(uintptr_t(c.script->getName(pc)));
      for (uint32_t i = 0; i < kWJFieldPromoN; i++) if (c.fcField[i] == wf) c.fcRecv[i] = -1;
      break;
    }
    case JSOp::SetElem: case JSOp::StrictSetElem: case JSOp::InitProp: case JSOp::InitElem:
    case JSOp::SetName: case JSOp::StrictSetName: case JSOp::SetGName: case JSOp::StrictSetGName:
    case JSOp::SetAliasedVar:
    case JSOp::Call: case JSOp::CallContent: case JSOp::CallIgnoresRv: case JSOp::CallContentIter:
    case JSOp::New: case JSOp::SuperCall:
      for (uint32_t i = 0; i < kWJFieldPromoN; i++) c.fcRecv[i] = -1;
      break;
    default: break;
  }
  bool cacheThis = (op == JSOp::GetProp && recv >= 0);
  uint32_t dBefore = c.depth;
  if (!WJEmitOpVSInner(e, pc, c)) return false;
  int32_t newLastFP = -1;
  if (c.depth == dBefore + 1) {
    if (op == JSOp::GetArg) {
      newLastFP = int32_t(c.inlined ? c.inlineArgBase + GET_ARGNO(pc) : GET_ARGNO(pc));
    } else if (op == JSOp::GetLocal) {
      newLastFP = int32_t(c.localBaseS + GET_LOCALNO(pc));
    } else if (op == JSOp::FunctionThis) {
      // `this` is a stable receiver within a frame -> track it so `this.field` reads promote.
      newLastFP = c.inlined ? int32_t(c.inlineThis) : kWJThisRecvSentinel;
    }
  }
  c.cseLastSlot = newLastFP;
  if (cacheThis && c.depth == dBefore && c.depth > 0) {  // cache the result (numeric f64 or boxed)
    uint32_t slot = kWJFieldPromoN;
    for (uint32_t i = 0; i < kWJFieldPromoN; i++) if (c.fcRecv[i] < 0) { slot = i; break; }
    if (slot == kWJFieldPromoN) slot = field % kWJFieldPromoN;  // evict
    uint32_t topD = c.depth - 1;
    if (c.unbox && c.repr[topD] == 1) {  // numeric f64 result -> f64 cache (GC-safe: no pointer)
      if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(WJFLoc(topD)) ||
          !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSfcBaseF + slot)) {
        return false;
      }
      c.fcRepr[slot] = 1;
    } else {  // boxed result
      if (!WJSLoad(e, c, c.stackBaseS + topD) || !e.writeOp(Op::LocalSet) ||
          !e.writeVarU32(kVSfcBase + slot)) {
        return false;
      }
      c.fcRepr[slot] = 0;
    }
    c.fcRecv[slot] = recv; c.fcField[slot] = field;
  }
  return true;
}
static bool WJEmitOpVS(Encoder& e, jsbytecode* pc, WJVSCtx& c) {
  if (c.useFieldPromo) return WJEmitOpVSFieldPromo(e, pc, c);
  if (!c.useCSE || c.inlined || c.unbox) return WJEmitOpVSInner(e, pc, c);
  JSOp op = JSOp(*pc);
  int32_t recvSlot = c.cseLastSlot;  // receiver source slot iff prev op was a slot load
  if (op == JSOp::GetProp && c.depth > 0 && c.cseValid && recvSlot >= 0 &&
      c.cseRecvSlot == recvSlot &&
      c.cseField == uint32_t(uintptr_t(c.script->getName(pc)))) {
    uint32_t topSlot = c.stackBaseS + c.depth - 1;  // receiver slot -> becomes the result
    if (!WJSStorePre(e, c, topSlot) || !e.writeOp(Op::LocalGet) ||
        !e.writeVarU32(kVScse) || !WJSStorePost(e, c, topSlot)) {
      return false;
    }
    c.cseLastSlot = -1;
    gWJCseHits++;
    return true;
  }
  bool cacheThis = (op == JSOp::GetProp && recvSlot >= 0);
  uint32_t field = (op == JSOp::GetProp) ? uint32_t(uintptr_t(c.script->getName(pc))) : 0;
  if (op != JSOp::GetProp && !WJCseTransparent(op)) c.cseValid = false;
  uint32_t dBefore = c.depth;
  if (!WJEmitOpVSInner(e, pc, c)) return false;
  int32_t newLast = -1;
  if (c.depth == dBefore + 1) {
    if (op == JSOp::GetArg) {
      newLast = int32_t(c.inlined ? c.inlineArgBase + GET_ARGNO(pc) : GET_ARGNO(pc));
    } else if (op == JSOp::GetLocal) {
      newLast = int32_t(c.localBaseS + GET_LOCALNO(pc));
    }
  }
  c.cseLastSlot = newLast;
  if (cacheThis && c.depth == dBefore && c.depth > 0) {  // GetProp: net-0 depth, result on top
    uint32_t topSlot = c.stackBaseS + c.depth - 1;
    if (!WJSLoad(e, c, topSlot) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVScse)) {
      return false;
    }
    c.cseValid = true;
    c.cseRecvSlot = recvSlot;
    c.cseField = field;
  }
  return true;
}

// GECKO_WJVS_TYPEDLOC=1 (requires UNBOX): keep numeric-only arg/local slots
// unboxed as f64 across the whole function (no frame box). A/B / bring-up gate.
static bool WJTypedLoc() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_NOTYPEDLOC") ? 0 : 1;  // default ON (correct as of 2026-06-18)
  return v != 0;
}
// UNBOX typed-locals analysis. Returns a bitmask of frame slots (arg k -> bit k,
// local k -> bit nargs+k) that are NUMERIC-ONLY and SOUND to keep unboxed: a slot
// qualifies iff it is consumed ONLY by numeric ops and NEVER (a) used as an object
// (property/element receiver, call callee/this) nor (b) escapes with identity
// (returned, stored as a property/element value, or passed as a call argument) --
// because such a slot is only ever observed via ToNumber, so coercing it to f64 at
// every store is semantically safe (boxing back yields the same Number). Provenance
// is simulated per basic block (empty operand stack at block starts). Any unmodeled
// op or stack overflow -> taint everything seen so far is unsafe -> bail to 0.
static uint64_t WJAnalyzeNumericSlots(JSScript* script, uint32_t nargs,
                                      uint32_t nfixed,
                                      const std::vector<bool>& isStart) {
  uint32_t nslot = nargs + nfixed;
  if (nslot == 0 || nslot > 64 || nslot > kWJVSMaxTLocals) return 0;
  // Typed ARGS are UNSOUND in general: a slot is typed from how it is USED, but an arg's VALUE
  // comes from the caller and need not be a number. Even restricting to "every use ToNumbers it"
  // is not enough -- entry coercion is EAGER (ToNumber once at entry vs the interpreter's per-use,
  // differing on throw/valueOf side effects and across-block uses the straight-line provenance
  // can't see), which miscompiles e.g. octane-typescript (`this.checker is null`). LOCALS are
  // sound (a typed local's value is a PROVEN number: every def is -1). So by DEFAULT type LOCALS
  // ONLY; GECKO_WJVS_TYPEDARGS=1 re-enables typed args (faster crypto ~+10%) for experiments.
  static bool typedArgs = []{ const char* e = getenv("GECKO_WJVS_TYPEDARGS"); return e && atoi(e); }();
  int tldrop = 0;  // debug: force an op-category to -2 (bit0 arith,1 unary,2 const,3 Add,4 args)
  if (const char* e = getenv("GECKO_WJVS_TLDROP")) tldrop = atoi(e);
  jsbytecode* const start = script->code();
  jsbytecode* const end = script->codeEnd();
  uint64_t tainted = 0;
  auto taint = [&](int32_t s) { if (s >= 0 && s < 64) tainted |= (uint64_t(1) << s); };
  // An ARG slot is typed based on how it is USED, but its VALUE comes from the caller and
  // may not be a number. Typing it as f64 (entry ToNumber-coerced) is observationally
  // equivalent ONLY when every use genuinely ToNumbers it (arithmetic). Uses where the
  // boxed-back value's type/identity is observable are UNSOUND for a non-number arg:
  // truthiness (NaN is falsy, an object is truthy -> wrong branch), a copy to another slot
  // (the copy boxes the f64, losing the object), and (strict)equality (identity differs).
  // So taint an arg consumed that way. LOCALS are exempt: a typed local's value is a PROVEN
  // number (every def is -1), so boxing it back / its NaN-falsiness / copies are all correct.
  auto taintIfArg = [&](int32_t v) { if (v >= 0 && uint32_t(v) < nargs) taint(v); };
  std::vector<int32_t> st;  // -1 = NUM, -2 = OTHER, >=0 = copy of frame slot s
  auto pop = [&]() -> int32_t {
    if (st.empty()) return -2;
    int32_t v = st.back();
    st.pop_back();
    return v;
  };
  auto push = [&](int32_t v) { st.push_back(v); };
  // Fixpoint: `tainted` only grows; re-run the walk until it stabilizes so a slot whose
  // taint-causing def follows its use (across a back-edge) is still caught (the Add numeric
  // check reads `tainted`, so an under-taint in one pass is corrected in the next).
  if (tldrop & 16) { for (uint32_t a = 0; a < nargs && a < 64; a++) taint(int32_t(a)); }
  uint64_t prevTainted = ~uint64_t(0);
  for (int iter = 0; tainted != prevTainted && iter <= 64; iter++) {
    prevTainted = tainted;
    st.clear();
  for (jsbytecode* pc = start; pc < end;) {
    uint32_t cur = uint32_t(pc - start);
    if (isStart[cur]) st.clear();  // block boundary: stack empty (WJStackSafe)
    JSOp op = JSOp(*pc);
    uint32_t ol = GetBytecodeLength(pc);
    switch (op) {
      case JSOp::GetArg: push(int32_t(GET_ARGNO(pc))); break;
      case JSOp::GetLocal: push(int32_t(nargs + GET_LOCALNO(pc))); break;
      case JSOp::Zero: case JSOp::One: case JSOp::Int8: case JSOp::Int32:
      case JSOp::Uint16: case JSOp::Uint24: case JSOp::Double:
        push((tldrop & 4) ? -2 : -1); break;
      case JSOp::Sub: case JSOp::Mul: case JSOp::Div:
      case JSOp::Mod: case JSOp::Pow: case JSOp::BitOr: case JSOp::BitAnd:
      case JSOp::BitXor: case JSOp::Lsh: case JSOp::Rsh: case JSOp::Ursh:
        pop(); pop(); push((tldrop & 1) ? -2 : -1); break;  // ToNumber both operands -> Number
      case JSOp::Add: {
        // `+` is NUMERIC add only if BOTH operands are provably numbers; otherwise it may be
        // STRING concatenation (or object valueOf/toString), so the result is NOT provably a
        // number. Marking it -1 unconditionally was unsound: a local assigned a string `+`
        // result got typed f64 and corrupted (typescript `this.checker`). An operand is provably
        // numeric if it is a NUM (-1) or a frame-slot copy that is not (yet) tainted.
        int32_t r = pop(), l = pop();
        auto provablyNum = [&](int32_t v) {
          return v == -1 || (v >= 0 && v < 64 && !(tainted & (uint64_t(1) << v)));
        };
        push(((tldrop & 8) == 0 && provablyNum(l) && provablyNum(r)) ? -1 : -2);
        break;
      }
      case JSOp::Inc: case JSOp::Dec: case JSOp::Neg: case JSOp::Pos:
      case JSOp::BitNot: case JSOp::ToNumeric:
        pop(); push((tldrop & 2) ? -2 : -1); break;
      case JSOp::Lt: case JSOp::Le: case JSOp::Gt: case JSOp::Ge:
        pop(); pop(); push(-2); break;  // relational ToNumbers/ToPrimitive both -> arg-safe
      case JSOp::Eq: case JSOp::Ne: case JSOp::StrictEq: case JSOp::StrictNe: {
        int32_t b = pop(), a = pop();  // (strict)eq is identity/type-sensitive -> unsafe for an arg
        taintIfArg(a); taintIfArg(b); push(-2); break;
      }
      case JSOp::GetProp: { taint(pop()); push(-2); break; }     // receiver = object
      case JSOp::GetElem: { pop(); taint(pop()); push(-2); break; }  // [recv][idx]; idx numeric
      case JSOp::SetProp: case JSOp::StrictSetProp: {
        int32_t v = pop(); taint(pop()); taint(v); push(v); break;  // value escapes; recv object
      }
      case JSOp::SetElem: case JSOp::StrictSetElem: {
        int32_t v = pop(); pop(); taint(pop()); taint(v); push(v); break;
      }
      case JSOp::Dup: push(st.empty() ? -2 : st.back()); break;
      case JSOp::Pop: pop(); break;
      case JSOp::Swap:
        if (st.size() >= 2) std::swap(st[st.size() - 1], st[st.size() - 2]);
        break;
      // tee: leaves the value. A slot is numeric only if EVERY def is numeric too
      // -- not just every use. If the stored value is not provably a number (-1),
      // taint the destination: coercing it to f64 at store would lose the real
      // (e.g. object) value, which then leaks through copies (`next = peek`) or any
      // later boxed read. Leave the value on the stack with its original provenance
      // so downstream uses still taint the true source slot.
      case JSOp::SetLocal: {
        int32_t dst = int32_t(nargs + GET_LOCALNO(pc));
        int32_t src = st.empty() ? -2 : st.back();
        if (src != -1) taint(dst);
        taintIfArg(src);  // copy boxes the source's f64; a typed non-number arg would corrupt
        break;
      }
      case JSOp::SetArg: {
        int32_t dst = int32_t(GET_ARGNO(pc));
        int32_t src = st.empty() ? -2 : st.back();
        if (src != -1) taint(dst);
        taintIfArg(src);
        break;
      }
      case JSOp::SetRval: taint(pop()); break;       // escapes (return value)
      case JSOp::GetRval: push(-2); break;
      case JSOp::Return: taint(pop()); break;        // escapes
      case JSOp::RetRval: break;
      case JSOp::JumpIfTrue: case JSOp::JumpIfFalse:
        taintIfArg(pop()); break;  // truthiness: ToNumber(obj)=NaN is falsy but obj is truthy
      case JSOp::Goto: case JSOp::JumpTarget: case JSOp::LoopHead:
      case JSOp::Nop: case JSOp::NopDestructuring: case JSOp::Lineno:
      case JSOp::DebugCheckSelfHosted: break;
      case JSOp::Call: case JSOp::CallContent: case JSOp::CallIgnoresRv:
      case JSOp::CallContentIter: case JSOp::New: case JSOp::SuperCall: {
        uint32_t argc = GET_ARGC(pc);
        for (uint32_t k = 0; k < argc; k++) taint(pop());  // args escape with identity
        taint(pop());  // this/newTarget
        taint(pop());  // callee
        push(-2);
        break;
      }
      case JSOp::FunctionThis: case JSOp::GlobalThis: case JSOp::GetGName:
      case JSOp::GetAliasedVar:
      case JSOp::Null: case JSOp::Undefined: case JSOp::True: case JSOp::False:
      case JSOp::String: case JSOp::CallSiteObj: case JSOp::Object:
        push(-2); break;
      default:
        return 0;  // unmodeled op -> bail (no typed locals; safe)
    }
    pc += ol;
  }
  }  // fixpoint loop
  uint64_t all = (nslot >= 64) ? ~uint64_t(0) : ((uint64_t(1) << nslot) - 1);
  uint64_t result = all & ~tainted;
  if (!typedArgs) {  // soundness: never type ARG slots (bits [0,nargs)) unless opted in
    uint64_t argMask = (nargs >= 64) ? ~uint64_t(0) : ((uint64_t(1) << nargs) - 1);
    result &= ~argMask;
  }
  return result;
}

// Mode VS body emitter: relooper dispatcher over a GC-traced frame-memory stack.
// ============================ Mode VS — SSA IR (Phase A) =====================
// A small block-local SSA value graph for the straight-line regions of a Mode VS body.
// Phase A builds it for analysis and lowers each region back to wasm by delegating to the
// per-op emitter (parity by construction); Phases B-D will read this graph (def/use + the
// type lattice) and make the lowerer node-aware to elide redundant guards/loads/boxes.
enum class WJTy : uint8_t {
  Top = 0, Int32, Double, Number, Boolean, Null, Undef, String, Object, Value
};
enum class WJIROp : uint8_t {
  ConstInt, ConstDouble, ConstBool, ConstNull, ConstUndef, ConstString,
  GetArg, GetLocal, GetRval, SetLocal, SetArg, SetRval,
  GetProp, GetGName, GetAliased, SetProp, GetElem, SetElem,
  Add, Sub, Mul, Div, Inc, Dec, BitOr, BitAnd, BitXor, Lsh, Rsh, Ursh, BitNot,
  Call, FunctionThis, Pop, Dup, Swap, Other
};
struct WJIRNode {
  WJIROp op;
  jsbytecode* pc;
  int16_t in0 = -1, in1 = -1, in2 = -1;  // operand value-ids (SSA), -1 = none
  int16_t result = -1;                   // value-id produced, -1 = none
  uint32_t aux = 0;                       // arg/local slot, field-name low32, or argc
  WJTy ty = WJTy::Value;
  int16_t reuseOf = -1;     // PHASE B: this GetProp reuses node[reuseOf]'s cached result
  uint8_t cacheSlot = 0xFF; // PHASE B: this node's result is captured to GVN frame slot N
};
struct WJIRValue {
  int16_t def = -1;  // defining node index (-1 = block-live-in, e.g. a Dup source)
  WJTy ty = WJTy::Value;
};
struct WJIRRegion {
  std::vector<WJIRNode> nodes;
  std::vector<WJIRValue> values;
  bool opaque = false;  // graph tracking stopped at an unmodeled op (lowering unaffected)
};
// Per-op classification for the value-graph builder. Returns false for an op the graph
// does not model (the region is still lowered correctly, the graph is just truncated).
struct WJIRClass {
  WJIROp op;
  uint8_t pops, pushes;  // operand-stack effect for the value graph
  uint32_t aux;
  WJTy ty;
};
static bool WJIRClassify(jsbytecode* pc, WJIRClass& k) {
  JSOp op = JSOp(*pc);
  k.aux = 0;
  switch (op) {
    case JSOp::Zero: case JSOp::One: case JSOp::Int8: case JSOp::Int32:
    case JSOp::Uint16: case JSOp::Uint24:
      k.op = WJIROp::ConstInt; k.pops = 0; k.pushes = 1; k.ty = WJTy::Int32; return true;
    case JSOp::Double:
      k.op = WJIROp::ConstDouble; k.pops = 0; k.pushes = 1; k.ty = WJTy::Double; return true;
    case JSOp::True: case JSOp::False:
      k.op = WJIROp::ConstBool; k.pops = 0; k.pushes = 1; k.ty = WJTy::Boolean; return true;
    case JSOp::Null:
      k.op = WJIROp::ConstNull; k.pops = 0; k.pushes = 1; k.ty = WJTy::Null; return true;
    case JSOp::Undefined:
      k.op = WJIROp::ConstUndef; k.pops = 0; k.pushes = 1; k.ty = WJTy::Undef; return true;
    case JSOp::String:
      k.op = WJIROp::ConstString; k.pops = 0; k.pushes = 1; k.ty = WJTy::String; return true;
    case JSOp::GetArg:
      k.op = WJIROp::GetArg; k.pops = 0; k.pushes = 1; k.aux = GET_ARGNO(pc); k.ty = WJTy::Value; return true;
    case JSOp::GetLocal:
      k.op = WJIROp::GetLocal; k.pops = 0; k.pushes = 1; k.aux = GET_LOCALNO(pc); k.ty = WJTy::Value; return true;
    case JSOp::GetRval:
      k.op = WJIROp::GetRval; k.pops = 0; k.pushes = 1; k.ty = WJTy::Value; return true;
    case JSOp::SetLocal:
      k.op = WJIROp::SetLocal; k.pops = 0; k.pushes = 0; k.aux = GET_LOCALNO(pc); k.ty = WJTy::Value; return true;
    case JSOp::SetArg:
      k.op = WJIROp::SetArg; k.pops = 0; k.pushes = 0; k.aux = GET_ARGNO(pc); k.ty = WJTy::Value; return true;
    case JSOp::SetRval:
      k.op = WJIROp::SetRval; k.pops = 1; k.pushes = 0; k.ty = WJTy::Value; return true;
    case JSOp::Pop:
      k.op = WJIROp::Pop; k.pops = 1; k.pushes = 0; k.ty = WJTy::Value; return true;
    case JSOp::Dup:
      k.op = WJIROp::Dup; k.pops = 0; k.pushes = 1; k.ty = WJTy::Value; return true;
    case JSOp::Swap:
      k.op = WJIROp::Swap; k.pops = 0; k.pushes = 0; k.ty = WJTy::Value; return true;
    case JSOp::Add: k.op = WJIROp::Add; k.pops = 2; k.pushes = 1; k.ty = WJTy::Value; return true;
    case JSOp::Sub: k.op = WJIROp::Sub; k.pops = 2; k.pushes = 1; k.ty = WJTy::Number; return true;
    case JSOp::Mul: k.op = WJIROp::Mul; k.pops = 2; k.pushes = 1; k.ty = WJTy::Number; return true;
    case JSOp::Div: k.op = WJIROp::Div; k.pops = 2; k.pushes = 1; k.ty = WJTy::Number; return true;
    case JSOp::Inc: k.op = WJIROp::Inc; k.pops = 1; k.pushes = 1; k.ty = WJTy::Number; return true;
    case JSOp::Dec: k.op = WJIROp::Dec; k.pops = 1; k.pushes = 1; k.ty = WJTy::Number; return true;
    case JSOp::BitOr: k.op = WJIROp::BitOr; k.pops = 2; k.pushes = 1; k.ty = WJTy::Int32; return true;
    case JSOp::BitAnd: k.op = WJIROp::BitAnd; k.pops = 2; k.pushes = 1; k.ty = WJTy::Int32; return true;
    case JSOp::BitXor: k.op = WJIROp::BitXor; k.pops = 2; k.pushes = 1; k.ty = WJTy::Int32; return true;
    case JSOp::Lsh: k.op = WJIROp::Lsh; k.pops = 2; k.pushes = 1; k.ty = WJTy::Int32; return true;
    case JSOp::Rsh: k.op = WJIROp::Rsh; k.pops = 2; k.pushes = 1; k.ty = WJTy::Int32; return true;
    case JSOp::Ursh: k.op = WJIROp::Ursh; k.pops = 2; k.pushes = 1; k.ty = WJTy::Number; return true;
    case JSOp::BitNot: k.op = WJIROp::BitNot; k.pops = 1; k.pushes = 1; k.ty = WJTy::Int32; return true;
    case JSOp::GetProp:
      k.op = WJIROp::GetProp; k.pops = 1; k.pushes = 1; k.ty = WJTy::Value; return true;
    case JSOp::GetGName:
      k.op = WJIROp::GetGName; k.pops = 0; k.pushes = 1; k.ty = WJTy::Value; return true;
    case JSOp::GetAliasedVar:
      k.op = WJIROp::GetAliased; k.pops = 0; k.pushes = 1; k.ty = WJTy::Value; return true;
    case JSOp::SetProp: case JSOp::StrictSetProp:
      k.op = WJIROp::SetProp; k.pops = 2; k.pushes = 1; k.ty = WJTy::Value; return true;
    case JSOp::GetElem:
      k.op = WJIROp::GetElem; k.pops = 2; k.pushes = 1; k.ty = WJTy::Value; return true;
    case JSOp::SetElem: case JSOp::StrictSetElem:
      k.op = WJIROp::SetElem; k.pops = 3; k.pushes = 1; k.ty = WJTy::Value; return true;
    case JSOp::FunctionThis:
      k.op = WJIROp::FunctionThis; k.pops = 0; k.pushes = 1; k.ty = WJTy::Object; return true;
    case JSOp::Call: case JSOp::CallContent: case JSOp::CallIgnoresRv:
      k.op = WJIROp::Call; k.aux = GET_ARGC(pc); k.pops = uint8_t(k.aux + 2);
      k.pushes = 1; k.ty = WJTy::Value; return true;
    case JSOp::Nop: case JSOp::NopIsAssignOp: case JSOp::NopDestructuring:
    case JSOp::JumpTarget: case JSOp::LoopHead: case JSOp::Pos: case JSOp::ToNumeric:
      k.op = WJIROp::Other; k.pops = 0; k.pushes = 0; k.ty = WJTy::Value; return true;
    default:
      return false;
  }
}
// Build the block-local SSA value graph for a straight-line region. Value-numbers frame
// loads (GetArg/GetLocal/GetRval/FunctionThis) so a redundant load pushes the SAME value-id
// as its earlier equivalent -> a GetProp's receiver has a stable SSA identity that Phase B's
// GVN matches on. Records the field-name id on GetProp nodes. Analysis only: never affects
// emitted code unless GVN consumes it. Tolerant: stops tracking at the first unmodeled op.
// PHASE B: an op clobbers cached heap loads if it can mutate the heap or run user code that
// could mutate a cached field: a call, a property/element store, an element read (proxy /
// index coercion), or an arith op whose operand could ToPrimitive. (Cmp ops never appear in
// a region: they terminate it.) NOTE: a data-property GetProp is treated as NON-clobbering so
// repeated/chained reads (a.b.c ... a.b.c) compose -- SOUND ONLY for side-effect-free data
// properties (the same assumption the kVScse path makes). A property with a getter that has
// side effects would be miscompiled; making this default-on needs a no-getter guard or the
// Phase F deopt path. This is why GECKO_WJVS_GVN is a gated, default-OFF bring-up knob.
static bool WJIRClobbers(WJIROp op) {
  switch (op) {
    case WJIROp::Call: case WJIROp::SetProp: case WJIROp::SetElem: case WJIROp::GetElem:
    case WJIROp::Add: case WJIROp::Sub: case WJIROp::Mul: case WJIROp::Div:
    case WJIROp::Inc: case WJIROp::Dec: case WJIROp::BitOr: case WJIROp::BitAnd:
    case WJIROp::BitXor: case WJIROp::Lsh: case WJIROp::Rsh: case WJIROp::Ursh:
    case WJIROp::BitNot:
      return true;
    default:
      return false;
  }
}
static void WJIRBuild(const WJVSCtx& c, const std::vector<jsbytecode*>& pcs, WJIRRegion& r,
                      bool doGvn, bool* anyReuse) {
  std::vector<int16_t> vstack;                         // operand stack of value-ids (VNs)
  std::vector<int16_t> slotCur(c.rvalS + 1, -1);       // frame slot -> current value-id
  int16_t thisVal = -1;                                 // canonical `this` value-id
  const bool vn = !c.inlined;  // slot value-numbering is only valid for a non-inlined frame
  // PHASE B GVN: hash-cons of available heap loads. (recvVN, field) -> {result VN, def node}.
  // Cleared on any clobber. A GetProp whose key is present reuses it -- and crucially gets the
  // SAME result VN, so a chained `a.b.c` reuse composes (the inner `a.b` VN matches downstream).
  struct Avail { int16_t recv; uint32_t field; int16_t vn; int16_t def; };
  std::vector<Avail> avail;
  uint32_t nextCache = 0;
  if (anyReuse) *anyReuse = false;
  auto pop = [&]() -> int16_t {
    if (vstack.empty()) return -1;
    int16_t v = vstack.back();
    vstack.pop_back();
    return v;
  };
  auto newVal = [&](int16_t def, WJTy ty) -> int16_t {
    int16_t id = int16_t(r.values.size());
    r.values.push_back({def, ty});
    return id;
  };
  for (jsbytecode* pc : pcs) {
    WJIRClass k;
    if (!WJIRClassify(pc, k)) { r.opaque = true; break; }
    int16_t ni = int16_t(r.nodes.size());
    WJIRNode n;
    n.op = k.op;
    n.pc = pc;
    n.aux = k.aux;
    n.ty = k.ty;
    // Frame loads: reuse the slot's current value-id if known (value numbering).
    if (vn && (k.op == WJIROp::GetArg || k.op == WJIROp::GetLocal || k.op == WJIROp::GetRval)) {
      uint32_t slot = k.op == WJIROp::GetArg    ? k.aux
                      : k.op == WJIROp::GetLocal ? c.localBaseS + k.aux
                                                 : c.rvalS;
      int16_t cur = (slot < slotCur.size()) ? slotCur[slot] : -1;
      if (cur < 0) { cur = newVal(ni, k.ty); if (slot < slotCur.size()) slotCur[slot] = cur; }
      n.result = cur;
      r.nodes.push_back(n);
      vstack.push_back(cur);
      continue;
    }
    if (vn && k.op == WJIROp::FunctionThis) {
      if (thisVal < 0) thisVal = newVal(ni, k.ty);
      n.result = thisVal;
      r.nodes.push_back(n);
      vstack.push_back(thisVal);
      continue;
    }
    if (k.op == WJIROp::Dup) {
      int16_t t = vstack.empty() ? -1 : vstack.back();
      n.in0 = t;
      r.nodes.push_back(n);
      vstack.push_back(t);
      continue;
    }
    if (k.op == WJIROp::Swap) {
      r.nodes.push_back(n);
      if (vstack.size() >= 2) std::swap(vstack[vstack.size() - 1], vstack[vstack.size() - 2]);
      continue;
    }
    if (k.op == WJIROp::SetLocal || k.op == WJIROp::SetArg) {
      // tee: records the stored value (top stays); updates the slot's value number.
      n.in0 = vstack.empty() ? -1 : vstack.back();
      r.nodes.push_back(n);
      if (vn) {
        uint32_t slot = k.op == WJIROp::SetArg ? k.aux : c.localBaseS + k.aux;
        if (slot < slotCur.size()) slotCur[slot] = n.in0;
      }
      continue;
    }
    int16_t ins[3] = {-1, -1, -1};
    // Pop k.pops operands, capturing the deepest 3 into in0..in2 (deepest = in0); discard
    // the rest (e.g. a Call's args beyond the receiver/callee aren't tracked individually).
    for (uint8_t i = 0; i < k.pops; i++) {
      int16_t v = pop();
      if (k.pops - 1 - i < 3) ins[k.pops - 1 - i] = v;
    }
    n.in0 = ins[0];
    n.in1 = ins[1];
    n.in2 = ins[2];
    if (k.op == WJIROp::GetProp) {
      n.aux = uint32_t(uintptr_t(c.script->getName(pc)));  // field-name id for GVN matching
    }
    if (k.op == WJIROp::SetRval && vn && c.rvalS < slotCur.size()) {
      slotCur[c.rvalS] = ins[0];
    }
    // PHASE B GVN: redundant-load elimination via hash-consing of GetProp results.
    if (doGvn && k.op == WJIROp::GetProp && n.in0 >= 0) {
      int matchVN = -1, matchDef = -1;
      for (const Avail& a : avail) {
        if (a.recv == n.in0 && a.field == n.aux) { matchVN = a.vn; matchDef = a.def; break; }
      }
      if (matchVN >= 0) {  // reuse: same result VN so chained loads off it also match
        WJIRNode& def = r.nodes[matchDef];
        if (def.cacheSlot == 0xFF && nextCache < kWJGvnSlots) def.cacheSlot = uint8_t(nextCache++);
        if (def.cacheSlot != 0xFF) {
          n.reuseOf = int16_t(matchDef);
          n.result = int16_t(matchVN);
          if (anyReuse) *anyReuse = true;
          r.nodes.push_back(n);
          vstack.push_back(int16_t(matchVN));
          continue;  // no clobber: the reused load doesn't re-execute
        }
      }
      // miss (or no cache slot): the load executes -> its getter could mutate prior loads.
      avail.clear();
      n.result = newVal(ni, k.ty);
      avail.push_back({n.in0, n.aux, n.result, ni});
      r.nodes.push_back(n);
      vstack.push_back(n.result);
      continue;
    }
    if (doGvn && WJIRClobbers(k.op)) avail.clear();  // heap mutation / user code invalidates loads
    if (k.pushes) {
      n.result = newVal(ni, k.ty);
      r.nodes.push_back(n);
      vstack.push_back(n.result);
    } else {
      r.nodes.push_back(n);
    }
  }
}
// Lower one straight-line region. Without GVN: emit each op via the same WJEmitOpVS the
// per-op path uses (byte-identical). With GVN: a reused GetProp copies its cached result
// (in a GC-traced frame slot) into the receiver slot instead of re-emitting guard+load.
static bool WJEmitOpVS(Encoder& e, jsbytecode* pc, WJVSCtx& c);
static bool WJIRLowerRegion(Encoder& e, WJVSCtx& c, const std::vector<jsbytecode*>& pcs) {
  if (pcs.empty()) return true;
  WJIRRegion r;
  bool doGvn = WJVSGvn() && !c.unbox && !c.inlined;
  bool gvn = false;
  WJIRBuild(c, pcs, r, doGvn, &gvn);
  gWJIRRegions++;
  gWJIRNodes += uint64_t(r.nodes.size());
  if (!gvn) {
    for (jsbytecode* pc : pcs) {
      if (!WJEmitOpVS(e, pc, c)) return false;
    }
    return true;
  }
  for (size_t i = 0; i < r.nodes.size(); i++) {
    WJIRNode& n = r.nodes[i];
    if (n.reuseOf >= 0) {  // redundant GetProp: receiver is on top; overwrite with cached value
      uint32_t topS = c.stackBaseS + c.depth - 1;
      uint32_t cacheAbs = c.gvnBase + r.nodes[n.reuseOf].cacheSlot;
      if (!WJSStorePre(e, c, topS) || !WJSAddr(e, cacheAbs) ||
          !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0) ||
          !WJSStorePost(e, c, topS)) {
        return false;
      }
      gWJGvnHits++;
      continue;  // GetProp is net-0 depth -> c.depth unchanged
    }
    if (!WJEmitOpVS(e, n.pc, c)) return false;
    if (n.cacheSlot != 0xFF) {  // capture this load's result for later reuse
      uint32_t resS = c.stackBaseS + c.depth - 1;
      uint32_t cacheAbs = c.gvnBase + n.cacheSlot;
      if (!WJSAddr(e, cacheAbs) || !WJSLoad(e, c, resS) || !WJSStoreEnd(e)) return false;
    }
  }
  return true;
}

// ===== STRUCTURED CONTROL FLOW (GECKO_WJVS_STRUCTCF) =========================
// The relooper lowers control flow as `loop { if(pc==0).. if(pc==1).. }`, so every block
// transition re-enters the loop and re-scans the pc if-chain -- TurboFan cannot recognize the
// real JS loop, so no host-level LICM/regalloc across iterations, and a long if-chain for big
// (inlined) functions. Because the operand stack lives in wasm LOCALS (not the value stack),
// control flow is "pure" and a reducible CFG can be emitted as real nested wasm loop/block/if.
// This struct holds the per-function CFG + the recovered structure; WJAnalyzeCFG fills it, and
// returns false (-> relooper fallback) for anything not cleanly reducible/structurable.
static bool WJStructCF() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WJVS_STRUCTCF") ? 1 : 0;
  return v != 0;
}
struct WJCFG {
  uint32_t K = 0;                          // block count
  std::vector<int32_t> succ0, succ1;       // up to 2 successors per block (block ids; -1 = none)
  std::vector<int32_t> rpo;                // blocks in reverse-postorder
  std::vector<int32_t> rpoIndex;           // block id -> its position in rpo (-1 = unreachable)
  std::vector<uint8_t> isLoopHeader;       // block is the target of a back-edge
  std::vector<int32_t> idom;               // immediate dominator (block id; entry's = itself)
  std::vector<uint8_t> isMerge;            // >=2 forward predecessors -> needs a `block` scope
  std::vector<int32_t> loopLast;           // for a header, the highest rpoIndex of any block in its loop
  bool ok = false;                         // reducible + structurable
};
// Compute successors (from each block's terminator), RPO, back-edges, loop headers, and verify
// REDUCIBILITY (every back-edge target dominates its source). `ofId[off]` maps a bytecode offset
// to its block id; `blockOff[i]` the reverse. Returns false if irreducible or unsupported.
static bool WJAnalyzeCFG(JSScript* script, const std::vector<bool>& isStart,
                         const std::vector<int32_t>& ofId,
                         const std::vector<uint32_t>& blockOff, WJCFG& g) {
  jsbytecode* const start = script->code();
  const uint32_t len = uint32_t(script->codeEnd() - start);
  g.K = uint32_t(blockOff.size());
  if (g.K == 0 || g.K > 512) return false;
  g.succ0.assign(g.K, -1);
  g.succ1.assign(g.K, -1);
  // successors: scan each block to its terminator (a jump/return, or fallthrough at the next
  // block boundary).
  for (uint32_t i = 0; i < g.K; i++) {
    jsbytecode* pc = start + blockOff[i];
    jsbytecode* const e = script->codeEnd();
    while (pc < e) {
      JSOp op = JSOp(*pc);
      uint32_t cur = uint32_t(pc - start);
      uint32_t ol = GetBytecodeLength(pc);
      uint32_t nextOff = cur + ol;
      if (op == JSOp::Goto) {
        int64_t t = int64_t(cur) + GET_JUMP_OFFSET(pc);
        if (t < 0 || uint64_t(t) > len) return false;
        g.succ0[i] = ofId[t];
        break;
      }
      if (op == JSOp::JumpIfTrue || op == JSOp::JumpIfFalse || op == JSOp::And ||
          op == JSOp::Or || op == JSOp::Coalesce) {
        int64_t t = int64_t(cur) + GET_JUMP_OFFSET(pc);
        if (t < 0 || uint64_t(t) > len || nextOff > len) return false;
        g.succ0[i] = ofId[nextOff];  // fallthrough
        g.succ1[i] = ofId[t];        // taken
        break;
      }
      if (op == JSOp::Return || op == JSOp::RetRval) break;  // no successor
      if (nextOff <= len && isStart[nextOff]) {  // fallthrough to the next block
        g.succ0[i] = ofId[nextOff];
        break;
      }
      pc += ol;
    }
  }
  // DFS for postorder + back-edge detection.
  g.rpo.clear();
  g.rpoIndex.assign(g.K, -1);
  std::vector<uint8_t> state(g.K, 0);  // 0=unseen,1=on-stack,2=done
  std::vector<std::pair<int32_t, int>> stk;  // (block, child-index)
  stk.push_back({0, 0});
  state[0] = 1;
  std::vector<int32_t> post;
  while (!stk.empty()) {
    int32_t b = stk.back().first;    // VALUE copies: stk.push_back below may realloc, so a
    int ci = stk.back().second;      // reference into stk would dangle (UB).
    stk.back().second++;             // advance this frame's child cursor in place (pre-push)
    int32_t s = (ci == 0) ? g.succ0[b] : (ci == 1) ? g.succ1[b] : -2;
    if (s == -2) { state[b] = 2; post.push_back(b); stk.pop_back(); continue; }
    if (s < 0) continue;
    if (state[s] == 0) { state[s] = 1; stk.push_back({s, 0}); }
  }
  // post may be < K: bytecode commonly has unreachable trailing blocks after a Return/RetRval
  // (the frontend emits a trailing RetRval block). Those are never entered, so structure only
  // the REACHABLE blocks (in `post`); unreachable ones get rpoIndex -1 and are skipped at emit.
  for (size_t i = 0; i < post.size(); i++) {
    g.rpo.push_back(post[post.size() - 1 - i]);
  }
  for (uint32_t i = 0; i < g.rpo.size(); i++) g.rpoIndex[g.rpo[i]] = int32_t(i);  // rpo<=K (unreachable excluded)
  // Back-edge = edge u->v with rpoIndex[v] <= rpoIndex[u]. v is a loop header.
  g.isLoopHeader.assign(g.K, 0);
  auto isBack = [&](int32_t u, int32_t v) {
    return v >= 0 && g.rpoIndex[v] <= g.rpoIndex[u];
  };
  for (uint32_t i = 0; i < g.K; i++) {
    if (isBack(i, g.succ0[i])) g.isLoopHeader[g.succ0[i]] = 1;
    if (isBack(i, g.succ1[i])) g.isLoopHeader[g.succ1[i]] = 1;
  }
  // Dominators (Cooper-Harvey-Kennedy, iterate over reachable blocks in RPO). idom over the
  // FORWARD CFG (back-edges excluded, sound for reducible graphs). Also count forward preds for
  // merge-node detection, and compute each loop header's span (max rpoIndex of a loop member).
  g.idom.assign(g.K, -1);
  g.isMerge.assign(g.K, 0);
  g.loopLast.assign(g.K, -1);
  int32_t entry = g.rpo.empty() ? -1 : g.rpo[0];
  if (entry < 0) return false;
  g.idom[entry] = entry;
  std::vector<uint32_t> fwdPredCount(g.K, 0);
  auto eachFwdEdge = [&](auto fn) {
    for (int32_t u : g.rpo) {
      if (g.succ0[u] >= 0 && !isBack(u, g.succ0[u])) fn(u, g.succ0[u]);
      if (g.succ1[u] >= 0 && !isBack(u, g.succ1[u])) fn(u, g.succ1[u]);
    }
  };
  eachFwdEdge([&](int32_t, int32_t v) { fwdPredCount[v]++; });
  for (uint32_t b = 0; b < g.K; b++) if (fwdPredCount[b] >= 2) g.isMerge[b] = 1;
  auto intersect = [&](int32_t a, int32_t b) {
    while (a != b) {
      while (g.rpoIndex[a] > g.rpoIndex[b]) a = g.idom[a];
      while (g.rpoIndex[b] > g.rpoIndex[a]) b = g.idom[b];
    }
    return a;
  };
  bool changed = true;
  int guard = 0;
  while (changed && guard++ < 1000) {
    changed = false;
    for (int32_t b : g.rpo) {
      if (b == entry) continue;
      int32_t nd = -1;
      auto considerPred = [&](int32_t u, int32_t v) {
        if (v != b) return;
        if (g.idom[u] < 0 && u != entry) return;  // pred not yet processed
        nd = (nd < 0) ? u : intersect(nd, u);
      };
      eachFwdEdge(considerPred);
      if (nd >= 0 && g.idom[b] != nd) { g.idom[b] = nd; changed = true; }
    }
  }
  // Loop spans: for each back-edge u->h, every block on a path from h to u (rpoIndex in
  // [rpoIndex[h], rpoIndex[u]]) is in the loop; track the max rpoIndex per header.
  for (int32_t u : g.rpo) {
    for (int32_t v : {g.succ0[u], g.succ1[u]}) {
      if (isBack(u, v)) {  // v is the header
        if (g.rpoIndex[u] > g.loopLast[v]) g.loopLast[v] = g.rpoIndex[u];
      }
    }
  }
  g.ok = true;
  return true;
}
static uint32_t gWJVSFailLine = 0;  // DEBUG: line of the last structural WJEmitBodyVS bail
static inline bool WJVSFail(uint32_t line) { gWJVSFailLine = line; return false; }
static bool WJEmitBodyVS(JSScript* script, Encoder& e, uint32_t nargs,
                         uint32_t nfixed) {
  jsbytecode* const start = script->code();
  jsbytecode* const end = script->codeEnd();
  const uint32_t len = uint32_t(end - start);
  const uint8_t kVoid = 0x40, kI64 = uint8_t(TypeCode::I64),
                kF64 = uint8_t(TypeCode::F64), kI32 = uint8_t(TypeCode::I32);
  WJVSCtx c;
  c.script = script;
  c.start = start;
  c.nargs = nargs;
  c.nfixed = nfixed;
  c.localBaseS = nargs;
  c.rvalS = nargs + nfixed;
  c.stackBaseS = nargs + nfixed + 1;
  c.depth = 0;
  c.unbox = WJUnbox() && WJVSUseLocals();  // typed f64 operand stack (registers only)
  c.useCSE = WJVSCSE() && !c.unbox;  // PHASE 2b: within-block GetProp load-CSE (boxed path)
  c.useFieldPromo = WJFieldPromo();  // FIELDPROMO: multi-entry field read-cache (scalar replacement)
  const bool scEnabled = WJShortCircuit();  // && / || / ?? + value-typed branch conditions
  const bool irMode = WJVSIR() || WJVSGvn();  // PHASE A/B: route straight-line runs through the IR
  // Size the frame to the script's ACTUAL max operand-stack depth (nslots-nfixed),
  // not the worst-case kWJVSMaxStack: the prologue zero-inits the whole frame to
  // Undefined on every call, and with the call chain staying in wasm every nested
  // call pays that. +8 headroom for safety; bail if the static stack exceeds the
  // register budget (the per-op c.depth check is the runtime backstop).
  uint32_t maxStack = script->nslots() > nfixed ? script->nslots() - nfixed : 0;
  if (maxStack > kWJVSMaxStack) return WJVSFail(__LINE__);
  // Inlining deepens the operand stack by the inlined callee's stack -> reserve the full
  // register budget so the frame mirror covers any inlined spill.
  if (gWJEmitInline) maxStack = kWJVSMaxStack;
  // PHASE B: when GVN is active, reserve kWJGvnSlots GC-traced frame slots above the operand
  // stack for the load cache (init'd to Undefined by the prologue, traced by WJTraceRoots
  // like any slot). When GVN is off, gvnSlots=0 so the frame layout is byte-for-byte as
  // before (Phase A parity preserved; no unconditional init cost in the default build).
  const uint32_t gvnSlots = (WJVSGvn() && !c.unbox) ? kWJGvnSlots : 0;
  c.gvnBase = c.stackBaseS + maxStack;
  // PHASE F: a dedicated frame slot holding `this`, saved at the prologue (when forced-deopt
  // is active) so a bail AFTER a nested call -- which clobbers gWJScratch[kWJThisSlot] -- can
  // still restore the correct receiver for the resume. Lives in the +8 headroom.
  const uint32_t thisFrameSlot = c.stackBaseS + maxStack + gvnSlots;
  const uint32_t frameSize = c.stackBaseS + maxStack + gvnSlots + 8;
  if (frameSize > kWJFrameSlots / 8) return WJVSFail(__LINE__);
  // PHASE F bring-up: force a bail-to-interpreter at the top of block `fdeopt`. The bailing
  // function self-resumes the rest of its body in PBL via wjhelp(WJH_RESUME) and returns a
  // normal result -- so it works whether it was entered via WasmJitRunCall (C++) or
  // call_indirect (a wasm Mode VS caller); CALLS in the body are allowed (cross-frame).
  // Excludes: SetArg (stale formal args), aliased/lexical-env (env chain not reconstructed),
  // unbox (numeric locals in f64 regs, frame stale), inlined, nfixed>32 (resume staging cap),
  // no JitScript (resume runs in PBL, needs IC entries; must be created via the normal path).
  int fdeopt = WJForceDeopt();
  bool fdeoptOK = false;
  if (fdeopt >= 0 && !c.inlined && !c.unbox && nfixed <= kWJResumeLocalsMax &&
      script->hasJitScript()) {
    fdeoptOK = true;
    for (jsbytecode* p = start; p < end;) {
      JSOp o = JSOp(*p);
      if (o == JSOp::SetArg || o == JSOp::GetAliasedVar || o == JSOp::SetAliasedVar ||
          o == JSOp::PushLexicalEnv || o == JSOp::PushVarEnv || o == JSOp::EnterWith) {
        fdeoptOK = false;
        break;
      }
      p += GetBytecodeLength(p);
    }
  }
  const uint64_t kUndef = 0xFFFFFF83ULL << 32;
  uint32_t spAddr = uint32_t(uintptr_t(&gWJFrameSP));
  uint32_t frameAddr = uint32_t(uintptr_t(&gWJFrameMem[0]));
  uint32_t scratchBase = uint32_t(uintptr_t(static_cast<void*>(gWJScratch)));

  // Block starts (same as WJEmitBodyVCF).
  std::vector<bool> isStart(len + 1, false);
  isStart[0] = true;
  for (jsbytecode* pc = start; pc < end;) {
    JSOp op = JSOp(*pc);
    uint32_t ol = GetBytecodeLength(pc);
    uint32_t cur = uint32_t(pc - start);
    if (IsJumpOpcode(op)) {
      if ((op == JSOp::And || op == JSOp::Or || op == JSOp::Coalesce) && !scEnabled) {
        gWJBailOp = op;
        return false;
      }
      int64_t tgt = int64_t(cur) + GET_JUMP_OFFSET(pc);
      if (tgt < 0 || tgt > len) return false;
      isStart[tgt] = true;
      if (cur + ol <= len) isStart[cur + ol] = true;
    } else if (op == JSOp::Return || op == JSOp::RetRval) {
      if (cur + ol <= len) isStart[cur + ol] = true;
    }
    pc += ol;
  }
  std::vector<int32_t> ofId(len + 1, -1);
  std::vector<uint32_t> blockOff;
  for (uint32_t o = 0; o <= len; o++) {
    if (isStart[o]) {
      ofId[o] = int32_t(blockOff.size());
      blockOff.push_back(o);
    }
  }
  uint32_t K = uint32_t(blockOff.size());
  if (K == 0 || K > 1024) return WJVSFail(__LINE__);
  // STRUCTURED CF: if the CFG is reducible + MERGE-FREE (every join is a loop header or the
  // function exit) + has no short-circuit (&&/||/??) and we're not inlined/deopt, emit real
  // nested wasm loop/if instead of the relooper pc-dispatch. Merge-free covers the common
  // while/if-else-returning shapes (richards schedule, Packet.addTo); else use the relooper.
  WJCFG g;
  bool useStruct = false;
  if (WJStructCF() && !c.inlined && WJForceDeopt() < 0 &&
      WJAnalyzeCFG(script, isStart, ofId, blockOff, g) && g.ok) {
    useStruct = true;
    if (const char* only = getenv("GECKO_WJVS_STRUCT_ONLY"))  // debug: structure only this lineno
      if (uint32_t(atoi(only)) != uint32_t(script->lineno())) useStruct = false;
    bool noCallProbe = getenv("GECKO_WJVS_STRUCT_NOCALL") != nullptr;
    for (jsbytecode* p = start; p < end && useStruct; p += GetBytecodeLength(p)) {
      JSOp o = JSOp(*p);
      if (o == JSOp::And || o == JSOp::Or || o == JSOp::Coalesce) useStruct = false;
      if (noCallProbe && (o == JSOp::Call || o == JSOp::CallContent || o == JSOp::CallIgnoresRv))
        useStruct = false;
    }
    // Exclude functions with a VALUE-CROSSING MERGE (ternary `?:` etc.: a result value lives on the
    // operand stack across a join, entryDepth==1). The structured emitter resets depth per block and
    // has no phi; those fall back to the relooper (which spills to kVSphi). Value-branches (if(call()))
    // are fine -- the value is consumed at the branch (depth 0 after).
    if (useStruct) {
      std::vector<int> ed;
      if (!WJComputeEntryDepth(script, isStart, len, ed)) useStruct = false;
      else for (uint32_t bb = 0; bb < g.K && useStruct; bb++)
        if (blockOff[bb] <= len && ed[blockOff[bb]] == 1) useStruct = false;
    }
    if (getenv("GECKO_DEBUG_JIT")) {
      uint32_t nm = 0, nh = 0;
      for (uint32_t bb = 0; bb < g.K; bb++) { nm += g.isMerge[bb]; nh += g.isLoopHeader[bb]; }
      fprintf(stderr, "[wj-struct] %s:%u useStruct=%d K=%u merges=%u headers=%u nargs=%u nfixed=%u rvalS=%u stackBaseS=%u\n",
              script->filename() ? script->filename() : "?", uint32_t(script->lineno()),
              useStruct, g.K, nm, nh, nargs, nfixed, c.rvalS, c.stackBaseS);
      for (uint32_t bb = 0; bb < g.K; bb++)
        fprintf(stderr, "   blk%u: succ=%d,%d idom=%d rpo=%d merge=%d hdr=%d\n", bb,
                g.succ0[bb], g.succ1[bb], g.idom[bb], g.rpoIndex[bb], g.isMerge[bb], g.isLoopHeader[bb]);
    }
  }
  // SHORT-CIRCUIT: WJComputeEntryDepth validates the operand-stack depth (0/1) across the
  // CFG (handles && / || / ?? value-preserving joins) -- it subsumes WJStackSafe's depth
  // check and additionally permits value-typed branch conditions (ToBoolean'd at the jump).
  std::vector<int> entryDepth;
  if (useStruct) {
    // structured CF handles value-typed branches (ToBoolean) directly; the relooper's
    // WJStackSafe/entryDepth depth-discipline does not apply (operands never cross a join).
  } else if (scEnabled) {
    if (!WJComputeEntryDepth(script, isStart, len, entryDepth)) return WJVSFail(__LINE__);
  } else if (!WJStackSafe(script, isStart)) {
    return WJVSFail(__LINE__);
  }
  auto entryD = [&](uint32_t off) -> int {
    return (scEnabled && off <= len && entryDepth[off] == 1) ? 1 : 0;
  };
  if (c.unbox && WJTypedLoc() && !c.inlined) {
    c.numMask = WJAnalyzeNumericSlots(script, nargs, nfixed, isStart);
    if (getenv("GECKO_DEBUG_JIT")) {
      uint32_t pc8 = 0;
      for (uint32_t b = 0; b < 64; b++) pc8 += (c.numMask >> b) & 1;
      fprintf(stderr, "[wasm-jit]   typedloc %s:%u nargs=%u nfixed=%u typed=%u/%u mask=%llx\n",
              script->filename() ? script->filename() : "?",
              uint32_t(script->lineno()), nargs, nfixed, pc8, nargs + nfixed,
              (unsigned long long)c.numMask);
      fflush(stderr);
    }
  }

  // Locals: 2 i64 (t0,t1), 1 f64 (tf), 6 i32 (fb,pc,basesp,ti,ti2,pc2),
  // kWJVSMaxStack i64 (operand-stack registers s[0..kWJVSMaxStack)), then
  // kWJVSMaxStack f64 (UNBOX parallel stack sf[]; kVSsBaseF). All always declared
  // so local indices are stable regardless of WJVSUseLocals()/unbox.
  // All extended operand-register blocks are ALWAYS declared so kVSsBaseP / kVSfcBase /
  // kVSfcBaseF / kVSsBaseI32 have fixed, stable local indices regardless of which gates are on.
  // Unused locals are free (TurboFan ignores undefined-use locals); this removes the fragile
  // index-coupling between PTRUNBOX/FIELDPROMO/INTUNBOX.
  if (!e.writeVarU32(12)) return false;
  if (!e.writeVarU32(2) || !e.writeFixedU8(kI64)) return false;
  if (!e.writeVarU32(1) || !e.writeFixedU8(kF64)) return false;
  if (!e.writeVarU32(6) || !e.writeFixedU8(kI32)) return false;
  if (!e.writeVarU32(kWJVSMaxStack) || !e.writeFixedU8(kI64)) return false;
  if (!e.writeVarU32(kWJVSMaxStack) || !e.writeFixedU8(kF64)) return false;
  if (!e.writeVarU32(kWJVSMaxTLocals) || !e.writeFixedU8(kF64)) return false;
  if (!e.writeVarU32(1) || !e.writeFixedU8(kI64)) return false;  // kVSphi (short-circuit)
  if (!e.writeVarU32(1) || !e.writeFixedU8(kI64)) return false;  // kVScse (PHASE 2b CSE)
  if (!e.writeVarU32(kWJVSMaxStack + 1) || !e.writeFixedU8(kI32)) return false;  // kVSsBaseP + kVSthisShape
  if (!e.writeVarU32(kWJFieldPromoN) || !e.writeFixedU8(kI64)) return false;  // kVSfcBase (boxed)
  if (!e.writeVarU32(kWJFieldPromoN) || !e.writeFixedU8(kF64)) return false;  // kVSfcBaseF (numeric)
  if (!e.writeVarU32(kWJVSMaxStack) || !e.writeFixedU8(kI32)) return false;  // kVSsBaseI32 (unboxed int)

  // Prologue: basesp = gWJFrameSP; overflow -> deopt (pre-bump, sound).
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(spAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSbasesp)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSbasesp) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(frameSize)) ||
      !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(kWJFrameSlots)) || !e.writeOp(Op::I32GtU)) {
    return false;
  }
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
  if (!WJConst(e, 1.0) || !e.writeOp(Op::Return) || !e.writeOp(Op::End)) {
    return false;
  }
  // fb = frameAddr + basesp*8
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(frameAddr)) ||
      !e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSbasesp) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) ||
      !e.writeOp(Op::I32Add) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSfb)) {
    return false;
  }
  // gWJFrameSP = basesp + frameSize
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(spAddr)) ||
      !e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSbasesp) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(frameSize)) ||
      !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Store) || !e.writeVarU32(2) ||
      !e.writeVarU32(0)) {
    return false;
  }
  // Init the whole frame to Undefined (valid Values, safe to GC-trace). LEANINIT: for small
  // frames, UNROLL the init to straight-line stores (kUndef at each fb+i*8) -- this drops the
  // per-iteration loop branch/counter overhead that is pure per-call tax on call-heavy benches.
  // (memory.fill was tried but its fixed per-call cost regressed small-frame call-dense code.)
  // Default OFF; GECKO_WJVS_LEANINIT=1.
  constexpr uint32_t kUnrollMax = 48;
  static int noInitProbe = -1;  // PERF PROBE (unsound): GECKO_WJVS_NOINIT=1 skips frame zero-init
  if (noInitProbe < 0) noInitProbe = getenv("GECKO_WJVS_NOINIT") ? 1 : 0;
  if (noInitProbe) {
    // skip: leaves the frame uninitialized (GC-unsafe; for measuring the init cost only)
  } else if (WJLeanInit() && frameSize <= kUnrollMax) {
    for (uint32_t i = 0; i < frameSize; i++) {
      if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSfb) || !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(int64_t(kUndef)) || !e.writeOp(Op::I64Store) ||
          !e.writeVarU32(3) || !e.writeVarU32(i * 8)) {
        return false;
      }
    }
  } else {
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0) || !e.writeOp(Op::LocalSet) ||
        !e.writeVarU32(kVSti)) {
      return false;
    }
    if (!e.writeOp(Op::Block) || !e.writeFixedU8(kVoid)) return false;
    if (!e.writeOp(Op::Loop) || !e.writeFixedU8(kVoid)) return false;
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(frameSize)) || !e.writeOp(Op::I32GeU) ||
        !e.writeOp(Op::BrIf) || !e.writeVarU32(1)) {
      return false;
    }
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSfb) || !e.writeOp(Op::LocalGet) ||
        !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Const) || !e.writeVarS32(8) ||
        !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add) || !e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(kUndef)) || !WJSStoreEnd(e)) {
      return false;
    }
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSti) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(1) || !e.writeOp(Op::I32Add) || !e.writeOp(Op::LocalSet) ||
        !e.writeVarU32(kVSti) || !e.writeOp(Op::Br) || !e.writeVarU32(0)) {
      return false;
    }
    if (!e.writeOp(Op::End) || !e.writeOp(Op::End)) return false;  // loop, block
  }
  // Copy args: frame[i] = gWJScratch[i].
  for (uint32_t i = 0; i < nargs; i++) {
    if (!WJSAddr(e, i) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(scratchBase)) || !e.writeOp(Op::I64Load) ||
        !e.writeVarU32(3) || !e.writeVarU32(i * 8) || !WJSStoreEnd(e)) {
      return false;
    }
  }
  // PHASE F: save `this` to its frame slot (gWJScratch[kWJThisSlot] holds this fn's receiver
  // at entry, before any nested call clobbers it) so a later bail can restore it for resume.
  if (fdeoptOK) {
    if (!WJSAddr(e, thisFrameSlot) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(scratchBase + kWJThisOff)) || !e.writeOp(Op::I64Load) ||
        !e.writeVarU32(3) || !e.writeVarU32(0) || !WJSStoreEnd(e)) {
      return false;
    }
  }
  // UNBOX typed locals: seed each numeric-only slot's f64 register lf[s]. Typed
  // ARGS are coerced from the frame (isNum?unbox:ToNumber); typed LOCALS start
  // Undefined -> NaN (= ToNumber(undefined)), set directly (no helper).
  if (c.numMask) {
    uint32_t resAddr = uint32_t(uintptr_t(&gWJScratch[kWJResultSlot]));
    uint32_t helpA = uint32_t(uintptr_t(&gWJHelpA));
    uint32_t nslot = nargs + nfixed;
    for (uint32_t s = 0; s < nslot && s < kWJVSMaxTLocals; s++) {
      if (!(c.numMask & (uint64_t(1) << s))) continue;
      if (s >= nargs) {  // local: NaN
        double nan;
        uint64_t nanBits = kWJCanonNaN;
        memcpy(&nan, &nanBits, sizeof(double));
        if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(nan) ||
            !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSsBaseLF + s)) {
          return false;
        }
        continue;
      }
      // arg: t0 = frame[s]; lf[s] = isNum(t0) ? unbox(t0) : ToNumber(t0)
      if (!WJSAddr(e, s) || !e.writeOp(Op::I64Load) || !e.writeVarU32(3) ||
          !e.writeVarU32(0) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0)) {
        return false;
      }
      if (!WJSIsNum(e, kVSt0) || !e.writeOp(Op::If) || !e.writeFixedU8(kF64)) return false;
      if (!WJVUnboxNG(e, kVSt0)) return false;
      if (!e.writeOp(Op::Else)) return false;
      if (!WJVSStoreGlobal(e, helpA, kVSt0) || !WJVSCallHelper(e, c, WJH_TONUMBER, 0, 0) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(resAddr)) ||
          !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0) ||
          !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSt0) || !WJVUnboxNG(e, kVSt0)) {
        return false;
      }
      if (!e.writeOp(Op::End) || !e.writeOp(Op::LocalSet) ||
          !e.writeVarU32(kVSsBaseLF + s)) {
        return false;
      }
    }
  }

  // SHORT-CIRCUIT: before branching to a merge block (entryDepth 1), spill the operand
  // top to kVSphi so the merge reloads it as the phi value.
  auto spillIfPhi = [&](uint32_t targetOff) -> bool {
    if (entryD(targetOff) != 1) return true;
    if (c.unbox && !WJMaterializeAll(e, c)) return false;
    return WJSLoad(e, c, c.stackBaseS + c.depth - 1) && e.writeOp(Op::LocalSet) &&
           e.writeVarU32(kVSphi);
  };
  if (useStruct) {
    // ----- STRUCTURED emitter (Ramsey-style dom-tree, merge-free) -----
    enum { T_GOTO, T_BRANCH, T_RET };
    std::vector<int32_t> scopes;       // scope target block ids; -1 = $fnexit; header id = loop
    std::vector<uint8_t> emitted(g.K, 0);
    auto brDepth = [&](int32_t T) -> int {
      for (int i = int(scopes.size()) - 1; i >= 0; i--)
        if (scopes[i] == T) return int(scopes.size()) - 1 - i;
      return -1;
    };
    // emit block i's straight-line ops up to its terminator; for T_BRANCH leave the cond i32
    // on the wasm stack. term out-params: a/b = then/else (BRANCH) or target (GOTO).
    auto emitBlockBody = [&](int32_t i, int& term, int32_t& a, int32_t& b) -> bool {
      for (uint32_t d = 0; d < kWJVSMaxStack; d++) c.repr[d] = 0;
      c.cseValid = false; c.cseLastSlot = -1; c.thisShapeCached = false;
      c.depth = 0;
      jsbytecode* p = start + blockOff[i];
      bool prevCmp = false;
      while (p < end) {
        JSOp op = JSOp(*p);
        uint32_t ol = GetBytecodeLength(p), cur = uint32_t(p - start);
        if (op == JSOp::Goto) { term = T_GOTO; a = ofId[int64_t(cur) + GET_JUMP_OFFSET(p)]; return true; }
        if (op == JSOp::JumpIfFalse || op == JSOp::JumpIfTrue) {
          int32_t tgt = ofId[int64_t(cur) + GET_JUMP_OFFSET(p)], fl = ofId[cur + ol];
          if (!prevCmp) {  // value-branch: ToBoolean the operand (no phi needed; consumed here)
            if (c.unbox && !WJMaterializeAll(e, c)) return false;
            if (!WJSLoad(e, c, c.stackBaseS + c.depth - 1) || !e.writeOp(Op::LocalSet) ||
                !e.writeVarU32(kVSt0) || !WJVToBool(e, kVSt0, kVStf, kVSti)) {
              return false;
            }
            c.depth -= 1;
          }
          term = T_BRANCH;
          if (op == JSOp::JumpIfFalse) { a = fl; b = tgt; } else { a = tgt; b = fl; }
          return true;
        }
        if (op == JSOp::Return) {
          if (c.unbox && !WJMaterializeAll(e, c)) return false;
          if (!WJSAddr(e, c.rvalS) || !WJSLoad(e, c, c.stackBaseS + c.depth - 1) || !WJSStoreEnd(e))
            return false;
          term = T_RET; return true;
        }
        if (op == JSOp::RetRval) { term = T_RET; return true; }
        if (WJIsCmp(op)) {
          jsbytecode* nx = p + ol;
          if (!(nx < end && (JSOp(*nx) == JSOp::JumpIfFalse || JSOp(*nx) == JSOp::JumpIfTrue)))
            return false;  // value-compare: unsupported in structured mode
          if (c.unbox && !WJMaterializeAll(e, c)) return false;
          if (!WJVSCmp(e, c, op, /*asValue=*/false)) return false;
          prevCmp = true; p += ol; continue;
        }
        if (!WJEmitOpVS(e, p, c)) return false;
        uint32_t nextOff = cur + ol;
        if (nextOff <= len && isStart[nextOff]) { term = T_GOTO; a = ofId[nextOff]; return true; }
        p += ol;
      }
      return false;
    };
    std::function<bool(int32_t)> emitNode;
    auto codeForNode = [&](int32_t X) -> bool {
      int term; int32_t a = -1, b = -1;
      if (!emitBlockBody(X, term, a, b)) return false;
      if (getenv("GECKO_WJVS_STRUCT_TRACE"))
        fprintf(stderr, "   codeForNode blk%d term=%d a=%d b=%d depth=%u scopes=%zu\n",
                X, term, a, b, c.depth, scopes.size());
      auto doEdge = [&](int32_t S) -> bool {
        if (S >= 0 && g.isLoopHeader[S] && g.rpoIndex[S] <= g.rpoIndex[X]) {  // back-edge: continue
          int d = brDepth(S);
          return d >= 0 && e.writeOp(Op::Br) && e.writeVarU32(uint32_t(d));
        }
        if (S >= 0 && g.isMerge[S]) {  // merge node: br to its (enclosing) block scope
          int d = brDepth(S);
          return d >= 0 && e.writeOp(Op::Br) && e.writeVarU32(uint32_t(d));
        }
        return emitNode(S);  // forward, single-pred (dominated): inline
      };
      if (term == T_RET) {
        int d = brDepth(-1);
        return d >= 0 && e.writeOp(Op::Br) && e.writeVarU32(uint32_t(d));  // br $fnexit
      }
      if (term == T_GOTO) return doEdge(a);
      // T_BRANCH: cond i32 on the wasm stack. The `if` adds ONE wasm nesting level over both
      // arms, so push a sentinel scope (-2, nothing targets it) while emitting the arms -- else
      // every `br` inside an arm is off by one (fatal for loop back-edges / exits).
      if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
      scopes.push_back(-2);
      // FIELDPROMO: the two arms are ALTERNATIVE paths -- each must start from the entry cache,
      // and the join after them clears it (conservative merge). Without save/restore, arm b would
      // wrongly reuse arm a's cached fields.
      int32_t savR[kWJFieldPromoN]; uint32_t savF[kWJFieldPromoN]; uint8_t savP[kWJFieldPromoN];
      if (c.useFieldPromo) {
        memcpy(savR, c.fcRecv, sizeof savR); memcpy(savF, c.fcField, sizeof savF);
        memcpy(savP, c.fcRepr, sizeof savP);
      }
      bool ok = doEdge(a);
      if (c.useFieldPromo) {
        memcpy(c.fcRecv, savR, sizeof savR); memcpy(c.fcField, savF, sizeof savF);
        memcpy(c.fcRepr, savP, sizeof savP);
      }
      ok = ok && e.writeOp(Op::Else) && doEdge(b);
      scopes.pop_back();
      if (c.useFieldPromo) for (uint32_t i = 0; i < kWJFieldPromoN; i++) c.fcRecv[i] = -1;  // join
      return ok && e.writeOp(Op::End);
    };
    // emit X's merge-children (idom[M]==X) as nested `block` scopes (innermost = latest rpo);
    // each merge node M is emitted right after its block closes, so branches to M `br` out to it.
    std::function<bool(int32_t, std::vector<int32_t>&, size_t)> withBlocks =
        [&](int32_t X, std::vector<int32_t>& kids, size_t idx) -> bool {
      if (idx == kids.size()) return codeForNode(X);
      int32_t M = kids[idx];
      scopes.push_back(M);
      if (!e.writeOp(Op::Block) || !e.writeFixedU8(kVoid)) return false;
      if (!withBlocks(X, kids, idx + 1)) return false;
      if (!e.writeOp(Op::End)) return false;
      scopes.pop_back();
      if (c.useFieldPromo) for (uint32_t i = 0; i < kWJFieldPromoN; i++) c.fcRecv[i] = -1;  // merge: clear
      return emitNode(M);
    };
    emitNode = [&](int32_t X) -> bool {
      if (X < 0 || uint32_t(X) >= g.K || emitted[X]) return false;  // revisit -> abort (interp)
      emitted[X] = 1;
      bool hdr = g.isLoopHeader[X];
      if (hdr) {
        scopes.push_back(X);
        if (!e.writeOp(Op::Loop) || !e.writeFixedU8(kVoid)) return false;
        // FIELDPROMO: the back-edge carries stale field values (object/iteration changed) ->
        // the loop body must start with an empty cache.
        if (c.useFieldPromo) for (uint32_t i = 0; i < kWJFieldPromoN; i++) c.fcRecv[i] = -1;
      }
      std::vector<int32_t> kids;  // merge nodes immediately dominated by X, not yet emitted
      for (uint32_t m = 0; m < g.K; m++)
        if (g.isMerge[m] && g.idom[m] == X && !emitted[m]) kids.push_back(int32_t(m));
      std::sort(kids.begin(), kids.end(),
                [&](int32_t p, int32_t q) { return g.rpoIndex[p] > g.rpoIndex[q]; });
      bool ok = withBlocks(X, kids, 0);
      if (hdr) { if (ok && !e.writeOp(Op::End)) ok = false; scopes.pop_back(); }
      return ok;
    };
    scopes.push_back(-1);  // $fnexit (outermost)
    if (!e.writeOp(Op::Block) || !e.writeFixedU8(kVoid)) return false;
    if (!emitNode(0)) return WJVSFail(__LINE__);
    if (!e.writeOp(Op::End)) return false;  // $fnexit
    scopes.pop_back();
  } else {
  if (!e.writeOp(Op::Block) || !e.writeFixedU8(kVoid)) return false;  // $exit
  if (!e.writeOp(Op::Loop) || !e.writeFixedU8(kVoid)) return false;
  for (uint32_t i = 0; i < K; i++) {
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSpc) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(i)) || !e.writeOp(Op::I32Eq) || !e.writeOp(Op::If) ||
        !e.writeFixedU8(kVoid)) {
      return false;
    }
    jsbytecode* pc = start + blockOff[i];
    for (uint32_t d = 0; d < kWJVSMaxStack; d++) c.repr[d] = 0;  // empty stack at block start
    c.cseValid = false;  // PHASE 2b: CSE cache + provenance do not cross block boundaries
    c.cseLastSlot = -1;
    c.thisShapeCached = false;  // PTRUNBOX: hoisted this-shape does not cross block boundaries
    for (uint32_t fi = 0; fi < kWJFieldPromoN; fi++) c.fcRecv[fi] = -1;  // FIELDPROMO: within-block
    c.depth = entryD(blockOff[i]);
    // PHASE F: forced bail at the top of block `fdeopt` (empty-stack boundary only). The
    // function SELF-RESUMES: it calls wjhelp(WJH_RESUME), which finishes the body in PBL from
    // the recorded pc + this fn's frame (args/locals) + `this`, writes the result to
    // gWJScratch[kWJResultSlot], and returns 0 (or 1 on exception). The wasm then restores
    // gWJFrameSP and returns 0.0 -- a NORMAL result, so a call_indirect or C++ caller is
    // oblivious (this is what makes NON-leaf / cross-frame bailout work). Restricted to
    // JumpTarget/LoopHead targets (the first resumed op resyncs the interpreter IC pointer).
    bool fdeoptHere = fdeoptOK && int(i) == fdeopt && c.depth == 0 &&
                      (JSOp(*pc) == JSOp::JumpTarget || JSOp(*pc) == JSOp::LoopHead);
    if (fdeoptHere) {
      uint32_t helpA = uint32_t(uintptr_t(&gWJHelpA));
      uint32_t helpB = uint32_t(uintptr_t(&gWJHelpB));
      // gWJScratch[kWJResumePcSlot] = blockOff
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(scratchBase + kWJResumePcOff)) ||
          !e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(blockOff[i])) ||
          !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
        return false;
      }
      // gWJScratch[kWJThisSlot] = frame[thisFrameSlot]  (restore the real receiver)
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(scratchBase + kWJThisOff)) ||
          !WJSLoad(e, c, thisFrameSlot) || !e.writeOp(Op::I64Store) || !e.writeVarU32(3) ||
          !e.writeVarU32(0)) {
        return false;
      }
      // gWJHelpA = script low32 (raw, not a Value); gWJHelpB = basesp
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(helpA)) ||
          !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(int64_t(uint32_t(uintptr_t(script)))) ||
          !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
        return false;
      }
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(helpB)) ||
          !e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSbasesp) ||
          !e.writeOp(Op::I64ExtendI32U) || !e.writeOp(Op::I64Store) ||
          !e.writeVarU32(3) || !e.writeVarU32(0)) {
        return false;
      }
      // wjhelp(WJH_RESUME, 0): runs PBL resume; status f64 on stack (0 ok / nonzero exc).
      // WJVSExcCheck restores SP + returns 2.0 on a thrown exception.
      if (!WJConst(e, double(WJH_RESUME)) || !WJConst(e, 0.0) || !e.writeOp(Op::Call) ||
          !e.writeVarU32(kWJVSHelpIdx) || !WJVSExcCheck(e)) {
        return false;
      }
      // success: restore gWJFrameSP and return 0.0 (result already in gWJScratch[result]).
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(spAddr)) ||
          !e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSbasesp) ||
          !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
        return false;
      }
      if (!WJConst(e, 0.0) || !e.writeOp(Op::Return)) return false;
      if (!e.writeOp(Op::End)) return false;  // close if (pc == i)
      continue;
    }
    if (c.depth == 1) {  // short-circuit merge: reload the phi value into operand slot 0
      if (!WJSStorePre(e, c, c.stackBaseS + 0) || !e.writeOp(Op::LocalGet) ||
          !e.writeVarU32(kVSphi) || !WJSStorePost(e, c, c.stackBaseS + 0)) {
        return false;
      }
    }
    bool prevCmp = false;
    bool terminated = false;
    // PHASE A: buffer of straight-line ops for the current IR region; lowered (emitted) at
    // the next control-flow op / block boundary. irDepth shadows the operand depth so the
    // per-op overflow backstop fires at exactly the same point as the non-IR path.
    std::vector<jsbytecode*> irRegion;
    int irDepth = 0;
    auto flushIR = [&]() -> bool {
      if (irRegion.empty()) return true;
      bool ok = WJIRLowerRegion(e, c, irRegion);
      irRegion.clear();
      return ok;
    };
    while (pc < end && !terminated) {
      JSOp op = JSOp(*pc);
      uint32_t ol = GetBytecodeLength(pc);
      uint32_t cur = uint32_t(pc - start);
      // bail BEFORE a push goes OOB. With a pending IR region c.depth is stale (lowering is
      // deferred), so use the shadow depth so this backstop fires at exactly the same op as
      // the non-IR path -> the same set of functions compile.
      uint32_t effDepth = (irMode && !irRegion.empty()) ? uint32_t(irDepth) : c.depth;
      if (effDepth >= kWJVSMaxStack) return WJVSFail(__LINE__);
      bool wasCmp = prevCmp;  // prev op was a jump-mode cmp (i32 on the wasm stack)
      prevCmp = false;
      // PHASE A: a control-flow / cmp op ends the straight-line region -> lower it first so
      // c.depth and the operand slots are current before the control-flow handler reads them.
      if (irMode && !irRegion.empty() &&
          (IsJumpOpcode(op) || op == JSOp::Return || op == JSOp::RetRval || WJIsCmp(op))) {
        if (!flushIR()) return false;
      }
      if (IsJumpOpcode(op)) {
        uint32_t fall = cur + ol;
        uint32_t tgtOff = uint32_t(int64_t(cur) + GET_JUMP_OFFSET(pc));
        int32_t tgtId = ofId[tgtOff];
        int32_t fallId = (fall <= len) ? ofId[fall] : -1;
        if (op == JSOp::Goto) {
          if (tgtId < 0) return false;
          if (!spillIfPhi(tgtOff)) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(tgtId) ||
              !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSpc) ||
              !e.writeOp(Op::Br) || !e.writeVarU32(1)) {
            return false;
          }
        } else if (scEnabled &&
                   (op == JSOp::And || op == JSOp::Or || op == JSOp::Coalesce)) {
          // short-circuit: spill v to kVSphi (the merge reloads it); branch on the
          // truthiness (And/Or) / non-nullishness (Coalesce) of v. depth==1 here.
          if (tgtId < 0 || fallId < 0) return false;
          if (c.unbox && !WJMaterializeAll(e, c)) return false;
          if (!WJSLoad(e, c, c.stackBaseS + c.depth - 1) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(kVSphi)) {
            return false;
          }
          if (op == JSOp::Coalesce) {
            if (!WJVIsNullish(e, kVSphi, kVSti)) return false;
          } else if (!WJVToBool(e, kVSphi, kVStf, kVSti)) {
            return false;
          }
          // Or: jump-to-target (keep v) when truthy; And: when falsy; Coalesce: when
          // NOT nullish. aId = the "keep value" successor, bId = the "evaluate b" one.
          int32_t aId = (op == JSOp::Or) ? tgtId : fallId;
          int32_t bId = (op == JSOp::Or) ? fallId : tgtId;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid) ||
              !e.writeOp(Op::I32Const) || !e.writeVarS32(aId) ||
              !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSpc) ||
              !e.writeOp(Op::Else) || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(bId) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(kVSpc) || !e.writeOp(Op::End) || !e.writeOp(Op::Br) ||
              !e.writeVarU32(1)) {
            return false;
          }
          c.depth -= 1;  // v consumed off the operand stack (lives in kVSphi)
        } else {  // JumpIfFalse / JumpIfTrue
          if (tgtId < 0 || fallId < 0) return false;
          if (!wasCmp) {  // condition is a Value (call/getprop result): ToBoolean it
            if (!scEnabled) return false;  // (non-cmp branch bails without short-circuit)
            if (c.unbox && !WJMaterializeAll(e, c)) return false;
            if (!WJSLoad(e, c, c.stackBaseS + c.depth - 1) || !e.writeOp(Op::LocalSet) ||
                !e.writeVarU32(kVSt0) || !WJVToBool(e, kVSt0, kVStf, kVSti)) {
              return false;
            }
            c.depth -= 1;
          }
          int32_t thenId = (op == JSOp::JumpIfTrue) ? tgtId : fallId;
          int32_t elseId = (op == JSOp::JumpIfTrue) ? fallId : tgtId;
          // condition (i32) on the wasm stack (cmp result or the ToBoolean above)
          if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid) ||
              !e.writeOp(Op::I32Const) || !e.writeVarS32(thenId) ||
              !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSpc) ||
              !e.writeOp(Op::Else) || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(elseId) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(kVSpc) || !e.writeOp(Op::End) || !e.writeOp(Op::Br) ||
              !e.writeVarU32(1)) {
            return false;
          }
        }
        terminated = true;
      } else if (op == JSOp::Return) {
        // frame[rvalS] = frame[top-1]; br $exit
        if (c.unbox && !WJMaterializeAll(e, c)) return false;  // box the result operand
        if (!WJSAddr(e, c.rvalS) || !WJSLoad(e, c, c.stackBaseS + c.depth - 1) ||
            !WJSStoreEnd(e) || !e.writeOp(Op::Br) || !e.writeVarU32(2)) {
          return false;
        }
        terminated = true;
      } else if (op == JSOp::RetRval) {
        if (!e.writeOp(Op::Br) || !e.writeVarU32(2)) return false;
        terminated = true;
      } else if (WJIsCmp(op)) {
        jsbytecode* nx = pc + ol;
        bool toJump = (nx < end) && (JSOp(*nx) == JSOp::JumpIfFalse ||
                                     JSOp(*nx) == JSOp::JumpIfTrue);
        if (!toJump && !scEnabled) return false;  // value-compare needs short-circuit
        if (c.unbox && !WJMaterializeAll(e, c)) return false;  // box cmp operands
        if (!WJVSCmp(e, c, op, /*asValue=*/!toJump)) return false;
        prevCmp = toJump;  // jump-mode leaves the i32 on the wasm stack for the JumpIf
      } else {
        if (irMode) {
          WJIRClass k;
          if (WJIRClassify(pc, k)) {  // straight-line op: buffer into the IR region
            if (irRegion.empty()) irDepth = int(c.depth);
            irRegion.push_back(pc);  // overflow backstop is the effDepth check at loop top
            irDepth += int(k.pushes) - int(k.pops);
          } else {  // op the IR doesn't model: flush then emit directly
            if (!flushIR()) return false;
            if (!WJEmitOpVS(e, pc, c)) return false;
          }
        } else {
          if (!WJEmitOpVS(e, pc, c)) return false;
        }
        uint32_t nextOff = cur + ol;
        if (nextOff <= len && isStart[nextOff] && nextOff != blockOff[i]) {
          if (!flushIR()) return false;  // lower the region before the block-boundary branch
          int32_t nid = ofId[nextOff];
          if (nid < 0) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(nid) ||
              !e.writeOp(Op::LocalSet) || !e.writeVarU32(kVSpc) ||
              !e.writeOp(Op::Br) || !e.writeVarU32(1)) {
            return false;
          }
          terminated = true;
        }
      }
      pc += ol;
    }
    if (!flushIR()) return false;  // PHASE A: lower any region pending at block end
    if (!terminated) {
      if (!e.writeOp(Op::Br) || !e.writeVarU32(2)) return false;
    }
    if (!e.writeOp(Op::End)) return false;  // end if (pc == i)
  }
  if (!e.writeOp(Op::End) || !e.writeOp(Op::End)) return false;  // loop, $exit
  }  // end else (relooper path)
  // Epilogue: restore gWJFrameSP, store rval to gWJScratch[result], return 0.
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(spAddr)) ||
      !e.writeOp(Op::LocalGet) || !e.writeVarU32(kVSbasesp) ||
      !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(0) ||
      !e.writeOp(MiscOp::I32TruncSatF64U) || !WJSLoad(e, c, c.rvalS) ||
      !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(kWJResultOff)) {
    return false;
  }
  if (!WJConst(e, 0.0)) return false;
  return e.writeOp(Op::End);
}

static bool WJCompile(JSScript* script, WasmJitEntry& entry);
static WasmJitEntry* WJEntryFor(JSScript* script);
static bool WJModeVSSupported(JSOp op);

// METHOD_INLINING diagnostic: a callee is inline-eligible (Phase A) if it is a small,
// straight-line LEAF of Mode-VS-supported ops (no calls/jumps). Counts the inlinable
// fraction of hot call sites to size the opportunity before building emission.
static uint64_t gWJCallFills = 0, gWJCallInlinable = 0;
static bool WJIsCmp(JSOp op);
static bool WJStackSafe(JSScript* script, const std::vector<bool>& isStart);
// A callee is inline-eligible if it is a small LEAF of Mode-VS-supported ops with no
// calls. Phase A: straight-line only. Phase B: also simple control flow (the relooper
// emits the callee's own block/loop/if(pc2==i) dispatch inline). On success *hasCF is
// set if the callee contains jumps (-> the caller uses the CFG inline emitter). The
// validation must be strict enough that emission cannot fail (else the whole caller's
// compile aborts rather than falling back to call_indirect).
static bool WJCallInlinable(JSScript* cs, bool* hasCF) {
  // INLINE budget: raised aggressively so richards' whole hot tree (run/task.run/scheduler
  // methods, each >60 bytecodes) can inline into schedule() -- the only way to kill the
  // per-call bridge overhead that makes compiled-schedule slower than the interpreter.
  static int sMaxLen = -1, sMaxFixed = -1;
  if (sMaxLen < 0) { const char* s = getenv("GECKO_WJVS_INLMAXLEN"); sMaxLen = s ? atoi(s) : 60; }
  if (sMaxFixed < 0) { const char* s = getenv("GECKO_WJVS_INLMAXFIXED"); sMaxFixed = s ? atoi(s) : 8; }
  if (cs->length() > uint32_t(sMaxLen)) return false;
  if (cs->nfixed() > uint32_t(sMaxFixed)) return false;
  jsbytecode* const start = cs->code();
  jsbytecode* const end = cs->codeEnd();
  const uint32_t len = uint32_t(end - start);
  bool cf = false;
  std::vector<bool> isStart(len + 1, false);
  isStart[0] = true;
  for (jsbytecode* pc = start; pc < end; pc += GetBytecodeLength(pc)) {
    JSOp op = JSOp(*pc);
    if (op == JSOp::New || op == JSOp::SuperCall || op == JSOp::SpreadCall) {
      return false;  // construct / spread: different operand-stack shape, not inlined
    }
    if (op == JSOp::Call || op == JSOp::CallContent || op == JSOp::CallIgnoresRv) {
      // NON-LEAF: a nested call is allowed (it emits via WJEmitOpVS->WJVSCall in the
      // inlined sub-context, which -- guarded by !c.inlined -- does NOT recurse and so
      // emits a normal call_indirect/helper). WJStackSafe models the call stack effect
      // via js::StackUses/StackDefs. Gated to leaf-only for A/B.
      if (WJLeafOnly()) return false;
      continue;  // accept; skip the WJModeVSSupported check below (handled by WJVSCall)
    }
    if (op == JSOp::String) return false;      // non-atom strings can't be baked
    if (op == JSOp::SetArg) return false;      // would mutate the caller's shared arg slot
    if (IsJumpOpcode(op)) {
      // Short-circuit ops keep a Value on the stack across the branch -> the relooper
      // bails on them (gWJBailOp); reject here too so emission can't fail.
      if (op == JSOp::And || op == JSOp::Or || op == JSOp::Coalesce) return false;
      uint32_t cur = uint32_t(pc - start);
      uint32_t ol = GetBytecodeLength(pc);
      int64_t tgt = int64_t(cur) + GET_JUMP_OFFSET(pc);
      if (tgt < 0 || tgt > len) return false;
      isStart[tgt] = true;
      if (cur + ol <= len) isStart[cur + ol] = true;
      cf = true;
      continue;
    }
    if (op == JSOp::Return || op == JSOp::RetRval) {
      uint32_t cur = uint32_t(pc - start);
      uint32_t ol = GetBytecodeLength(pc);
      if (cur + ol <= len) isStart[cur + ol] = true;
      continue;
    }
    if (op == JSOp::SetRval || op == JSOp::GetRval) continue;
    if (WJIsCmp(op)) {                          // must be consumed by a following branch
      jsbytecode* nx = pc + GetBytecodeLength(pc);
      if (nx >= end ||
          (JSOp(*nx) != JSOp::JumpIfFalse && JSOp(*nx) != JSOp::JumpIfTrue)) {
        return false;
      }
      continue;
    }
    if (!WJModeVSSupported(op)) return false;
  }
  if (!WJStackSafe(cs, isStart)) return false;
  if (hasCF) *hasCF = cf;
  return true;
}

// Fill the IC/call cache for a site that missed, so the next call runs inline.
// Property/element sites cache {shape, offset}; call sites compile the callee
// and cache {JSFunction*, handle, nargs}. Called from WasmJitRunCall on a miss.
static void WJFillIC(uint32_t site) {
  if (site >= gWJSiteCount) return;
  WJSite& s = gWJSites[site];
  if (!s.script) return;
  jsbytecode* pc = s.script->offsetToPC(s.pcOff);
  JSOp op = JSOp(*pc);

  if (op == JSOp::GetGName) {
    // Resolve the name on the current global and cache {holder, shape, slot}.
    JSContext* cx = js::TlsContext.get();
    if (!cx) return;
    js::NativeObject* g = cx->global();
    js::PropertyName* name = s.script->getName(pc);
    jsid id = js::NameToId(name);
    mozilla::Maybe<js::PropertyInfo> prop = g->lookupPure(id);
    if (prop.isNothing() || !prop->isDataProperty()) return;
    uint32_t slot = prop->slot();
    uint32_t nfixed = g->numFixedSlots();
    if (slot < nfixed) {
      gWJICTable[2 * site + 1] = 16 + slot * 8;  // fixed-slot byte offset
    } else {
      gWJICTable[2 * site + 1] = ((slot - nfixed) * 8) | kWJDynSlot;
    }
    gWJICTable[2 * site] = uint32_t(uintptr_t(g->shape()));
    gWJGNameHolder[site] = uint32_t(uintptr_t(g));  // set last: enables fast path
    return;
  }

  JS::Value v = JS::Value::fromRawBits(gWJMissObj);
  if (!v.isObject()) return;
  JSObject* obj = &v.toObject();

  if (op == JSOp::Call || op == JSOp::CallContent ||
      op == JSOp::CallIgnoresRv) {
    // Call site: the missed value is the callee. Compile it + cache its shared-
    // table index (the call_indirect target).
    if (!obj->is<JSFunction>()) return;
    JSFunction* fun = &obj->as<JSFunction>();
    if (!fun->isInterpreted() || !fun->baseScript() ||
        !fun->baseScript()->hasBytecode()) {
      // Native callee: recognize Math.* intrinsics for inline f64-op emission (a later recompile
      // reads gWJMathRec by script+pcOff); other natives stay on the generic helper path.
      if (WJMathInlineEnabled() && fun->isNativeFun()) {
        int mop = WJMathIntrinsic(fun);
        if (mop) {
          gWJMathRec[WJInlineKey(s.script, s.pcOff)] =
              WJMathRec{uint8_t(mop), uint32_t(uintptr_t(fun))};
          if (WasmJitEntry* me = WJEntryFor(s.script)) me->hasMathCall = true;
        }
      }
      return;
    }
    JSScript* cs = fun->baseScript()->asJSScript();
    // INLINE-RECORD EARLY: register the observed callee for this site BEFORE the
    // standalone-compile / table-index / arity / mode checks below. Inlining works on
    // the callee's BYTECODE (WJCallInlinable inspects the script) and does NOT need the
    // callee compiled as its own wasm function -- so an outermost caller (e.g. richards
    // schedule, which compiles before its callees are tabled) can still inline its tree.
    // Without this the record was only created after those checks, losing the inline
    // candidate for the hottest (first-compiled) functions. Dedup is in WJRecordInlineCallee.
    if (WJVSInline()) WJRecordInlineCallee(s.script, s.pcOff, uint32_t(uintptr_t(fun)));
    WasmJitEntry* ce = WJEntryFor(cs);
    if (!ce) return;
    if (ce->state == WasmJitEntry::State::Cold) {
      if (!cs->function() || cs->isModule() || cs->length() > 4096 ||
          !WJCompile(cs, *ce)) {
        ce->state = WasmJitEntry::State::Failed;
      }
    }
    if (ce->state != WasmJitEntry::State::Compiled) return;
    if (ce->tableIdx < 0) return;                 // callee not in the table
    if (ce->nargs != gWJCallArgc[site]) return;   // arity mismatch: keep deopting
    // A restart-based (Mode N/V) caller must NOT call_indirect a Mode VS callee
    // (double-mutation on restart + GC-rooting of its untraced frame). Such calls
    // deopt to the interpreter. A Mode VS caller MAY call_indirect a VS callee.
    if (ce->modeVS) {
      WasmJitEntry* callerE = WJEntryFor(s.script);
      if (!callerE || !callerE->modeVS) return;
    }
    uint32_t fnLow = uint32_t(uintptr_t(fun));
    int32_t handle = ce->tableIdx;
    if (WJNoPolyCall()) {  // monomorphic baseline: rewrite way 0 on every miss
      gWJCallHandle[site] = handle;
      gWJCallNargs[site] = ce->nargs;
      gWJCallFn[site] = fnLow;
      gWJCallWaysFilled[site] = 1;
      gWJCallFills++;
      WJRecordInlineCallee(s.script, s.pcOff, fnLow);
      return;
    }
    // Add to the N-way call IC. Skip if this callee is already cached (a guard
    // can miss for a non-callee reason). Way 0 reuses gWJCallFn/gWJCallHandle.
    uint8_t nw = gWJCallWaysFilled[site];
    if (gWJCallFn[site] == fnLow && nw >= 1) return;
    for (uint8_t w = 1; w < nw; w++) {
      if (gWJCallFnX[(w - 1) * kWJMaxSites + site] == fnLow) return;
    }
    if (nw == 0) {
      gWJCallHandle[site] = handle;
      gWJCallNargs[site] = ce->nargs;
      gWJCallFn[site] = fnLow;  // set last: enables the fast path
      gWJCallWaysFilled[site] = 1;
    } else if (nw < kWJCallWays) {
      gWJCallHandleX[(nw - 1) * kWJMaxSites + site] = handle;
      gWJCallFnX[(nw - 1) * kWJMaxSites + site] = fnLow;  // set last
      gWJCallWaysFilled[site] = nw + 1;
      gWJCallPolyFills++;
    } else {
      gWJCallMegaMiss++;  // all ways full -> stays on the generic helper path
    }
    gWJCallFills++;
    if (WJCallInlinable(cs)) gWJCallInlinable++;
    // Record callee by (caller script, pcOff) so a later recompile can inline it
    // (accumulates up to 4 distinct callees for polymorphic inlining).
    WJRecordInlineCallee(s.script, s.pcOff, fnLow);
    return;
  }

  if (!obj->is<js::NativeObject>()) {
    if (op == JSOp::GetProp) gWJResolveFail[0]++;
    return;
  }
  js::NativeObject& nobj = obj->as<js::NativeObject>();
  if (op == JSOp::GetElem || op == JSOp::SetElem || op == JSOp::StrictSetElem) {
    // Typed-array site: cache the kind (Float64/Int32) + shape; the inline path
    // loads/stores raw elements (no boxing, no barrier).
    if (obj->is<js::TypedArrayObject>()) {
      js::Scalar::Type ty = obj->as<js::TypedArrayObject>().type();
      uint8_t kind = ty == js::Scalar::Float64 ? kWJElemF64
                     : ty == js::Scalar::Int32 ? kWJElemI32
                     : ty == js::Scalar::Uint8 ? kWJElemU8
                     : ty == js::Scalar::Int8 ? kWJElemI8
                     : ty == js::Scalar::Uint16 ? kWJElemU16
                     : ty == js::Scalar::Int16 ? kWJElemI16
                     : ty == js::Scalar::Uint32 ? kWJElemU32
                     : ty == js::Scalar::Float32 ? kWJElemF32
                                                 : 0;  // Uint8Clamped/BigInt/... -> helper
      if (!kind) return;  // other element types -> stay on the helper
      gWJICTable[2 * site] = uint32_t(uintptr_t(nobj.shape()));
      gWJICTable[2 * site + 1] = 0;
      gWJElemKind[site] = kind;  // set last: enables the typed fast path
      return;
    }
    // Dense element site: a stable shape guarantees the dense-array layout.
    if (!obj->is<js::ArrayObject>()) return;
    gWJICTable[2 * site] = uint32_t(uintptr_t(nobj.shape()));
    gWJICTable[2 * site + 1] = 0;
    return;
  }
  // GetProp / SetProp: cache {recv shape, byte offset} for a data property.
  // OWN -> holder=0 (load from the receiver). A Mode VS GetProp also caches a
  // PROTOTYPE-chain data property (e.g. a method on Foo.prototype): the receiver-
  // shape guard fixes the proto chain, the holder-shape guard fixes the slot.
  js::PropertyName* name = s.script->getName(pc);
  jsid id = js::NameToId(name);
  // Encode a slot as a byte offset: fixed slot -> 16+slot*8; dynamic slot ->
  // ((slot-nfixed)*8) | kWJDynSlot (the wasm loads it via slots_).
  auto encodeOff = [](uint32_t slot, uint32_t nfixed) -> uint32_t {
    return slot < nfixed ? (16 + slot * 8) : (((slot - nfixed) * 8) | kWJDynSlot);
  };
  // GetProp (Mode V AND Mode VS) supports dynamic slots + the prototype chain
  // (both emitters' slot-load handles fixed/dyn + a holder-shape guard). Enabling
  // this for Mode V too eliminates a major deopt source: proto-method / dynamic-
  // slot reads in non-mutating functions used to miss + deopt on every call. The
  // old proto miscompile was the deopt-double-execution of a MUTATING function
  // (now Mode VS / no-restart), not the proto load itself. ALL SetProp sites stay
  // OWN fixed-slot only (the SetProp emitters inline-store fixed slots).
  bool vsGet = (op == JSOp::GetProp);
  // Resolve the property to {recvShape, off, holder(0=own), holderShape}. Returns
  // (via the lambda) false on anything unsupported (accessor, missing, etc.).
  uint32_t recvShape = 0, offVal = 0, holderVal = 0, holderShapeVal = 0;
  uint32_t methodFnLow = 0;  // proto-method JSFunction low32 (for poly dispatch)
  bool resolved = false;
  if (gWJSiteLen[site]) {
    // Length site: cache the shape only for dense arrays (the emit loads the
    // length inline from the elements header, off/holder unused). Non-arrays deopt.
    if (!obj->is<js::ArrayObject>()) return;
    recvShape = uint32_t(uintptr_t(nobj.shape()));
    resolved = true;
  }
  mozilla::Maybe<js::PropertyInfo> prop =
      resolved ? mozilla::Nothing() : nobj.lookupPure(id);
  if (!resolved && prop.isSome()) {
    if (!prop->isDataProperty()) {
      if (op == JSOp::GetProp) {
        gWJResolveFail[1]++;
        JSContext* cxn = js::TlsContext.get();
        if (cxn && name == cxn->names().length) gWJResolveFail[5]++;
      }
      return;  // accessor -> deopt to the helper
    }
    uint32_t nfixed = nobj.numFixedSlots();
    if (!vsGet && prop->slot() >= nfixed) return;  // own dynamic slot: VS only
    recvShape = uint32_t(uintptr_t(nobj.shape()));
    offVal = vsGet ? encodeOff(prop->slot(), nfixed) : (16 + prop->slot() * 8);
    resolved = true;
  } else if (!resolved && vsGet) {
    // Walk the prototype chain for a data property (e.g. a method).
    JSObject* holder = obj;
    for (int depth = 0; depth < 8 && !resolved; depth++) {
      holder = holder->staticPrototype();
      if (!holder || !holder->is<js::NativeObject>()) { gWJResolveFail[3]++; return; }
      js::NativeObject& hobj = holder->as<js::NativeObject>();
      mozilla::Maybe<js::PropertyInfo> hp = hobj.lookupPure(id);
      if (hp.isSome()) {
        if (!hp->isDataProperty()) { gWJResolveFail[2]++; return; }  // proto accessor
        recvShape = uint32_t(uintptr_t(nobj.shape()));  // receiver shape
        offVal = encodeOff(hp->slot(), hobj.numFixedSlots());
        holderVal = uint32_t(uintptr_t(&hobj));
        holderShapeVal = uint32_t(uintptr_t(hobj.shape()));
        JS::Value mval = hobj.getSlot(hp->slot());
        if (mval.isObject() && mval.toObject().is<JSFunction>()) {
          methodFnLow = uint32_t(uintptr_t(&mval.toObject()));
        }
        resolved = true;
      }
    }
  }
  if (!resolved) {
    if (op == JSOp::GetProp) gWJResolveFail[4]++;
    return;
  }
  // Per-way cell accessors (way 0 = the shared arrays; ways 1+ = the extra arrays).
  auto shapeCell = [&](uint32_t w) -> uint32_t& {
    return w == 0 ? gWJICTable[2 * site]
                  : gWJICTableX[((w - 1) * kWJMaxSites + site) * 2];
  };
  auto offCell = [&](uint32_t w) -> uint32_t& {
    return w == 0 ? gWJICTable[2 * site + 1]
                  : gWJICTableX[((w - 1) * kWJMaxSites + site) * 2 + 1];
  };
  auto holderCell = [&](uint32_t w) -> uint32_t& {
    return w == 0 ? gWJProtoHolder[site]
                  : gWJProtoHolderX[(w - 1) * kWJMaxSites + site];
  };
  auto holderShapeCell = [&](uint32_t w) -> uint32_t& {
    return w == 0 ? gWJProtoHolderShape[site]
                  : gWJProtoHolderShapeX[(w - 1) * kWJMaxSites + site];
  };
  // Pick the IC way. Monomorphic sites (SetProp, Mode VS GetProp) always fill way
  // 0. An N-way Mode V GetProp site fills the way already holding this shape (self-
  // heal) or the first empty way; if all ways are full with other shapes it evicts
  // way 0 (megamorphic thrash, no worse than the old monomorphic IC). The shape is
  // written LAST so a concurrently-read half-built way is never matched.
  uint32_t useWay = 0;
  if (gWJSitePoly[site]) {
    int chosen = -1, empty = -1;
    for (uint32_t w = 0; w < kWJICWays; w++) {
      uint32_t sw = shapeCell(w);
      if (sw == recvShape) { chosen = int(w); break; }
      if (sw == 0 && empty < 0) empty = int(w);
    }
    useWay = chosen >= 0 ? uint32_t(chosen) : (empty >= 0 ? uint32_t(empty) : 0);
    if (chosen < 0 && empty < 0) gWJMegaMiss++;  // all ways full, none match: >N shapes
  }
  offCell(useWay) = offVal;
  holderCell(useWay) = holderVal;
  holderShapeCell(useWay) = holderShapeVal;
  shapeCell(useWay) = recvShape;  // shape last
  // Capture a representative live instance of this shape for the Ion oracle's
  // field-type reading (unboxing). Overwrite freely -- the latest is freshest.
  gWJShapeSample[recvShape] = uint32_t(uintptr_t(&nobj));
  // Polymorphic method dispatch: record (recvShape -> methodFn) for a proto-method
  // GetProp so the Ion front-end can inline each task type's body behind a shape
  // guard. Also capture a sample receiver (method sites have no own-field record).
  if (op == JSOp::GetProp && holderVal != 0 && methodFnLow) {
    WJMethodPolyRec& m = gWJMethodPoly[WJInlineKey(s.script, s.pcOff)];
    bool seen = false;
    for (uint8_t i = 0; i < m.n; i++) {
      if (m.shapes[i] == recvShape) { m.fns[i] = methodFnLow; seen = true; break; }
    }
    if (!seen && m.n < 4) {
      m.shapes[m.n] = recvShape;
      m.fns[m.n] = methodFnLow;
      m.n++;
    }
    if (gWJSampleRecv.find(WJInlineKey(s.script, s.pcOff)) == gWJSampleRecv.end()) {
      gWJSampleRecv[WJInlineKey(s.script, s.pcOff)] = uint32_t(uintptr_t(&nobj));
    }
  }
  // LEAN EMISSION foundation: record own-monomorphic {shape,off} by (script,pcOff) for a
  // specialized recompile to bake as constants. Own data prop only (holder==0), non-poly,
  // non-length, fixed-or-dyn offset. Skipped for proto/poly/length sites (not bakeable as a
  // single direct load). The consumer (baked direct-field emission) reads this at recompile.
  // Record own-data {shape,off} keyed by (script,pcOff) for baking. Mode-VS GetProp sites
  // are flagged poly even when monomorphic in practice (richards), so do NOT exclude poly --
  // record the observed shape; the baked guard falls to the helper on any shape mismatch.
  if (holderVal == 0 && !gWJSiteLen[site] &&
      (op == JSOp::GetProp || op == JSOp::SetProp || op == JSOp::StrictSetProp)) {
    // Record the field's observed value type so the Ion front-end can specialize
    // the unboxed load: int32-valued fields (richards' common case) are stored as
    // int32-boxed Values, NOT double bits, so a raw f64 load would read NaN.
    uint8_t vty = 2;  // other: not f64-able (bail)
    {
      // Slot index from the byte offset (fixed: 16+slot*8; dynamic: encodes
      // slot-nfixed). Read its observed value type for the Ion FE's unboxing.
      uint32_t slot = (offVal & kWJDynSlot)
                          ? (((offVal & ~kWJDynSlot) / 8) + nobj.numFixedSlots())
                          : ((offVal - 16) / 8);
      JS::Value fv = nobj.getSlot(slot);
      if (fv.isInt32()) vty = 1;
      else if (fv.isDouble()) vty = 0;
      else if (fv.isObject() || fv.isNull()) vty = 3;  // object-or-null ref field
    }
    gWJShapeRec[WJInlineKey(s.script, s.pcOff)] = WJShapeRec{recvShape, offVal, vty};
    gWJSampleRecv[WJInlineKey(s.script, s.pcOff)] = uint32_t(uintptr_t(&nobj));
    // Share the observed value type by (shape, offset) so the Baseline-IC oracle
    // can type fields of functions the WJ JIT never compiled.
    if (vty != 2) {
      gWJFieldVty[(uint64_t(recvShape) << 32) | offVal] = vty;
    }
  }
}

// Build a complete wasm module that imports the guest memory as "m"."mem" and
// exports a single function "f": (f64 scratchPtr) -> f64 deopt. `sharedMem`
// selects the imported-memory limits form (shared memory needs a max).
// `useModeV` selects the Value-typed body emitter (objects) over the numeric one.
static bool WJBuildModule(JSScript* script, uint32_t nargs, uint32_t nfixed,
                          bool sharedMem, bool useModeV, bool hasCall,
                          bool modeVS, Bytes& out) {
  Encoder e(out);
  const uint8_t kF64 = uint8_t(TypeCode::F64);
  if (!e.writeFixedU32(MagicNumber) ||
      !e.writeFixedU32(EncodingVersionModule)) {
    return false;
  }
  // Mode VS imports the C++ helper `wjhelp:(f64,f64)->f64` as function index 0,
  // which shifts the body function to index 1.
  const uint32_t bodyFnIdx = modeVS ? 1 : 0;
  size_t s;
  // Type section: type 0 = (f64)->f64 (body + call_indirect); for Mode VS also
  // type 1 = (f64,f64)->f64 (wjhelp).
  if (!e.startSection(SectionId::Type, &s)) return false;
  if (!e.writeVarU32(modeVS ? 2 : 1)) return false;
  if (!e.writeFixedU8(0x60) || !e.writeVarU32(1) || !e.writeFixedU8(kF64) ||
      !e.writeVarU32(1) || !e.writeFixedU8(kF64)) {
    return false;
  }
  if (modeVS) {
    if (!e.writeFixedU8(0x60) || !e.writeVarU32(2) || !e.writeFixedU8(kF64) ||
        !e.writeFixedU8(kF64) || !e.writeVarU32(1) || !e.writeFixedU8(kF64)) {
      return false;
    }
  }
  e.finishSection(s);
  // Import section: (Mode VS) wjhelp function first, then guest memory "m"."mem",
  // and (if calls) the shared JIT funcref table "m"."tbl".
  uint32_t nImports = (hasCall ? 2 : 1) + (modeVS ? 1 : 0);
  if (!e.startSection(SectionId::Import, &s)) return false;
  if (!e.writeVarU32(nImports)) return false;
  if (modeVS) {
    if (!e.writeBytes("m", 1) || !e.writeBytes("help", 4) ||
        !e.writeFixedU8(0x00) || !e.writeVarU32(1)) {  // func import, type 1
      return false;
    }
  }
  if (!e.writeBytes("m", 1) || !e.writeBytes("mem", 3) ||
      !e.writeFixedU8(0x02)) {  // kind: memory
    return false;
  }
  if (sharedMem) {
    if (!e.writeFixedU8(0x03) || !e.writeVarU32(1) || !e.writeVarU32(65536)) {
      return false;
    }
  } else {
    if (!e.writeFixedU8(0x00) || !e.writeVarU32(0)) return false;
  }
  if (hasCall) {
    if (!e.writeBytes("m", 1) || !e.writeBytes("tbl", 3) ||
        !e.writeFixedU8(0x01) ||                  // kind: table
        !e.writeFixedU8(uint8_t(TypeCode::FuncRef)) ||  // elemtype funcref
        !e.writeFixedU8(0x00) || !e.writeVarU32(0)) {   // limits: no max, min 0
      return false;
    }
  }
  e.finishSection(s);
  // Function section: one function of type 0.
  if (!e.startSection(SectionId::Function, &s)) return false;
  if (!e.writeVarU32(1) || !e.writeVarU32(0)) return false;
  e.finishSection(s);
  // Export section: export the body function.
  if (!e.startSection(SectionId::Export, &s)) return false;
  if (!e.writeVarU32(1) || !e.writeBytes("f", 1) || !e.writeFixedU8(0x00) ||
      !e.writeVarU32(bodyFnIdx)) {
    return false;
  }
  e.finishSection(s);
  // Code section: one body (the emitter writes the locals + ops + closing end).
  if (!e.startSection(SectionId::Code, &s)) return false;
  if (!e.writeVarU32(1)) return false;
  size_t bodyOff;
  if (!e.writePatchableVarU32(&bodyOff)) return false;
  size_t bodyStart = e.currentOffset();
  if (modeVS) {
    if (!WJEmitBodyVS(script, e, nargs, nfixed)) return false;
  } else if (useModeV) {
    if (!WJEmitBodyVCF(script, e, nargs, nfixed)) return false;
  } else {
    if (!WJEmitBodyCF(script, e, nargs, nfixed)) return false;
  }
  e.patchVarU32(bodyOff, uint32_t(e.currentOffset() - bodyStart));
  e.finishSection(s);
  return true;
}

// GC-safety: a major GC can move or free shapes, global objects, and functions,
// invalidating the raw pointers cached in the inline caches. Clear them at GC
// end so every site re-resolves with live pointers on its next execution (a
// single miss + refill per site). The script->wasm map persists (its entries are
// validated by bcLen against JSScript* address reuse). Additive callback -- does
// not replace Gecko's.
static void WJFinalizeCB(JS::GCContext*, JSFinalizeStatus status, void*) {
  if (status != JSFINALIZE_COLLECTION_END) {
    return;
  }
  for (uint32_t i = 0; i < 2 * kWJMaxSites; i++) {
    gWJICTable[i] = 0;
  }
  for (uint32_t i = 0; i < kWJMaxSites; i++) {
    gWJCallFn[i] = 0;
    gWJGNameHolder[i] = 0;
    gWJProtoHolder[i] = 0;
    gWJProtoHolderShape[i] = 0;
  }
  gWJMissSite = kWJNoMiss;
  // Recorded inline callees hold raw JSFunction* addresses; a major GC may move/free
  // them. Drop them (re-recorded on the next miss) so a recompile never derefs a stale
  // address.
  gWJInlineCallee.clear();
  gWJMethodPoly.clear();  // raw JSFunction*/shape addresses; re-recorded on next miss
  gWJFieldVty.clear();    // shape addresses can be reused across GC -> drop
  gWJElemShape.clear();
  gWJPropByName.clear();  // shape/atom addresses can be reused across GC
  gWJShapeSample.clear();  // raw object pointers; dead after GC
  gWJLenSite.clear();
  gWJFieldPoly.clear();
}

extern "C" void WJTraceRoots(JSTracer* trc, void*);  // defined near wjhelp (below)

// Ops handled by the Mode VS (mutating, no-restart) emitter. A mutating function
// compiles only if EVERY op is supported (any unsupported op -> not compiled, so
// it runs in the interpreter where mutation happens exactly once).
static bool WJModeVSSupported(JSOp op) {
  switch (op) {
    case JSOp::GetArg: case JSOp::SetArg: case JSOp::GetLocal: case JSOp::SetLocal:
    case JSOp::Pop: case JSOp::Dup:
    case JSOp::Zero: case JSOp::One: case JSOp::Int8: case JSOp::Int32:
    case JSOp::Uint16: case JSOp::Uint24:
    case JSOp::Add: case JSOp::Sub: case JSOp::Mul: case JSOp::Div:
    case JSOp::Inc: case JSOp::Dec:
    case JSOp::BitOr: case JSOp::BitAnd: case JSOp::BitXor:
    case JSOp::Lsh: case JSOp::Rsh: case JSOp::Ursh: case JSOp::BitNot:
    case JSOp::Lt: case JSOp::Le: case JSOp::Gt: case JSOp::Ge:
    case JSOp::Eq: case JSOp::Ne: case JSOp::StrictEq: case JSOp::StrictNe:
    case JSOp::JumpIfFalse: case JSOp::JumpIfTrue: case JSOp::Goto:
    case JSOp::And: case JSOp::Or: case JSOp::Coalesce:  // short-circuit (WJShortCircuit)
    case JSOp::GetProp: case JSOp::SetProp: case JSOp::StrictSetProp:
    case JSOp::GetElem: case JSOp::SetElem: case JSOp::StrictSetElem:
    case JSOp::GetGName: case JSOp::GetAliasedVar:
    case JSOp::Null: case JSOp::Undefined: case JSOp::True: case JSOp::False:
    case JSOp::Double: case JSOp::String:
    case JSOp::Call: case JSOp::CallContent: case JSOp::CallIgnoresRv:
    case JSOp::FunctionThis: case JSOp::Swap:
    case JSOp::SetRval: case JSOp::GetRval: case JSOp::Return: case JSOp::RetRval:
    case JSOp::Nop: case JSOp::NopIsAssignOp: case JSOp::NopDestructuring:
    case JSOp::JumpTarget: case JSOp::LoopHead: case JSOp::Pos: case JSOp::ToNumeric:
      return true;
    case JSOp::NewObject: case JSOp::NewInit: case JSOp::InitProp:
      return WJVSInlineAlloc();  // INLINEALLOC (plan §8.3)
    default:
      return false;
  }
}

// ===================== ION MIDDLE-END SMOKE TEST =====================
// Gate for the Ion-tier rewrite. The plan: drive Ion's MIR optimizer
// (OptimizeMIR -> GVN/LICM/ScalarReplacement/AliasAnalysis) from a JS-bytecode
// -> MIR front-end, then emit wasm from the optimized SSA graph. That only works
// if the middle-end actually RUNS in this JS_CODEGEN_NONE build (no native
// backend). This builds a trivial wasm-mode MIR graph and runs OptimizeMIR on it
// to confirm it executes (vs assert/crash). Gated by GECKO_WJVS_IONSMOKE.
static void WJIonSmokeTest() {
  using namespace js::jit;
  fprintf(stderr, "[ion-smoke] begin\n");
  fflush(stderr);

  LifoAlloc lifo(TempAllocator::PreferredLifoChunkSize,
                 js::BackgroundMallocArena);
  TempAllocator alloc(&lifo);
  JitContext jitContext;  // wasm-compilation ctor; compilingWasm() comes from CompileInfo

  CompileInfo compileInfo(/*nlocals=*/0);  // script()==nullptr => compilingWasm()
  MIRGraph graph(&alloc);
  JitCompileOptions options;
  MIRGenerator mirGen(/*realm=*/nullptr, options, &alloc, &graph, &compileInfo,
                      IonOptimizations.get(OptimizationLevel::Wasm),
                      /*wasmCodeMeta=*/nullptr);

  // Trivial but non-empty graph: two congruent int32 constants + an add, then a
  // control terminator. GVN should fold c1==c2; DCE should drop the dead add.
  MBasicBlock* entry =
      MBasicBlock::New(graph, compileInfo, /*pred=*/nullptr, MBasicBlock::NORMAL);
  if (!entry) {
    fprintf(stderr, "[ion-smoke] block alloc FAILED\n");
    fflush(stderr);
    return;
  }
  graph.addBlock(entry);

  MConstant* c1 = MConstant::NewInt32(alloc, 21);
  MConstant* c2 = MConstant::NewInt32(alloc, 21);
  entry->add(c1);
  entry->add(c2);
  MAdd* sum = MAdd::NewWasm(alloc, c1, c2, MIRType::Int32);
  entry->add(sum);
  entry->end(MUnreachable::New(alloc));

  fprintf(stderr, "[ion-smoke] graph built (%zu blocks), calling OptimizeMIR...\n",
          graph.numBlocks());
  fflush(stderr);

  bool ok = OptimizeMIR(&mirGen);

  fprintf(stderr, "[ion-smoke] OptimizeMIR returned %d, blocks now=%zu\n",
          (int)ok, graph.numBlocks());
  fflush(stderr);
}

// ===================== ION MIR -> WASM BACK-END =====================
// The novel half of the Ion-reuse rewrite: walk an OptimizeMIR'd SSA graph and
// emit a wasm function body. Strategy: every MDefinition that is used gets its own
// wasm LOCAL; each instruction computes its value (pushing operands via local.get)
// then local.set's its own local. This "value-per-local" form is trivially correct
// and lets the HOST engine (V8/TurboFan) re-run real register allocation + folding.
// Control-flow + phis come later; slice 1 handles a single block ending in a return.

static uint8_t WJWasmValType(js::jit::MIRType t) {
  using js::jit::MIRType;
  switch (t) {
    case MIRType::Int32:   return uint8_t(TypeCode::I32);
    case MIRType::Int64:   return uint8_t(TypeCode::I64);
    case MIRType::Double:  return uint8_t(TypeCode::F64);
    case MIRType::Float32: return uint8_t(TypeCode::F32);
    case MIRType::Pointer: return uint8_t(TypeCode::I32);  // wasm32
    default:               return 0;  // unsupported
  }
}

namespace {
// Per-compilation back-end state: maps each MDefinition id() to a wasm local slot.
struct WJIonBackend {
  std::vector<int32_t> localOf;   // def id -> wasm local index (-1 = none)
  std::vector<uint8_t> localTy;   // wasm valtype per assigned local (in index order)
  uint32_t paramCount = 0;        // wasm function params occupy the low local indices

  int32_t local(const js::jit::MDefinition* d) const {
    uint32_t id = d->id();
    return id < localOf.size() ? localOf[id] : -1;
  }
  void ensureSize(uint32_t id) {
    if (id >= localOf.size()) localOf.resize(id + 1, -1);
  }
  // Assign a fresh local for def d (call in emission order so indices are stable).
  int32_t assign(const js::jit::MDefinition* d) {
    ensureSize(d->id());
    if (localOf[d->id()] >= 0) return localOf[d->id()];
    uint8_t ty = WJWasmValType(d->type());
    int32_t idx = int32_t(paramCount + localTy.size());
    localTy.push_back(ty);
    localOf[d->id()] = idx;
    return idx;
  }
};
}  // namespace

// Emit `local.get` for an operand's value (it must already have a local assigned).
static bool WJIonGetOperand(Encoder& e, WJIonBackend& be,
                            const js::jit::MDefinition* d) {
  int32_t l = be.local(d);
  if (l < 0) return false;
  return e.writeOp(Op::LocalGet) && e.writeVarU32(uint32_t(l));
}

// Emit the value-computation for one instruction, leaving its result on the wasm
// stack. Returns false on an unsupported node.
static bool WJIonEmitValue(Encoder& e, WJIonBackend& be,
                           js::jit::MDefinition* ins) {
  using namespace js::jit;
  switch (ins->op()) {
    case MDefinition::Opcode::Constant: {
      MConstant* c = ins->toConstant();
      if (c->type() == MIRType::Int32) {
        return e.writeOp(Op::I32Const) && e.writeVarS32(c->toInt32());
      }
      if (c->type() == MIRType::Int64) {
        return e.writeOp(Op::I64Const) && e.writeVarS64(c->toInt64());
      }
      if (c->type() == MIRType::Double) {
        return e.writeOp(Op::F64Const) && e.writeFixedF64(c->numberToDouble());
      }
      return false;
    }
    case MDefinition::Opcode::ReinterpretCast: {
      // Bit-cast between same-width int/float. Used to unbox a boxed double Value
      // (i64) to f64, and to re-box an f64 number into a double Value (i64).
      if (!WJIonGetOperand(e, be, ins->getOperand(0))) return false;
      MIRType from = ins->getOperand(0)->type();
      if (from == MIRType::Int64 && ins->type() == MIRType::Double) {
        return e.writeOp(Op::F64ReinterpretI64);
      }
      if (from == MIRType::Double && ins->type() == MIRType::Int64) {
        return e.writeOp(Op::I64ReinterpretF64);
      }
      return false;
    }
    case MDefinition::Opcode::ExtendInt32ToInt64: {
      if (!WJIonGetOperand(e, be, ins->getOperand(0))) {
        MDefinition* od = ins->getOperand(0);
        fprintf(stderr, "[ion-be] Extend operand op#%u type%u no-local\n",
                unsigned(od->op()), unsigned(od->type()));
        if (od->isPhi()) {
          MPhi* p = od->toPhi();
          for (size_t pi = 0; pi < p->numOperands(); pi++) {
            MDefinition* in = p->getOperand(pi);
            fprintf(stderr, "[ion-be]   phi in#%zu op#%u type%u\n", pi,
                    unsigned(in->op()), unsigned(in->type()));
          }
        }
        return false;
      }
      return e.writeOp(ins->toExtendInt32ToInt64()->isUnsigned()
                           ? Op::I64ExtendI32U
                           : Op::I64ExtendI32S);
    }
    case MDefinition::Opcode::TruncateToInt32: {
      // JS ToInt32 of a number. Use the SATURATING truncation (trunc_sat): JS
      // ToInt32 must NEVER trap (NaN -> 0), whereas i32.trunc_f64_s TRAPS on NaN
      // or out-of-range. Exact for in-range integers (richards' small ints); full
      // JS modulo-2^32 wrap-around is not modeled (clamps instead).
      if (!WJIonGetOperand(e, be, ins->getOperand(0))) return false;
      if (ins->getOperand(0)->type() == MIRType::Double) {
        return e.writeOp(MiscOp::I32TruncSatF64S);
      }
      return ins->getOperand(0)->type() == MIRType::Int32;  // already i32
    }
    case MDefinition::Opcode::WasmBinaryBitwise: {
      if (!WJIonGetOperand(e, be, ins->getOperand(0)) ||
          !WJIonGetOperand(e, be, ins->getOperand(1))) {
        return false;
      }
      auto sub = ins->toWasmBinaryBitwise()->subOpcode();
      bool i64 = ins->type() == MIRType::Int64;
      switch (sub) {
        case MWasmBinaryBitwise::SubOpcode::And:
          return e.writeOp(i64 ? Op::I64And : Op::I32And);
        case MWasmBinaryBitwise::SubOpcode::Or:
          return e.writeOp(i64 ? Op::I64Or : Op::I32Or);
        case MWasmBinaryBitwise::SubOpcode::Xor:
          return e.writeOp(i64 ? Op::I64Xor : Op::I32Xor);
      }
      return false;
    }
    case MDefinition::Opcode::WJIonCall: {
      // Non-inlined call: callee operand -> gWJHelpA, argc -> gWJHelpB, then
      // wjhelp(WJH_IONCALL,0). Args/this were marshalled to gWJScratch by the
      // preceding stores. On deopt (!=0) return 1.0; else load the i64 result.
      MWJIonCall* call = ins->toWJIonCall();
      uint32_t helpA = uint32_t(uintptr_t(&gWJHelpA));
      uint32_t helpB = uint32_t(uintptr_t(&gWJHelpB));
      uint32_t resAddr = uint32_t(uintptr_t(&gWJScratch[kWJResultSlot]));
      if (call->isPostBarrier()) {
        // GC post-write barrier: receiver operand -> gWJHelpA; the stored value was
        // already written to gWJHelpB by the builder; wjhelp(WJH_POSTBARRIER, 0)
        // runs the generational barrier for the obj->value edge. Never deopts; the
        // i64 result is a dummy (the node's result local is unused).
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(helpA)) ||
            !WJIonGetOperand(e, be, call->callee()) ||  // boxed receiver (i64)
            !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
          return false;
        }
        if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_POSTBARRIER)) ||
            !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
            !e.writeOp(Op::Call) || !e.writeVarU32(kWJVSHelpIdx)) {
          return false;
        }
        if (!e.writeOp(Op::Drop)) return false;  // discard the f64 barrier result
        if (!e.writeOp(Op::I64Const) || !e.writeVarS64(0)) return false;  // dummy i64
        return true;
      }
      // GETPROP-helper mode: store the boxed receiver to gWJHelpA, then
      // wjhelp(WJH_GETPROP, propSite). A generic property load that keeps the
      // (possibly stateful) caller running compiled instead of deopt-restarting
      // on an off-shape receiver -- the dominant deltablue cost.
      if (call->propSite() != 0) {
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(helpA)) ||
            !WJIonGetOperand(e, be, call->callee()) ||  // boxed receiver (i64)
            !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
          return false;
        }
        if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_GETPROP)) ||
            !e.writeOp(Op::F64Const) ||
            !e.writeFixedF64(double(call->propSite())) ||
            !e.writeOp(Op::Call) || !e.writeVarU32(kWJVSHelpIdx)) {
          return false;  // -> f64 deopt code
        }
        if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
            !e.writeOp(Op::F64Ne) || !e.writeOp(Op::If) ||
            !e.writeFixedU8(0x40)) {
          return false;  // if (deopt != 0)
        }
        if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(1.0) ||
            !e.writeOp(Op::Return) || !e.writeOp(Op::End)) {
          return false;  //   genuine failure (exception) -> deopt
        }
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(resAddr)) ||
            !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
          return false;  // result (i64) on stack
        }
        return true;
      }
      if (call->setPropSite() != 0) {
        // SETPROP-helper: receiver operand -> gWJHelpA; the value was already stored
        // to gWJHelpB by the builder; wjhelp(WJH_SETPROP, site) does SetProperty and
        // returns the value in the result slot. A cold/no-record field WRITE that
        // keeps the function compiled instead of bailing.
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(helpA)) ||
            !WJIonGetOperand(e, be, call->callee()) ||  // boxed receiver (i64)
            !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
          return false;
        }
        if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_SETPROP)) ||
            !e.writeOp(Op::F64Const) ||
            !e.writeFixedF64(double(call->setPropSite())) ||
            !e.writeOp(Op::Call) || !e.writeVarU32(kWJVSHelpIdx)) {
          return false;
        }
        if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
            !e.writeOp(Op::F64Ne) || !e.writeOp(Op::If) ||
            !e.writeFixedU8(0x40)) {
          return false;
        }
        if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(1.0) ||
            !e.writeOp(Op::Return) || !e.writeOp(Op::End)) {
          return false;
        }
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(resAddr)) ||
            !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
          return false;  // result (the stored value, i64) on stack
        }
        return true;
      }
      bool isMeth = call->methName() != 0;
      // gWJHelpA = callee fn (IONCALL) OR the PropertyName* (METHCALL).
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(helpA))) return false;
      if (isMeth) {
        if (!e.writeOp(Op::I64Const) ||
            !e.writeVarS64(int64_t(uint64_t(call->methName())))) {
          return false;
        }
      } else {
        if (!WJIonGetOperand(e, be, call->callee())) return false;  // i64 callee
      }
      if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
        return false;
      }
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(helpB)) ||
          !e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(call->argc())) ||
          !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
        return false;
      }
      uint32_t helpKind = call->isConstruct() ? uint32_t(WJH_CONSTRUCT)
                          : isMeth             ? uint32_t(WJH_METHCALL)
                                               : uint32_t(WJH_IONCALL);
      if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(helpKind)) ||
          !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
          !e.writeOp(Op::Call) || !e.writeVarU32(kWJVSHelpIdx)) {
        return false;  // -> f64 deopt code
      }
      if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
          !e.writeOp(Op::F64Ne) || !e.writeOp(Op::If) ||
          !e.writeFixedU8(0x40)) {
        return false;  // if (deopt != 0)
      }
      if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(1.0) ||
          !e.writeOp(Op::Return) || !e.writeOp(Op::End)) {
        return false;  //   return 1.0 (whole fn deopts to interpreter)
      }
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(resAddr)) ||
          !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0)) {
        return false;  // result (i64) on stack
      }
      return true;
    }
    case MDefinition::Opcode::Lsh:
    case MDefinition::Opcode::Rsh:
    case MDefinition::Opcode::Ursh: {
      if (!WJIonGetOperand(e, be, ins->getOperand(0)) ||
          !WJIonGetOperand(e, be, ins->getOperand(1))) {
        return false;
      }
      return e.writeOp(ins->isLsh()   ? Op::I32Shl
                       : ins->isRsh() ? Op::I32ShrS
                                      : Op::I32ShrU);
    }
    case MDefinition::Opcode::WasmSelect: {
      MWasmSelect* sel = ins->toWasmSelect();
      if (!WJIonGetOperand(e, be, sel->trueExpr()) ||
          !WJIonGetOperand(e, be, sel->falseExpr()) ||
          !WJIonGetOperand(e, be, sel->condExpr())) {
        return false;
      }
      return e.writeOp(Op::SelectNumeric);
    }
    case MDefinition::Opcode::WasmFloatConstant: {
      MWasmFloatConstant* c = ins->toWasmFloatConstant();
      if (c->type() == MIRType::Double) {
        return e.writeOp(Op::F64Const) && e.writeFixedF64(c->toDouble());
      }
      return false;
    }
    case MDefinition::Opcode::Add: {
      if (!WJIonGetOperand(e, be, ins->getOperand(0)) ||
          !WJIonGetOperand(e, be, ins->getOperand(1))) {
        return false;
      }
      if (ins->type() == MIRType::Int32) return e.writeOp(Op::I32Add);
      if (ins->type() == MIRType::Double) return e.writeOp(Op::F64Add);
      return false;
    }
    case MDefinition::Opcode::Sub: {
      if (!WJIonGetOperand(e, be, ins->getOperand(0)) ||
          !WJIonGetOperand(e, be, ins->getOperand(1))) {
        return false;
      }
      if (ins->type() == MIRType::Int32) return e.writeOp(Op::I32Sub);
      if (ins->type() == MIRType::Double) return e.writeOp(Op::F64Sub);
      return false;
    }
    case MDefinition::Opcode::Mul: {
      if (!WJIonGetOperand(e, be, ins->getOperand(0)) ||
          !WJIonGetOperand(e, be, ins->getOperand(1))) {
        return false;
      }
      if (ins->type() == MIRType::Int32) return e.writeOp(Op::I32Mul);
      if (ins->type() == MIRType::Double) return e.writeOp(Op::F64Mul);
      return false;
    }
    case MDefinition::Opcode::Div: {
      if (!WJIonGetOperand(e, be, ins->getOperand(0)) ||
          !WJIonGetOperand(e, be, ins->getOperand(1))) {
        return false;
      }
      if (ins->type() == MIRType::Double) return e.writeOp(Op::F64Div);
      return false;
    }
    case MDefinition::Opcode::Compare: {
      MCompare* c = ins->toCompare();
      if (!WJIonGetOperand(e, be, c->getOperand(0)) ||
          !WJIonGetOperand(e, be, c->getOperand(1))) {
        return false;
      }
      if (c->compareType() == MCompare::Compare_Double) {
        switch (c->jsop()) {
          case JSOp::Lt: return e.writeOp(Op::F64Lt);
          case JSOp::Le: return e.writeOp(Op::F64Le);
          case JSOp::Gt: return e.writeOp(Op::F64Gt);
          case JSOp::Ge: return e.writeOp(Op::F64Ge);
          case JSOp::Eq:
          case JSOp::StrictEq: return e.writeOp(Op::F64Eq);
          case JSOp::Ne:
          case JSOp::StrictNe: return e.writeOp(Op::F64Ne);
          default: return false;
        }
      }
      if (c->compareType() == MCompare::Compare_Int32 ||
          c->compareType() == MCompare::Compare_UInt32) {
        bool u = c->compareType() == MCompare::Compare_UInt32;
        switch (c->jsop()) {
          case JSOp::Lt: return e.writeOp(u ? Op::I32LtU : Op::I32LtS);
          case JSOp::Le: return e.writeOp(u ? Op::I32LeU : Op::I32LeS);
          case JSOp::Gt: return e.writeOp(u ? Op::I32GtU : Op::I32GtS);
          case JSOp::Ge: return e.writeOp(u ? Op::I32GeU : Op::I32GeS);
          case JSOp::Eq:
          case JSOp::StrictEq: return e.writeOp(Op::I32Eq);
          case JSOp::Ne:
          case JSOp::StrictNe: return e.writeOp(Op::I32Ne);
          default: return false;
        }
      }
      if (c->compareType() == MCompare::Compare_Int64 ||
          c->compareType() == MCompare::Compare_UInt64) {
        switch (c->jsop()) {
          case JSOp::Eq:
          case JSOp::StrictEq: return e.writeOp(Op::I64Eq);
          case JSOp::Ne:
          case JSOp::StrictNe: return e.writeOp(Op::I64Ne);
          default: return false;
        }
      }
      return false;
    }
    case MDefinition::Opcode::ToDouble: {
      MDefinition* in = ins->getOperand(0);
      if (!WJIonGetOperand(e, be, in)) return false;
      if (in->type() == MIRType::Int32) return e.writeOp(Op::F64ConvertI32S);
      if (in->type() == MIRType::Double) return true;  // already f64
      return false;
    }
    case MDefinition::Opcode::WasmLoad: {
      MWasmLoad* l = ins->toWasmLoad();
      if (!WJIonGetOperand(e, be, l->base())) return false;
      const wasm::MemoryAccessDesc& acc = l->access();
      uint32_t a = acc.align();
      uint32_t alignLog2 = a >= 8 ? 3 : a >= 4 ? 2 : a >= 2 ? 1 : 0;
      Op loadOp;
      switch (acc.type()) {
        case js::Scalar::Int32:
        case js::Scalar::Uint32: loadOp = Op::I32Load; break;
        case js::Scalar::Float64: loadOp = Op::F64Load; break;
        case js::Scalar::Int64: loadOp = Op::I64Load; break;
        default: return false;
      }
      return e.writeOp(loadOp) && e.writeVarU32(alignLog2) &&
             e.writeVarU32(uint32_t(acc.offset64()));
    }
    case MDefinition::Opcode::WrapInt64ToInt32: {
      // Extract a half of an i64. bottomHalf=true -> low 32 (i32.wrap_i64);
      // bottomHalf=false -> HIGH 32 (shift right 32, then wrap). The high half is
      // needed for tag-aware truthiness (the NaN-box tag lives in the high word);
      // emitting low32 unconditionally (the old behaviour) silently broke `if(obj)`.
      if (!WJIonGetOperand(e, be, ins->getOperand(0))) return false;
      if (!ins->toWrapInt64ToInt32()->bottomHalf()) {
        if (!e.writeOp(Op::I64Const) || !e.writeVarU64(32) ||
            !e.writeOp(Op::I64ShrU)) {
          return false;
        }
      }
      return e.writeOp(Op::I32WrapI64);
    }
    default:
      return false;
  }
}

// Emit an effectful MWasmStore: push base + value, then iN.store. Has no result
// local (its effect is the heap write), so the body emitter calls this directly
// instead of the value+local.set path.
static bool WJIonEmitStore(Encoder& e, WJIonBackend& be,
                           js::jit::MWasmStore* st) {
  using namespace js::jit;
  if (!WJIonGetOperand(e, be, st->base()) ||
      !WJIonGetOperand(e, be, st->value())) {
    return false;
  }
  const wasm::MemoryAccessDesc& acc = st->access();
  uint32_t a = acc.align();
  uint32_t alignLog2 = a >= 8 ? 3 : a >= 4 ? 2 : a >= 2 ? 1 : 0;
  Op storeOp;
  switch (acc.type()) {
    case js::Scalar::Int32:
    case js::Scalar::Uint32: storeOp = Op::I32Store; break;
    case js::Scalar::Float64: storeOp = Op::F64Store; break;
    case js::Scalar::Int64: storeOp = Op::I64Store; break;
    default: return false;
  }
  return e.writeOp(storeOp) && e.writeVarU32(alignLog2) &&
         e.writeVarU32(uint32_t(acc.offset64()));
}

// Emit phi-destruction copies for the edge (from -> to): for each phi in `to`,
// copy the operand corresponding to `from`'s predecessor slot into the phi's
// PARALLEL-COPY correct: a phi's source may BE another phi's destination at the
// same join (loop headers, and -- the bug -- the poly-field / dispatch diamonds that
// add joins with many live-slot phis). Sequential `dst = src` then `dst2 = src2`
// loses values when src2 is dst (it reads the already-overwritten new value). Fix:
// read ALL sources onto the wasm value stack FIRST (so every read sees the
// predecessor's OLD value), THEN pop into the dests in reverse (the stack is LIFO).
// This is a correct parallel copy for any dependency pattern, no cycle detection.
static bool WJIonEmitEdgeCopies(Encoder& e, WJIonBackend& be,
                                js::jit::MBasicBlock* from,
                                js::jit::MBasicBlock* to) {
  using namespace js::jit;
  uint32_t k = to->getPredecessorIndex(from);
  std::vector<int32_t> dsts;
  for (MPhiIterator p = to->phisBegin(); p != to->phisEnd(); p++) {
    MPhi* phi = *p;
    int32_t dst = be.local(phi);
    if (dst < 0) continue;
    MDefinition* src = phi->getOperand(k);
    if (!WJIonGetOperand(e, be, src)) return false;  // push old value
    dsts.push_back(dst);
  }
  for (size_t i = dsts.size(); i-- > 0;) {  // pop in reverse (LIFO) into dests
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(uint32_t(dsts[i]))) return false;
  }
  return true;
}

// Set $bid to `target`'s dispatch index and branch back to the dispatch loop.
static bool WJIonGotoBlock(Encoder& e, uint32_t bidLocal, uint32_t loopDepthFromHere,
                           uint32_t targetIndex) {
  return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(targetIndex)) &&
         e.writeOp(Op::LocalSet) && e.writeVarU32(bidLocal) &&
         e.writeOp(Op::Br) && e.writeVarU32(loopDepthFromHere);
}

// Walk the optimized graph and emit the wasm function body. Control flow is
// lowered to a single dispatch loop: each MBasicBlock becomes a case selected by
// a $bid i32 local via br_table; block transitions set $bid + br back to the loop;
// phis are destructed into per-edge local copies. Correct for any reducible CFG;
// TurboFan re-optimizes the shape. argParams[0..nargs) are the wasm parameters.
static bool WJIonEmitBody(js::jit::MIRGraph& graph, Encoder& e,
                          js::jit::MWasmParameter** argParams, uint32_t nargs) {
  using namespace js::jit;
  const uint8_t kVoid = 0x40;
  WJIonBackend be;
  be.paramCount = nargs;
  for (uint32_t i = 0; i < nargs; i++) {
    be.ensureSize(argParams[i]->id());
    be.localOf[argParams[i]->id()] = int32_t(i);
  }

  // Collect blocks in RPO and assign dispatch indices.
  std::vector<MBasicBlock*> blocks;
  std::unordered_map<MBasicBlock*, uint32_t> blockIdx;
  for (ReversePostorderIterator b = graph.rpoBegin(); b != graph.rpoEnd(); b++) {
    blockIdx[*b] = uint32_t(blocks.size());
    blocks.push_back(*b);
  }
  const uint32_t n = uint32_t(blocks.size());

  // Assign locals: phis first (so edge copies can target them), then each
  // value-producing instruction. Record per-block emission order.
  for (MBasicBlock* b : blocks) {
    for (MPhiIterator p = b->phisBegin(); p != b->phisEnd(); p++) {
      if (WJWasmValType((*p)->type()) != 0) be.assign(*p);
    }
  }
  for (MBasicBlock* b : blocks) {
    for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
      MInstruction* ins = *it;
      if (ins->isControlInstruction() || ins->isWasmParameter()) continue;
      if (WJWasmValType(ins->type()) == 0) continue;
      be.assign(ins);
    }
  }

  // $bid selector local is the LAST local (index nargs + localTy.size()).
  const uint32_t bidLocal = be.paramCount + uint32_t(be.localTy.size());

  // Locals declaration: all value/phi locals (f64/...) + the $bid i32.
  if (!e.writeVarU32(uint32_t(be.localTy.size()) + 1)) return false;
  for (uint8_t ty : be.localTy) {
    if (!e.writeVarU32(1) || !e.writeFixedU8(ty)) return false;
  }
  if (!e.writeVarU32(1) || !e.writeFixedU8(uint8_t(TypeCode::I32))) return false;

  // Fast path: single block, no control flow -> emit straight-line (no loop).
  if (n == 1) {
    MBasicBlock* b = blocks[0];
    for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
      MInstruction* ins = *it;
      if (ins->isControlInstruction() || ins->isWasmParameter()) continue;
      if (ins->isWasmStore()) {
        if (!WJIonEmitStore(e, be, ins->toWasmStore())) return false;
        continue;
      }
      if (be.local(ins) < 0) continue;
      if (!WJIonEmitValue(e, be, ins)) {
        fprintf(stderr, "[ion-be] unsupported node op#%u type%u nOperands%u\n", unsigned(ins->op()), unsigned(ins->type()), unsigned(ins->numOperands()));
        return false;
      }
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(uint32_t(be.local(ins)))) {
        return false;
      }
    }
    MControlInstruction* t = b->lastIns();
    if (t->isWasmReturn()) {
      if (!WJIonGetOperand(e, be, t->getOperand(0))) return false;
      if (!e.writeOp(Op::Return)) return false;
    } else if (t->isUnreachable()) {
      if (!e.writeOp(Op::Unreachable)) return false;
    } else {
      return false;
    }
    return e.writeOp(Op::End);
  }

  // Dispatch loop. Nesting (outer->inner): loop $L { block $b0 { block $b1 {
  // ... { block $b_{n-1} { br_table } } body_{n-1} } body_{n-2} } ... } body_0.
  // From inside the innermost block, $b_i is at depth (n-1-i); $L is at depth n.
  if (!e.writeOp(Op::Loop) || !e.writeFixedU8(kVoid)) return false;  // $L
  for (uint32_t i = 0; i < n; i++) {
    if (!e.writeOp(Op::Block) || !e.writeFixedU8(kVoid)) return false;  // $b_i
  }
  // br_table on $bid: entry i -> depth (n-1-i); default -> 0 (=$b_{n-1}).
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(bidLocal)) return false;
  if (!e.writeOp(Op::BrTable) || !e.writeVarU32(n)) return false;
  for (uint32_t i = 0; i < n; i++) {
    if (!e.writeVarU32(n - 1 - i)) return false;
  }
  if (!e.writeVarU32(0)) return false;  // default

  // Bodies in reverse order: B_{n-1} (right after innermost end) down to B0.
  for (uint32_t ri = 0; ri < n; ri++) {
    uint32_t bi = n - 1 - ri;
    MBasicBlock* b = blocks[bi];
    if (!e.writeOp(Op::End)) return false;  // close block $b_{bi}; body starts here

    for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
      MInstruction* ins = *it;
      if (ins->isControlInstruction() || ins->isWasmParameter()) continue;
      if (ins->isWasmStore()) {
        if (!WJIonEmitStore(e, be, ins->toWasmStore())) return false;
        continue;
      }
      if (be.local(ins) < 0) continue;
      if (!WJIonEmitValue(e, be, ins)) {
        fprintf(stderr, "[ion-be] unsupported node op#%u type%u nOperands%u\n", unsigned(ins->op()), unsigned(ins->type()), unsigned(ins->numOperands()));
        return false;
      }
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(uint32_t(be.local(ins)))) {
        return false;
      }
    }

    // Depth of $L from this body: we have closed (ri+1) blocks, so the remaining
    // enclosing blocks are (n-1-ri) plus the loop => $L at depth (n-1-ri).
    uint32_t loopDepthHere = n - 1 - ri;
    MControlInstruction* t = b->lastIns();
    if (t->isWasmReturn()) {
      if (!WJIonGetOperand(e, be, t->getOperand(0)) || !e.writeOp(Op::Return)) {
        return false;
      }
    } else if (t->isUnreachable()) {
      if (!e.writeOp(Op::Unreachable)) return false;
    } else if (t->isGoto()) {
      MBasicBlock* s = t->getSuccessor(0);
      if (!WJIonEmitEdgeCopies(e, be, b, s)) return false;
      if (!WJIonGotoBlock(e, bidLocal, loopDepthHere, blockIdx[s])) return false;
    } else if (t->isTest()) {
      MTest* test = t->toTest();
      MBasicBlock* tb = test->ifTrue();
      MBasicBlock* fb = test->ifFalse();
      // if (cond) { copies(true); bid=true; br $L } else { ...false... }
      if (!WJIonGetOperand(e, be, test->getOperand(0))) return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
      if (!WJIonEmitEdgeCopies(e, be, b, tb)) return false;
      // inside the If, $L is one deeper.
      if (!WJIonGotoBlock(e, bidLocal, loopDepthHere + 1, blockIdx[tb])) return false;
      if (!e.writeOp(Op::Else)) return false;
      if (!WJIonEmitEdgeCopies(e, be, b, fb)) return false;
      if (!WJIonGotoBlock(e, bidLocal, loopDepthHere + 1, blockIdx[fb])) return false;
      if (!e.writeOp(Op::End)) return false;  // end If
      if (!e.writeOp(Op::Unreachable)) return false;  // both arms branched
    } else {
      return false;
    }
  }

  if (!e.writeOp(Op::Unreachable)) return false;  // fell out of dispatch
  if (!e.writeOp(Op::End)) return false;          // end loop $L
  // The loop never exits (all paths return/br); the function-level fallthru is
  // unreachable, but wasm still requires the body to be stack-valid at end.
  if (!e.writeOp(Op::Unreachable)) return false;
  return e.writeOp(Op::End);                       // end function body
}

// End-to-end back-end test: build a tiny MIR graph (() -> f64 returning 84.0),
// OptimizeMIR it, emit a complete wasm module, host-compile + instantiate + call,
// and verify the result. Proves the MIR->wasm output stage. Gated GECKO_WJVS_IONBE.
static void WJIonBackendTest() {
  using namespace js::jit;
  fprintf(stderr, "[ion-be] begin\n");
  fflush(stderr);

  LifoAlloc lifo(TempAllocator::PreferredLifoChunkSize,
                 js::BackgroundMallocArena);
  TempAllocator alloc(&lifo);
  JitContext jitContext;

  CompileInfo compileInfo(/*nlocals=*/0);
  MIRGraph graph(&alloc);
  JitCompileOptions options;
  MIRGenerator mirGen(nullptr, options, &alloc, &graph, &compileInfo,
                      IonOptimizations.get(OptimizationLevel::Wasm), nullptr);

  MBasicBlock* entry =
      MBasicBlock::New(graph, compileInfo, nullptr, MBasicBlock::NORMAL);
  if (!entry) { fprintf(stderr, "[ion-be] block FAILED\n"); return; }
  graph.addBlock(entry);

  // instance param so MWasmReturn is well-formed; never materialized in wasm.
  MWasmParameter* instance =
      MWasmParameter::New(alloc, ABIArg(InstanceReg), MIRType::Pointer);
  entry->add(instance);

  // (42.0 + 42.0) = 84.0; two congruent consts let GVN dedup.
  MWasmFloatConstant* d1 = MWasmFloatConstant::NewDouble(alloc, 42.0);
  MWasmFloatConstant* d2 = MWasmFloatConstant::NewDouble(alloc, 42.0);
  entry->add(d1);
  entry->add(d2);
  MAdd* sum = MAdd::NewWasm(alloc, d1, d2, MIRType::Double);
  entry->add(sum);
  entry->end(MWasmReturn::New(alloc, sum, instance));

  if (!OptimizeMIR(&mirGen)) {
    fprintf(stderr, "[ion-be] OptimizeMIR FAILED\n");
    return;
  }
  fprintf(stderr, "[ion-be] optimized, blocks=%zu\n", graph.numBlocks());

  // Assemble a complete module: type ()->f64, one func, export "f", code = body.
  Bytes out;
  Encoder e(out);
  const uint8_t kF64 = uint8_t(TypeCode::F64);
  if (!e.writeFixedU32(MagicNumber) ||
      !e.writeFixedU32(EncodingVersionModule)) {
    fprintf(stderr, "[ion-be] header FAILED\n");
    return;
  }
  size_t s;
  // Type section: type0 = () -> f64
  if (!e.startSection(SectionId::Type, &s) || !e.writeVarU32(1) ||
      !e.writeFixedU8(0x60) || !e.writeVarU32(0) || !e.writeVarU32(1) ||
      !e.writeFixedU8(kF64)) {
    fprintf(stderr, "[ion-be] type sec FAILED\n");
    return;
  }
  e.finishSection(s);
  // Function section: func0 : type0
  if (!e.startSection(SectionId::Function, &s) || !e.writeVarU32(1) ||
      !e.writeVarU32(0)) {
    return;
  }
  e.finishSection(s);
  // Export section: "f" = func0
  if (!e.startSection(SectionId::Export, &s) || !e.writeVarU32(1) ||
      !e.writeBytes("f", 1) || !e.writeFixedU8(0x00) || !e.writeVarU32(0)) {
    return;
  }
  e.finishSection(s);
  // Code section
  if (!e.startSection(SectionId::Code, &s) || !e.writeVarU32(1)) return;
  size_t bodyOff;
  if (!e.writePatchableVarU32(&bodyOff)) return;
  size_t bodyStart = e.currentOffset();
  if (!WJIonEmitBody(graph, e, /*argParams=*/nullptr, /*nargs=*/0)) {
    fprintf(stderr, "[ion-be] EMIT BODY FAILED\n");
    return;
  }
  e.patchVarU32(bodyOff, uint32_t(e.currentOffset() - bodyStart));
  e.finishSection(s);

  fprintf(stderr, "[ion-be] module built (%zu bytes), host-compiling...\n",
          size_t(out.length()));
  fflush(stderr);

  int handle = wasmhost_compile(out.begin(), int(out.length()));
  if (handle < 0) {
    fprintf(stderr, "[ion-be] HOST COMPILE FAILED (invalid wasm)\n");
    return;
  }
  int rc = wasmhost_instantiate(handle, nullptr, 0);
  if (rc != 0) {
    fprintf(stderr, "[ion-be] INSTANTIATE FAILED rc=%d\n", rc);
    return;
  }
  double result = wasmhost_call(handle, 0, nullptr, 0);
  fprintf(stderr, "[ion-be] RESULT = %g (expect 84) %s\n", result,
          result == 84.0 ? "OK" : "*** MISMATCH ***");
  fflush(stderr);
}

// ===================== ION BYTECODE -> MIR FRONT-END =====================
// Slice 2: numeric functions with reducible control flow (if/else + while loops).
// Every JS arg/local/value/rval is f64 (matching the existing Mode N/V model).
// Locals + rval live in MIR SLOTS (CompileInfo) so MBasicBlock's slot machinery
// auto-creates phis at joins/loop headers; the JS operand stack is tracked via
// block push/pop. Control flow uses WarpBuilder's pending-edge algorithm:
// JumpTarget/LoopHead start blocks; forward branches register pending edges that
// resolve into phis; loop backedges call setBackedgeWasm. Bails on anything
// outside the supported set (property access, calls, &&/||, SetArg, bitops, ...).
namespace {
struct WJPendEdge {
  js::jit::MBasicBlock* block;
  uint32_t successor;  // index into block->lastIns() successors
};
}  // namespace

static constexpr int kWJInlineMaxDepth = 6;

// Resolve a recorded inline callee (monomorphic way 0) for a call site, with an
// arg-count check. Returns nullptr if not statically inlinable.
static JSScript* WJResolveInlineCallee(JSScript* fscript, uint32_t pcOff,
                                       uint32_t argc) {
  auto it = gWJInlineCallee.find(WJInlineKey(fscript, pcOff));
  if (it == gWJInlineCallee.end() || it->second.n == 0) return nullptr;
  if (!WJGuestPtrOk(it->second.fns[0])) return nullptr;  // stale fn ptr -> no inline
  JSFunction* fun = reinterpret_cast<JSFunction*>(uintptr_t(it->second.fns[0]));
  if (!fun || !fun->isInterpreted() || !fun->baseScript() ||
      !fun->baseScript()->hasBytecode() || fun->nargs() != argc) {
    return nullptr;
  }
  JSScript* cs = fun->baseScript()->asJSScript();
  // Don't INLINE a callee that reads closed-over vars (JSOp::GetAliasedVar): an
  // inlined frame's environment differs from gWJCurEnv, so an inlined GetAliasedVar
  // bails the whole caller (navier's lin_solve/advect, which inline grid helpers).
  // Returning null here makes the caller emit a NON-INLINED call instead -- the
  // standalone callee runs with its own gWJCurEnv (set by WasmJitRunCall), so its
  // GetAliasedVar at depth 0 is correct -- letting the hot caller compile. DEFAULT
  // OFF (GECKO_WJVS_NOINLINEALIASED to enable): measured net-NEGATIVE on navier
  // (0.99x -> 0.89x) -- it lets the solver fns compile but the non-inlined
  // wjhelp->interpreter calls cost more than the compiled body saves. Needs
  // compiled-to-compiled call_indirect (not wjhelp) to be a win.
  if (getenv("GECKO_WJVS_NOINLINEALIASED")) {
    for (jsbytecode* p = cs->code(); p < cs->codeEnd(); p += GetBytecodeLength(p)) {
      if (JSOp(*p) == JSOp::GetAliasedVar) return nullptr;
    }
  }
  return cs;
}

// Upper bound on the extra CompileInfo local slots needed to inline the whole
// call graph rooted at `cs`: one frame's (args + locals + rval) per inlined
// callee instance, summed over the inline expansion up to the depth bound. The
// builder allocates inline-frame slot ranges sequentially and must stay within
// info.nlocals(); sizing CompileInfo with this count guarantees that.
static uint32_t WJCountInlineSlots(JSScript* cs, int depth) {
  if (depth > kWJInlineMaxDepth) return 0;
  uint32_t total = 0;
  jsbytecode* start = cs->code();
  jsbytecode* end = cs->codeEnd();
  // Sum a frame's slots (args+locals+rval) plus its inline subtree.
  auto addFn = [&](uint32_t fnLow, int wantArgc) {
    if (!WJGuestPtrOk(fnLow)) return;  // stale/desynced fn ptr -> don't deref
    JSFunction* fun = reinterpret_cast<JSFunction*>(uintptr_t(fnLow));
    if (!fun || !fun->isInterpreted() || !fun->baseScript() ||
        !fun->baseScript()->hasBytecode()) {
      return;
    }
    if (wantArgc >= 0 && int(fun->nargs()) != wantArgc) return;
    JSScript* callee = fun->baseScript()->asJSScript();
    total += fun->nargs() + callee->nfixed() + 1 +
             WJCountInlineSlots(callee, depth + 1);
  };
  for (jsbytecode* pc = start; pc < end; pc += GetBytecodeLength(pc)) {
    JSOp op = JSOp(*pc);
    uint32_t off = uint32_t(pc - start);
    if (op == JSOp::Call || op == JSOp::CallContent ||
        op == JSOp::CallIgnoresRv) {
      uint32_t argc = GET_ARGC(pc);
      auto it = gWJInlineCallee.find(WJInlineKey(cs, off));
      if (it == gWJInlineCallee.end() || it->second.n == 0) continue;
      total += 1;  // dispatch merge slot
      for (uint8_t w = 0; w < it->second.n; w++) addFn(it->second.fns[w], argc);
    } else if (op == JSOp::GetProp) {
      // Field-read helper-fallback scratch slot -- ONLY when the (default-off)
      // poly-field helper diamond is enabled (it merges the inline load and the
      // wjhelp(WJH_GETPROP) fallback in one scratch slot). Off by default so the
      // slot budget matches the legacy mono path exactly (don't inflate it, which
      // lets more functions inline and exposes latent miscompiles e.g. Earley).
      // The poly-field diamond consumes 1 scratch slot for the merge + up to N for
      // operand-stack spill/reload across the join. Reserve generous headroom.
      if (getenv("GECKO_WJVS_POLYFIELD") || getenv("GECKO_WJVS_POLYALL")) total += 10;
      // Method-load site -> polymorphic dispatch inlines every way's body.
      auto mit = gWJMethodPoly.find(WJInlineKey(cs, off));
      if (mit == gWJMethodPoly.end() || mit->second.n == 0) continue;
      total += 1;  // dispatch merge slot
      for (uint8_t w = 0; w < mit->second.n; w++) addFn(mit->second.fns[w], -1);
    } else if (op == JSOp::Or || op == JSOp::And || op == JSOp::Coalesce) {
      total += 1;  // logical-operator result slot (operand stack across a branch)
    } else if (op == JSOp::Goto) {
      total += 1;  // headroom for a ternary join slot (value carried over a Goto)
    }
  }
  return total;
}

static bool WJIonBuildMIR(JSScript* script, js::jit::MIRGenerator& mir,
                          js::jit::TempAllocator& alloc, js::jit::MIRGraph& graph,
                          js::jit::CompileInfo& info,
                          js::jit::MBasicBlock* entry,
                          js::jit::MDefinition* instance,
                          js::jit::MWasmParameter** argParams, uint32_t nargs,
                          uint32_t nfixed,
                          js::jit::MDefinition* topThisDef = nullptr,
                          js::jit::MDefinition* scratchResultBase = nullptr,
                          js::jit::MDefinition** scratchArgs = nullptr) {
  using namespace js::jit;
  jsbytecode* const start = script->code();
  jsbytecode* const end = script->codeEnd();

  auto konstIn = [&](MBasicBlock* b, double v) -> MDefinition* {
    MWasmFloatConstant* c = MWasmFloatConstant::NewDouble(alloc, v);
    b->add(c);
    return c;
  };
  // Entry-block slot init: args from params; every other local slot (this
  // frame's locals + rval AND all inline-frame slot ranges) to 0.0, so blocks
  // created before an inline frame is entered have non-null slots to copy/phi.
  for (uint32_t i = 0; i < nargs; i++) {
    // Scratch-ABI args are i64 boxed Values loaded from gWJScratch; direct-ABI
    // args are the wasm parameters.
    entry->initSlot(info.localSlot(i),
                    scratchArgs ? scratchArgs[i] : argParams[i]);
  }
  // UNIFORM i64 SLOTS: every local slot holds a NaN-boxed i64 Value, so a loop
  // header's slot phis always merge i64==i64 (never Double-vs-Value, which the
  // NDEBUG build silently mis-builds since setBackedgeWasm's type assert is
  // compiled out -> OptimizeMIR then miscompiles the loop). The boxed double 0.0
  // is bit-identical to i64 0, so the slot-zero init is an i64 const 0.
  for (uint32_t i = nargs; i < info.nlocals(); i++) {
    MConstant* z = MConstant::NewInt64(alloc, 0);
    entry->add(z);
    entry->initSlot(info.localSlot(i), z);
  }

  MBasicBlock* cur = entry;
  // pending/loopHeader/loopHeadOff are keyed by NAMESPACED offset (frame.offBase
  // + local pc offset) so inline frames built into the same graph don't collide.
  std::unordered_map<uint32_t, std::vector<WJPendEdge>> pending;
  std::unordered_map<uint32_t, MBasicBlock*> loopHeader;
  // Logical operators (||/&&): the result value spans a branch (the operand stack
  // is normally empty at block boundaries). We spill it to a per-op slot at the
  // branch; at the target join the slot is phi-merged and reloaded. Maps the
  // (namespaced) target offset -> that slot.
  std::unordered_map<uint32_t, uint32_t> logicalJoinSlot;
  std::unordered_set<uint32_t> loopHeadOff;
  uint32_t loopDepth = 0;
  // Front-end CSE for property access (Ion's GVN can't CSE MWasmLoad -- it has
  // no congruentTo). Within a linear dominator region: remember which receivers
  // have had their shape guarded (skip the redundant guard) and cache loaded
  // field values (reuse instead of re-loading). Both are invalidated wherever
  // dominance breaks (joins, loop headers) and by any heap mutation (SetProp).
  std::unordered_map<MDefinition*, uint32_t> guardedShape;  // recv -> guarded shape
  std::vector<std::tuple<MDefinition*, uint32_t, MDefinition*>> fieldCache;
  // Memoized object-pointer unbox (boxed i64 recv -> i32 ptr), so repeated field
  // accesses on the same receiver share one wrap node and stay CSE-able.
  std::unordered_map<MDefinition*, MDefinition*> objPtrMemo;
  auto invalidateCSE = [&]() {
    guardedShape.clear();
    fieldCache.clear();
    objPtrMemo.clear();
  };
  // Method-call idiom: maps a callee-placeholder def (the re-pushed receiver) to
  // the method-load GetProp pcOff, so the Call can find its polymorphic dispatch
  // record (gWJMethodPoly) and inline each receiver type's body behind a guard.
  std::unordered_map<MDefinition*, uint32_t> methodOffOf;
  // The JS operand stack lives OUTSIDE block slots (empty at every block
  // boundary, so it never crosses blocks / needs phis). Each inline frame gets
  // its OWN operand stack (the callee's stack is independent of the caller's),
  // tracked via `curStk`; inline-frame args/locals/rval live in dedicated
  // CompileInfo slots starting at `nextSlotBase`.
  std::vector<MDefinition*> rootStk;
  std::vector<MDefinition*>* curStk = &rootStk;
  uint32_t nextSlotBase = nargs + nfixed + 1;
  uint32_t nextOffBase = uint32_t(end - start) + 1;
  uint64_t gWJBuildOps = 0;  // decode-iteration guard (catch builder infinite loops)
  uint32_t gWJBuildFrames = 0;  // total inlined frames (catch inline explosion)

  auto konst = [&](double v) -> MDefinition* { return konstIn(cur, v); };
  auto push = [&](MDefinition* d) { curStk->push_back(d); };
  auto pop = [&]() -> MDefinition* {
    MDefinition* d = curStk->back();
    curStk->pop_back();
    return d;
  };
  // Boxed-value unbox helpers (assigned below, after konstI32). Forward-declared
  // so binF64/cmp can use them; identity on Double/Int32 (existing typed values).
  std::function<MDefinition*(MDefinition*)> asNumber, asObjPtr, boxObj,
      boxForStore;
  auto binF64 = [&](MInstruction* (*mk)(TempAllocator&, MDefinition*,
                                        MDefinition*)) {
    MDefinition* b = pop();
    MDefinition* a = pop();
    a = asNumber(a);
    b = asNumber(b);
    MInstruction* n = mk(alloc, a, b);
    cur->add(n);
    push(n);
  };
  auto cmp = [&](JSOp jsop) {
    MDefinition* b = pop();
    MDefinition* a = pop();
    bool rel = (jsop == JSOp::Lt || jsop == JSOp::Le || jsop == JSOp::Gt ||
                jsop == JSOp::Ge);
    bool objA = a->type() == MIRType::Int32 || a->type() == MIRType::Int64;
    bool objB = b->type() == MIRType::Int32 || b->type() == MIRType::Int64;
    MCompare::CompareType ct;
    if (!rel && objA && objB) {
      // Reference / null identity compare on object pointers.
      a = asObjPtr(a);
      b = asObjPtr(b);
      ct = MCompare::Compare_Int32;
    } else {
      a = asNumber(a);
      b = asNumber(b);
      ct = MCompare::Compare_Double;
    }
    MCompare* c = MCompare::NewWasm(alloc, a, b, jsop, ct);
    cur->add(c);
    push(c);
  };
  auto konstI32 = [&](int32_t v) -> MDefinition* {
    MConstant* c = MConstant::NewInt32(alloc, v);
    cur->add(c);
    return c;
  };
  // BOXED VALUE MODEL: object/unknown-typed fields are carried as the raw i64
  // NaN-boxed Value and unboxed lazily at use. boxedVty maps such an i64 def to
  // its observed value type (0 double / 1 int32 / 3 object) or 2 (unknown) so the
  // unbox can be specialized. The unbox helpers are IDENTITY on Double/Int32 (the
  // pre-existing typed values), so wiring them into consumers is a no-op for those
  // -- only genuinely boxed (i64) values trigger real unboxing.
  std::unordered_map<MDefinition*, uint8_t> boxedVty;
  // Unbox a value to an f64 number.
  asNumber = [&](MDefinition* d) -> MDefinition* {
    if (d->type() == MIRType::Double) return d;
    if (d->type() == MIRType::Int32) {
      MToDouble* t = MToDouble::New(alloc, d);
      t->setMovableUnchecked();
      cur->add(t);
      return t;
    }
    if (d->type() != MIRType::Int64) return d;  // unexpected
    auto vit = boxedVty.find(d);
    uint8_t vty = vit != boxedVty.end() ? vit->second : 2;
    if (vty == 0) {  // double-boxed: bit-reinterpret
      MReinterpretCast* r = MReinterpretCast::New(alloc, d, MIRType::Double);
      cur->add(r);
      return r;
    }
    MWrapInt64ToInt32* low = MWrapInt64ToInt32::New(alloc, d, /*bottomHalf=*/true);
    cur->add(low);
    if (vty == 1) {  // int32-boxed: convert payload
      MToDouble* t = MToDouble::New(alloc, low);
      t->setMovableUnchecked();
      cur->add(t);
      return t;
    }
    // Unknown: runtime tag dispatch. isInt = (d == (int32-box of low32)).
    MToDouble* asI = MToDouble::New(alloc, low);
    cur->add(asI);
    MReinterpretCast* asD = MReinterpretCast::New(alloc, d, MIRType::Double);
    cur->add(asD);
    MExtendInt32ToInt64* ext =
        MExtendInt32ToInt64::New(alloc, low, /*isUnsigned=*/true);
    cur->add(ext);
    MConstant* tagK = MConstant::NewInt64(alloc, int64_t(kWJTagInt32 << 32));
    cur->add(tagK);
    MWasmBinaryBitwise* cand = MWasmBinaryBitwise::New(
        alloc, ext, tagK, MIRType::Int64, MWasmBinaryBitwise::SubOpcode::Or);
    cur->add(cand);
    MCompare* isInt = MCompare::NewWasm(alloc, d, cand, JSOp::Eq,
                                        MCompare::Compare_Int64);
    cur->add(isInt);
    MWasmSelect* sel = MWasmSelect::New(alloc, asI, asD, isInt);
    cur->add(sel);
    return sel;
  };
  // Unbox a value to an i32 object pointer (low32 of a boxed Value; 0 for null).
  asObjPtr = [&](MDefinition* d) -> MDefinition* {
    if (d->type() != MIRType::Int64) return d;  // already an i32 pointer
    auto it = objPtrMemo.find(d);
    if (it != objPtrMemo.end()) return it->second;
    MWrapInt64ToInt32* w = MWrapInt64ToInt32::New(alloc, d, /*bottomHalf=*/true);
    cur->add(w);
    objPtrMemo[d] = w;
    return w;
  };
  // Box an i32 object pointer (0 == null) into an i64 Value. CRITICAL: a 0 pointer
  // must become the NULL Value (tag kWJTagNull), NOT an object Value with payload 0
  // -- otherwise GC would trace a reference to address 0 and crash. Pick the tag by
  // (ptr==0): null vs object, then OR in the (zero-extended) pointer payload.
  boxObj = [&](MDefinition* ptr) -> MDefinition* {
    MExtendInt32ToInt64* lo =
        MExtendInt32ToInt64::New(alloc, ptr, /*isUnsigned=*/true);
    cur->add(lo);
    MCompare* isNull = MCompare::NewWasm(alloc, ptr, konstI32(0), JSOp::Eq,
                                         MCompare::Compare_Int32);
    cur->add(isNull);
    MConstant* nullHi = MConstant::NewInt64(alloc, int64_t(kWJTagNull) << 32);
    cur->add(nullHi);
    MConstant* objHi = MConstant::NewInt64(alloc, int64_t(kWJTagObject) << 32);
    cur->add(objHi);
    MWasmSelect* hi = MWasmSelect::New(alloc, nullHi, objHi, isNull);
    cur->add(hi);
    MWasmBinaryBitwise* boxed = MWasmBinaryBitwise::New(
        alloc, hi, lo, MIRType::Int64, MWasmBinaryBitwise::SubOpcode::Or);
    cur->add(boxed);
    return boxed;
  };
  // Direct-param mode (no scratchArgs) passes object args + `this` as raw i32
  // pointers, but boxed object fields (vty2/vty3) are uniform i64 Values. If an
  // i32 arg and an i64 boxed field ever merge into the same wasm local the phi
  // degrades to MIRType::Value (no wasm local) and the backend can't emit it.
  // Re-box object params to i64 here (cur==entry) so every object-typed value
  // has one representation. Scratch-ABI args are already boxed i64.
  if (!scratchArgs) {
    for (uint32_t i = 0; i < nargs; i++) {
      if (!argParams[i]) continue;
      // Direct ABI: object args are i32 pointers (-> object Value via boxObj),
      // number args are f64 (-> double Value). Box both so every slot is i64.
      if (argParams[i]->type() == MIRType::Int32) {
        entry->setSlot(info.localSlot(i), boxObj(argParams[i]));
      } else if (argParams[i]->type() == MIRType::Double) {
        entry->setSlot(info.localSlot(i), boxForStore(argParams[i]));
      }
    }
    if (topThisDef && topThisDef->type() == MIRType::Int32) {
      topThisDef = boxObj(topThisDef);
    }
  }
  // Box a value into the raw i64 Value to store into a slot: a boxed i64 is
  // stored as-is; an f64 number is a double Value (bit-reinterpret); an i32
  // Coerce a value to an i32 for a JS bitwise op. A boxed int Value's payload is
  // its low32; a float constant folds; an i32 passes through; a non-constant f64
  // truncates (i32.trunc_f64_s -- exact for the small ints richards uses).
  auto asInt32 = [&](MDefinition* d) -> MDefinition* {
    if (d->type() == MIRType::Int32) return d;
    if (d->type() == MIRType::Int64) {
      // A boxed i64. For an INT-boxed Value (vty1) the low32 payload IS ToInt32.
      // For a DOUBLE-boxed number (vty0) or an unknown box (vty2, e.g. a slot
      // value) the low32 is the low bits of the f64 -- garbage -- so go through
      // ToNumber and truncate (JS ToInt32 = trunc-to-int32 of ToNumber; objects
      // -> NaN -> 0, which is also correct ToInt32).
      auto vit = boxedVty.find(d);
      uint8_t vty = vit != boxedVty.end() ? vit->second : 2;
      if (vty == 1) {
        MWrapInt64ToInt32* w =
            MWrapInt64ToInt32::New(alloc, d, /*bottomHalf=*/true);
        cur->add(w);
        return w;
      }
      MTruncateToInt32* t = MTruncateToInt32::New(alloc, asNumber(d));
      cur->add(t);
      return t;
    }
    if (d->isWasmFloatConstant()) {
      return konstI32(int32_t(d->toWasmFloatConstant()->toDouble()));
    }
    if (d->isConstant() && d->type() == MIRType::Double) {
      return konstI32(int32_t(d->toConstant()->numberToDouble()));
    }
    MTruncateToInt32* t = MTruncateToInt32::New(alloc, d);
    cur->add(t);
    return t;
  };
  // JS truthiness as an i32 (nonzero=truthy) for an MTest branch condition. asInt32 is
  // WRONG for a boxed OBJECT (ToInt32(obj)=0, but objects are truthy) and a boxed
  // boolean. Tag-aware: a boxed i64 is a double iff high32(unsigned) < 0xFFFFFF81 ->
  // truthy=(d!=0 && !NaN); else a tagged value -> truthy=low32!=0 (object ptr!=0 /
  // int,bool!=0; null,undef have payload 0). (Relies on the now-fixed high-half
  // MWrapInt64ToInt32.) GECKO_WJVS_NOCONDTRUTHY reverts to asInt32.
  auto condTruthy = [&](MDefinition* v) -> MDefinition* {
    if (getenv("GECKO_WJVS_NOCONDTRUTHY")) return asInt32(v);
    if (v->type() == MIRType::Int32) return v;
    if (v->type() == MIRType::Double) {
      MCompare* nz = MCompare::NewWasm(alloc, v, konst(0.0), JSOp::Ne,
                                       MCompare::Compare_Double);
      cur->add(nz);
      MCompare* nn = MCompare::NewWasm(alloc, v, v, JSOp::Eq,
                                       MCompare::Compare_Double);
      cur->add(nn);
      MWasmBinaryBitwise* t = MWasmBinaryBitwise::New(
          alloc, nz, nn, MIRType::Int32, MWasmBinaryBitwise::SubOpcode::And);
      cur->add(t);
      return t;
    }
    if (v->type() != MIRType::Int64) return asInt32(v);
    MWrapInt64ToInt32* hi = MWrapInt64ToInt32::New(alloc, v, /*bottomHalf=*/false);
    cur->add(hi);
    MWrapInt64ToInt32* lo = MWrapInt64ToInt32::New(alloc, v, /*bottomHalf=*/true);
    cur->add(lo);
    MCompare* isDouble = MCompare::NewWasm(alloc, hi, konstI32(int32_t(0xFFFFFF81)),
                                           JSOp::Lt, MCompare::Compare_UInt32);
    cur->add(isDouble);
    MReinterpretCast* dval = MReinterpretCast::New(alloc, v, MIRType::Double);
    cur->add(dval);
    MCompare* dNz = MCompare::NewWasm(alloc, dval, konst(0.0), JSOp::Ne,
                                      MCompare::Compare_Double);
    cur->add(dNz);
    MCompare* dNn = MCompare::NewWasm(alloc, dval, dval, JSOp::Eq,
                                      MCompare::Compare_Double);
    cur->add(dNn);
    MWasmBinaryBitwise* dT = MWasmBinaryBitwise::New(
        alloc, dNz, dNn, MIRType::Int32, MWasmBinaryBitwise::SubOpcode::And);
    cur->add(dT);
    MCompare* tT = MCompare::NewWasm(alloc, lo, konstI32(0), JSOp::Ne,
                                     MCompare::Compare_Int32);
    cur->add(tT);
    MWasmSelect* truthy = MWasmSelect::New(alloc, dT, tT, isDouble);
    cur->add(truthy);
    return truthy;
  };
  // object pointer gets the object tag.
  boxForStore = [&](MDefinition* d) -> MDefinition* {
    if (d->type() == MIRType::Int64) return d;
    if (d->type() == MIRType::Double) {
      MReinterpretCast* r = MReinterpretCast::New(alloc, d, MIRType::Int64);
      cur->add(r);
      return r;
    }
    // An Int32 here is ALWAYS a numeric (bitwise / ToInt32) result -- object
    // pointers always flow as boxed i64 Values, never raw i32. So box it as an
    // int32-tagged NUMBER Value (kWJTagInt32 | u32), NOT an object Value. Boxing
    // it as an object (the old behaviour) made `this.x = a & b` store a bogus
    // object Value into a numeric field, which then read back as NaN/garbage.
    MExtendInt32ToInt64* lo =
        MExtendInt32ToInt64::New(alloc, d, /*isUnsigned=*/true);
    cur->add(lo);
    MConstant* hi = MConstant::NewInt64(alloc, int64_t(kWJTagInt32) << 32);
    cur->add(hi);
    MWasmBinaryBitwise* boxed = MWasmBinaryBitwise::New(
        alloc, hi, lo, MIRType::Int64, MWasmBinaryBitwise::SubOpcode::Or);
    cur->add(boxed);
    return boxed;
  };
  // Box a number as an INT32-tagged Value: (kWJTagInt32 << 32) | (u32)ToInt32(d).
  // Used to store into a field OBSERVED as int (vty1), which is READ back as the
  // low32 int payload. boxForStore would make a DOUBLE Value (raw f64 bits) whose
  // low32 is garbage when re-read as an int -- so an int field mutated by compiled
  // code (richards' `this.v2++`) must round-trip through the int box to stay
  // consistent with both the int read path and the interpreter's int Values.
  auto boxForStoreInt = [&](MDefinition* d) -> MDefinition* {
    if (d->type() == MIRType::Int64) return d;  // already a boxed Value
    MDefinition* i32 = asInt32(d);
    MExtendInt32ToInt64* lo =
        MExtendInt32ToInt64::New(alloc, i32, /*isUnsigned=*/true);
    cur->add(lo);
    MConstant* hi = MConstant::NewInt64(alloc, int64_t(kWJTagInt32) << 32);
    cur->add(hi);
    MWasmBinaryBitwise* boxed = MWasmBinaryBitwise::New(
        alloc, hi, lo, MIRType::Int64, MWasmBinaryBitwise::SubOpcode::Or);
    cur->add(boxed);
    return boxed;
  };
  // Field/heap load into `cur`. The field offset is folded into the address
  // (base = ptr + off) with a 0 access-offset, so two loads of the SAME field
  // share an address node (GVN CSEs them; distinct fields never merge).
  auto loadAt = [&](MDefinition* ptr, uint32_t off, js::Scalar::Type sty,
                    MIRType rty) -> MDefinition* {
    MDefinition* base = ptr;
    if (off != 0) {
      MInstruction* add = MAdd::NewWasm(alloc, ptr, konstI32(int32_t(off)),
                                        MIRType::Int32);
      cur->add(add);
      base = add;
    }
    uint32_t align =
        (sty == js::Scalar::Float64 || sty == js::Scalar::Int64) ? 8 : 4;
    wasm::MemoryAccessDesc acc(/*memoryIndex=*/0, sty, align, /*offset=*/0,
                               wasm::TrapSiteDesc(), /*hugeMemory=*/false);
    MWasmLoad* l = MWasmLoad::New(alloc, /*memoryBase=*/nullptr, base, acc, rty);
    // MWasmLoad is created Guard-but-not-Movable. Mark it movable so GVN can CSE
    // repeated loads and LICM can hoist loop-invariant ones out of loops --
    // correctness is preserved by AliasAnalysis (any aliasing WasmHeap store in
    // the loop sets a dependency that blocks the hoist). This is THE enabler for
    // the richards win (shape-guard + field-load hoisting out of the hot loop).
    if (!getenv("GECKO_WJVS_NOMOVE")) l->setMovableUnchecked();
    cur->add(l);
    return l;
  };
  // Field/heap store into `cur` (offset folded into the address, like loadAt).
  auto storeAt = [&](MDefinition* ptr, uint32_t off, js::Scalar::Type sty,
                     MDefinition* val) {
    MDefinition* base = ptr;
    if (off != 0) {
      MInstruction* add = MAdd::NewWasm(alloc, ptr, konstI32(int32_t(off)),
                                        MIRType::Int32);
      cur->add(add);
      base = add;
    }
    uint32_t align =
        (sty == js::Scalar::Float64 || sty == js::Scalar::Int64) ? 8 : 4;
    wasm::MemoryAccessDesc acc(/*memoryIndex=*/0, sty, align, /*offset=*/0,
                               wasm::TrapSiteDesc(), /*hugeMemory=*/false);
    MWasmStore* st = MWasmStore::New(alloc, /*memoryBase=*/nullptr, base, acc, val);
    cur->add(st);
  };
  // Guard `recv`'s shape (deopt -> sentinel 1.0 on miss) unless it is already
  // guarded for this shape in the current dominator region; advances `cur` into
  // the fast continuation block.
  auto ensureGuard = [&](MDefinition* recv, uint32_t recShape,
                         double dcode = 6.0) -> bool {
    auto git = guardedShape.find(recv);
    if (git != guardedShape.end() && git->second == recShape) return true;
    MDefinition* shapeWord = loadAt(recv, 0, js::Scalar::Int32, MIRType::Int32);
    MCompare* g = MCompare::NewWasm(alloc, shapeWord, konstI32(int32_t(recShape)),
                                    JSOp::Eq, MCompare::Compare_Int32);
    cur->add(g);
    MBasicBlock* cont = MBasicBlock::New(graph, info, cur, MBasicBlock::NORMAL);
    MBasicBlock* deopt = MBasicBlock::New(graph, info, cur, MBasicBlock::NORMAL);
    if (!cont || !deopt) return false;
    cont->setLoopDepth(loopDepth);
    deopt->setLoopDepth(loopDepth);
    graph.addBlock(cont);
    graph.addBlock(deopt);
    cur->end(MTest::New(alloc, g, cont, deopt));
    deopt->end(MWasmReturn::New(alloc, konstIn(deopt, dcode), instance));
    cur = cont;
    guardedShape[recv] = recShape;
    return true;
  };
  // Bounds guard for dense element access: deopt (return to interpreter) when
  // `idx` is NOT unsigned-less-than `limit` (initializedLength for reads at
  // elements_-12, capacity for writes at elements_-8), instead of letting the raw
  // i64 load/store trap with "memory access out of bounds". `elemsPtr` points at
  // element 0; the header words sit just before it. Speculative element access
  // (richards' fixed a2, deltablue's growable OrderedCollection) is only safe with
  // this guard -- a growable collection routinely indexes past the speculated
  // capacity, which without the guard is an OOB wasm trap that aborts the engine.
  auto boundsGuard = [&](MDefinition* elemsPtr, MDefinition* idx,
                         int32_t hdrOff) -> bool {
    MAdd* hAddr = MAdd::NewWasm(alloc, elemsPtr, konstI32(hdrOff), MIRType::Int32);
    cur->add(hAddr);
    MDefinition* limit = loadAt(hAddr, 0, js::Scalar::Int32, MIRType::Int32);
    MCompare* g = MCompare::NewWasm(alloc, idx, limit, JSOp::Lt,
                                    MCompare::Compare_UInt32);  // unsigned: idx<0 fails
    cur->add(g);
    MBasicBlock* cont = MBasicBlock::New(graph, info, cur, MBasicBlock::NORMAL);
    MBasicBlock* deopt = MBasicBlock::New(graph, info, cur, MBasicBlock::NORMAL);
    if (!cont || !deopt) return false;
    cont->setLoopDepth(loopDepth);
    deopt->setLoopDepth(loopDepth);
    graph.addBlock(cont);
    graph.addBlock(deopt);
    cur->end(MTest::New(alloc, g, cont, deopt));
    deopt->end(MWasmReturn::New(alloc, konstIn(deopt, 7.0), instance));
    cur = cont;
    return true;
  };
  // Resolve a slot byte-offset (possibly carrying the kWJDynSlot bit) to the base
  // pointer + real offset for a field access: a fixed slot is at recv+off; a
  // dynamic slot is at slots_(= *(recv+8)) + (off & ~kWJDynSlot).
  auto fieldBase = [&](MDefinition* recv, uint32_t off,
                       uint32_t* realOff) -> MDefinition* {
    if (off & kWJDynSlot) {
      *realOff = off & ~kWJDynSlot;
      return loadAt(recv, 8, js::Scalar::Int32, MIRType::Int32);  // slots_ ptr
    }
    *realOff = off;
    return recv;
  };
  // Generic property load via wjhelp(WJH_GETPROP) -- the NO-DEOPT fallback for an
  // off-shape receiver. Returns the boxed-i64 value (or nullptr on site overflow).
  // This is what makes a polymorphic field site keep running compiled instead of
  // deopt-restarting the whole (stateful) caller on every off-shape iteration --
  // the dominant deltablue cost.
  auto emitPropHelper = [&](MDefinition* recvBoxed, JSScript* fscript,
                            uint32_t pcOff) -> MDefinition* {
    uint32_t s = gWJSiteCount++;
    if (s == 0) s = gWJSiteCount++;  // 0 is the "not a getprop" sentinel
    if (s >= kWJMaxSites) return nullptr;
    gWJSites[s].script = fscript;
    gWJSites[s].pcOff = pcOff;
    // The receiver must reach the helper as a boxed OBJECT Value. A field receiver
    // is always an object, but `recvBoxed` may arrive as a raw i32 pointer OR as an
    // i64 that an inlined frame mistagged int32 (object ptr 0xPP boxForStore'd to
    // 0xFFFFFF81_000000PP) -- the helper's ToObject would then wrap a Number and
    // read `undefined`. Re-derive the object Value from the pointer payload
    // (asObjPtr keeps the low32 ptr regardless of tag; boxObj re-tags it object).
    MDefinition* recvVal = boxObj(asObjPtr(recvBoxed));
    MWJIonCall* h = MWJIonCall::New(alloc, recvVal, 0);
    h->setPropSite(s);
    cur->add(h);
    return h;  // i64 boxed Value
  };
  // GC generational post-write barrier for an inline heap store `recv.<slot> = val`
  // (SetProp/SetElem). The reused-Ion inline store writes raw memory with no barrier;
  // when `val` is a nursery object pointer stored into a tenured object, the GC must
  // record the edge or it collects/moves the nursery cell out from under the store
  // (splay's tree churn -- fresh nodes linked into older nodes). Skipped when `val`
  // is provably a number (no GC pointer). The helper itself re-checks
  // tenured(obj) && nursery(val), so a conservative call is always safe.
  auto emitPostBarrier = [&](MDefinition* recv, MDefinition* val) {
    // Gated default-OFF: a helper call per object store is a real perf cost and the
    // working benches don't need it (their objects are tenured / no collected edges).
    // Kept as infrastructure for allocation-churn code (splay) once a cheaper inline
    // nursery-check barrier exists. GECKO_WJVS_POSTBARRIER enables it.
    if (!getenv("GECKO_WJVS_POSTBARRIER")) return;
    if (val->type() == MIRType::Double || val->type() == MIRType::Int32) return;
    if (val->type() == MIRType::Int64) {
      auto vit = boxedVty.find(val);
      uint8_t vty = vit != boxedVty.end() ? vit->second : 2;
      if (vty == 0 || vty == 1) return;  // double/int box -> not a GC pointer
    }
    MDefinition* objp = asObjPtr(recv);
    auto emitHelperCall = [&]() {
      uint32_t helpBAddr = uint32_t(uintptr_t(&gWJHelpB));
      storeAt(konstI32(int32_t(helpBAddr)), 0, js::Scalar::Int64, boxForStore(val));
      MWJIonCall* bar = MWJIonCall::New(alloc, boxObj(objp), 0);
      bar->setPostBarrier();
      cur->add(bar);
    };
    // On a non-empty operand stack a diamond is unsafe (operand-stack invariant), so
    // fall back to the unconditional helper there. Splay's stores are statement-level.
    if (!curStk->empty()) { emitHelperCall(); return; }
    // INLINE fast path: only call the (expensive) barrier helper for a genuine
    // tenured-obj <- nursery-val edge. A cell is in the nursery iff its chunk's
    // storeBuffer pointer is non-null (chunk = ptr & ~ChunkMask). Most splay stores
    // are tenured<-tenured -> skipped here without the helper call. Loads are safe:
    // `valp` falls back to `objp` (a valid object address) when val isn't an object,
    // so the chunk load never dereferences garbage.
    uint32_t maskInv = ~uint32_t(js::gc::ChunkMask);
    uint32_t sbOff = uint32_t(js::gc::ChunkStoreBufferOffset);
    MWrapInt64ToInt32* valHi =
        MWrapInt64ToInt32::New(alloc, val, /*bottomHalf=*/false);
    cur->add(valHi);
    MCompare* isObj = MCompare::NewWasm(alloc, valHi,
                                        konstI32(int32_t(uint32_t(kWJTagObject))),
                                        JSOp::Eq, MCompare::Compare_Int32);
    cur->add(isObj);
    MWrapInt64ToInt32* valLo =
        MWrapInt64ToInt32::New(alloc, val, /*bottomHalf=*/true);
    cur->add(valLo);
    MWasmSelect* valp = MWasmSelect::New(alloc, valLo, objp, isObj);
    cur->add(valp);
    MWasmBinaryBitwise* valChunk =
        MWasmBinaryBitwise::New(alloc, valp, konstI32(int32_t(maskInv)),
                                MIRType::Int32, MWasmBinaryBitwise::SubOpcode::And);
    cur->add(valChunk);
    MDefinition* valSB = loadAt(valChunk, sbOff, js::Scalar::Int32, MIRType::Int32);
    MWasmBinaryBitwise* objChunk =
        MWasmBinaryBitwise::New(alloc, objp, konstI32(int32_t(maskInv)),
                                MIRType::Int32, MWasmBinaryBitwise::SubOpcode::And);
    cur->add(objChunk);
    MDefinition* objSB = loadAt(objChunk, sbOff, js::Scalar::Int32, MIRType::Int32);
    MCompare* valNursery = MCompare::NewWasm(alloc, valSB, konstI32(0), JSOp::Ne,
                                             MCompare::Compare_Int32);
    cur->add(valNursery);
    MCompare* objTenured = MCompare::NewWasm(alloc, objSB, konstI32(0), JSOp::Eq,
                                             MCompare::Compare_Int32);
    cur->add(objTenured);
    MWasmBinaryBitwise* need = MWasmBinaryBitwise::New(
        alloc, valNursery, objTenured, MIRType::Int32,
        MWasmBinaryBitwise::SubOpcode::And);
    cur->add(need);
    MBasicBlock* helperB = MBasicBlock::New(graph, info, cur, MBasicBlock::NORMAL);
    MBasicBlock* skipB = MBasicBlock::New(graph, info, cur, MBasicBlock::NORMAL);
    if (!helperB || !skipB) return;
    helperB->setLoopDepth(loopDepth);
    skipB->setLoopDepth(loopDepth);
    graph.addBlock(helperB);
    graph.addBlock(skipB);
    cur->end(MTest::New(alloc, need, helperB, skipB));
    std::vector<WJPendEdge> bedges;
    cur = helperB;
    emitHelperCall();
    { MGoto* go = MGoto::New(alloc, nullptr); cur->end(go);
      bedges.push_back({cur, MGoto::TargetIndex}); }
    cur = skipB;
    { MGoto* go = MGoto::New(alloc, nullptr); cur->end(go);
      bedges.push_back({cur, MGoto::TargetIndex}); }
    MBasicBlock* bjoin = nullptr;
    for (const WJPendEdge& e : bedges) {
      if (!bjoin) {
        bjoin = MBasicBlock::New(graph, info, e.block, MBasicBlock::NORMAL);
        if (!bjoin) return;
        bjoin->setLoopDepth(loopDepth);
        graph.addBlock(bjoin);
      } else if (!bjoin->addPredecessor(alloc, e.block)) {
        return;
      }
      e.block->lastIns()->initSuccessor(e.successor, bjoin);
    }
    if (!bjoin) return;
    cur = bjoin;
    invalidateCSE();
  };
  // Shared GetProp(field) for own fixed double slots: shape-guard `recv` (CSE'd)
  // then load the field (CSE'd on recv+off). Returns the value def, or nullptr if
  // this site has no own-fixed-slot record (caller treats that as "method load /
  // bail"). Keyed by the FRAME's script so inlined callees resolve their own ICs.
  auto getPropField = [&](JSScript* fscript, MDefinition* recv,
                          uint32_t pcOff) -> MDefinition* {
    // Dense-array `.length` (oracle saw LoadInt32ArrayLengthResult): shape-guard
    // the array, then load the i32 length from the ObjectElements header
    // (elements_-4) -> ToDouble. NOT a slot, so it must NOT go through the normal
    // field path (that read a bogus offset and trapped).
    {
      auto lit = gWJLenSite.find(WJInlineKey(fscript, pcOff));
      if (lit != gWJLenSite.end() && !getenv("GECKO_WJVS_NOLEN")) {
        MDefinition* objp = asObjPtr(recv);
        if (!ensureGuard(objp, lit->second, 6.1)) return nullptr;
        MDefinition* elemsPtr =
            loadAt(objp, 12, js::Scalar::Int32, MIRType::Int32);
        MAdd* lenAddr =
            MAdd::NewWasm(alloc, elemsPtr, konstI32(-4), MIRType::Int32);
        cur->add(lenAddr);
        MDefinition* len = loadAt(lenAddr, 0, js::Scalar::Int32, MIRType::Int32);
        MToDouble* d = MToDouble::New(alloc, len);
        d->setMovableUnchecked();
        cur->add(d);
        return d;
      }
    }
    // Polymorphic field site (receiver has several shapes -- deltablue's
    // constraint/Variable fields): emit a shape-guard chain that loads each shape's
    // slot inline, merging into one slot. An off-shape access no longer
    // deopt-restarts the whole (stateful) function -- the dominant deltablue deopt.
    // Each way loads the raw boxed i64 Value (unboxed generically at use, vty2).
    {
      // DEFAULT OFF: the helper-diamond has a latent MIR-construction bug when
      // deeply nested in a complex inlined call/dispatch tree (reproduces on
      // deltablue's Plan.execute under GECKO_WJVS_POLYALL, even with NOOPT -- a
      // call_indirect table-index OOB / mis-propagation). Gate behind
      // GECKO_WJVS_POLYFIELD (poly-at-compile sites) / GECKO_WJVS_POLYALL (all
      // sites) until that construction bug is found+fixed, so the default build
      // keeps the known-good legacy typed mono path (richards 18x, all benches OK).
      auto fpit = gWJFieldPoly.find(WJInlineKey(fscript, pcOff));
      bool polyOn = getenv("GECKO_WJVS_POLYFIELD") || getenv("GECKO_WJVS_POLYALL");
      uint8_t polyMin = getenv("GECKO_WJVS_POLYALL") ? 1 : 2;
      // INVARIANT (root-cause fix): the JS operand stack must be empty at a block
      // boundary -- it lives outside block slots, so values on it do NOT cross
      // blocks / get phis. The diamond creates a mid-expression boundary. To allow
      // the diamond on a NON-empty stack (the common case -- field reads inside
      // expressions), SPILL each live operand-stack value to a fresh slot before the
      // diamond and RELOAD it from that slot after the join, so it crosses the
      // boundary via slots (with phis). GECKO_WJVS_NOSPILL falls back to empty-stack-
      // only (the conservative correct subset).
      bool canSpill = !getenv("GECKO_WJVS_NOSPILL");
      bool stackOk = curStk->empty() || canSpill;
      if (polyOn && stackOk && fpit != gWJFieldPoly.end() && fpit->second.n >= polyMin &&
          !getenv("GECKO_WJVS_NOPOLYFIELD")) {
        const WJFieldPolyRec fp = fpit->second;
        MDefinition* objp = asObjPtr(recv);
        MDefinition* shapeWord =
            loadAt(objp, 0, js::Scalar::Int32, MIRType::Int32);
        // Spill the operand stack to slots (reload after the join). Box each value
        // to a uniform i64 in the SAME encoding its reload-vty expects: Double ->
        // raw double bits (vty0), Int32 -> int32-tagged (vty1), i64 -> as-is (its
        // own boxedVty). The consumer unboxes via asNumber/asObjPtr/asInt32.
        std::vector<std::pair<uint32_t, uint8_t>> spilled;  // (slot, reloadVty)
        for (MDefinition* sv : *curStk) {
          uint32_t ss = nextSlotBase++;
          if (ss >= info.nlocals()) return nullptr;
          uint8_t rvty;
          MDefinition* boxed;
          if (sv->type() == MIRType::Double) { boxed = boxForStore(sv); rvty = 0; }
          else if (sv->type() == MIRType::Int32) { boxed = boxForStoreInt(sv); rvty = 1; }
          else {
            boxed = sv;  // already a boxed i64
            auto bit = boxedVty.find(sv);
            rvty = bit != boxedVty.end() ? bit->second : 2;
          }
          cur->setSlot(info.localSlot(ss), boxed);
          spilled.push_back({ss, rvty});
        }
        uint32_t tmp = nextSlotBase++;
        if (tmp >= info.nlocals()) return nullptr;
        std::vector<WJPendEdge> edges;
        for (uint8_t w = 0; w < fp.n; w++) {
          MCompare* g = MCompare::NewWasm(alloc, shapeWord,
                                          konstI32(int32_t(fp.shapes[w])),
                                          JSOp::Eq, MCompare::Compare_Int32);
          cur->add(g);
          MBasicBlock* body = MBasicBlock::New(graph, info, cur, MBasicBlock::NORMAL);
          MBasicBlock* next = MBasicBlock::New(graph, info, cur, MBasicBlock::NORMAL);
          if (!body || !next) return nullptr;
          body->setLoopDepth(loopDepth);
          next->setLoopDepth(loopDepth);
          graph.addBlock(body);
          graph.addBlock(next);
          cur->end(MTest::New(alloc, g, body, next));
          cur = body;
          uint32_t fo;
          MDefinition* base = fieldBase(objp, fp.offs[w], &fo);
          MDefinition* fv = loadAt(base, fo, js::Scalar::Int64, MIRType::Int64);
          cur->setSlot(info.localSlot(tmp), fv);
          MGoto* go = MGoto::New(alloc, nullptr);
          cur->end(go);
          edges.push_back({cur, MGoto::TargetIndex});
          cur = next;
          invalidateCSE();
        }
        // No recorded shape matched -> generic helper load (NO deopt/restart).
        // GECKO_WJVS_POLYNOHELP: A/B -- deopt instead of helper (no helper block
        // emitted) to isolate whether the helper-block emission is the corruption.
        {
          MDefinition* hv = emitPropHelper(recv, fscript, pcOff);
          if (!hv) return nullptr;
          cur->setSlot(info.localSlot(tmp), hv);
          MGoto* go = MGoto::New(alloc, nullptr);
          cur->end(go);
          edges.push_back({cur, MGoto::TargetIndex});
        }
        MBasicBlock* join = nullptr;
        for (const WJPendEdge& e : edges) {
          if (!join) {
            join = MBasicBlock::New(graph, info, e.block, MBasicBlock::NORMAL);
            if (!join) return nullptr;
            join->setLoopDepth(loopDepth);
            graph.addBlock(join);
          } else if (!join->addPredecessor(alloc, e.block)) {
            return nullptr;
          }
          e.block->lastIns()->initSuccessor(e.successor, join);
        }
        if (!join) return nullptr;
        cur = join;
        invalidateCSE();
        // Reload the spilled operand-stack values, preserving each value's EXACT
        // original MIR type/representation (Double->reinterpret back to Double,
        // Int32->extract low32 to Int32, i64->keep with its boxedVty) so downstream
        // consumers see an identical value -- not a re-typed one that a type-checking
        // consumer would mishandle.
        for (size_t i = 0; i < spilled.size(); i++) {
          MDefinition* raw = cur->getSlot(info.localSlot(spilled[i].first));
          uint8_t rvty = spilled[i].second;
          MDefinition* rv;
          if (rvty == 0) {  // was Double
            MReinterpretCast* d = MReinterpretCast::New(alloc, raw, MIRType::Double);
            cur->add(d);
            rv = d;
          } else if (rvty == 1) {  // was Int32 (numeric)
            MWrapInt64ToInt32* w =
                MWrapInt64ToInt32::New(alloc, raw, /*bottomHalf=*/true);
            cur->add(w);
            rv = w;
          } else {  // boxed i64
            rv = raw;
            boxedVty[rv] = rvty;
          }
          (*curStk)[i] = rv;
        }
        MDefinition* res = cur->getSlot(info.localSlot(tmp));
        boxedVty[res] = 2;
        return res;
      }
    }
    uint32_t recShape, recOff;
    uint8_t recVty;
    auto rec = gWJShapeRec.find(WJInlineKey(fscript, pcOff));
    if (rec != gWJShapeRec.end() && rec->second.shape != 0) {
      recShape = rec->second.shape;
      recOff = rec->second.off;
      recVty = rec->second.vty;
    } else {
      // Cold site (fallback-only IC, e.g. a rarely-taken branch): resolve a field
      // that is monomorphic program-wide by its property name, behind a shape
      // guard. Proto methods aren't indexed here -> still return null (method idiom).
      if (getenv("GECKO_WJVS_NOCOLDPROP")) return nullptr;
      jsbytecode* spc = fscript->offsetToPC(pcOff);
      uint32_t nameLow = uint32_t(uintptr_t(fscript->getName(spc)));
      auto pit = gWJPropByName.find(nameLow);
      if (pit == gWJPropByName.end() || pit->second.ambig ||
          pit->second.shape == 0) {
        // Cold/ambiguous field site (no inline record): instead of bailing the
        // whole function, emit a generic wjhelp(WJH_GETPROP) load -- correct (runs
        // GetProperty in the interpreter), no block boundary (single call, unlike
        // the diamond), so no operand-stack hazard. Lets field-heavy functions
        // (raytrace's vector/color ops) compile. GECKO_WJVS_NOCOLDHELP reverts.
        if (getenv("GECKO_WJVS_NOCOLDGET")) return nullptr;
        MDefinition* hv = emitPropHelper(recv, fscript, pcOff);
        if (!hv) return nullptr;
        boxedVty[hv] = 2;
        return hv;
      }
      recShape = pit->second.shape;
      recOff = pit->second.off;
      recVty = pit->second.vty;
    }
    MDefinition* objp = asObjPtr(recv);  // boxed-i64 receiver -> i32 object ptr
    // Monomorphic field site: shape-guard (deopt-restart on miss) + typed load.
    // This is richards' hot path -- kept at its native typed representation (vty0/1
    // -> Double, vty2/3 -> boxed i64) with no boxing round-trip, so OptimizeMIR sees
    // the same graph as before. (The helper-fallback path above handles sites that
    // are polymorphic AT COMPILE TIME; a mono-at-compile site that turns out
    // polymorphic at runtime still deopt-restarts here.)
    if (!ensureGuard(objp, recShape, 6.2)) return nullptr;
    uint32_t foff = recOff;
    for (auto& t : fieldCache) {
      if (std::get<0>(t) == objp && std::get<1>(t) == foff) return std::get<2>(t);
    }
    uint32_t fo;
    MDefinition* base = fieldBase(objp, foff, &fo);  // fixed: objp+off; dyn: slots_
    MDefinition* fv;
    if (recVty == 2) {
      fv = loadAt(base, fo, js::Scalar::Int64, MIRType::Int64);
      boxedVty[fv] = 2;
    } else if (recVty == 1) {
      MDefinition* iv = loadAt(base, fo, js::Scalar::Int32, MIRType::Int32);
      MToDouble* d = MToDouble::New(alloc, iv);
      d->setMovableUnchecked();
      cur->add(d);
      fv = d;
    } else if (recVty == 3) {
      fv = loadAt(base, fo, js::Scalar::Int64, MIRType::Int64);
      boxedVty[fv] = 3;
    } else {
      fv = loadAt(base, fo, js::Scalar::Float64, MIRType::Double);
    }
    fieldCache.emplace_back(objp, foff, fv);
    return fv;
  };
  // One inline (or top-level) frame to decode. Inline frames get their own
  // CompileInfo slot range [slotBase, slotBase+nargs+nfixed+1) and their own
  // namespace `offBase` for pending/loopHeader/loopHeadOff keys.
  struct WJFrameDesc {
    JSScript* script;
    jsbytecode* start;
    jsbytecode* end;
    uint32_t slotBase;
    uint32_t nargs;
    uint32_t nfixed;
    uint32_t offBase;
    MDefinition* thisDef;
    bool isInline;
    int depth;
  };
  // Decode one frame's bytecode into the shared MIRGraph, with full control flow
  // (branches + loops via block slots/phis). Inline callees recurse here, so a
  // whole call graph -- INCLUDING callees with if/loops -- collapses into one
  // MIRGraph that GVN/LICM optimize across. Each inline frame routes its
  // returns through a continuation block (the return value via its rval slot);
  // the caller reads that slot to continue. This is richards' real lever.
  std::function<bool(const WJFrameDesc&, std::vector<MDefinition*>&)> buildFrame;
  // Polymorphic method dispatch: guard the receiver shape and inline the matching
  // method body for each observed way; a shape miss deopts (sentinel). All way
  // bodies merge their return value into one dispatch slot read at the join.
  auto emitMethodDispatch = [&](const WJMethodPolyRec& poly, MDefinition* recv,
                                std::vector<MDefinition*>& cargs, uint32_t argc,
                                int callerDepth, uint32_t methNameLow) -> bool {
    MDefinition* recvBoxed = recv;  // keep the boxed receiver for the METHCALL marshal
    recv = asObjPtr(recv);  // boxed-i64 receiver -> i32 object pointer
    MDefinition* shapeWord = loadAt(recv, 0, js::Scalar::Int32, MIRType::Int32);
    uint32_t dispSlot = nextSlotBase++;
    if (dispSlot >= info.nlocals()) {
      if (WJIonLog()) fprintf(stderr, "[ion-fe] dispatch slot overflow\n");
      return false;
    }
    // The dispatch slot merges all ways' results; box each to a uniform i64 so
    // the join phi is i64 (not MIRType::Value from mixed return types).
    cur->setSlot(info.localSlot(dispSlot), boxForStore(konst(0.0)));
    std::vector<WJPendEdge> contEdges;
    for (uint8_t w = 0; w < poly.n; w++) {
      if (!WJGuestPtrOk(poly.fns[w])) return false;  // stale fn ptr -> bail safely
      JSFunction* fun = reinterpret_cast<JSFunction*>(uintptr_t(poly.fns[w]));
      if (!fun || !fun->isInterpreted() || !fun->baseScript() ||
          !fun->baseScript()->hasBytecode() || fun->nargs() != argc) {
        if (WJIonLog()) fprintf(stderr, "[ion-fe] dispatch way %u not inlinable\n", w);
        return false;
      }
      JSScript* cs = fun->baseScript()->asJSScript();
      MCompare* g = MCompare::NewWasm(alloc, shapeWord,
                                      konstI32(int32_t(poly.shapes[w])), JSOp::Eq,
                                      MCompare::Compare_Int32);
      cur->add(g);
      MBasicBlock* body = MBasicBlock::New(graph, info, cur, MBasicBlock::NORMAL);
      MBasicBlock* next = MBasicBlock::New(graph, info, cur, MBasicBlock::NORMAL);
      if (!body || !next) return false;
      body->setLoopDepth(loopDepth);
      next->setLoopDepth(loopDepth);
      graph.addBlock(body);
      graph.addBlock(next);
      cur->end(MTest::New(alloc, g, body, next));
      cur = body;
      // The dispatch compare already proved recv's shape on this (true) edge --
      // record it so the inlined method body's field accesses skip the redundant
      // shape guard (the hottest per-node work in richards).
      guardedShape[recv] = poly.shapes[w];
      WJFrameDesc cfr;
      cfr.script = cs;
      cfr.start = cs->code();
      cfr.end = cs->codeEnd();
      cfr.nargs = fun->nargs();
      cfr.nfixed = cs->nfixed();
      cfr.slotBase = nextSlotBase;
      cfr.offBase = nextOffBase;
      cfr.thisDef = recv;
      cfr.isInline = true;
      cfr.depth = callerDepth + 1;
      if (cfr.slotBase + cfr.nargs + cfr.nfixed + 1 > info.nlocals()) {
        if (WJIonLog()) fprintf(stderr, "[ion-fe] inline slot overflow (dispatch)\n");
        return false;
      }
      nextSlotBase += cfr.nargs + cfr.nfixed + 1;
      nextOffBase += uint32_t(cfr.end - cfr.start) + 1;
      uint32_t crvalIdx = cfr.slotBase + cfr.nargs + cfr.nfixed;
      std::vector<MDefinition*> argsCopy = cargs;
      if (!buildFrame(cfr, argsCopy)) return false;
      cur->setSlot(info.localSlot(dispSlot),
                   boxForStore(cur->getSlot(info.localSlot(crvalIdx))));
      MGoto* go = MGoto::New(alloc, nullptr);
      cur->end(go);
      contEdges.push_back({cur, MGoto::TargetIndex});
      cur = next;
      invalidateCSE();  // each way body is a separate dominator region
    }
    // No observed shape matched. Instead of deopting the whole (possibly stateful)
    // caller -- which for a MEGAMORPHIC site in a loop means a deopt-restart on every
    // off-type iteration (deltablue's 388 deopts) -- do a NON-INLINED dynamic method
    // call: marshal recv+args to gWJScratch and wjhelp(WJH_METHCALL) resolves
    // recv[name] and calls it. The result merges into the dispatch slot like an
    // inlined way, so the loop keeps running compiled. Falls back to deopt only if
    // the method name is unknown.
    if (methNameLow) {
      uint32_t scratchAddr = uint32_t(uintptr_t(static_cast<void*>(gWJScratch)));
      for (uint32_t i = 0; i < argc; i++) {
        storeAt(konstI32(int32_t(scratchAddr)), i * 8, js::Scalar::Int64,
                boxForStore(cargs[i]));
      }
      storeAt(konstI32(int32_t(scratchAddr)), kWJThisOff, js::Scalar::Int64,
              boxForStore(recvBoxed));
      MConstant* dummy = MConstant::NewInt64(alloc, 0);
      cur->add(dummy);
      MWJIonCall* mc = MWJIonCall::New(alloc, dummy, argc);
      mc->setMethName(methNameLow);
      cur->add(mc);
      cur->setSlot(info.localSlot(dispSlot), boxForStore(mc));
      MGoto* go = MGoto::New(alloc, nullptr);
      cur->end(go);
      contEdges.push_back({cur, MGoto::TargetIndex});
    } else {
      cur->end(MWasmReturn::New(alloc, konst(9.0), instance));
    }
    MBasicBlock* cont = nullptr;
    for (const WJPendEdge& e : contEdges) {
      if (!cont) {
        cont = MBasicBlock::New(graph, info, e.block, MBasicBlock::NORMAL);
        if (!cont) return false;
        cont->setLoopDepth(loopDepth);
        graph.addBlock(cont);
      } else {
        if (!cont->addPredecessor(alloc, e.block)) return false;
      }
      e.block->lastIns()->initSuccessor(e.successor, cont);
    }
    if (!cont) return false;
    cur = cont;
    invalidateCSE();
    MDefinition* res = cur->getSlot(info.localSlot(dispSlot));
    boxedVty[res] = 2;  // dispatch result is a boxed Value -> unbox lazily at use
    push(res);
    return true;
  };
  // Non-inlined call (like real Ion: inline what you can, call the rest). Marshal
  // this/args (boxed i64) into gWJScratch, then MWJIonCall(callee) which the
  // backend lowers to wjhelp(WJH_IONCALL) (the interpreter runs the callee) +
  // deopt check + result load. `callee` must be the real callee Value (i64).
  auto emitNonInlinedCall = [&](MDefinition* callee, MDefinition* thisv,
                                std::vector<MDefinition*>& cargs,
                                uint32_t argc) -> bool {
    uint32_t scratchAddr = uint32_t(uintptr_t(static_cast<void*>(gWJScratch)));
    for (uint32_t i = 0; i < argc; i++) {
      storeAt(konstI32(int32_t(scratchAddr)), i * 8, js::Scalar::Int64,
              boxForStore(cargs[i]));
    }
    storeAt(konstI32(int32_t(scratchAddr)), kWJThisOff, js::Scalar::Int64,
            boxForStore(thisv ? thisv : konst(0.0)));
    MWJIonCall* call = MWJIonCall::New(alloc, boxForStore(callee), argc);
    cur->add(call);
    boxedVty[call] = 2;  // result is a boxed Value
    push(call);
    return true;
  };
  // `new callee(args)`: marshal args to gWJScratch, then wjhelp(WJH_CONSTRUCT) which
  // does JS::Construct and returns the new object. Lets a hot fn that constructs
  // (splay's `new Node`, deltablue's ctors) compile instead of bailing on JSOp::New.
  auto emitConstructCall = [&](MDefinition* callee,
                               std::vector<MDefinition*>& cargs,
                               uint32_t argc) -> bool {
    uint32_t scratchAddr = uint32_t(uintptr_t(static_cast<void*>(gWJScratch)));
    for (uint32_t i = 0; i < argc; i++) {
      storeAt(konstI32(int32_t(scratchAddr)), i * 8, js::Scalar::Int64,
              boxForStore(cargs[i]));
    }
    MWJIonCall* call = MWJIonCall::New(alloc, boxForStore(callee), argc);
    call->setConstruct();
    cur->add(call);
    boxedVty[call] = 3;  // result is a fresh object Value
    // NOTE (latent): a construct ALLOCATES -> may GC-move objects, so cached field
    // values / unboxed pointers held across it are technically stale (like SetProp).
    // Clearing fieldCache/objPtrMemo here is correct but cost richards ~10% with no
    // demonstrated bench fix (splay's deeper issue is the JIT holding non-GC-rooted
    // pointers across the construct, which a cache-clear does NOT fix), so left as-is
    // to preserve the flagship; revisit with the GC-rooting rewrite.
    push(call);
    return true;
  };
  buildFrame = [&](const WJFrameDesc& fr,
                   std::vector<MDefinition*>& inArgs) -> bool {
    if (fr.depth > kWJInlineMaxDepth) return false;
    if (++gWJBuildFrames > 20000) {
      fprintf(stderr,
              "[ion-be] FRAME GUARD tripped (%u frames) at %s:%u depth=%u "
              "(inline explosion)\n",
              gWJBuildFrames, fr.script->filename() ? fr.script->filename() : "?",
              uint32_t(fr.script->lineno()), fr.depth);
      return false;
    }
    const uint32_t rvalSlotIdx = fr.slotBase + fr.nargs + fr.nfixed;
    jsbytecode* const start = fr.start;
    jsbytecode* const end = fr.end;
    // Record this frame's loop heads (namespaced) for backedge detection.
    for (jsbytecode* p = start; p < end;) {
      uint32_t bl = GetBytecodeLength(p);
      if (bl == 0) {
        fprintf(stderr, "[ion-be] zero bytecode length at line=%u off=%u op=%s\n",
                uint32_t(fr.script->lineno()), uint32_t(p - start),
                js::CodeName(JSOp(*p)));
        return false;
      }
      if (JSOp(*p) == JSOp::LoopHead) {
        loopHeadOff.insert(fr.offBase + uint32_t(p - start));
      }
      p += bl;
    }
    // An inline frame runs on its own operand stack and seeds its arg/local/rval
    // slots in the (live) current block before decoding its body.
    std::vector<MDefinition*> frameStk;
    std::vector<MDefinition*>* savedStk = curStk;
    std::vector<WJPendEdge> retEdges;
    if (fr.isInline) {
      curStk = &frameStk;
      // Uniform i64 slots: args/locals are NaN-boxed Values (boxForStore handles
      // f64/i32/i64) so every slot phi is i64==i64.
      for (uint32_t i = 0; i < fr.nargs; i++) {
        cur->setSlot(info.localSlot(fr.slotBase + i),
                     i < inArgs.size() ? boxForStore(inArgs[i]) : boxForStore(konst(0.0)));
      }
      for (uint32_t j = 0; j < fr.nfixed; j++) {
        cur->setSlot(info.localSlot(fr.slotBase + fr.nargs + j),
                     boxForStore(konst(0.0)));
      }
      cur->setSlot(info.localSlot(rvalSlotIdx), boxForStore(konst(0.0)));
    }

    for (jsbytecode* pc = start; pc < end;) {
      if (++gWJBuildOps > 2000000) {
        fprintf(stderr,
                "[ion-be] BUILD GUARD tripped at %s:%u loff=%u op=%s depth=%u "
                "(infinite builder loop)\n",
                fr.script->filename() ? fr.script->filename() : "?",
                uint32_t(fr.script->lineno()), uint32_t(pc - start),
                js::CodeName(JSOp(*pc)), fr.depth);
        return false;
      }
      uint32_t loff = uint32_t(pc - start);
      uint32_t off = fr.offBase + loff;
      JSOp op = JSOp(*pc);
      uint32_t len = GetBytecodeLength(pc);

      // --- Block-entry transitions ---
      if (op == JSOp::LoopHead) {
        // Start a pending loop header; current (if live) is its entry predecessor.
        if (cur) {
          MBasicBlock* header = MBasicBlock::New(graph, info, cur,
                                                 MBasicBlock::PENDING_LOOP_HEADER);
          if (!header) return false;
          loopDepth++;
          header->setLoopDepth(loopDepth);
          graph.addBlock(header);
          cur->end(MGoto::New(alloc, header));
          loopHeader[off] = header;
          cur = header;
          curStk->clear();  // operand stack is empty at a loop header
          invalidateCSE();  // dominance breaks at a loop header
        }
      } else if (op == JSOp::JumpTarget) {
        // Logical-operator join: the fall-through (RHS) path has its result on the
        // stack; stash it into the logical slot so it phi-merges with the LHS
        // value carried on the short-circuit edge.
        auto lj = logicalJoinSlot.find(off);
        uint32_t logicalL = 0;
        bool isLogical = lj != logicalJoinSlot.end();
        if (isLogical) {
          logicalL = lj->second;
          logicalJoinSlot.erase(lj);
          if (cur) cur->setSlot(info.localSlot(logicalL), boxForStore(pop()));
        }
        auto it = pending.find(off);
        bool haveEdges = it != pending.end() && !it->second.empty();
        if (haveEdges || cur == nullptr) {
          std::vector<WJPendEdge> edges =
              haveEdges ? std::move(it->second) : std::vector<WJPendEdge>();
          if (haveEdges) pending.erase(off);
          MBasicBlock* join = nullptr;
          if (cur) {
            // Fall-through edge from cur into a fresh join block.
            join = MBasicBlock::New(graph, info, cur, MBasicBlock::NORMAL);
            if (!join) return false;
            join->setLoopDepth(loopDepth);
            graph.addBlock(join);
            cur->end(MGoto::New(alloc, join));
          }
          for (const WJPendEdge& e : edges) {
            if (!join) {
              join = MBasicBlock::New(graph, info, e.block, MBasicBlock::NORMAL);
              if (!join) return false;
              join->setLoopDepth(loopDepth);
              graph.addBlock(join);
            } else {
              if (!join->addPredecessor(alloc, e.block)) return false;
            }
            e.block->lastIns()->initSuccessor(e.successor, join);
          }
          cur = join;
          curStk->clear();  // operand stack is empty at a join
          invalidateCSE();  // dominance breaks at a join
          if (isLogical) push(cur->getSlot(info.localSlot(logicalL)));
        }
      }

      if (cur == nullptr) {  // dead code until a target opens a block
        pc += len;
        continue;
      }

      // --- Op emission ---
      switch (op) {
        case JSOp::GetArg: {
          MDefinition* v = cur->getSlot(info.localSlot(fr.slotBase + GET_ARGNO(pc)));
          boxedVty[v] = 2;  // slots are uniform boxed i64 -> unbox lazily at use
          push(v);
          break;
        }
        case JSOp::SetArg:
          cur->setSlot(info.localSlot(fr.slotBase + GET_ARGNO(pc)),
                       boxForStore(curStk->back()));
          break;
        case JSOp::GetLocal: {
          MDefinition* v = cur->getSlot(
              info.localSlot(fr.slotBase + fr.nargs + GET_LOCALNO(pc)));
          boxedVty[v] = 2;
          push(v);
          break;
        }
        case JSOp::SetLocal:
          cur->setSlot(info.localSlot(fr.slotBase + fr.nargs + GET_LOCALNO(pc)),
                       boxForStore(curStk->back()));
          break;
        case JSOp::GetAliasedVar: {
          // Read a closed-over variable inline (hoistable -- the key to navier's
          // numeric loops, which close over the grid arrays/width). gWJCurEnv is the
          // top frame's environment chain (set by WasmJitRunCall). Only valid at
          // depth 0: an inlined callee has a DIFFERENT env that gWJCurEnv doesn't
          // reflect, so bail there. Walk `hops` enclosing envs (ENCLOSING_ENV_SLOT=0
          // -> Value at obj+16, object ptr = low32), then load the var's boxed Value
          // at the fixed slot (obj + 16 + slot*8). Dynamic-slot envs bail.
          // depth>0 (inlined callee): the callee's GetAliasedVar hops are relative
          // to the callee's environment. For a SIBLING closure (same enclosing
          // scope as the top frame, no own per-frame env object) the callee's
          // environment == the top frame's == gWJCurEnv, so walking gWJCurEnv by the
          // callee's own hops is correct -- this is navier's lin_solve/set_bnd case,
          // and unblocks the richards-style full inlining that makes numeric code
          // fast. DEFAULT ON (verified correct on all octane benches incl. navier's
          // own checksum + the 11 gate tests). GECKO_WJVS_NOINLINEALIASED reverts to
          // bail; unsound only for a callee whose enclosing scope differs from the
          // top frame's, which these benches don't hit.
          if (fr.depth != 0 && getenv("GECKO_WJVS_NOINLINEALIASED")) {
            if (WJIonLog()) fprintf(stderr, "[ion-fe] GetAliasedVar inlined -> bail\n");
            return false;
          }
          js::EnvironmentCoordinate ec(pc);
          MDefinition* env = loadAt(
              konstI32(int32_t(uint32_t(uintptr_t(&gWJCurEnv)))), 0,
              js::Scalar::Int32, MIRType::Int32);
          for (unsigned h = ec.hops(); h; h--) {
            // ENCLOSING_ENV_SLOT is reserved slot 0 -> always a fixed slot
            // (obj+16); environments are non-extensible so this never moves.
            MDefinition* encVal = loadAt(env, 16, js::Scalar::Int64, MIRType::Int64);
            MWrapInt64ToInt32* w =
                MWrapInt64ToInt32::New(alloc, encVal, /*bottomHalf=*/true);
            cur->add(w);
            env = w;
          }
          // Non-extensible environment slot layout: slot < MAX_FIXED_SLOTS (16) is a
          // FIXED slot inline at obj+16+slot*8; a slot >= 16 lives in the dynamic
          // slots_ buffer (ptr at obj+8) at index (slot-16). FluidField (navier) has
          // >16 closed-over bindings, so its later function bindings (e.g. set_bnd,
          // lin_solve) land in dynamic slots -- reading them as fixed gave garbage
          // (a non-function), so calling set_bnd via the non-inlined IONCALL threw.
          uint32_t aslot = ec.slot();
          MDefinition* v;
          if (aslot < js::NativeObject::MAX_FIXED_SLOTS) {
            v = loadAt(env, 16 + aslot * 8, js::Scalar::Int64, MIRType::Int64);
          } else {
            MDefinition* slots =
                loadAt(env, 8, js::Scalar::Int32, MIRType::Int32);  // slots_
            v = loadAt(slots, (aslot - js::NativeObject::MAX_FIXED_SLOTS) * 8,
                       js::Scalar::Int64, MIRType::Int64);
          }
          boxedVty[v] = 2;
          push(v);
          break;
        }
        case JSOp::Zero: push(konst(0.0)); break;
        case JSOp::One: push(konst(1.0)); break;
        case JSOp::True:
        case JSOp::False: {
          // A real BOOLEAN Value (tag kWJTagBoolean, payload 1/0), NOT a double --
          // scheme/earley does `x === false` identity checks, and a double 0.0 is
          // not === false. boxedVty=1 (int-like) so asNumber/asInt32 read the low32
          // payload (1/0); the boolean tag is preserved through boxForStore for the
          // identity compare.
          MConstant* b = MConstant::NewInt64(
              alloc, (int64_t(kWJTagBoolean) << 32) | (op == JSOp::True ? 1 : 0));
          cur->add(b);
          boxedVty[b] = 1;
          push(b);
          break;
        }
        case JSOp::Not: {
          // !x. Correct JS truthiness of a boxed value needs tag-aware handling
          // (object -> truthy, number -> !=0 && !NaN, null/undef -> falsy);
          // asInt32 gives 0 for objects (-> wrongly truthy), so only handle the
          // cases we can do correctly and bail otherwise.
          MDefinition* v = pop();
          if (v->type() == MIRType::Int32) {
            MCompare* c = MCompare::NewWasm(alloc, v, konstI32(0), JSOp::Eq,
                                            MCompare::Compare_Int32);
            cur->add(c);
            push(c);
            break;
          }
          if (v->type() == MIRType::Double) {
            MCompare* c = MCompare::NewWasm(alloc, v, konst(0.0), JSOp::Eq,
                                            MCompare::Compare_Double);
            cur->add(c);
            push(c);
            break;
          }
          // Boxed i64: tag-aware JS truthiness. NaN-boxing -- a value is a double
          // iff its high32 (unsigned) is below the tag base (0xFFFFFF81); tags
          // (int/bool/undef/null/object 0xFFFFFF81..8C) are at/above it. For a
          // double, falsy = (==0.0 || NaN). For a tagged value, falsy = (low32==0)
          // -- correct for object(ptr!=0 -> truthy), int/bool(!=0), null/undef
          // (payload 0 -> falsy). (Empty-string truthiness is not modeled; strings
          // don't reach `!x` in these benches.) Pushes i32 (1 = !x is true).
          {
            MWrapInt64ToInt32* hi =
                MWrapInt64ToInt32::New(alloc, v, /*bottomHalf=*/false);
            cur->add(hi);
            MWrapInt64ToInt32* lo =
                MWrapInt64ToInt32::New(alloc, v, /*bottomHalf=*/true);
            cur->add(lo);
            MCompare* isDouble = MCompare::NewWasm(
                alloc, hi, konstI32(int32_t(0xFFFFFF81)), JSOp::Lt,
                MCompare::Compare_UInt32);
            cur->add(isDouble);
            MReinterpretCast* dval =
                MReinterpretCast::New(alloc, v, MIRType::Double);
            cur->add(dval);
            MCompare* dIsZero = MCompare::NewWasm(alloc, dval, konst(0.0),
                                                  JSOp::Eq, MCompare::Compare_Double);
            cur->add(dIsZero);
            MCompare* dIsNaN = MCompare::NewWasm(alloc, dval, dval, JSOp::Ne,
                                                 MCompare::Compare_Double);
            cur->add(dIsNaN);
            MWasmBinaryBitwise* dFalsy = MWasmBinaryBitwise::New(
                alloc, dIsZero, dIsNaN, MIRType::Int32,
                MWasmBinaryBitwise::SubOpcode::Or);
            cur->add(dFalsy);
            MCompare* tFalsy = MCompare::NewWasm(alloc, lo, konstI32(0), JSOp::Eq,
                                                 MCompare::Compare_Int32);
            cur->add(tFalsy);
            MWasmSelect* falsy =
                MWasmSelect::New(alloc, dFalsy, tFalsy, isDouble);
            cur->add(falsy);
            push(falsy);  // i32 0/1 -- !x
            break;
          }
        }
        case JSOp::IsConstructing:
          // Used by functions callable as both `f()` and `new f()`. We can't tell
          // at compile which; pushing a constant is unsound (earley's constructors
          // get the wrong path). Bail to PBL until construct-awareness exists.
          if (getenv("GECKO_WJVS_ISCTOR")) { push(konst(0.0)); break; }
          if (WJIonLog()) fprintf(stderr, "[ion-fe] unsupported op IsConstructing\n");
          return false;
        case JSOp::Int8: push(konst(double(GET_INT8(pc)))); break;
        case JSOp::Int32: push(konst(double(GET_INT32(pc)))); break;
        case JSOp::Uint16: push(konst(double(GET_UINT16(pc)))); break;
        case JSOp::Uint24: push(konst(double(GET_UINT24(pc)))); break;
        case JSOp::Double: {
          double d;
          memcpy(&d, pc + 1, sizeof(double));
          push(konst(d));
          break;
        }
        case JSOp::Add:
          binF64([](TempAllocator& a, MDefinition* l, MDefinition* r) -> MInstruction* {
            return MAdd::NewWasm(a, l, r, MIRType::Double);
          });
          break;
        case JSOp::Sub:
          binF64([](TempAllocator& a, MDefinition* l, MDefinition* r) -> MInstruction* {
            return MSub::NewWasm(a, l, r, MIRType::Double, /*preserveNaN=*/true);
          });
          break;
        case JSOp::Mul:
          binF64([](TempAllocator& a, MDefinition* l, MDefinition* r) -> MInstruction* {
            return MMul::NewWasm(a, l, r, MIRType::Double, MMul::Normal,
                                 /*preserveNaN=*/true);
          });
          break;
        case JSOp::Div:
          binF64([](TempAllocator& a, MDefinition* l, MDefinition* r) -> MInstruction* {
            return MDiv::New(a, l, r, MIRType::Double);  // f64.div (no trap)
          });
          break;
        case JSOp::Neg: {
          // -x : multiply by -1.0 so a zero negates to -0.0 (JS-correct), unlike 0-x.
          MDefinition* x = asNumber(pop());
          MMul* n = MMul::NewWasm(alloc, x, konst(-1.0), MIRType::Double,
                                  MMul::Normal, /*preserveNaN=*/true);
          cur->add(n);
          push(n);
          break;
        }
        case JSOp::BitAnd:
        case JSOp::BitOr:
        case JSOp::BitXor: {
          MDefinition* b = asInt32(pop());
          MDefinition* a = asInt32(pop());
          auto sub = op == JSOp::BitAnd  ? MWasmBinaryBitwise::SubOpcode::And
                     : op == JSOp::BitOr ? MWasmBinaryBitwise::SubOpcode::Or
                                         : MWasmBinaryBitwise::SubOpcode::Xor;
          MWasmBinaryBitwise* n =
              MWasmBinaryBitwise::New(alloc, a, b, MIRType::Int32, sub);
          cur->add(n);
          push(n);  // JS bitwise result is an int32
          break;
        }
        case JSOp::Lsh:
        case JSOp::Rsh:
        case JSOp::Ursh: {
          MDefinition* b = asInt32(pop());
          MDefinition* a = asInt32(pop());
          MInstruction* n =
              op == JSOp::Lsh
                  ? static_cast<MInstruction*>(MLsh::New(alloc, a, b, MIRType::Int32))
              : op == JSOp::Rsh
                  ? static_cast<MInstruction*>(MRsh::New(alloc, a, b, MIRType::Int32))
                  : static_cast<MInstruction*>(
                        MUrsh::NewWasm(alloc, a, b, MIRType::Int32));
          cur->add(n);
          push(n);  // JS shift result is an int32
          break;
        }
        case JSOp::Lt: cmp(JSOp::Lt); break;
        case JSOp::Le: cmp(JSOp::Le); break;
        case JSOp::Gt: cmp(JSOp::Gt); break;
        case JSOp::Ge: cmp(JSOp::Ge); break;
        case JSOp::Eq:
        case JSOp::StrictEq: cmp(JSOp::Eq); break;
        case JSOp::Ne:
        case JSOp::StrictNe: cmp(JSOp::Ne); break;
        case JSOp::StrictConstantEq:
        case JSOp::StrictConstantNe: {
          // Fused `x === <const>` / `x !== <const>`. In the object-pointer model a
          // null/undefined check is an i32 compare against 0; an Int32 const
          // compares against the (numeric) value.
          using ET = js::ConstantCompareOperand::EncodedType;
          js::ConstantCompareOperand cco =
              js::ConstantCompareOperand::fromRawValue(GET_UINT16(pc));
          bool isEq = (op == JSOp::StrictConstantEq);
          MDefinition* val = pop();
          MDefinition* rhs = nullptr;
          MCompare::CompareType ct = MCompare::Compare_Int32;
          ET t = cco.type();
          if (t == ET::Null || t == ET::Undefined) {
            // Object-pointer / null check: unbox to the i32 pointer (0 == null).
            val = asObjPtr(val);
            if (val->type() != MIRType::Int32) {
              if (WJIonLog()) fprintf(stderr, "[ion-fe] StrictConstant null on non-ptr\n");
              return false;
            }
            rhs = konstI32(0);
          } else if (t == ET::Int32) {
            if (val->type() == MIRType::Int32) {
              rhs = konstI32(cco.toInt32());
            } else {
              val = asNumber(val);
              rhs = konst(double(cco.toInt32()));
              ct = MCompare::Compare_Double;
            }
          } else {
            if (WJIonLog()) fprintf(stderr, "[ion-fe] StrictConstant unsupported operand\n");
            return false;
          }
          MCompare* c = MCompare::NewWasm(alloc, val, rhs,
                                          isEq ? JSOp::Eq : JSOp::Ne, ct);
          cur->add(c);
          push(c);
          break;
        }
        case JSOp::Inc:
        case JSOp::Dec: {
          MDefinition* v = asNumber(pop());
          MDefinition* one = konst(1.0);
          MInstruction* n =
              op == JSOp::Inc
                  ? static_cast<MInstruction*>(MAdd::NewWasm(alloc, v, one,
                                                             MIRType::Double))
                  : static_cast<MInstruction*>(MSub::NewWasm(
                        alloc, v, one, MIRType::Double, /*preserveNaN=*/true));
          cur->add(n);
          push(n);
          break;
        }
        case JSOp::Pos:
        case JSOp::ToNumeric:
          break;  // identity on a number
        case JSOp::Pop: pop(); break;
        case JSOp::Dup: push(curStk->back()); break;
        case JSOp::DupAt: {
          uint32_t n = GET_UINT24(pc);
          if (n >= curStk->size()) return false;
          push((*curStk)[curStk->size() - 1 - n]);
          break;
        }
        case JSOp::FunctionThis:
        case JSOp::GlobalThis:
          push(fr.thisDef ? fr.thisDef : konst(0.0));
          break;
        case JSOp::SetRval:
          cur->setSlot(info.localSlot(rvalSlotIdx), boxForStore(pop()));
          break;
        case JSOp::GetRval: push(cur->getSlot(info.localSlot(rvalSlotIdx))); break;
        case JSOp::JumpIfFalse:
        case JSOp::JumpIfTrue: {
          // MTest needs an i32 condition. A boxed i64 (e.g. a boxed boolean from a
          // dispatched method, or an object-ref for `if (obj)`) is reduced to its
          // Tag-aware JS truthiness (object/bool correct); the old asInt32 read an
          // object as ToInt32=0 -> `if(obj)` wrongly false (the high-half WrapInt64
          // backend bug that broke condTruthy before is now fixed).
          MDefinition* condv = condTruthy(pop());
          // A value still on the stack after the condition spans BOTH branch edges
          // (e.g. nested-expression control flow) -- not supported by the per-join
          // single-slot spill; bail to PBL rather than desync the operand stack.
          if (!curStk->empty()) return false;
          uint32_t tgt = uint32_t(int64_t(off) + GET_JUMP_OFFSET(pc));
          uint32_t fall = off + len;
          MTest* test = MTest::New(alloc, condv, nullptr, nullptr);
          cur->end(test);
          // JumpIfTrue: true->tgt, false->fall. JumpIfFalse: true->fall, false->tgt.
          uint32_t trueOff = (op == JSOp::JumpIfTrue) ? tgt : fall;
          uint32_t falseOff = (op == JSOp::JumpIfTrue) ? fall : tgt;
          auto wire = [&](uint32_t edgeOff, uint32_t succIdx) -> bool {
            if (loopHeadOff.count(edgeOff) && edgeOff <= off) {  // backedge
              MBasicBlock* h = loopHeader[edgeOff];
              test->initSuccessor(succIdx, h);
              return h->setBackedgeWasm(cur, 0);
            }
            pending[edgeOff].push_back({cur, succIdx});
            return true;
          };
          if (!wire(trueOff, MTest::TrueBranchIndex)) return false;
          if (!wire(falseOff, MTest::FalseBranchIndex)) return false;
          cur = nullptr;
          break;
        }
        case JSOp::Goto: {
          uint32_t tgt = uint32_t(int64_t(off) + GET_JUMP_OFFSET(pc));
          // Ternary `a ? b : c`: the then-branch evaluates `b` then Goto's the join
          // with `b` on the operand stack; the else-branch falls through with `c`.
          // Spill the carried value to the join slot (the JumpTarget handler merges
          // it with the fall-through `c` and reloads) -- same machinery as ||/&&.
          // Only depth-1 is supported; bail on deeper stacks across the branch.
          if (!curStk->empty()) {
            if (curStk->size() > 1) return false;
            auto lj = logicalJoinSlot.find(tgt);
            uint32_t L;
            if (lj != logicalJoinSlot.end()) {
              L = lj->second;
            } else {
              L = nextSlotBase++;
              if (L >= info.nlocals()) return false;
              logicalJoinSlot[tgt] = L;
            }
            cur->setSlot(info.localSlot(L), boxForStore(pop()));
          }
          MGoto* go = MGoto::New(alloc, nullptr);
          cur->end(go);
          if (loopHeadOff.count(tgt) && tgt <= off) {  // backedge
            MBasicBlock* h = loopHeader[tgt];
            go->initSuccessor(MGoto::TargetIndex, h);
            if (!h->setBackedgeWasm(cur, 0)) return false;
            if (loopDepth) loopDepth--;
          } else {
            pending[tgt].push_back({cur, MGoto::TargetIndex});
          }
          cur = nullptr;
          break;
        }
        case JSOp::Or:
        case JSOp::And: {
          // Short-circuit: keep the LHS value (jump to target) when truthy (Or) /
          // falsy (And); else fall through to evaluate the RHS. The result spans
          // the branch, so spill it to a slot (reloaded at the join).
          MDefinition* v = pop();
          uint32_t L = nextSlotBase++;
          if (L >= info.nlocals()) {
            if (WJIonLog()) fprintf(stderr, "[ion-fe] logical slot overflow\n");
            return false;
          }
          cur->setSlot(info.localSlot(L), boxForStore(v));
          uint32_t tgt = uint32_t(int64_t(off) + GET_JUMP_OFFSET(pc));
          MDefinition* condv = condTruthy(v);
          MTest* test = MTest::New(alloc, condv, nullptr, nullptr);
          MBasicBlock* fall =
              MBasicBlock::New(graph, info, cur, MBasicBlock::NORMAL);
          if (!fall) return false;
          fall->setLoopDepth(loopDepth);
          graph.addBlock(fall);
          cur->end(test);
          uint32_t keepSucc = (op == JSOp::Or) ? MTest::TrueBranchIndex
                                               : MTest::FalseBranchIndex;
          uint32_t fallSucc = (op == JSOp::Or) ? MTest::FalseBranchIndex
                                               : MTest::TrueBranchIndex;
          pending[tgt].push_back({cur, keepSucc});
          test->initSuccessor(fallSucc, fall);
          logicalJoinSlot[tgt] = L;
          cur = fall;
          // SpiderMonkey's JSOp::Or/And PEEK the condition (leave it on the
          // stack); the fall-through path has an explicit JSOp::Pop that removes
          // it before evaluating the RHS. We consumed it via pop() above to stash
          // it into slot L for the short-circuit edge -- re-push it on the
          // fall-through so that JSOp::Pop balances (else the stack underflows to
          // size -1 and corrupts, hanging the builder).
          push(v);
          break;
        }
        case JSOp::Return: {
          MDefinition* r = pop();
          if (!fr.isInline) {
            if (scratchResultBase) {
              storeAt(scratchResultBase, kWJResultOff, js::Scalar::Int64,
                      boxForStore(r));
              cur->end(MWasmReturn::New(alloc, konst(0.0), instance));
            } else {
              cur->end(MWasmReturn::New(alloc, asNumber(r), instance));
            }
          } else {
            // Route to the frame's return-continuation: stash the value (boxed to
            // a uniform i64) in the rval slot and Goto a join built after the body.
            cur->setSlot(info.localSlot(rvalSlotIdx), boxForStore(r));
            MGoto* g = MGoto::New(alloc, nullptr);
            cur->end(g);
            retEdges.push_back({cur, MGoto::TargetIndex});
          }
          cur = nullptr;
          break;
        }
        case JSOp::RetRval: {
          if (!fr.isInline) {
            MDefinition* rv = cur->getSlot(info.localSlot(rvalSlotIdx));
            if (scratchResultBase) {
              storeAt(scratchResultBase, kWJResultOff, js::Scalar::Int64,
                      boxForStore(rv));
              cur->end(MWasmReturn::New(alloc, konst(0.0), instance));
            } else {
              cur->end(MWasmReturn::New(alloc, asNumber(rv), instance));
            }
          } else {
            MGoto* g = MGoto::New(alloc, nullptr);  // rval slot already holds value
            cur->end(g);
            retEdges.push_back({cur, MGoto::TargetIndex});
          }
          cur = nullptr;
          break;
        }
        case JSOp::GetProp: {
          MDefinition* recv = pop();  // receiver object pointer (i32)
          // Method-call idiom (Dup; GetProp method; Swap; Call) takes PRIORITY over
          // own-field resolution: a method load must never be resolved as a
          // same-named field. richards has "queue" as a METHOD on Scheduler but a
          // FIELD on TaskControlBlock -- resolving `this.scheduler.queue` as the
          // TCB.queue field guards the wrong shape and deopts (corrupting the
          // stateful scheduler). The trailing Swap unambiguously marks the idiom.
          if (JSOp(*(pc + len)) == JSOp::Swap) {
            push(recv);  // re-push recv as the callee placeholder for the Call
            methodOffOf[recv] = loff;
            break;
          }
          // Own fixed-slot field -> shape-guarded load (GVN/LICM-able).
          MDefinition* fv = getPropField(fr.script, recv, loff);
          if (fv) {
            push(fv);
          } else {
            if (WJIonLog())
              fprintf(stderr,
                      "[ion-fe] GetProp@%u (%s:%u): no record, next=%s\n", loff,
                      fr.script->filename() ? fr.script->filename() : "?",
                      uint32_t(fr.script->lineno()),
                      js::CodeName(JSOp(*(pc + len))));
            return false;
          }
          break;
        }
        case JSOp::GetElem: {
          if (getenv("GECKO_WJVS_NOELEM")) return false;
          // Dense array element read: shape-guard the array, then load the boxed
          // Value at elements_[index] (elements_ points at element 0). Unchecked
          // bounds (speculative; valid indices in richards). Result is boxed i64.
          auto eit = gWJElemShape.find(WJInlineKey(fr.script, loff));
          if (eit == gWJElemShape.end()) {
            if (WJIonLog()) fprintf(stderr, "[ion-fe] GetElem@%u: no dense-array record\n", loff);
            return false;
          }
          MDefinition* idx = asInt32(pop());
          MDefinition* arr = asObjPtr(pop());
          if (!ensureGuard(arr, eit->second, 6.3)) return false;
          MDefinition* elemsPtr =
              loadAt(arr, 12, js::Scalar::Int32, MIRType::Int32);  // elements_
          // Read bounds: idx u< initializedLength (elements_-12), else deopt (a
          // hole / OOB -> interpreter, which yields undefined).
          if (!boundsGuard(elemsPtr, idx, -12)) return false;
          MMul* byteOff = MMul::NewWasm(alloc, idx, konstI32(8), MIRType::Int32,
                                        MMul::Normal, /*preserveNaN=*/false);
          cur->add(byteOff);
          MAdd* addr =
              MAdd::NewWasm(alloc, elemsPtr, byteOff, MIRType::Int32);
          cur->add(addr);
          MDefinition* el = loadAt(addr, 0, js::Scalar::Int64, MIRType::Int64);
          boxedVty[el] = 2;
          push(el);
          break;
        }
        case JSOp::SetElem:
        case JSOp::StrictSetElem: {
          if (getenv("GECKO_WJVS_NOELEM")) return false;
          // Dense array element write: [arr, idx, val] -> [val].
          auto eit = gWJElemShape.find(WJInlineKey(fr.script, loff));
          if (eit == gWJElemShape.end()) {
            if (WJIonLog()) fprintf(stderr, "[ion-fe] SetElem@%u: no dense-array record\n", loff);
            return false;
          }
          MDefinition* val = pop();
          MDefinition* idx = asInt32(pop());
          MDefinition* arr = asObjPtr(pop());
          if (!ensureGuard(arr, eit->second, 6.4)) return false;
          MDefinition* elemsPtr =
              loadAt(arr, 12, js::Scalar::Int32, MIRType::Int32);
          // Write bounds: idx u< capacity (elements_-8), else deopt -- the inline
          // store can fill an allocated-but-uninitialized slot (dense append) but
          // cannot grow the buffer; a grow goes to the interpreter.
          if (!boundsGuard(elemsPtr, idx, -8)) return false;
          MMul* byteOff = MMul::NewWasm(alloc, idx, konstI32(8), MIRType::Int32,
                                        MMul::Normal, /*preserveNaN=*/false);
          cur->add(byteOff);
          MAdd* addr = MAdd::NewWasm(alloc, elemsPtr, byteOff, MIRType::Int32);
          cur->add(addr);
          storeAt(addr, 0, js::Scalar::Int64, boxForStore(val));
          emitPostBarrier(arr, val);  // GC barrier for an object-valued element store
          // Dense append bookkeeping: writing the raw element slot is not enough --
          // a JS-level read respects the ObjectElements header. Grow
          // initializedLength (elements_ - 12) and length (elements_ - 4) to
          // max(cur, idx+1) so the written element is live instead of a hole
          // (e.g. `new Array(n)` then filled sequentially, richards' packet data).
          // Branch-free select keeps it monotonic (a write below the watermark
          // never shrinks the array).
          MAdd* idxp1 = MAdd::NewWasm(alloc, idx, konstI32(1), MIRType::Int32);
          cur->add(idxp1);
          auto growField = [&](int32_t hdrOff) {
            MAdd* hAddr =
                MAdd::NewWasm(alloc, elemsPtr, konstI32(hdrOff), MIRType::Int32);
            cur->add(hAddr);
            MDefinition* old = loadAt(hAddr, 0, js::Scalar::Int32, MIRType::Int32);
            MCompare* grow = MCompare::NewWasm(alloc, idxp1, old, JSOp::Gt,
                                               MCompare::Compare_Int32);
            cur->add(grow);
            MWasmSelect* nv = MWasmSelect::New(alloc, idxp1, old, grow);
            cur->add(nv);
            storeAt(hAddr, 0, js::Scalar::Int32, nv);
          };
          growField(-12);  // initializedLength
          growField(-4);   // length
          fieldCache.clear();
          objPtrMemo.clear();
          push(val);
          break;
        }
        case JSOp::SetProp:
        case JSOp::StrictSetProp: {
          // Mutating an own slot (fixed or dynamic). Stack: [recv, val] -> [val].
          // Guard the shape, box the value to its raw NaN-boxed i64 Value, and
          // store it. (Boxing makes int/object stores correct, not just doubles.)
          auto rec = gWJShapeRec.find(WJInlineKey(fr.script, loff));
          if (rec == gWJShapeRec.end() || rec->second.shape == 0) {
            // Cold/no-record field WRITE: emit a generic wjhelp(WJH_SETPROP) (runs
            // SetProperty in the interpreter -- correct), keeping the function
            // compiled instead of bailing. Single call, no block boundary.
            if (!getenv("GECKO_WJVS_COLDHELPSET")) {
              if (WJIonLog()) fprintf(stderr, "[ion-fe] SetProp@%u: no shape record\n", loff);
              return false;
            }
            uint32_t s = gWJSiteCount++;
            if (s == 0) s = gWJSiteCount++;
            if (s >= kWJMaxSites) return false;
            gWJSites[s].script = fr.script;
            gWJSites[s].pcOff = loff;
            MDefinition* val = pop();
            MDefinition* recv = pop();
            uint32_t helpBAddr = uint32_t(uintptr_t(&gWJHelpB));
            storeAt(konstI32(int32_t(helpBAddr)), 0, js::Scalar::Int64,
                    boxForStore(val));
            MWJIonCall* mc = MWJIonCall::New(alloc, boxObj(asObjPtr(recv)), 0);
            mc->setSetPropSite(s);
            cur->add(mc);
            fieldCache.clear();
            objPtrMemo.clear();
            push(val);  // SetProp leaves the assigned value on the stack
            break;
          }
          uint32_t recShape = rec->second.shape, foff = rec->second.off;
          // Box to match the field's READ representation. getPropField reads a field
          // observed as int (vty1) as a low32 int payload, so a store must use the
          // int32-tagged Value (not a double Value, whose low32 is garbage). Use the
          // SAME authoritative source the read does -- gWJFieldVty[(shape,offset)] --
          // so write and read representations always agree (rec->second.vty as
          // fallback). gWJPropByName must NOT be used here: it can disagree with the
          // shape+offset field type and silently corrupt the field.
          uint8_t setVty = rec->second.vty;
          {
            auto vit = gWJFieldVty.find((uint64_t(recShape) << 32) | foff);
            if (vit != gWJFieldVty.end()) setVty = vit->second;
          }
          MDefinition* val = pop();
          MDefinition* recv = pop();
          MDefinition* objp = asObjPtr(recv);
          // NO-DEOPT SetProp (GECKO_WJVS_POLYSET): a mono-at-compile SetProp whose
          // receiver is POLY at runtime deopt-restarts on every off-shape store
          // (splay_'s tree restructuring: deopt=6.5 storm -> corruption). Instead emit
          // a diamond: shape-match -> inline typed store; miss -> wjhelp(WJH_SETPROP)
          // (SetProperty, any shape, no restart). The store writes the HEAP (no slot
          // phis) and `val` is pre-diamond (dominates the join), so no spill needed;
          // requires an empty operand stack (statement-level store, splay's case).
          if (getenv("GECKO_WJVS_POLYSET") && curStk->empty()) {
            MDefinition* shapeWord =
                loadAt(objp, 0, js::Scalar::Int32, MIRType::Int32);
            MCompare* g = MCompare::NewWasm(alloc, shapeWord,
                                            konstI32(int32_t(recShape)), JSOp::Eq,
                                            MCompare::Compare_Int32);
            cur->add(g);
            MBasicBlock* matchB = MBasicBlock::New(graph, info, cur, MBasicBlock::NORMAL);
            MBasicBlock* missB = MBasicBlock::New(graph, info, cur, MBasicBlock::NORMAL);
            if (!matchB || !missB) return false;
            matchB->setLoopDepth(loopDepth);
            missB->setLoopDepth(loopDepth);
            graph.addBlock(matchB);
            graph.addBlock(missB);
            cur->end(MTest::New(alloc, g, matchB, missB));
            std::vector<WJPendEdge> sedges;
            cur = matchB;
            {
              uint32_t fo;
              MDefinition* base = fieldBase(objp, foff, &fo);
              storeAt(base, fo, js::Scalar::Int64,
                      setVty == 1 ? boxForStoreInt(val) : boxForStore(val));
              emitPostBarrier(recv, val);
              MGoto* go = MGoto::New(alloc, nullptr);
              cur->end(go);
              sedges.push_back({cur, MGoto::TargetIndex});
            }
            cur = missB;
            {
              uint32_t s = gWJSiteCount++;
              if (s == 0) s = gWJSiteCount++;
              if (s >= kWJMaxSites) return false;
              gWJSites[s].script = fr.script;
              gWJSites[s].pcOff = loff;
              uint32_t helpBAddr = uint32_t(uintptr_t(&gWJHelpB));
              storeAt(konstI32(int32_t(helpBAddr)), 0, js::Scalar::Int64,
                      boxForStore(val));
              MWJIonCall* mc = MWJIonCall::New(alloc, boxObj(objp), 0);
              mc->setSetPropSite(s);
              cur->add(mc);
              MGoto* go = MGoto::New(alloc, nullptr);
              cur->end(go);
              sedges.push_back({cur, MGoto::TargetIndex});
            }
            MBasicBlock* sjoin = nullptr;
            for (const WJPendEdge& e : sedges) {
              if (!sjoin) {
                sjoin = MBasicBlock::New(graph, info, e.block, MBasicBlock::NORMAL);
                if (!sjoin) return false;
                sjoin->setLoopDepth(loopDepth);
                graph.addBlock(sjoin);
              } else if (!sjoin->addPredecessor(alloc, e.block)) {
                return false;
              }
              e.block->lastIns()->initSuccessor(e.successor, sjoin);
            }
            if (!sjoin) return false;
            cur = sjoin;
            invalidateCSE();
            fieldCache.clear();
            objPtrMemo.clear();
            push(val);
            break;
          }
          if (!ensureGuard(objp, recShape, 6.5)) return false;
          uint32_t fo;
          MDefinition* base = fieldBase(objp, foff, &fo);
          storeAt(base, fo, js::Scalar::Int64,
                  setVty == 1 ? boxForStoreInt(val) : boxForStore(val));
          emitPostBarrier(recv, val);  // GC barrier for an object-valued field store
          // Heap mutated: drop cached field values (conservative). Re-reads reload.
          fieldCache.clear();
          objPtrMemo.clear();
          push(val);
          break;
        }
        case JSOp::GetGName:
        case JSOp::GetName: {
          // A global read used as a VALUE (richards' STATE_*/ID_*/KIND_* numeric
          // constants) is baked as a constant (de-facto-const globals). A global
          // that is an object/function is left as a 0.0 placeholder: it is the
          // callee of an inlined call, resolved statically at the Call. (FUTURE:
          // a global-shape guard to deopt if a baked global is ever reassigned.)
          bool baked = false;
          if (JSContext* cx = js::TlsContext.get()) {
            if (js::NativeObject* g = cx->global()) {
              js::PropertyName* nm = fr.script->getName(pc);
              mozilla::Maybe<js::PropertyInfo> prop = g->lookupPure(js::NameToId(nm));
              if (prop.isSome() && prop->isDataProperty()) {
                JS::Value gv = g->getSlot(prop->slot());
                if (gv.isInt32()) { push(konst(double(gv.toInt32()))); baked = true; }
                else if (gv.isDouble()) { push(konst(gv.toDouble())); baked = true; }
                else if (gv.isObject()) {
                  // A function/object global. Baking the raw pointer as a constant
                  // is UNSOUND: a reassigned global (splay's `splayTree = new
                  // SplayTree()` each setup) or a GC move leaves a stale pointer ->
                  // calling through it traps ("table index out of bounds"). Instead
                  // LOAD the global's CURRENT slot value at runtime: the global
                  // object `g` is stable (tenured), the slot index is fixed, and the
                  // slot always holds the live (GC-updated, reassignment-current)
                  // boxed Value. (GECKO_WJVS_BAKEGLOBAL reverts to the old baking.)
                  // Baking the raw pointer is UNSOUND in general (a compacting GC
                  // moves the object, or the global is reassigned -> stale ptr ->
                  // "table index out of bounds" when called). It works in practice
                  // for the default benches (de-facto-const tenured globals), so
                  // keep it as the DEFAULT to preserve richards' 18x (it bakes hot
                  // constructor-global reads as constants). GECKO_WJVS_GLOBALLOAD
                  // switches to the correct runtime slot load (needed for splay's
                  // reassigned splayTree under METHFALL); costs richards ~18x->~7x.
                  // DEFAULT: runtime slot-load (correct -- handles reassignment + GC
                  // moves; richards confirmed still ~18x since loadAt is movable and
                  // these reads aren't hot-loop-bound). GECKO_WJVS_BAKEGLOBAL reverts
                  // to the (unsound) baked pointer.
                  if (getenv("GECKO_WJVS_BAKEGLOBAL")) {
                    uint32_t p = uint32_t(uintptr_t(&gv.toObject()));
                    MConstant* c = MConstant::NewInt64(
                        alloc, (int64_t(kWJTagObject) << 32) | int64_t(uint64_t(p)));
                    cur->add(c);
                    boxedVty[c] = 3;
                    push(c);
                  } else {
                    uint32_t gaddr = uint32_t(uintptr_t(g));
                    uint32_t nfixed = g->numFixedSlots();
                    uint32_t slot = prop->slot();
                    MDefinition* v;
                    if (slot < nfixed) {
                      v = loadAt(konstI32(int32_t(gaddr + 16 + slot * 8)), 0,
                                 js::Scalar::Int64, MIRType::Int64);
                    } else {
                      MDefinition* slots =
                          loadAt(konstI32(int32_t(gaddr + 8)), 0,
                                 js::Scalar::Int32, MIRType::Int32);
                      v = loadAt(slots, (slot - nfixed) * 8, js::Scalar::Int64,
                                 MIRType::Int64);
                    }
                    boxedVty[v] = 3;
                    push(v);
                  }
                  baked = true;
                }
              }
            }
          }
          if (!baked) push(konst(0.0));  // unresolved -> placeholder
          break;
        }
        case JSOp::Undefined:
          // Placeholder for an inlined call's `this`/callee (popped at the Call).
          push(konst(0.0));
          break;
        case JSOp::Null: {
          // null as a BOXED i64 Value (tag kWJTagNull, payload 0), so it has the
          // same representation as object/reference fields. A local that holds a
          // field load on one path and `null` on another (e.g. richards'
          // `packet = this.queue ... else packet = null`) then merges to a uniform
          // i64 phi instead of degrading to MIRType::Value (which has no wasm
          // local). asObjPtr extracts low32==0 so reference/null compares stay i32.
          MConstant* n = MConstant::NewInt64(alloc, int64_t(kWJTagNull) << 32);
          cur->add(n);
          boxedVty[n] = 3;
          push(n);
          break;
        }
        case JSOp::Swap: {
          MDefinition* a = pop();
          MDefinition* b = pop();
          push(a);
          push(b);
          break;
        }
        case JSOp::New:
        case JSOp::NewContent: {
          // Stack: callee, isConstructing, args[0..argc-1], newTarget => rval.
          // Emit a non-inlined construct (JS::Construct in the interpreter); lets a
          // hot fn that constructs compile instead of bailing. (No inlining of the
          // ctor body -> avoids its IsConstructing/apply/arguments.)
          uint32_t argc = GET_ARGC(pc);
          pop();  // newTarget
          std::vector<MDefinition*> cargs(argc, nullptr);
          for (uint32_t i = 0; i < argc; i++) cargs[argc - 1 - i] = pop();
          pop();           // isConstructing
          MDefinition* callee = pop();
          if (callee->type() != MIRType::Int64) {
            if (WJIonLog()) fprintf(stderr, "[ion-fe] New: callee not boxed\n");
            return false;
          }
          if (!emitConstructCall(callee, cargs, argc)) return false;
          break;
        }
        case JSOp::Call:
        case JSOp::CallContent:
        case JSOp::CallIgnoresRv: {
          uint32_t argc = GET_ARGC(pc);
          std::vector<MDefinition*> cargs(argc, nullptr);
          for (uint32_t i = 0; i < argc; i++) cargs[argc - 1 - i] = pop();
          MDefinition* thisv = pop();      // this (receiver for methods)
          MDefinition* calleeDef = pop();  // callee placeholder
          // Method polymorphic dispatch: guard the receiver shape, inline each
          // observed task type's body (richards' megamorphic task.run).
          auto moit = methodOffOf.find(calleeDef);
          if (moit != methodOffOf.end()) {
            auto mit = gWJMethodPoly.find(WJInlineKey(fr.script, moit->second));
            if (mit != gWJMethodPoly.end() && mit->second.n >= 1) {
              // The method name (from the GetProp method-load site) lets the
              // no-match path do a dynamic recv[name]() call instead of deopting.
              jsbytecode* gpc = fr.script->offsetToPC(moit->second);
              uint32_t methNameLow =
                  uint32_t(uintptr_t(fr.script->getName(gpc)));
              if (!emitMethodDispatch(mit->second, thisv, cargs, argc, fr.depth,
                                      methNameLow)) {
                return false;
              }
              break;
            }
          }
          // Free-function monomorphic inline.
          JSScript* cs = WJResolveInlineCallee(fr.script, loff, argc);
          if (!cs) {
            // Can't inline (no recorded callee). If this is a FREE function call
            // (calleeDef is the real callee Value, not a method receiver), emit a
            // non-inlined call instead of bailing the whole function.
            bool isMethod = methodOffOf.find(calleeDef) != methodOffOf.end();
            if (!isMethod && calleeDef->type() == MIRType::Int64) {
              if (!emitNonInlinedCall(calleeDef, thisv, cargs, argc)) return false;
              break;
            }
            // Method call with no inlinable record (raytrace's shape.intersect,
            // light loops): instead of bailing the whole function to PBL, emit a
            // dynamic recv[name]() via wjhelp(WJH_METHCALL) -- the interpreter runs
            // the method, the compiled caller keeps running. Same machinery as the
            // dispatch no-match arm. Gated default-off until validated broadly.
            if (isMethod && getenv("GECKO_WJVS_METHFALL")) {
              uint32_t moff = methodOffOf[calleeDef];
              jsbytecode* gpc = fr.script->offsetToPC(moff);
              uint32_t methNameLow = uint32_t(uintptr_t(fr.script->getName(gpc)));
              if (methNameLow) {
                uint32_t scratchAddr =
                    uint32_t(uintptr_t(static_cast<void*>(gWJScratch)));
                for (uint32_t i = 0; i < argc; i++) {
                  storeAt(konstI32(int32_t(scratchAddr)), i * 8,
                          js::Scalar::Int64, boxForStore(cargs[i]));
                }
                // Marshal the receiver as a faithful boxed Value (object recv ->
                // boxObj; already-boxed i64 -> as-is; number -> boxForStore).
                MDefinition* recvVal =
                    thisv->type() == MIRType::Int64       ? thisv
                    : thisv->type() == MIRType::Int32     ? boxObj(thisv)
                                                          : boxForStore(thisv);
                storeAt(konstI32(int32_t(scratchAddr)), kWJThisOff,
                        js::Scalar::Int64, recvVal);
                MConstant* dummy = MConstant::NewInt64(alloc, 0);
                cur->add(dummy);
                MWJIonCall* mc = MWJIonCall::New(alloc, dummy, argc);
                mc->setMethName(methNameLow);
                cur->add(mc);
                boxedVty[mc] = 2;
                push(mc);
                break;
              }
            }
            if (WJIonLog()) fprintf(stderr, "[ion-fe] Call@%u: callee not inlinable\n", loff);
            return false;
          }
          JSFunction* cf = cs->function();
          WJFrameDesc cfr;
          cfr.script = cs;
          cfr.start = cs->code();
          cfr.end = cs->codeEnd();
          cfr.nargs = cf ? cf->nargs() : argc;
          cfr.nfixed = cs->nfixed();
          cfr.slotBase = nextSlotBase;
          cfr.offBase = nextOffBase;
          cfr.thisDef = thisv;
          cfr.isInline = true;
          cfr.depth = fr.depth + 1;
          if (cfr.slotBase + cfr.nargs + cfr.nfixed + 1 > info.nlocals()) {
            if (WJIonLog()) fprintf(stderr, "[ion-fe] inline slot overflow\n");
            return false;
          }
          nextSlotBase += cfr.nargs + cfr.nfixed + 1;
          nextOffBase += uint32_t(cfr.end - cfr.start) + 1;
          uint32_t crvalIdx = cfr.slotBase + cfr.nargs + cfr.nfixed;
          if (!buildFrame(cfr, cargs)) return false;
          // buildFrame left cur = the callee's return-continuation and restored
          // curStk to this frame's stack; the rval slot holds the (boxed) return.
          MDefinition* cret = cur->getSlot(info.localSlot(crvalIdx));
          boxedVty[cret] = 2;  // inline return is a boxed i64 -> unbox lazily
          push(cret);
          break;
        }
        case JSOp::JumpTarget:
        case JSOp::LoopHead:
        case JSOp::Nop:
        case JSOp::NopIsAssignOp:
        case JSOp::NopDestructuring:
          break;
        default:
          if (WJIonLog()) fprintf(stderr, "[ion-fe] unsupported op %s\n", js::CodeName(op));
          return false;
      }
      pc += len;
    }

    // --- Inline-frame return continuation ---
    if (fr.isInline) {
      MBasicBlock* cont = nullptr;
      uint32_t nEdges = 0;
      auto addRet = [&](MBasicBlock* b, uint32_t succ) -> bool {
        if (!cont) {
          cont = MBasicBlock::New(graph, info, b, MBasicBlock::NORMAL);
          if (!cont) return false;
          cont->setLoopDepth(loopDepth);
          graph.addBlock(cont);
        } else {
          if (!cont->addPredecessor(alloc, b)) return false;
        }
        b->lastIns()->initSuccessor(succ, cont);
        nEdges++;
        return true;
      };
      for (const WJPendEdge& e : retEdges) {
        if (!addRet(e.block, e.successor)) return false;
      }
      if (cur) {  // fell off the end without an explicit return
        MGoto* g = MGoto::New(alloc, nullptr);
        cur->end(g);
        if (!addRet(cur, MGoto::TargetIndex)) return false;
      }
      if (!cont) return false;  // callee has no reachable return
      cur = cont;
      curStk = savedStk;  // restore caller's operand stack
      // The continuation joins multiple branches only when there was >1 return
      // edge; dominance breaks there, so the CSE state no longer holds. A single
      // straight-line return keeps the (linear) CSE state for the caller.
      if (nEdges > 1) invalidateCSE();
    }
    return true;
  };

  WJFrameDesc top;
  top.script = script;
  top.start = start;
  top.end = end;
  top.slotBase = 0;
  top.nargs = nargs;
  top.nfixed = nfixed;
  top.offBase = 0;
  top.thisDef = topThisDef;
  top.isInline = false;
  top.depth = 0;
  std::vector<MDefinition*> noArgs;
  return buildFrame(top, noArgs);
}


// Populate gWJShapeRec / gWJInlineCallee for `script` by reading its Baseline ICs
// (CacheIR) directly -- the same data source real Ion's WarpOracle uses. This is
// DECOUPLED from whether the WJ JIT compiled the function, so even complex hot fns
// (richards schedule) that the WJ JIT skips still get shape/field/callee data
// (they always run in the Baseline interpreter, which fills CacheIR ICs). Existing
// runtime records are not clobbered. Each IC stub is one observed shape/callee way.
static void WJReadBaselineICs(JSScript* script) {
  using namespace js::jit;
  bool dbg = getenv("GECKO_WJVS_ORACLE_DBG") != nullptr;
  if (!script->hasJitScript()) {
    if (dbg) fprintf(stderr, "[oracle] %s:%u NO JitScript\n",
                     script->filename() ? script->filename() : "?",
                     uint32_t(script->lineno()));
    return;
  }
  JitScript* jitScript = script->jitScript();
  jsbytecode* start = script->code();
  jsbytecode* end = script->codeEnd();
  for (jsbytecode* pc = start; pc < end; pc += GetBytecodeLength(pc)) {
    JSOp op = JSOp(*pc);
    bool isProp = (op == JSOp::GetProp || op == JSOp::SetProp ||
                   op == JSOp::StrictSetProp);
    bool isCall = (op == JSOp::Call || op == JSOp::CallContent ||
                   op == JSOp::CallIgnoresRv);
    bool isElem = (op == JSOp::GetElem || op == JSOp::SetElem ||
                   op == JSOp::StrictSetElem);
    if (!isProp && !isCall && !isElem) continue;
    uint32_t pcOff = uint32_t(pc - start);
    // SAFE IC-entry lookup: icEntryFromPCOffset is a binary search whose only
    // bounds/exact-match safety is a MOZ_ASSERT -- COMPILED OUT in this NDEBUG
    // build -- so on a JitScript whose IC-entry table doesn't contain this pcOff
    // (or is empty) it derefs out of the array and reads OOB guest memory (a hard
    // wasm trap that aborts the engine; hit while reading deltablue's ICs). Scan
    // the entries by their fallback-stub pcOffset instead: bounded by
    // numICEntries(), no OOB, and skips any op that genuinely has no IC entry.
    ICEntry* entryp = nullptr;
    uint32_t nIC = jitScript->numICEntries();
    for (uint32_t ei = 0; ei < nIC; ei++) {
      if (jitScript->fallbackStub(ei)->pcOffset() == pcOff) {
        entryp = &jitScript->icEntry(ei);
        break;
      }
    }
    if (!entryp) continue;
    ICEntry& entry = *entryp;
    if (dbg) {
      fprintf(stderr, "[oracle] %s:%u off=%u op=%s firstStubFallback=%d\n",
              script->filename() ? script->filename() : "?",
              uint32_t(script->lineno()), pcOff, js::CodeName(op),
              entry.firstStub()->isFallback());
    }
    for (ICStub* s = entry.firstStub(); s && !s->isFallback();
         s = s->maybeNext()) {
      ICCacheIRStub* cir = s->toCacheIRStub();
      const CacheIRStubInfo* info = cir->stubInfo();
      CacheIRReader reader(info->code(), info->code() + info->codeLength());
      uint32_t shape = 0, foff = 0, calleeFn = 0;
      bool haveShape = false, haveOff = false;
      int recvObjId = -1, loadObjId = -1;
      JSObject* holderObj = nullptr;
      // Bounds-checked stub-data reads: a CacheIR reader desync (an op whose
      // operands we don't consume exactly) yields an out-of-range stubOffset, and
      // getStubRawWord/Int32 then read OOB guest memory -> a hard wasm trap that
      // aborts the engine (hit reading deltablue's megamorphic ICs). Validate the
      // offset against stubDataSize() and bail the stub instead of trapping.
      size_t stubSz = info->stubDataSize();
      bool stubBail = false;
      auto rawWord = [&](uint32_t off) -> uintptr_t {
        if (off + sizeof(uintptr_t) > stubSz) { stubBail = true; return 0; }
        return info->getStubRawWord(cir, off);
      };
      auto rawI32 = [&](uint32_t off) -> int32_t {
        if (off + sizeof(int32_t) > stubSz) { stubBail = true; return 0; }
        return info->getStubRawInt32(cir, off);
      };
      while (reader.more() && !stubBail) {
        CacheOp cop = reader.readOp();
        if (dbg && pcOff >= 55 && pcOff <= 72) {
          fprintf(stderr, "[oracle]   off=%u cop=%s\n", pcOff,
                  CacheIRCodeName(cop));
        }
        switch (cop) {
          case CacheOp::GuardShape: {
            uint32_t oid = reader.objOperandId().id();
            uint32_t shp = uint32_t(rawWord(reader.stubOffset()));
            // Only the FIRST GuardShape (the receiver). A proto/method load also
            // guards the holder shape -- ignore that one.
            if (recvObjId < 0) {
              recvObjId = int(oid);
              shape = shp;
              haveShape = true;
            }
            break;
          }
          case CacheOp::LoadFixedSlotResult: {
            loadObjId = int(reader.objOperandId().id());
            foff = uint32_t(rawI32(reader.stubOffset()));
            haveOff = true;
            break;
          }
          case CacheOp::LoadDynamicSlotResult: {
            loadObjId = int(reader.objOperandId().id());
            foff = uint32_t(rawI32(reader.stubOffset())) |
                   kWJDynSlot;
            haveOff = true;
            break;
          }
          case CacheOp::StoreFixedSlot: {
            loadObjId = int(reader.objOperandId().id());
            foff = uint32_t(rawI32(reader.stubOffset()));
            haveOff = true;
            reader.skip(CacheIROpInfos[size_t(cop)].argLength - 2);
            break;
          }
          case CacheOp::StoreDynamicSlot: {
            loadObjId = int(reader.objOperandId().id());
            foff = uint32_t(rawI32(reader.stubOffset())) |
                   kWJDynSlot;
            haveOff = true;
            reader.skip(CacheIROpInfos[size_t(cop)].argLength - 2);
            break;
          }
          case CacheOp::LoadObject: {
            reader.objOperandId();  // result id
            holderObj = reinterpret_cast<JSObject*>(rawWord(reader.stubOffset()));
            break;
          }
          case CacheOp::LoadInt32ArrayLengthResult: {
            // array.length: record the (receiver-shape-guarded) site so the FE
            // loads the ObjectElements header length instead of a bogus slot.
            reader.objOperandId();
            if (haveShape) gWJLenSite[WJInlineKey(script, pcOff)] = shape;
            break;
          }
          case CacheOp::GuardSpecificFunction: {
            reader.objOperandId();
            calleeFn = uint32_t(rawWord(reader.stubOffset()));
            reader.skip(CacheIROpInfos[size_t(cop)].argLength - 2);
            break;
          }
          default:
            reader.skip(CacheIROpInfos[size_t(cop)].argLength);
            break;
        }
      }
      if (stubBail) continue;  // desynced/unsafe stub -> ignore it, don't crash
      // Only record own-slot accesses (slot loaded from the shape-guarded
      // receiver). A proto/method load reads from a different (holder) object --
      // skip it so the FE's method-call idiom + polymorphic dispatch still fire.
      if (isProp && haveShape && haveOff && loadObjId == recvObjId) {
        // typeData is unpopulated in the portable-baseline build, so take the vty
        // observed for this (shape, offset) by any wj-compiled function; fall back
        // to typeData, else 2 (bail).
        uint8_t vty = 2;
        auto vit = gWJFieldVty.find((uint64_t(shape) << 32) | foff);
        if (vit != gWJFieldVty.end()) {
          vty = vit->second;
        } else {
          switch (cir->typeData().type()) {
            case JSVAL_TYPE_INT32: vty = 1; break;
            case JSVAL_TYPE_DOUBLE: vty = 0; break;
            case JSVAL_TYPE_OBJECT:
            case JSVAL_TYPE_NULL: vty = 3; break;
            default: vty = 2; break;
          }
        }
        // Still unknown -> read the field's value type from a live sample instance
        // of this shape (validated), so unobserved fields are still unboxed.
        if (vty == 2 && !getenv("GECKO_WJVS_NOSAMPLE")) {
          auto sit = gWJShapeSample.find(shape);
          if (sit != gWJShapeSample.end()) {
            js::NativeObject* so =
                reinterpret_cast<js::NativeObject*>(uintptr_t(sit->second));
            if (so && uint32_t(uintptr_t(so->shape())) == shape) {  // still valid
              uint32_t slot = (foff & kWJDynSlot)
                                  ? (((foff & ~kWJDynSlot) / 8) + so->numFixedSlots())
                                  : ((foff - 16) / 8);
              if (slot < so->slotSpan()) {
                JS::Value fv = so->getSlot(slot);
                if (fv.isInt32()) vty = 1;
                else if (fv.isDouble()) vty = 0;
                else if (fv.isObject() || fv.isNull()) vty = 3;
                if (vty != 2) gWJFieldVty[(uint64_t(shape) << 32) | foff] = vty;
              }
            }
          }
        }
        uint64_t key = WJInlineKey(script, pcOff);
        if (dbg) fprintf(stderr, "[oracle]   -> shape=%#x off=%#x vty=%u rawtype=%#x\n",
                         shape, foff, vty, uint32_t(cir->typeData().type()));
        if (gWJShapeRec.find(key) == gWJShapeRec.end()) {
          gWJShapeRec[key] = WJShapeRec{shape, foff, vty};
        }
        // Accumulate every distinct receiver shape this site observes (one per
        // stub) so a polymorphic field site can be inlined as a shape-guard chain.
        {
          WJFieldPolyRec& fp = gWJFieldPoly[key];
          bool seen = false;
          for (uint8_t i = 0; i < fp.n; i++) {
            if (fp.shapes[i] == shape) { seen = true; break; }
          }
          if (!seen && fp.n < 4) {
            fp.shapes[fp.n] = shape;
            fp.offs[fp.n] = foff;
            fp.vtys[fp.n] = vty;
            fp.n++;
          }
        }
        // Index by property name for cold-site (fallback-IC) resolution.
        jsbytecode* spc = script->offsetToPC(pcOff);
        uint32_t nameLow = uint32_t(uintptr_t(script->getName(spc)));
        auto pit = gWJPropByName.find(nameLow);
        if (pit == gWJPropByName.end()) {
          gWJPropByName[nameLow] = WJPropRec{shape, foff, vty, false};
        } else if (pit->second.shape != shape || pit->second.off != foff) {
          pit->second.ambig = true;  // same name on different layouts
        }
      } else if (isProp && haveShape && haveOff && holderObj &&
                 holderObj->is<js::NativeObject>()) {
        // Proto-property load (method): read the function from the holder slot and
        // record (recvShape -> methodFn) so the FE's polymorphic dispatch can
        // inline each receiver type's method. Each stub is one observed way.
        js::NativeObject& nh = holderObj->as<js::NativeObject>();
        uint32_t slot = (foff & kWJDynSlot)
                            ? (((foff & ~kWJDynSlot) / 8) + nh.numFixedSlots())
                            : ((foff - 16) / 8);
        if (slot < nh.slotSpan()) {
          JS::Value mv = nh.getSlot(slot);
          if (mv.isObject() && mv.toObject().is<JSFunction>()) {
            uint32_t fnLow = uint32_t(uintptr_t(&mv.toObject()));
            WJMethodPolyRec& m = gWJMethodPoly[WJInlineKey(script, pcOff)];
            bool seen = false;
            for (uint8_t i = 0; i < m.n; i++) {
              if (m.shapes[i] == shape) { seen = true; break; }
            }
            if (!seen && m.n < 4) {
              m.shapes[m.n] = shape;
              m.fns[m.n] = fnLow;
              m.n++;
            }
            if (dbg) {
              fprintf(stderr,
                      "[oracle]   method-poly off=%u shape=%#x fn=%#x (n=%u)\n",
                      pcOff, shape, fnLow, m.n);
            }
          }
        }
      }
      if (isElem && haveShape) {
        // Dense element access: record the array shape (the FE shape-guards then
        // loads elements_[index] directly). Skip typed-array/other element ICs
        // (no plain GuardShape+dense pattern -> haveShape stays from the recv).
        uint64_t key = WJInlineKey(script, pcOff);
        if (gWJElemShape.find(key) == gWJElemShape.end()) {
          gWJElemShape[key] = shape;
        }
      }
      if (isCall && calleeFn) WJRecordInlineCallee(script, pcOff, calleeFn);
    }
  }
}

// Read Baseline ICs for `script` and (depth-bounded) for every inlinable callee
// in its call graph, so the Ion FE has shape/field/callee data for the whole
// inline tree even when the WJ JIT compiled none of them. `seen` MEMOIZES visited
// scripts: deltablue's call graph is densely connected (constraints<->variables),
// so without memoization the depth-bounded recursion fans out EXPONENTIALLY -> a
// C++ shadow-stack overflow that manifests as a guest "memory access out of
// bounds" trap. Memoizing makes it linear and also breaks cycles.
static void WJReadICsRecursive(JSScript* script, int depth,
                               std::unordered_set<JSScript*>& seen) {
  if (depth > kWJInlineMaxDepth) return;
  if (!seen.insert(script).second && !getenv("GECKO_WJVS_NOMEMO"))
    return;  // already read this script (memoize -> avoid exponential fan-out)
  WJReadBaselineICs(script);
  jsbytecode* start = script->code();
  jsbytecode* end = script->codeEnd();
  auto recurse = [&](uint32_t fnLow) {
    if (!WJGuestPtrOk(fnLow)) return;  // stale/desynced fn pointer -> don't deref
    JSFunction* fun = reinterpret_cast<JSFunction*>(uintptr_t(fnLow));
    if (fun && fun->isInterpreted() && fun->baseScript() &&
        fun->baseScript()->hasBytecode()) {
      WJReadICsRecursive(fun->baseScript()->asJSScript(), depth + 1, seen);
    }
  };
  for (jsbytecode* pc = start; pc < end; pc += GetBytecodeLength(pc)) {
    JSOp op = JSOp(*pc);
    uint32_t off = uint32_t(pc - start);
    if (op == JSOp::Call || op == JSOp::CallContent ||
        op == JSOp::CallIgnoresRv) {
      auto it = gWJInlineCallee.find(WJInlineKey(script, off));
      if (it != gWJInlineCallee.end()) {
        for (uint8_t w = 0; w < it->second.n; w++) recurse(it->second.fns[w]);
      }
    } else if (op == JSOp::GetProp) {
      // Method-load site: recurse into each polymorphic dispatch target.
      auto mit = gWJMethodPoly.find(WJInlineKey(script, off));
      if (mit != gWJMethodPoly.end()) {
        for (uint8_t w = 0; w < mit->second.n; w++) recurse(mit->second.fns[w]);
      }
    }
  }
}

// Front-end test: compile one real JSScript (straight-line numeric) through the
// full bytecode->MIR->OptimizeMIR->wasm pipeline, call it with args parsed from
// GECKO_WJVS_IONFE_ARGS ("a,b,..."), and print the result. Gated GECKO_WJVS_IONFE
// (=target lineno). Verifies the FRONT-END produces correct, runnable wasm.
static void WJIonFrontendTest(JSScript* script) {
  using namespace js::jit;
  JSFunction* fun = script->function();
  if (!fun) return;
  uint32_t nargs = fun->nargs();
  uint32_t nfixed = script->nfixed();
  // Seed shape/field/callee data from Baseline ICs (the decoupled oracle), so the
  // FE works on functions the WJ JIT never compiled (e.g. richards schedule).
  { std::unordered_set<JSScript*> wjSeen; WJReadICsRecursive(script, 1, wjSeen); }
  fprintf(stderr, "[ion-fe] compiling %s:%u nargs=%u nfixed=%u\n",
          script->filename() ? script->filename() : "?",
          uint32_t(script->lineno()), nargs, nfixed);
  if (getenv("GECKO_WJVS_IONFE_DUMP")) {
    jsbytecode* st = script->code();
    jsbytecode* en = script->codeEnd();
    for (jsbytecode* p = st; p < en; p += GetBytecodeLength(p)) {
      JSOp o = JSOp(*p);
      int tgt = -1;
      if (o == JSOp::Goto || o == JSOp::JumpIfFalse || o == JSOp::JumpIfTrue ||
          o == JSOp::And || o == JSOp::Or || o == JSOp::Coalesce) {
        tgt = int(int64_t(p - st) + GET_JUMP_OFFSET(p));
      }
      fprintf(stderr, "  %4d: %s%s", int(p - st), js::CodeName(o),
              o == JSOp::LoopHead ? " [LOOPHEAD]" : "");
      if (tgt >= 0) fprintf(stderr, " -> %d", tgt);
      fprintf(stderr, "\n");
    }
  }
  fflush(stderr);

  LifoAlloc lifo(TempAllocator::PreferredLifoChunkSize,
                 js::BackgroundMallocArena);
  TempAllocator alloc(&lifo);
  JitContext jitContext;
  // Slots: this frame's args + locals + rval, plus a slot range for every
  // inlined callee frame in the call graph (see WJCountInlineSlots).
  CompileInfo compileInfo(nargs + nfixed + 1 + WJCountInlineSlots(script, 1));
  MIRGraph graph(&alloc);
  JitCompileOptions options;
  MIRGenerator mirGen(nullptr, options, &alloc, &graph, &compileInfo,
                      IonOptimizations.get(OptimizationLevel::Wasm), nullptr);

  MBasicBlock* entry =
      MBasicBlock::New(graph, compileInfo, nullptr, MBasicBlock::NORMAL);
  if (!entry) return;
  graph.addBlock(entry);

  // Classify args: one used as a `GetArg k; GetProp` receiver is an object
  // pointer (i32); the rest are numeric (f64). Also find the first GetProp's
  // pcOff so the call can pass a real captured receiver for arg0.
  uint32_t objArgMask = 0;
  uint32_t firstGetPropOff = UINT32_MAX;
  {
    jsbytecode* st = script->code();
    jsbytecode* en = script->codeEnd();
    // A local is an "object local" if it is ever a GetProp receiver (GetLocal m;
    // GetProp, or the method idiom GetLocal m; Dup; GetProp). An arg is an object
    // pointer if it is a GetProp receiver directly, OR is copied into an object
    // local (GetArg k; SetLocal m) -- the common `var c = head; ... c.next` cursor.
    std::unordered_set<uint32_t> objLocals;
    std::unordered_map<uint32_t, uint32_t> localFromArg;  // local <- arg
    jsbytecode* prev = nullptr;
    jsbytecode* prev2 = nullptr;
    for (jsbytecode* p = st; p < en; p += GetBytecodeLength(p)) {
      JSOp o = JSOp(*p);
      if (o == JSOp::GetProp) {
        if (firstGetPropOff == UINT32_MAX) firstGetPropOff = uint32_t(p - st);
        if (prev && JSOp(*prev) == JSOp::GetArg) {
          objArgMask |= (1u << GET_ARGNO(prev));
        } else if (prev && JSOp(*prev) == JSOp::GetLocal) {
          objLocals.insert(GET_LOCALNO(prev));
        } else if (prev && JSOp(*prev) == JSOp::Dup && prev2 &&
                   JSOp(*prev2) == JSOp::GetArg) {
          objArgMask |= (1u << GET_ARGNO(prev2));
        } else if (prev && JSOp(*prev) == JSOp::Dup && prev2 &&
                   JSOp(*prev2) == JSOp::GetLocal) {
          objLocals.insert(GET_LOCALNO(prev2));
        }
      } else if (o == JSOp::SetLocal && prev && JSOp(*prev) == JSOp::GetArg) {
        localFromArg[GET_LOCALNO(p)] = GET_ARGNO(prev);
      }
      prev2 = prev;
      prev = p;
    }
    for (uint32_t m : objLocals) {
      auto it = localFromArg.find(m);
      if (it != localFromArg.end()) objArgMask |= (1u << it->second);
    }
  }

  // Detect a method that reads `this` (FunctionThis/GlobalThis). If so, `this` is
  // passed as an extra trailing object param (i32 pointer) -- NOT a CompileInfo
  // slot, since it is accessed via the thisDef operand.
  bool useThis = false;
  for (jsbytecode* p = script->code(); p < script->codeEnd();
       p += GetBytecodeLength(p)) {
    JSOp o = JSOp(*p);
    if (o == JSOp::FunctionThis || o == JSOp::GlobalThis) { useThis = true; break; }
  }

  std::vector<MWasmParameter*> argParams(nargs ? nargs : 1, nullptr);
  for (uint32_t i = 0; i < nargs; i++) {
    MIRType ty = (objArgMask & (1u << i)) ? MIRType::Int32 : MIRType::Double;
    argParams[i] = MWasmParameter::New(alloc, ABIArg(), ty);
    entry->add(argParams[i]);
  }
  MWasmParameter* thisParam = nullptr;
  if (useThis) {
    thisParam = MWasmParameter::New(alloc, ABIArg(), MIRType::Int32);
    entry->add(thisParam);
  }
  MWasmParameter* instance =
      MWasmParameter::New(alloc, ABIArg(InstanceReg), MIRType::Pointer);
  entry->add(instance);

  // Full wasm param list = normal args, then `this` (if used). The backend binds
  // these to param locals 0..paramCount-1 in this order.
  std::vector<MWasmParameter*> paramList(argParams.begin(),
                                         argParams.begin() + nargs);
  if (useThis) paramList.push_back(thisParam);
  uint32_t paramCount = uint32_t(paramList.size());

  if (!WJIonBuildMIR(script, mirGen, alloc, graph, compileInfo, entry, instance,
                     argParams.data(), nargs, nfixed, thisParam)) {
    fprintf(stderr, "[ion-fe] BUILD MIR bailed\n");
    return;
  }
  fprintf(stderr, "[ion-fe] MIR built (%zu blocks), optimizing...\n",
          graph.numBlocks());
  fflush(stderr);
  if (getenv("GECKO_WJVS_IONFE_NOOPT")) {
    fprintf(stderr, "[ion-fe] skipping OptimizeMIR (NOOPT)\n");
  } else if (!OptimizeMIR(&mirGen)) {
    fprintf(stderr, "[ion-fe] OptimizeMIR FAILED\n");
    return;
  }
  fprintf(stderr, "[ion-fe] optimized, blocks=%zu\n", graph.numBlocks());

  if (getenv("GECKO_WJVS_IONFE_DUMP")) {
    // Custom dump (JS_JITSPEW is off in NDEBUG): per-block loop depth + counts
    // of the nodes the property-access optimization targets. WasmLoads/compares
    // sitting at loopDepth 0 means GVN/LICM hoisted them out of the loop.
    uint32_t totLoad = 0, loopLoad = 0, totCmp = 0, loopCmp = 0;
    for (ReversePostorderIterator b = graph.rpoBegin(); b != graph.rpoEnd();
         b++) {
      uint32_t ld = b->loopDepth();
      uint32_t nLoad = 0, nCmp = 0, nAdd = 0, nPhi = 0;
      for (MPhiIterator p = b->phisBegin(); p != b->phisEnd(); p++) nPhi++;
      for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
        if (it->isWasmLoad()) nLoad++;
        else if (it->isCompare()) nCmp++;
        else if (it->isAdd()) nAdd++;
      }
      totLoad += nLoad; totCmp += nCmp;
      if (ld > 0) { loopLoad += nLoad; loopCmp += nCmp; }
      fprintf(stderr,
              "  [blk %u] loopDepth=%u phis=%u load=%u cmp=%u add=%u\n",
              b->id(), ld, nPhi, nLoad, nCmp, nAdd);
    }
    fprintf(stderr,
            "[ion-fe] node summary: wasmLoad total=%u inLoop=%u | "
            "compare total=%u inLoop=%u\n",
            totLoad, loopLoad, totCmp, loopCmp);
    fflush(stderr);
  }

  Bytes out;
  Encoder e(out);
  const uint8_t kF64 = uint8_t(TypeCode::F64);
  if (!e.writeFixedU32(MagicNumber) ||
      !e.writeFixedU32(EncodingVersionModule)) {
    return;
  }
  size_t s;
  // Type: (param... ) -> f64, each param i32 (object) or f64 (numeric).
  if (!e.startSection(SectionId::Type, &s) || !e.writeVarU32(1) ||
      !e.writeFixedU8(0x60) || !e.writeVarU32(paramCount)) {
    return;
  }
  for (uint32_t i = 0; i < paramCount; i++) {
    if (!e.writeFixedU8(WJWasmValType(paramList[i]->type()))) return;
  }
  if (!e.writeVarU32(1) || !e.writeFixedU8(kF64)) return;
  e.finishSection(s);
  // Import the guest linear memory as "m"."mem" so MWasmLoad can read objects.
  int memId = wasmhost_guest_mem_objid();
  bool sharedMem = memId >= 0 && wasmhost_guest_mem_shared() != 0;
  if (memId < 0) {
    fprintf(stderr, "[ion-fe] no guest memory to import\n");
    return;
  }
  if (!e.startSection(SectionId::Import, &s) || !e.writeVarU32(1) ||
      !e.writeBytes("m", 1) || !e.writeBytes("mem", 3) ||
      !e.writeFixedU8(0x02)) {  // kind: memory
    return;
  }
  if (sharedMem) {
    if (!e.writeFixedU8(0x03) || !e.writeVarU32(1) || !e.writeVarU32(65536)) {
      return;
    }
  } else {
    if (!e.writeFixedU8(0x00) || !e.writeVarU32(0)) return;
  }
  e.finishSection(s);
  if (!e.startSection(SectionId::Function, &s) || !e.writeVarU32(1) ||
      !e.writeVarU32(0)) {
    return;
  }
  e.finishSection(s);
  if (!e.startSection(SectionId::Export, &s) || !e.writeVarU32(1) ||
      !e.writeBytes("f", 1) || !e.writeFixedU8(0x00) || !e.writeVarU32(0)) {
    return;
  }
  e.finishSection(s);
  if (!e.startSection(SectionId::Code, &s) || !e.writeVarU32(1)) return;
  size_t bodyOff;
  if (!e.writePatchableVarU32(&bodyOff)) return;
  size_t bodyStart = e.currentOffset();
  if (!WJIonEmitBody(graph, e, paramList.data(), paramCount)) {
    fprintf(stderr, "[ion-fe] EMIT BODY FAILED\n");
    return;
  }
  e.patchVarU32(bodyOff, uint32_t(e.currentOffset() - bodyStart));
  e.finishSection(s);

  if (getenv("GECKO_WJVS_IONFE_DUMP")) {
    if (FILE* f = fopen("/tmp/ionfe.wasm", "wb")) {
      fwrite(out.begin(), 1, out.length(), f);
      fclose(f);
      fprintf(stderr, "[ion-fe] wrote /tmp/ionfe.wasm (%zu bytes)\n",
              size_t(out.length()));
      fflush(stderr);
    }
  }
  int handle = wasmhost_compile(out.begin(), int(out.length()));
  if (handle < 0) {
    fprintf(stderr, "[ion-fe] HOST COMPILE FAILED (%zu bytes)\n",
            size_t(out.length()));
    return;
  }
  const int importIds[1] = {memId};
  if (wasmhost_instantiate(handle, importIds, 1) != 0) {
    fprintf(stderr, "[ion-fe] INSTANTIATE FAILED\n");
    return;
  }
  // Build call args: object args get a captured live receiver pointer (passed
  // as a double; the i32 param coerces it back); numeric args come from
  // GECKO_WJVS_IONFE_ARGS in order.
  double numArgs[16] = {0};
  uint32_t nNum = 0;
  if (const char* a = getenv("GECKO_WJVS_IONFE_ARGS")) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s", a);
    char* tok = strtok(buf, ",");
    while (tok && nNum < 16) {
      numArgs[nNum++] = atof(tok);
      tok = strtok(nullptr, ",");
    }
  }
  uint32_t sampleRecv = 0;
  if (firstGetPropOff != UINT32_MAX) {
    auto it = gWJSampleRecv.find(WJInlineKey(script, firstGetPropOff));
    if (it != gWJSampleRecv.end()) sampleRecv = it->second;
  }
  double callArgs[16] = {0};
  uint32_t argc = paramCount, ni = 0;
  for (uint32_t i = 0; i < nargs && i < 16; i++) {
    if (objArgMask & (1u << i)) {
      callArgs[i] = double(sampleRecv);
    } else {
      callArgs[i] = ni < nNum ? numArgs[ni] : 0.0;
      ni++;
    }
  }
  // `this` is the trailing param: pass the captured sample receiver.
  if (useThis && nargs < 16) callArgs[nargs] = double(sampleRecv);
  double result = wasmhost_call(handle, 0, callArgs, int(argc));
  fprintf(stderr, "[ion-fe] RESULT = %.17g (args:", result);
  for (uint32_t i = 0; i < argc; i++) fprintf(stderr, " %g", callArgs[i]);
  fprintf(stderr, ")\n");
  fflush(stderr);
}

// Compile `script` through the Ion pipeline with the PRODUCTION scratch ABI
// ((f64 scratchPtr) -> f64 deopt; args/this from gWJScratch as boxed i64; result
// to gWJScratch[kWJResultSlot]), then exercise it by staging a sample receiver and
// calling it exactly like WasmJitRunCall does. Validates the integration ABI
// end-to-end (gated GECKO_WJVS_IONSCRATCH = target lineno).
// Compile `script` through the Ion pipeline with the production scratch ABI and
// return the instantiated wasm handle (or -1 on any bail/failure). `quiet`
// suppresses diagnostics for the hot install path. Shared by the scratch test +
// the production install hook.
static int WJIonCompileInstall(JSScript* script, bool quiet) {
  using namespace js::jit;
  JSFunction* fun = script->function();
  if (!fun) return -1;
  uint32_t nargs = fun->nargs();
  uint32_t nfixed = script->nfixed();
  // Clear the raw-pointer IC records (fns/shapes/holders) before re-reading them
  // from the CURRENT ICs. They PERSIST across compiles and accumulate
  // (gWJMethodPoly never updates an existing shape's fn); a minor/nursery GC moves
  // a recorded JSFunction WITHOUT triggering WJFinalizeCB's clear (that only fires
  // on a major COLLECTION_END), leaving a stale in-range fn pointer that the build
  // derefs (fun->baseScript()->code()) into out-of-bounds memory. Re-reading fresh
  // each compile guarantees every pointer the build trusts is currently live.
  gWJMethodPoly.clear();
  gWJInlineCallee.clear();
  gWJShapeRec.clear();
  gWJElemShape.clear();
  gWJPropByName.clear();
  gWJLenSite.clear();
  gWJFieldPoly.clear();
  { std::unordered_set<JSScript*> wjSeen; WJReadICsRecursive(script, 1, wjSeen); }
  if (!quiet) {
    fprintf(stderr, "[ion-scratch] compiling %s:%u nargs=%u\n",
            script->filename() ? script->filename() : "?",
            uint32_t(script->lineno()), nargs);
    fflush(stderr);
  }

  // Safety cap: a pathologically large inline tree (deltablue/raytrace's deeply
  // recursive constraint/vector graphs expand to thousands of slots as ICs warm)
  // makes the build read far more speculative IC/heap state, where a single
  // residual unsafe guest deref OOB-traps and aborts the engine. Bail such
  // functions to PBL (correct, just not JIT'd) -- richards' schedule is ~417 slots,
  // well under the cap, so the big wins are kept. Tunable via GECKO_WJVS_SLOTCAP.
  uint32_t inlineSlots = WJCountInlineSlots(script, 1);
  uint32_t totalSlots = nargs + nfixed + 1 + inlineSlots;
  uint32_t slotCap =
      getenv("GECKO_WJVS_SLOTCAP") ? atoi(getenv("GECKO_WJVS_SLOTCAP")) : 8192;
  if (totalSlots > slotCap) {
    if (!quiet)
      fprintf(stderr, "[ion-scratch] slot cap: %u > %u -> bail (stay PBL)\n",
              totalSlots, slotCap);
    return -1;
  }
  LifoAlloc lifo(TempAllocator::PreferredLifoChunkSize, js::BackgroundMallocArena);
  TempAllocator alloc(&lifo);
  JitContext jitContext;
  CompileInfo compileInfo(totalSlots);
  MIRGraph graph(&alloc);
  JitCompileOptions options;
  MIRGenerator mirGen(nullptr, options, &alloc, &graph, &compileInfo,
                      IonOptimizations.get(OptimizationLevel::Wasm), nullptr);
  MBasicBlock* entry =
      MBasicBlock::New(graph, compileInfo, nullptr, MBasicBlock::NORMAL);
  if (!entry) return -1;
  graph.addBlock(entry);

  MWasmParameter* scratchPtr =
      MWasmParameter::New(alloc, ABIArg(), MIRType::Double);
  entry->add(scratchPtr);
  MWasmParameter* instance =
      MWasmParameter::New(alloc, ABIArg(InstanceReg), MIRType::Pointer);
  entry->add(instance);
  MTruncateToInt32* scratchBase = MTruncateToInt32::New(alloc, scratchPtr);
  entry->add(scratchBase);

  auto loadScratch = [&](uint32_t off) -> MDefinition* {
    wasm::MemoryAccessDesc acc(/*memoryIndex=*/0, js::Scalar::Int64, 8, off,
                               wasm::TrapSiteDesc(), /*hugeMemory=*/false);
    MWasmLoad* l = MWasmLoad::New(alloc, /*memoryBase=*/nullptr, scratchBase, acc,
                                  MIRType::Int64);
    entry->add(l);
    return l;
  };
  std::vector<MDefinition*> scratchArgs(nargs ? nargs : 1, nullptr);
  for (uint32_t i = 0; i < nargs; i++) scratchArgs[i] = loadScratch(i * 8);
  bool useThis = false;
  for (jsbytecode* p = script->code(); p < script->codeEnd();
       p += GetBytecodeLength(p)) {
    JSOp o = JSOp(*p);
    if (o == JSOp::FunctionThis || o == JSOp::GlobalThis) { useThis = true; break; }
  }
  MDefinition* thisDef = useThis ? loadScratch(kWJThisOff) : nullptr;

  if (!WJIonBuildMIR(script, mirGen, alloc, graph, compileInfo, entry, instance,
                     /*argParams=*/nullptr, nargs, nfixed, thisDef,
                     /*scratchResultBase=*/scratchBase, scratchArgs.data())) {
    if (!quiet) fprintf(stderr, "[ion-scratch] BUILD bailed\n");
    return -1;
  }
  // DIAGNOSTIC (GECKO_WJVS_VALIDATE): replicate the phi-type-consistency assert that
  // NDEBUG drops from MBasicBlock::setBackedgeWasm -- a malformed phi (inputs of
  // mixed MIRType, or an operand that doesn't dominate its use slot) is the class of
  // bug behind the poly-diamond array-iteration corruption. Logs each offender so it
  // can be pinned without a full debug engine build.
  if (getenv("GECKO_WJVS_VALIDATE")) {
    uint32_t bad = 0;
    for (ReversePostorderIterator b = graph.rpoBegin(); b != graph.rpoEnd(); b++) {
      for (MPhiIterator p = b->phisBegin(); p != b->phisEnd(); p++) {
        MIRType pt = p->type();
        for (size_t i = 0; i < p->numOperands(); i++) {
          MDefinition* op = p->getOperand(i);
          if (op->type() != pt) {
            bad++;
            if (bad <= 40)
              fprintf(stderr,
                      "[validate] %s:%u block%u phi(type=%d) operand %zu "
                      "type=%d (id=%u)\n",
                      script->filename() ? script->filename() : "?",
                      uint32_t(script->lineno()), b->id(),
                      int(pt), i, int(op->type()), op->id());
          }
        }
      }
    }
    if (bad)
      fprintf(stderr, "[validate] %s:%u TOTAL %u malformed phi operand(s)\n",
              script->filename() ? script->filename() : "?",
              uint32_t(script->lineno()), bad);
    // Dominance check: an operand's defining block must dominate the use (for a phi,
    // it must dominate the matching predecessor). A violation = a def used outside
    // its dominance region -> reads a stale/uninitialized wasm local at NOOPT =
    // deterministic wrong value. This is the prime remaining suspect for the
    // poly-diamond array miscompile.
    RenumberBlocks(graph);
    if (BuildDominatorTree(&mirGen, graph)) {
      uint32_t dom = 0;
      for (ReversePostorderIterator b = graph.rpoBegin(); b != graph.rpoEnd(); b++) {
        for (MPhiIterator p = b->phisBegin(); p != b->phisEnd(); p++) {
          for (size_t i = 0; i < p->numOperands(); i++) {
            MBasicBlock* pred = b->getPredecessor(i);
            MBasicBlock* db = p->getOperand(i)->block();
            if (db && pred && db != pred && !db->dominates(pred)) {
              dom++;
              if (dom <= 40)
                fprintf(stderr,
                        "[validate-dom] %s:%u PHI block%u op%zu def-block%u does "
                        "NOT dominate pred-block%u\n",
                        script->filename() ? script->filename() : "?",
                        uint32_t(script->lineno()), b->id(), i, db->id(),
                        pred->id());
            }
          }
        }
        for (MInstructionIterator ins = b->begin(); ins != b->end(); ins++) {
          for (size_t i = 0; i < ins->numOperands(); i++) {
            MBasicBlock* db = ins->getOperand(i)->block();
            if (db && db != *b && !db->dominates(*b)) {
              dom++;
              if (dom <= 40)
                fprintf(stderr,
                        "[validate-dom] %s:%u INS op#%d block%u op%zu def-block%u "
                        "does NOT dominate use-block%u\n",
                        script->filename() ? script->filename() : "?",
                        uint32_t(script->lineno()), int(ins->op()), b->id(), i,
                        db->id(), b->id());
            }
          }
        }
      }
      if (dom)
        fprintf(stderr, "[validate-dom] %s:%u TOTAL %u dominance violation(s)\n",
                script->filename() ? script->filename() : "?",
                uint32_t(script->lineno()), dom);
    }
  }
  // OptimizeMIR (GVN/LICM/AliasAnalysis/...) runs on every function. It requires
  // a well-formed SSA graph where loop-header phis merge same-typed slots; that
  // holds because ALL slots are uniform NaN-boxed i64 Values (see the slot init +
  // boxForStore on every SetLocal/SetArg). GECKO_WJVS_NOOPT disables it.
  if (!getenv("GECKO_WJVS_NOOPT") && !OptimizeMIR(&mirGen)) return -1;

  Bytes out;
  Encoder e(out);
  const uint8_t kF64 = uint8_t(TypeCode::F64);
  if (!e.writeFixedU32(MagicNumber) || !e.writeFixedU32(EncodingVersionModule)) {
    return -1;
  }
  size_t s;
  // Type 0 = body (f64)->f64; type 1 = wjhelp (f64,f64)->f64 (for non-inlined
  // calls via WJH_IONCALL). wjhelp is imported as func 0 -> the body is func 1.
  if (!e.startSection(SectionId::Type, &s) || !e.writeVarU32(2) ||
      !e.writeFixedU8(0x60) || !e.writeVarU32(1) || !e.writeFixedU8(kF64) ||
      !e.writeVarU32(1) || !e.writeFixedU8(kF64) ||
      !e.writeFixedU8(0x60) || !e.writeVarU32(2) || !e.writeFixedU8(kF64) ||
      !e.writeFixedU8(kF64) || !e.writeVarU32(1) || !e.writeFixedU8(kF64)) {
    return -1;
  }
  e.finishSection(s);
  int memId = wasmhost_guest_mem_objid();
  bool sharedMem = memId >= 0 && wasmhost_guest_mem_shared() != 0;
  if (memId < 0) return -1;
  if (!e.startSection(SectionId::Import, &s) || !e.writeVarU32(2) ||
      !e.writeBytes("m", 1) || !e.writeBytes("help", 4) || !e.writeFixedU8(0x00) ||
      !e.writeVarU32(1) ||  // wjhelp: func import, type 1
      !e.writeBytes("m", 1) || !e.writeBytes("mem", 3) || !e.writeFixedU8(0x02)) {
    return -1;
  }
  if (sharedMem) {
    if (!e.writeFixedU8(0x03) || !e.writeVarU32(1) || !e.writeVarU32(65536)) return -1;
  } else {
    if (!e.writeFixedU8(0x00) || !e.writeVarU32(0)) return -1;
  }
  e.finishSection(s);
  if (!e.startSection(SectionId::Function, &s) || !e.writeVarU32(1) ||
      !e.writeVarU32(0)) {
    return -1;
  }
  e.finishSection(s);
  if (!e.startSection(SectionId::Export, &s) || !e.writeVarU32(1) ||
      !e.writeBytes("f", 1) || !e.writeFixedU8(0x00) || !e.writeVarU32(1)) {
    return -1;  // export func index 1 (after the imported wjhelp at 0)
  }
  e.finishSection(s);
  if (!e.startSection(SectionId::Code, &s) || !e.writeVarU32(1)) return -1;
  size_t bodyOff;
  if (!e.writePatchableVarU32(&bodyOff)) return -1;
  size_t bodyStart = e.currentOffset();
  MWasmParameter* params1[1] = {scratchPtr};
  if (!WJIonEmitBody(graph, e, params1, 1)) {
    if (!quiet) fprintf(stderr, "[ion-scratch] EMIT BODY FAILED\n");
    return -1;
  }
  e.patchVarU32(bodyOff, uint32_t(e.currentOffset() - bodyStart));
  e.finishSection(s);

  if (getenv("GECKO_WJVS_IONFE_DUMP")) {
    if (FILE* f = fopen("/tmp/ionfe.wasm", "wb")) {
      fwrite(out.begin(), 1, out.length(), f);
      fclose(f);
      fprintf(stderr, "[ion-scratch] wrote /tmp/ionfe.wasm (%zu bytes) for %s:%u\n",
              size_t(out.length()),
              script->filename() ? script->filename() : "?",
              uint32_t(script->lineno()));
      fflush(stderr);
    }
  }
  int handle = wasmhost_compile(out.begin(), int(out.length()));
  if (handle < 0) {
    if (!quiet) fprintf(stderr, "[ion-scratch] HOST COMPILE FAILED (%zu bytes)\n",
                        size_t(out.length()));
    return -1;
  }
  const int importIds[2] = {-3, memId};  // -3 = wjhelp shim (bound by the bridge)
  if (wasmhost_instantiate(handle, importIds, 2) != 0) {
    if (!quiet) fprintf(stderr, "[ion-scratch] INSTANTIATE FAILED\n");
    return -1;
  }
  return handle;
}

static void WJIonScratchTest(JSScript* script) {
  using namespace js::jit;
  JSFunction* fun = script->function();
  if (!fun) return;
  uint32_t nargs = fun->nargs();
  int handle = WJIonCompileInstall(script, /*quiet=*/false);
  if (handle < 0) return;
  // Stage a sample receiver (boxed object Value) + object args, call via the
  // scratch ABI, decode the result Value.
  uint32_t sampleRecv = 0;
  for (jsbytecode* p = script->code(); p < script->codeEnd();
       p += GetBytecodeLength(p)) {
    if (JSOp(*p) == JSOp::GetProp) {
      auto it = gWJSampleRecv.find(WJInlineKey(script, uint32_t(p - script->code())));
      if (it != gWJSampleRecv.end()) { sampleRecv = it->second; break; }
    }
  }
  uint64_t boxedRecv = (uint64_t(kWJTagObject) << 32) | uint64_t(sampleRecv);
  gWJScratch[kWJThisSlot] = boxedRecv;
  for (uint32_t i = 0; i < nargs; i++) gWJScratch[i] = boxedRecv;
  double ptr = double(uintptr_t(static_cast<void*>(gWJScratch)));
  double deopt = wasmhost_call(handle, 0, &ptr, 1);
  JS::Value rv = JS::Value::fromRawBits(gWJScratch[kWJResultSlot]);
  fprintf(stderr, "[ion-scratch] deopt=%g sampleThis=%#x result=", deopt,
          sampleRecv);
  if (rv.isInt32()) {
    fprintf(stderr, "%d\n", rv.toInt32());
  } else if (rv.isDouble()) {
    fprintf(stderr, "%.17g\n", rv.toDouble());
  } else if (rv.isObject()) {
    fprintf(stderr, "object\n");
  } else {
    fprintf(stderr, "bits=%#llx\n", (unsigned long long)gWJScratch[kWJResultSlot]);
  }
  fflush(stderr);
}

static bool WJCompile(JSScript* script, WasmJitEntry& entry) {
  if (getenv("GECKO_WJVS_IONSMOKE")) {
    static bool sRan = false;
    if (!sRan) {
      sRan = true;
      WJIonSmokeTest();
    }
  }
  if (getenv("GECKO_WJVS_IONBE")) {
    static bool sRanBE = false;
    if (!sRanBE) {
      sRanBE = true;
      WJIonBackendTest();
    }
  }
  if (getenv("GECKO_WJVS_IONSMOKE")) {
    static bool sRan = false;
    if (!sRan) {
      sRan = true;
      WJIonSmokeTest();
    }
  }
  if (const char* fe = getenv("GECKO_WJVS_IONFE")) {
    // GECKO_WJVS_IONFE = the TRIGGER fn's lineno (fired once it compiles, after
    // warm-up has filled ICs). GECKO_WJVS_IONFE_TARGET (optional) = the lineno
    // of the fn to actually compile through the Ion path; its JSScript* is
    // captured here when it first compiles. If unset, target == trigger.
    static JSScript* sTargetScript = nullptr;
    if (const char* tgt = getenv("GECKO_WJVS_IONFE_TARGET")) {
      if (uint32_t(atoi(tgt)) == uint32_t(script->lineno())) {
        sTargetScript = script;
      }
    }
    if (uint32_t(atoi(fe)) == uint32_t(script->lineno())) {
      static bool sRanFE = false;
      if (!sRanFE) {
        sRanFE = true;
        if (getenv("GECKO_WJVS_IONSCRATCH")) {
          WJIonScratchTest(sTargetScript ? sTargetScript : script);
        } else {
          WJIonFrontendTest(sTargetScript ? sTargetScript : script);
        }
      }
    }
  }
  if (getenv("GECKO_DEBUG_JIT")) {
    fprintf(stderr, "[wj-enter] WJCompile %s:%u len=%u\n",
            script->filename() ? script->filename() : "?",
            uint32_t(script->lineno()), uint32_t(script->length()));
    fflush(stderr);
  }
  gWJEmitInline = entry.wantInline && WJVSInline();  // inline leaf calls this compile?
  gWJEmitBake = entry.wantBake && WJBake();  // LEAN EMISSION: bake shape/off constants this compile?
  JSFunction* fun = script->function();
  if (!fun) return false;
  uint32_t nargs = fun->nargs();
  uint32_t nfixed = script->nfixed();
  if (nargs > 32 || nfixed > 256) return false;
  static bool sFinalizeReg = false;
  if (!sFinalizeReg) {
    if (JSContext* cx = js::TlsContext.get()) {
      JS_AddFinalizeCallback(cx, WJFinalizeCB, nullptr);
      JS_AddExtraGCRootsTracer(cx, (JSTraceDataOp)WJTraceRoots, nullptr);
      sFinalizeReg = true;
    }
  }
  // ION INTEGRATION (gated): try the reused-Ion pipeline (boxed model, full
  // inlining + GVN/LICM) with the production scratch ABI. On success install it
  // as the entry; WasmJitRunCall runs it, deopting to the interpreter on a guard
  // miss. Functions the Ion FE can't fully build fall through to Mode V/VS.
  // Delay Ion install until SOME field value types have been observed (by the
  // Mode V/VS fallbacks / interpreter ICs), so boxed-i64 fields can instead load
  // as typed i32/f64 -- the big perf lever (the boxed tag-dispatch unbox dominates
  // the hot loop). GECKO_WJVS_IONINT_EAGER installs on the first compile (boxed).
  bool ionHasLoop = false;
  if (getenv("GECKO_WJVS_IONINT")) {
    for (jsbytecode* pc = script->code(); pc < script->codeEnd();
         pc += GetBytecodeLength(pc)) {
      if (JSOp(*pc) == JSOp::LoopHead) { ionHasLoop = true; break; }
    }
  }
  bool ionLog = getenv("GECKO_WJVS_IONINT_LOG") != nullptr;
  if (getenv("GECKO_WJVS_IONINT")) {
    // PBL -> Ion ONLY: no Mode V/VS baseline JIT (it's buggy + a net loss). A
    // function runs in the portable baseline interpreter (which warms the same
    // CacheIR ICs the Ion oracle reads) until Ion can compile it. Only loop-
    // bearing functions are worth installing (cross-module call tax); the rest
    // stay in PBL. Returning false here keeps the fn in PBL; WasmJitObserveCall
    // retries (with backoff) so it tiers up once its ICs are warm.
    // Bisection knob: GECKO_WJVS_IONINT_ONLY=<lineno> installs ONLY that one
    // function (everything else stays PBL) so a hang/miscompile can be pinned.
    const char* onlyLine = getenv("GECKO_WJVS_IONINT_ONLY");
    if (onlyLine && uint32_t(atoi(onlyLine)) != uint32_t(script->lineno())) {
      return false;
    }
    const char* maxLine = getenv("GECKO_WJVS_IONINT_MAXLINE");
    if (maxLine && uint32_t(script->lineno()) > uint32_t(atoi(maxLine))) {
      return false;  // bisection: only compile functions up to this source line
    }
    if (ionHasLoop || getenv("GECKO_WJVS_IONINT_ALL")) {
      int h = WJIonCompileInstall(script, /*quiet=*/!ionLog);
      if (ionLog) {
        fprintf(stderr, "[ion-int] %s:%u loop=%d -> %s\n",
                script->filename() ? script->filename() : "?",
                uint32_t(script->lineno()), ionHasLoop,
                h >= 0 ? "INSTALLED" : "ion-retry(stay PBL)");
        fflush(stderr);
      }
      if (h >= 0) {
        entry.handle = h;
        entry.nargs = nargs;
        entry.bcLen = script->length();  // ABA/validate key -- else WJValidateEntry
                                         // resets to Cold every call -> recompile thrash
        entry.modeVS = false;
        entry.vsCapable = false;
        entry.tableIdx = -1;
        entry.isIon = true;
        entry.state = WasmJitEntry::State::Compiled;
        return true;
      }
    }
    return false;  // not installed -> stay in PBL (no Mode V/VS fallback)
  }
  int memId = wasmhost_guest_mem_objid();
  if (memId < 0) return false;  // can't reach the guest memory object: no JIT
  bool sharedMem = wasmhost_guest_mem_shared() != 0;
  // Classify the body. A heap MUTATION normally bars compilation (deopt-by-restart
  // would double-execute it). Mode VS lifts that bar for the supported op set: it
  // NEVER restarts (misses call wjhelp) and keeps its frame in GC-traced memory,
  // so mutation is safe. Non-mutating functions use the faster Mode N/V (restart
  // is sound when there are no side effects).
  bool useModeV = false, hasCall = false, mutates = false, vsOK = true;
  bool usesAliased = false;  // reads a closed-over var (JSOp::GetAliasedVar)
  bool hasSC = false;        // has &&/||/?? (short-circuit) -> Mode V can't emit; needs Mode VS
  uint32_t nArith = 0, nLoop = 0;  // SKIPDISPATCH heuristic: numeric-work vs pure-dispatch signal
  JSOp firstUnsup = JSOp::Nop;  // first op Mode VS can't handle (diagnostic)
  for (jsbytecode* pc = script->code(); pc < script->codeEnd();
       pc += GetBytecodeLength(pc)) {
    JSOp op = JSOp(*pc);
    switch (op) {
      case JSOp::Add: case JSOp::Sub: case JSOp::Mul: case JSOp::Div: case JSOp::Mod:
      case JSOp::Pow: case JSOp::Lsh: case JSOp::Rsh: case JSOp::Ursh:
      case JSOp::BitOr: case JSOp::BitAnd: case JSOp::BitXor: case JSOp::Inc:
      case JSOp::Dec: case JSOp::Neg: case JSOp::BitNot: nArith++; break;
      case JSOp::LoopHead: nLoop++; break;
      default: break;
    }
    if (op == JSOp::SetProp || op == JSOp::StrictSetProp || op == JSOp::SetElem ||
        op == JSOp::StrictSetElem) {
      mutates = true;  // Mode VS handles these
    } else if (op == JSOp::InitProp && WJVSInlineAlloc()) {
      mutates = true;  // INLINEALLOC: Mode VS handles InitProp via WJVSInitProp
    } else if (op == JSOp::SetGName || op == JSOp::StrictSetGName ||
               op == JSOp::SetName || op == JSOp::StrictSetName ||
               op == JSOp::InitProp || op == JSOp::InitElem ||
               op == JSOp::SetAliasedVar) {
      mutates = true;
      vsOK = false;  // mutation not yet supported by Mode VS (later stages)
    }
    if (op == JSOp::GetProp || op == JSOp::GetElem || op == JSOp::GetGName ||
        op == JSOp::FunctionThis || op == JSOp::Null || op == JSOp::Undefined ||
        op == JSOp::True || op == JSOp::False) {
      // non-number constants/values must use Mode V (Mode N is f64-only and would
      // lose the boolean/null/undefined type).
      useModeV = true;
    } else if (op == JSOp::Call || op == JSOp::CallContent ||
               op == JSOp::CallIgnoresRv) {
      useModeV = true;
      hasCall = true;
    }
    if (op == JSOp::GetAliasedVar) {
      // Resolvable only when the running frame's environmentChain equals
      // fun->environment() (so the bytecode's hop count starts from the env we
      // pass to WasmJitRunCall). That holds iff no per-frame env object is built.
      if (WJNoAliased() || fun->needsSomeEnvironmentObject()) {
        vsOK = false;
        if (firstUnsup == JSOp::Nop) firstUnsup = op;
      } else {
        usesAliased = true;
      }
    }
    if (op == JSOp::And || op == JSOp::Or || op == JSOp::Coalesce) {
      hasSC = true;  // Mode V EMIT-FAILs on these; route to Mode VS when short-circuit on
    }
    if (!WJModeVSSupported(op)) {
      vsOK = false;
      if (firstUnsup == JSOp::Nop) firstUnsup = op;
    }
  }
  entry.vsCapable = vsOK;  // a future recompile may force Mode VS if every op fits
  entry.hasCall = hasCall;
  entry.usesAliased = usesAliased;
  // SKIPDISPATCH (GECKO_WJVS_SKIPDISPATCH=<maxArith>): a call-containing function with little
  // numeric work and no numeric loop gains nothing from Mode VS's only edge (unboxing numerics)
  // and its boxed-object dispatch is SLOWER than the PBL interpreter (richards: hybrid 80 vs PBL
  // 102). Leave such functions in the interpreter. Numeric/loop code (crypto/navier) is unaffected.
  static int skipDispatch = []{ const char* e = getenv("GECKO_WJVS_SKIPDISPATCH"); return e ? atoi(e) : -1; }();
  static int skipAllCall = []{ return getenv("GECKO_WJVS_SKIPALLCALL") ? 1 : 0; }();
  bool skipIt = (skipAllCall && hasCall) ||
                (skipDispatch >= 0 && hasCall && nLoop == 0 && int(nArith) <= skipDispatch);
  if (skipIt) {
    if (getenv("GECKO_DEBUG_JIT")) {
      fprintf(stderr, "[wj-compile] %s:%u SKIP-DISPATCH (nArith=%u nLoop=%u)\n",
              script->filename() ? script->filename() : "?", uint32_t(script->lineno()), nArith, nLoop);
      fflush(stderr);
    }
    return false;  // stay in the interpreter
  }
  bool modeVS = false;
  if (getenv("GECKO_DEBUG_JIT") && (mutates || hasCall)) {
    fprintf(stderr, "[wj-compile] %s:%u mutates=%d vsOK=%d hasCall=%d firstUnsup=%s\n",
            script->filename() ? script->filename() : "?", uint32_t(script->lineno()),
            mutates, vsOK, hasCall,
            firstUnsup == JSOp::Nop ? "-" : js::CodeName(firstUnsup));
    fflush(stderr);
  }
  if (mutates) {
    if (!vsOK) {
      gWJVSBlock[uint8_t(firstUnsup)]++;  // diagnostic: what blocked this mutating fn
      return false;  // a mutating op Mode VS can't handle yet
    }
    modeVS = true;
    useModeV = false;  // Mode VS keeps `hasCall` (call_indirect uses the table)
  } else if (entry.forceVS && vsOK && (!hasCall || WJVSHasCallRecompile())) {
    // Adaptive: a chronically-deopting NON-mutating function recompiles as no-restart
    // Mode VS. Historically restricted to call-free (deep VS->helper->interpreter->VS
    // chains crashed gecko's GC in TraceJitFrames). GECKO_WJVS_HASCALL=1 lifts that
    // restriction (under investigation) so the hot OO call chain can stay in wasm.
    modeVS = true;
    useModeV = false;
    if (hasCall && getenv("GECKO_DEBUG_JIT")) {
      fprintf(stderr, "[wasm-jit]   forceVS-recompile HASCALL fn %s:%u\n",
              script->filename() ? script->filename() : "?",
              uint32_t(script->lineno()));
    }
  } else if (vsOK && hasSC && WJShortCircuit() && (!hasCall || WJVSHasCallRecompile())) {
    // A NON-mutating function with &&/||/?? : Mode V EMIT-FAILs on the short-circuit op so the
    // fn never JITs (e.g. richards TaskControlBlock.isHeldOrSuspended, called every scheduler
    // iteration). Mode VS + short-circuit CAN emit it -> route it there so the whole hot call
    // chain stays in wasm (no per-iteration wasm<->interpreter boundary crossing).
    modeVS = true;
    useModeV = false;
  } else if (vsOK && usesAliased && WJAliasedVS()) {
    // A NON-mutating function that reads closed-over vars (GetAliasedVar): Mode V cannot emit
    // that op (it EMIT-FAILs and the fn never JITs), but Mode VS can. Route it to Mode VS so it
    // compiles at all. (navier-stokes' hot solver loops -- lin_solve/diffuse/advect/project --
    // close over the grid arrays; without this they run in the interpreter.) Mode VS is a strict
    // superset, so this is correctness-neutral. Opt out: GECKO_WJVS_NOALIASEDVS=1.
    modeVS = true;
    useModeV = false;
  }
  // PHASE F: provision a JitScript for a Mode VS body that may bail (deopt-resume runs the
  // rest in PBL, which needs IC entries). Do it HERE at compile -- but via the SAFE path the
  // normal tiering uses: in the script's realm (createJitScript asserts cx->check(script)) and
  // after ensureJitZoneExists (the jit-zone allocator must exist). Creating it without these
  // corrupted the zone IC LifoAlloc.
  if (WJForceDeopt() >= 0 && modeVS && !script->hasJitScript()) {
    JSContext* jcx = js::TlsContext.get();
    if (jcx && jcx->zone()->ensureJitZoneExists(jcx)) {
      js::AutoRealm ar(jcx, script);
      js::jit::AutoKeepJitScripts keep(jcx);
      (void)script->ensureHasJitScript(jcx, keep);
    }
  }
  int tableId = wasmhost_jit_table();  // shared call_indirect table (created once)
  if (hasCall && tableId < 0) return false;  // can't do calls without the table
  Bytes bytes;
  gWJBailOp = JSOp::Nop;  // diagnostic: an emitter sets this on an unsupported op
  gWJVSFailLine = 0;      // diagnostic: structural WJEmitBodyVS bail line
  if (!WJBuildModule(script, nargs, nfixed, sharedMem, useModeV, hasCall, modeVS,
                     bytes)) {
    gWJFailOp[uint8_t(gWJBailOp)]++;  // what blocked this (Mode N/V) function
    if (getenv("GECKO_DEBUG_JIT")) {
      fprintf(stderr, "[wj-compile] %s:%u EMIT-FAIL modeVS=%d bailOp=%s failLine=%u\n",
              script->filename() ? script->filename() : "?", uint32_t(script->lineno()),
              modeVS, gWJBailOp == JSOp::Nop ? "-" : js::CodeName(gWJBailOp), gWJVSFailLine);
      fflush(stderr);
    }
    return false;
  }
  if (getenv("GECKO_WJ_HASH")) {
    // PHASE A parity proof: hash the emitted module so an IR-on vs IR-off run can be
    // diffed for bit-for-bit identity per function (FNV-1a 64).
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes.length(); i++) {
      h ^= bytes[i];
      h *= 1099511628211ULL;
    }
    fprintf(stderr, "[wj-hash] %s:%u modeVS=%d len=%zu h=%016llx\n",
            script->filename() ? script->filename() : "?", uint32_t(script->lineno()),
            modeVS, size_t(bytes.length()), (unsigned long long)h);
    fflush(stderr);
  }
  if (const char* dl = getenv("GECKO_WJ_DUMP")) {  // DEBUG: dump module bytes for one lineno
    if (uint32_t(atoi(dl)) == uint32_t(script->lineno())) {
      char path[128];
      snprintf(path, sizeof(path), "/tmp/wjdump-%u.wasm", uint32_t(script->lineno()));
      if (FILE* f = fopen(path, "wb")) {
        fwrite(bytes.begin(), 1, bytes.length(), f);
        fclose(f);
        fprintf(stderr, "[wj-dump] wrote %s (%zu bytes, modeVS=%d)\n", path,
                size_t(bytes.length()), modeVS);
        fflush(stderr);
      }
    }
  }
  int handle = wasmhost_compile(bytes.begin(), int(bytes.length()));
  if (handle < 0) {
    if (getenv("GECKO_DEBUG_JIT")) {
      fprintf(stderr, "[wj-compile] %s:%u HOST-COMPILE-FAIL modeVS=%d (invalid wasm?)\n",
              script->filename() ? script->filename() : "?", uint32_t(script->lineno()),
              modeVS);
      fflush(stderr);
    }
    return false;
  }
  // Import order matches the module: (Mode VS) [wjhelp,memory,(table if calls)];
  // else [memory] then [table] when hasCall.
  int rc;
  if (modeVS && hasCall) {
    const int importIds[3] = {-3, memId, tableId};
    rc = wasmhost_instantiate(handle, importIds, 3);
  } else if (modeVS) {
    const int importIds[2] = {-3, memId};  // -3 = wjhelp shim (bound by the bridge)
    rc = wasmhost_instantiate(handle, importIds, 2);
  } else if (hasCall) {
    const int importIds[2] = {memId, tableId};
    rc = wasmhost_instantiate(handle, importIds, 2);
  } else {
    const int importIds[1] = {memId};
    rc = wasmhost_instantiate(handle, importIds, 1);
  }
  if (rc != 0) return false;
  entry.handle = handle;
  entry.nargs = nargs;
  entry.bcLen = script->length();
  entry.modeVS = modeVS;
  // Register this function in the shared table as a call_indirect target (a Mode
  // VS callee is allowed; WJFillIC only lets a Mode VS caller cache it). A function
  // that reads closed-over vars (usesAliased) is NOT registered: it must enter via
  // WasmJitRunCall (which sets gWJCurEnv to its environment), never the fast path.
  if (tableId >= 0 && gWJTableCount < kWJTableSize && !entry.usesAliased) {
    int idx = int(gWJTableCount);
    if (wasmhost_jit_table_set(handle, idx) == 0) {
      entry.tableIdx = idx;
      gWJTableCount++;
    }
  }
  entry.state = WasmJitEntry::State::Compiled;
  return true;
}

// Direct-mapped cache so the per-call check on the PBL fast path is a couple of
// loads (no hashmap probe) for already-classified scripts. Stores a pointer to
// the map entry, which is stable across std::unordered_map rehashes.
struct WJCacheSlot {
  JSScript* script = nullptr;
  WasmJitEntry* entry = nullptr;
};
static WJCacheSlot gWJCache[4096];

// If a finalized JSScript*'s address is reused by a different script, the cached
// entry would run the wrong wasm. Detect that ABA by comparing the compiled
// bytecode length to the script's current length; on mismatch, reset the entry
// so the (different) script recompiles. Cheap (one field compare on the path).
static void WJValidateEntry(WasmJitEntry* e, JSScript* script) {
  if (e->state == WasmJitEntry::State::Compiled && e->bcLen != script->length()) {
    e->state = WasmJitEntry::State::Cold;
    e->handle = -1;
    e->tableIdx = -1;
    e->bcLen = 0;
  }
}

static WasmJitEntry* WJEntryFor(JSScript* script) {
  WJCacheSlot& c = gWJCache[(uintptr_t(script) >> 3) & 4095];
  if (c.script == script) {
    WJValidateEntry(c.entry, script);
    return c.entry;
  }
  if (!gWasmJitMap) {
    gWasmJitMap = new (std::nothrow) WasmJitMap();
    if (!gWasmJitMap) return nullptr;
  }
  WasmJitEntry& e = (*gWasmJitMap)[script];  // inserts a Cold entry if absent
  WJValidateEntry(&e, script);
  c.script = script;
  c.entry = &e;
  return &e;
}

}  // namespace

namespace js {
namespace wasm {
extern bool WasmJitObserveCall(JSScript* script);
extern int WasmJitRunCall(JSScript* script, uint64_t thisBits,
                          const JS::Value* args, uint32_t argc,
                          JSObject* envChain, uint64_t* retBits);
}  // namespace wasm
}  // namespace js

// Called on every scripted call from the PBL fast path. Counts invocations and,
// once a script is hot, attempts to lower it to wasm. Returns true once the
// script has a wasm version -- the caller then misses the fast path so the call
// is routed through the IC, where WasmJitRunCall runs the wasm.
bool js::wasm::WasmJitObserveCall(JSScript* script) {
  // A/B toggle for benchmarking: GECKO_NOWASMJIT disables the JIT (checked once).
  static int sEnabled = -1;
  if (sEnabled < 0) sEnabled = getenv("GECKO_NOWASMJIT") ? 0 : 1;
  if (!sEnabled) return false;
  // RE-ENTRANCY GUARD: never compile while we are inside a JIT'd call. A compiled
  // function that calls out via wjhelp(WJH_IONCALL/METHCALL) runs the callee in the
  // interpreter, which can re-enter here and trigger WJIonCompileInstall -- that
  // clears/rebuilds the global gWJ* oracle maps and the shared funcref table WHILE
  // the outer frame's call is in flight (its args live in gWJScratch/gWJHelpA),
  // corrupting shared state -> a later call_indirect hits a clobbered handle ("table
  // index out of bounds"). Defer: stay in PBL now; the periodic retry compiles it
  // later at depth 0. (Disable via GECKO_WJVS_NOREENTRYGUARD for A/B.)
  if (gWJCallDepth > 0 && !getenv("GECKO_WJVS_NOREENTRYGUARD")) {
    WasmJitEntry* ce = WJEntryFor(script);
    return ce && ce->state == WasmJitEntry::State::Compiled;
  }
  // The caller (interpreter call hook) only invokes this for scripts whose warmup
  // counter is already past the threshold, so there is no per-call counting here:
  // attempt the compile on first sight of a hot script.
  WasmJitEntry* e = WJEntryFor(script);
  if (!e) return false;
  if (e->state == WasmJitEntry::State::Compiled) return true;
  if (e->state == WasmJitEntry::State::Failed) return false;
  if (!script->function() || script->isModule() || script->length() > 4096) {
    e->state = WasmJitEntry::State::Failed;
    return false;
  }
  if (getenv("GECKO_WJVS_IONINT")) {
    // PBL -> Ion: stay in the interpreter (warming CacheIR ICs) and retry the Ion
    // compile periodically until it succeeds. The Ion compile is EXPENSIVE
    // (recursive IC read + MIR build + optimize + emit), so cap retries: a fn that
    // keeps failing (e.g. its oracle data is wiped by richards' frequent GC before
    // it can compile) gives up and just runs in PBL -- otherwise the repeated
    // failed compiles dominate and sink the score below the interpreter.
    // Initial warmup delay: run in PBL first so the fn's own CacheIR ICs (which
    // WJReadICsRecursive reads) are warm -> the FIRST Ion attempt resolves shapes
    // and succeeds. A cold first attempt fails, then the backoff outlasts the fn's
    // remaining calls so it never retries warm -- which is what kept schedule off.
    // Initial delay of 2 observes: a hot fn warms its OWN CacheIR ICs within its
    // first call or two (richards' schedule does thousands of internal iterations
    // per call), so 2 is enough for the first Ion attempt to resolve shapes -- and
    // small enough that low-call-count fns (schedule is called only a few times,
    // each doing the whole loop) still reach the attempt before the bench ends.
    if (!e->ionWarmTried) { e->ionWarmTried = true; e->ionBackoff = 2; return false; }
    if (e->ionBackoff > 0) { e->ionBackoff--; return false; }
    if (WJCompile(script, *e)) return true;  // Ion installed -> route to wasm
    if (++e->ionFails >= 6) { e->state = WasmJitEntry::State::Failed; return false; }
    e->ionBackoff = 4u << e->ionFails;  // backoff between retries
    return false;
  }
  if (!WJCompile(script, *e)) {
    e->state = WasmJitEntry::State::Failed;
    return false;
  }
  return true;
}

// When GECKO_DEBUG_JIT is set, emit a heartbeat every ~10s with the deopt/run
// totals + current compiled/failed script counts. Gated on the env flag first so
// the hot path pays only a single predictable branch when debugging is off.
static void WJMaybeLogDeopts() {
  static int sDebug = -1;
  if (sDebug < 0) sDebug = getenv("GECKO_DEBUG_JIT") ? 1 : 0;
  if (!sDebug) return;
  static mozilla::TimeStamp sLast;
  static uint64_t sLastDeopts = 0;
  static uint64_t sTick = 0;
  uint32_t tickMod = 400000;
  if (const char* tm = getenv("GECKO_WJVS_TICKMOD")) tickMod = uint32_t(atoi(tm));
  if ((++sTick % tickMod) != 0) return;  // count-based: fires on short single-bench runs
  mozilla::TimeStamp now = mozilla::TimeStamp::Now();
  if (sLast.IsNull()) sLast = now;
  double elapsed = (now - sLast).ToSeconds();
  uint32_t nCompiled = 0, nFailed = 0;
  if (gWasmJitMap) {
    for (const auto& kv : *gWasmJitMap) {
      if (kv.second.state == WasmJitEntry::State::Compiled) nCompiled++;
      else if (kv.second.state == WasmJitEntry::State::Failed) nFailed++;
    }
  }
  fprintf(stderr,
          "[wasm-jit] +%llu deopts in last %.0fs (total deopts=%llu runs=%llu, "
          "%u compiled / %u failed, megamorphic-misses=%llu)\n",
          (unsigned long long)(gWJTotalDeopts - sLastDeopts), elapsed,
          (unsigned long long)gWJTotalDeopts, (unsigned long long)gWJTotalRuns,
          nCompiled, nFailed, (unsigned long long)gWJMegaMiss);
  fprintf(stderr,
          "[wasm-jit]   forceVS-recompiles=%llu  VS-string-ops=%llu  helper-calls=%llu\n",
          (unsigned long long)gWJForceVS, (unsigned long long)gWJVSStrOps,
          (unsigned long long)gWJHelpCalls);
  fprintf(stderr, "[wasm-jit]   runs ModeV(fast)=%llu  ModeVS(slow)=%llu\n",
          (unsigned long long)gWJRunsV, (unsigned long long)gWJRunsVS);
  fprintf(stderr,
          "[wasm-jit]   call-sites filled=%llu inlinable(leaf)=%llu inlined=%llu "
          "polyways=%llu megamiss=%llu\n",
          (unsigned long long)gWJCallFills, (unsigned long long)gWJCallInlinable,
          (unsigned long long)gWJInlinedCalls, (unsigned long long)gWJCallPolyFills,
          (unsigned long long)gWJCallMegaMiss);
  fprintf(stderr,
          "[wasm-jit]   phase2a typed-field-hits=%llu  phase2b cse-hits=%llu  "
          "phase4 lean-calls=%llu\n",
          (unsigned long long)gWJTypedFieldHits, (unsigned long long)gWJCseHits,
          (unsigned long long)gWJLeanCalls);
  fprintf(stderr,
          "[wasm-jit]   phaseA ir-regions=%llu ir-nodes=%llu  phaseB gvn-hits=%llu  "
          "phaseF deopt-resumes=%llu\n",
          (unsigned long long)gWJIRRegions, (unsigned long long)gWJIRNodes,
          (unsigned long long)gWJGvnHits, (unsigned long long)gWJDeoptResumes);
  {
    static const char* kHN[32] = {
        "Add","Sub","Mul","Div","Mod","Neg","Inc","Dec","Lt","Le","Gt","Ge","Eq",
        "Ne","StrictEq","StrictNe","GetProp","SetProp","GetElem","SetElem","GetGName",
        "Call","FunctionThis","BitOr","BitAnd","BitXor","Lsh","Rsh","Ursh","BitNot",
        "ToNumber","GetAliased"};
    for (int rank = 0; rank < 4; rank++) {
      int best = -1; uint64_t bestN = 0;
      for (int k = 0; k < 32; k++) {
        if (gWJHelpKind[k] > bestN) { bestN = gWJHelpKind[k]; best = k; }
      }
      if (best < 0) break;
      fprintf(stderr, "[wasm-jit]   helper-kind %s x%llu\n",
              kHN[best] ? kHN[best] : "?", (unsigned long long)bestN);
      gWJHelpKind[best] = 0;
    }
  }
  fprintf(stderr,
          "[wasm-jit]   getprop-fill-bail: notnative=%llu ownacc=%llu protoacc=%llu "
          "protononnative=%llu notfound=%llu (ownacc-is-length=%llu)\n",
          (unsigned long long)gWJResolveFail[0], (unsigned long long)gWJResolveFail[1],
          (unsigned long long)gWJResolveFail[2], (unsigned long long)gWJResolveFail[3],
          (unsigned long long)gWJResolveFail[4], (unsigned long long)gWJResolveFail[5]);
  // Top ops blocking MUTATING functions from compiling in Mode VS (what to add next).
  for (int rank = 0; rank < 6; rank++) {
    int best = -1;
    uint32_t bestN = 0;
    for (int op = 0; op < 256; op++) {
      if (gWJVSBlock[op] > bestN) { bestN = gWJVSBlock[op]; best = op; }
    }
    if (best < 0) break;
    fprintf(stderr, "[wasm-jit]   VS-blocked-by %s x%u\n",
            js::CodeName(JSOp(best)), bestN);
    gWJVSBlock[best] = 0;  // consume so the next rank finds the next-highest
  }
  // Top ops blocking NON-mutating (Mode N/V) functions from compiling.
  for (int rank = 0; rank < 6; rank++) {
    int best = -1;
    uint32_t bestN = 0;
    for (int op = 0; op < 256; op++) {
      if (gWJFailOp[op] > bestN) { bestN = gWJFailOp[op]; best = op; }
    }
    if (best < 0) break;
    fprintf(stderr, "[wasm-jit]   MV-blocked-by %s x%u\n",
            js::CodeName(JSOp(best)), bestN);
    gWJFailOp[best] = 0;
  }
  // Top deopting site ops (IC misses) + the type-guard deopt count.
  for (int rank = 0; rank < 5; rank++) {
    int best = -1;
    uint32_t bestN = 0;
    for (int op = 0; op < 256; op++) {
      if (gWJDeoptOp[op] > bestN) { bestN = gWJDeoptOp[op]; best = op; }
    }
    if (best < 0) break;
    fprintf(stderr, "[wasm-jit]   deopt-at %s x%u\n", js::CodeName(JSOp(best)),
            bestN);
    gWJDeoptOp[best] = 0;
  }
  fprintf(stderr, "[wasm-jit]   deopt-type(non-number) x%llu\n",
          (unsigned long long)gWJDeoptType);
  gWJDeoptType = 0;
  // Top individual deopting sites: what + where + IC state (to see why a site that
  // fills its IC still deopts -- proto-holder churn, shape instability, etc.).
  for (int rank = 0; rank < 4; rank++) {
    int best = -1;
    uint32_t bestN = 0;
    for (uint32_t st = 0; st < gWJSiteCount; st++) {
      if (gWJSiteDeopt[st] > bestN) { bestN = gWJSiteDeopt[st]; best = int(st); }
    }
    if (best < 0) break;
    WJSite& ws = gWJSites[best];
    JSOp wop = ws.script ? JSOp(*ws.script->offsetToPC(ws.pcOff)) : JSOp::Nop;
    int ways = 0;
    for (uint32_t w = 0; w < kWJICWays; w++) {
      uint32_t sh = w == 0 ? gWJICTable[2 * best]
                           : gWJICTableX[((w - 1) * kWJMaxSites + best) * 2];
      if (sh) ways++;
    }
    fprintf(stderr,
            "[wasm-jit]   site#%d %s x%u poly=%d len=%d proto=%d waysfilled=%d %s@%u\n",
            best, js::CodeName(wop), bestN, gWJSitePoly[best] ? 1 : 0,
            gWJSiteLen[best] ? 1 : 0, gWJProtoHolder[best] ? 1 : 0, ways,
            ws.script ? ws.script->filename() : "?", ws.pcOff);
    gWJSiteDeopt[best] = 0;
  }
  sLast = now;
  sLastDeopts = gWJTotalDeopts;
}

// Called from a scripted-call site. If `script` has a wasm version and all formal
// args are numbers, runs the wasm and writes the result's raw bits to `retBits`,
// returning true. Otherwise returns false (the interpreter handles it). `argv[i]`
// is formal arg i (callers pass the arg0 pointer, NOT a `this`-prefixed array).
// Returns 0 = not run (caller runs the interpreter), 1 = ran (result in retBits),
// 2 = ran but threw (a Mode VS helper raised; the exception is pending on cx and
// the caller must propagate it WITHOUT re-running -- the wasm may have mutated).
namespace js {
namespace pbl {
extern bool WasmJitResumeViaPBL(JSContext* cx, JSScript* script, uint64_t thisBits,
                                const JS::Value* args, uint32_t argc, JSObject* envChain,
                                const uint64_t* osrLocals, uint32_t nLocals,
                                uint32_t pcOff, uint64_t* retBits);
}  // namespace pbl
}  // namespace js

int js::wasm::WasmJitRunCall(JSScript* script, uint64_t thisBits,
                             const JS::Value* argv, uint32_t argc,
                             JSObject* envChain, uint64_t* retBits) {
  WJMaybeLogDeopts();
  WasmJitEntry* e = WJEntryFor(script);
  if (!e || e->state != WasmJitEntry::State::Compiled) return 0;
  if (argc < e->nargs) return 0;  // underflow -> let the interpreter pad
  // Stage formal args + the receiver (`this`) as raw Value bits in the guest-
  // heap scratch buffer; the wasm reads + type-guards them. Mismatches deopt.
  for (uint32_t i = 0; i < e->nargs; i++) {
    gWJScratch[i] = argv[i].asRawBits();
  }
  gWJScratch[kWJThisSlot] = thisBits;
  gWJMissSite = kWJNoMiss;  // cleared so a stale miss isn't re-read
  double ptr = double(uintptr_t(static_cast<void*>(gWJScratch)));
  if (gWJCallDepth >= kWJMaxCallDepth) return 0;  // too deep -> interpreter
  gWJCallDepth++;
  WasmJitEntry* savedCur = gWJCurEntry;
  // Set the env for GetAliasedVar reads; restore on exit so a nested entry can't
  // leave a stale env for this frame. Root the saved value across the wasm call:
  // a moving GC inside it would otherwise dangle the raw pointer.
  JS::Rooted<JSObject*> savedEnv(js::TlsContext.get(), gWJCurEnv);
  gWJCurEnv = envChain;
  if (e->modeVS) gWJCurEntry = e;  // attribute helper calls to this fn
  double deopt = wasmhost_call(e->handle, 0, &ptr, 1);
  gWJCurEnv = savedEnv;
  gWJCurEntry = savedCur;
  gWJCallDepth--;
  if (deopt == 2.0) {
    // Mode VS helper threw: exception pending on cx; the wasm already ran (and may
    // have mutated). Do NOT restart -- propagate. (Not counted as a deopt.)
    return 2;
  }
  // PHASE F: deopt-resume now self-resumes inside the wasm via wjhelp(WJH_RESUME) and returns
  // 0.0 with the result in gWJScratch[kWJResultSlot] -- handled by the normal success path
  // below, transparently to every caller (this is what makes cross-frame/non-leaf bailout
  // work: a call_indirect caller sees an ordinary return). No deopt code 3 reaches here.
  if (deopt != 0.0) {
    if (e->isIon && getenv("GECKO_WJVS_IONINT_LOG")) {
      fprintf(stderr, "[ion-deopt] %s:%u deopt=%g missSite=%d\n",
              script->filename() ? script->filename() : "?",
              uint32_t(script->lineno()), deopt, int(gWJMissSite));
    }
    // On a GetProp shape-guard miss the wasm recorded the site + object; fill
    // the IC so the site loads inline next time. Then let the interpreter run.
    if (gWJMissSite != kWJNoMiss) {
      WJSite& ms = gWJSites[gWJMissSite];
      if (ms.script) {  // diagnostic: which op's site keeps missing
        gWJDeoptOp[uint8_t(JSOp(*ms.script->offsetToPC(ms.pcOff)))]++;
        gWJSiteDeopt[gWJMissSite]++;
      }
      WJFillIC(gWJMissSite);
    } else {
      gWJDeoptType++;  // type-guard deopt (non-number arith / arg)
    }
    gWJTotalDeopts++;
    ++e->deopts;
    // Adaptive: a non-Mode-VS function that keeps deopting (a polymorphic site)
    // recompiles as no-restart Mode VS, where the miss calls the helper instead
    // of deopting + restarting (and it is GC-traced). Total-deopt threshold 24 is
    // the tuned point: aggressive enough to cut the dominant polymorphic deopts,
    // but conservative enough to avoid a latent miscompile in recompiled call-
    // heavy functions (consecutive-deopt or lower thresholds crash deltablue).
    if (++e->consecDeopts >= 6 && e->vsCapable && !e->modeVS) {
      e->forceVS = true;
      gWJForceVS++;
      e->state = WasmJitEntry::State::Cold;  // next observe recompiles as Mode VS
      e->handle = -1;
      e->tableIdx = -1;
      e->deopts = 0;
      e->runs = 0;
      return 0;
    }
    // Deopt-guard: a chronically-deopting function the JIT can't accelerate (and
    // can't move to Mode VS) is disabled so the JIT is never a net loss.
    if (e->deopts >= 64 && e->runs < e->deopts / 4) {
      e->state = WasmJitEntry::State::Failed;
    }
    return 0;
  }
  e->runs++;
  e->consecDeopts = 0;  // a success breaks a deopt streak (not polymorphic)
  gWJTotalRuns++;
  if (e->modeVS) gWJRunsVS++; else gWJRunsV++;
  // IONINT: a non-Ion fn (it fell back because its ICs were cold at first compile)
  // recompiles ONCE after its ICs have warmed, to retry the Ion pipeline (which can
  // now resolve shapes/fields/dispatch). This is the tier-up that lets the HOT
  // functions (schedule, task.run) actually reach Ion instead of the fallback.
  if (getenv("GECKO_WJVS_IONINT") && !e->isIon && !e->ionWarmTried &&
      e->runs == 64) {
    e->ionWarmTried = true;
    e->state = WasmJitEntry::State::Cold;
    e->handle = -1;
    e->tableIdx = -1;
    e->runs = 0;
    e->deopts = 0;
    *retBits = gWJScratch[kWJResultSlot];  // this call already ran; return its result
    return 1;
  }
  // METHOD_INLINING trigger: once a Mode VS function with calls has run enough for its
  // call ICs to warm (callees recorded in gWJInlineCallee), recompile it ONCE with leaf
  // inlining enabled. One-shot (triggeredInline) to avoid thrash.
  // Recompile once (warm ICs) for method inlining (gated) AND/OR Math-intrinsic emission
  // (gWJMathRec is populated by the first runs' observation and only readable at a recompile).
  if (e->modeVS && !e->triggeredInline && e->runs == 64 &&
      ((e->hasCall && (WJVSInline() || e->hasMathCall)) || WJBake())) {
    e->triggeredInline = true;
    e->wantInline = WJVSInline();  // only enable inlining when its gate is on
    e->wantBake = WJBake();        // LEAN EMISSION: bake shape/off constants (ICs now warm)
    e->state = WasmJitEntry::State::Cold;
    e->handle = -1;
    e->tableIdx = -1;
    e->runs = 0;          // judge the inlined version fresh (helper-dominated guard
    e->helperCalls = 0;   // uses cumulative helperCalls/runs; inlining cuts helpers)
    e->deopts = 0;
    *retBits = gWJScratch[kWJResultSlot];  // this call already ran; return its result
    return 1;
  }
  // A Mode VS function that is HELPER-DOMINATED (>4 wjhelp calls per run on average)
  // spends more on the wasm->C++ boundary than the interpreter would on the same ops,
  // so the JIT is a net loss for it (e.g. polymorphic dispatch / GetGName-heavy code).
  // Disable it -> it runs in the faster PBL interpreter. Checked every 128 runs.
  if (e->modeVS && (e->runs & 127) == 0 &&
      e->helperCalls > uint64_t(e->runs) * 1) {
    e->state = WasmJitEntry::State::Failed;
  }
  *retBits = gWJScratch[kWJResultSlot];
  return 1;
}

static int HostCompileBytes(JSContext* cx, HandleObject bytesObj);
static JSObject* HostMakeModuleObject(JSContext* cx, int handle);
static int HostModuleHandle(JSContext* cx, HandleObject obj);
static bool HostBindAndInstantiate(JSContext* cx, int handle,
                                   HandleObject importObj);
static JSObject* HostBuildInstanceObject(JSContext* cx, int handle);
static int HostObjId(JSContext* cx, HandleObject obj);
static JSObject* HostMakeMemoryWrapper(JSContext* cx, int objId, bool shared);
static JSObject* HostMakeObjIdWrapper(JSContext* cx, int objId);
#endif

/* static */
bool WasmModuleObject::construct(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs callArgs = CallArgsFromVp(argc, vp);

  Log(cx, "sync new Module() started");

  if (!ThrowIfNotConstructing(cx, callArgs, "Module")) {
    return false;
  }

#if defined(__EMSCRIPTEN__)
  if (wasm::UseHostPassthrough()) {
    if (!callArgs.requireAtLeast(cx, "WebAssembly.Module", 1)) {
      return false;
    }
    if (!callArgs.get(0).isObject()) {
      JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                               JSMSG_WASM_BAD_BUF_MOD_ARG);
      return false;
    }
    RootedObject bytesObj(cx, &callArgs.get(0).toObject());
    int handle = HostCompileBytes(cx, bytesObj);
    RootedObject moduleObj(cx);
    if (handle >= 0) {
      moduleObj = HostMakeModuleObject(cx, handle);
    }
    if (!moduleObj) {
      return false;
    }
    callArgs.rval().setObject(*moduleObj);
    return true;
  }
#endif

  JS::RootedVector<JSString*> parameterStrings(cx);
  JS::RootedVector<Value> parameterArgs(cx);
  bool canCompileStrings = false;
  if (!cx->isRuntimeCodeGenEnabled(JS::RuntimeCode::WASM, nullptr,
                                   JS::CompilationType::Undefined,
                                   parameterStrings, nullptr, parameterArgs,
                                   NullHandleValue, &canCompileStrings)) {
    return false;
  }
  if (!canCompileStrings) {
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_CSP_BLOCKED_WASM, "WebAssembly.Module");
    return false;
  }

  if (!callArgs.requireAtLeast(cx, "WebAssembly.Module", 1)) {
    return false;
  }

  if (!callArgs[0].isObject()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_BUF_ARG);
    return false;
  }

  FeatureOptions options;
  if (!options.init(cx, callArgs.get(1))) {
    return false;
  }

  SharedCompileArgs compileArgs =
      InitCompileArgs(cx, options, "WebAssembly.Module");
  if (!compileArgs) {
    return false;
  }

  UniqueChars error;
  UniqueCharsVector warnings;
  SharedModule module;
  {
    // Limit the lifetime of the bytecode to just compilation and ensure we pin
    // the buffer. No user code should be running here anyways, so this is very
    // conservative.
    BytecodeBufferOrSource bytecode;
    Rooted<JSObject*> sourceObj(cx, &callArgs[0].toObject());
    if (!GetBytecodeBufferOrSource(cx, sourceObj, JSMSG_WASM_BAD_BUF_ARG,
                                   &bytecode)) {
      return false;
    }
    AutoPinBufferSourceLength pin(cx, sourceObj.get());

    module = CompileModule(*compileArgs, bytecode, &error, &warnings, nullptr);
  }

  if (!ReportCompileWarnings(cx, warnings)) {
    return false;
  }
  if (!module) {
    if (error) {
      JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                               JSMSG_WASM_COMPILE_ERROR, error.get());
      return false;
    }
    return ThrowCompileOutOfMemory(cx);
  }

  RootedObject proto(
      cx, GetWasmConstructorPrototype(cx, callArgs, JSProto_WasmModule));
  if (!proto) {
    ReportOutOfMemory(cx);
    return false;
  }

  RootedObject moduleObj(cx, WasmModuleObject::create(cx, *module, proto));
  if (!moduleObj) {
    return false;
  }

  Log(cx, "sync new Module() succeded");

  callArgs.rval().setObject(*moduleObj);
  return true;
}

const Module& WasmModuleObject::module() const {
  MOZ_ASSERT(is<WasmModuleObject>());
  return *(const Module*)getReservedSlot(MODULE_SLOT).toPrivate();
}

// ============================================================================
// WebAssembly.Component class and methods

#ifdef ENABLE_WASM_COMPONENTS

const JSClassOps WasmComponentObject::classOps_ = {
    nullptr,                        // addProperty
    nullptr,                        // delProperty
    nullptr,                        // enumerate
    nullptr,                        // newEnumerate
    nullptr,                        // resolve
    nullptr,                        // mayResolve
    WasmComponentObject::finalize,  // finalize
    nullptr,                        // call
    nullptr,                        // construct
    nullptr,                        // trace
};

const JSClass WasmComponentObject::class_ = {
    "WebAssembly.Component",
    JSCLASS_DELAY_METADATA_BUILDER |
        JSCLASS_HAS_RESERVED_SLOTS(WasmComponentObject::RESERVED_SLOTS) |
        JSCLASS_FOREGROUND_FINALIZE,
    &WasmComponentObject::classOps_,
    &WasmComponentObject::classSpec_,
};

const JSClass& WasmComponentObject::protoClass_ = PlainObject::class_;

static constexpr char WasmComponentName[] = "Component";

const ClassSpec WasmComponentObject::classSpec_ = {
    CreateWasmConstructor<WasmComponentObject, WasmComponentName>,
    GenericCreatePrototype<WasmComponentObject>,
    WasmComponentObject::static_methods,
    nullptr,
    WasmComponentObject::methods,
    WasmComponentObject::properties,
    nullptr,
    ClassSpec::DontDefineConstructor,
};

const JSPropertySpec WasmComponentObject::properties[] = {
    JS_STRING_SYM_PS(toStringTag, "WebAssembly.Component", JSPROP_READONLY),
    JS_PS_END,
};

const JSFunctionSpec WasmComponentObject::methods[] = {
    JS_FS_END,
};

const JSFunctionSpec WasmComponentObject::static_methods[] = {
    JS_FS_END,
};

/* static */
WasmComponentObject* WasmComponentObject::create(JSContext* cx,
                                                 const Component& component,
                                                 HandleObject proto) {
  AutoSetNewObjectMetadata metadata(cx);
  auto* obj = NewObjectWithGivenProto<WasmComponentObject>(cx, proto);
  if (!obj) {
    return nullptr;
  }

  // See comment in WasmModuleObject::create. Because we also compile code when
  // creating components, we perform a flush here as well.
  jit::FlushExecutionContext();

  InitReservedSlot(obj, COMPONENT_SLOT, const_cast<Component*>(&component),
                   component.gcMallocBytesExcludingCode(),
                   MemoryUse::WasmComponent);
  component.AddRef();

  // TODO(wasm-cm): Not only is the amount being passed to incJitMemory probably
  // wrong here (per the comment on tier1CodeMemoryUsed), but we may also need
  // to separately account for any code memory allocated later, as bug 1569888
  // alludes to.
  size_t codeMemory = component.tier1CodeMemoryUsed();
  if (codeMemory) {
    cx->zone()->incJitMemory(codeMemory);
  }

  return obj;
}

/* static */
void WasmComponentObject::finalize(JS::GCContext* gcx, JSObject* obj) {
  const Component& component = obj->as<WasmComponentObject>().component();
  size_t codeMemory = component.tier1CodeMemoryUsed();
  if (codeMemory) {
    obj->zone()->decJitMemory(codeMemory);
  }
  gcx->release(obj, &component, component.gcMallocBytesExcludingCode(),
               MemoryUse::WasmComponent);
}

/* static */
bool WasmComponentObject::construct(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs callArgs = CallArgsFromVp(argc, vp);

  Log(cx, "sync new Component() started");

  if (!ThrowIfNotConstructing(cx, callArgs, "Component")) {
    return false;
  }

  JS::RootedVector<JSString*> parameterStrings(cx);
  JS::RootedVector<Value> parameterArgs(cx);
  bool canCompileStrings = false;
  if (!cx->isRuntimeCodeGenEnabled(JS::RuntimeCode::WASM, nullptr,
                                   JS::CompilationType::Undefined,
                                   parameterStrings, nullptr, parameterArgs,
                                   NullHandleValue, &canCompileStrings)) {
    return false;
  }
  if (!canCompileStrings) {
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_CSP_BLOCKED_WASM, "WebAssembly.Component");
    return false;
  }

  if (!callArgs.requireAtLeast(cx, "WebAssembly.Component", 1)) {
    return false;
  }

  if (!callArgs[0].isObject()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_BUF_ARG);
    return false;
  }

  FeatureOptions options;
  if (!options.init(cx, callArgs.get(1))) {
    return false;
  }

  SharedCompileArgs compileArgs =
      InitCompileArgs(cx, options, "WebAssembly.Component");
  if (!compileArgs) {
    return false;
  }

  BytecodeSource source;
  Rooted<JSObject*> sourceObj(cx, &callArgs[0].toObject());
  bool isShared;
  if (!GetBytecodeSource(cx, sourceObj, JSMSG_WASM_BAD_BUF_ARG, &source,
                         &isShared)) {
    return false;
  }

  UniqueChars error;
  UniqueCharsVector warnings;
  SharedComponent component;
  {
    AutoPinBufferSourceLength pin(cx, sourceObj.get());
    component = CompileComponent(*compileArgs, BytecodeBufferOrSource(source),
                                 &error, &warnings, nullptr);
  }

  if (!ReportCompileWarnings(cx, warnings)) {
    return false;
  }
  if (!component) {
    if (error) {
      JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                               JSMSG_WASM_COMPILE_ERROR, error.get());
      return false;
    }
    return ThrowCompileOutOfMemory(cx);
  }

  RootedObject proto(
      cx, GetWasmConstructorPrototype(cx, callArgs, JSProto_WasmComponent));
  if (!proto) {
    ReportOutOfMemory(cx);
    return false;
  }

  RootedObject componentObj(cx,
                            WasmComponentObject::create(cx, *component, proto));
  if (!componentObj) {
    return false;
  }

  Log(cx, "sync new Component() succeeded");

  callArgs.rval().setObject(*componentObj);
  return true;
}

const Component& WasmComponentObject::component() const {
  MOZ_ASSERT(is<WasmComponentObject>());
  return *(const Component*)getReservedSlot(COMPONENT_SLOT).toPrivate();
}

#endif  // ENABLE_WASM_COMPONENTS

// ============================================================================
// WebAssembly.Instance class and methods

const JSClassOps WasmInstanceObject::classOps_ = {
    .finalize = WasmInstanceObject::finalize,
    .trace = WasmInstanceObject::trace,
};

const JSClass WasmInstanceObject::class_ = {
    "WebAssembly.Instance",
    JSCLASS_DELAY_METADATA_BUILDER |
        JSCLASS_HAS_RESERVED_SLOTS(WasmInstanceObject::RESERVED_SLOTS) |
        JSCLASS_FOREGROUND_FINALIZE,
    &WasmInstanceObject::classOps_,
    &WasmInstanceObject::classSpec_,
};

const JSClass& WasmInstanceObject::protoClass_ = PlainObject::class_;

static constexpr char WasmInstanceName[] = "Instance";

const ClassSpec WasmInstanceObject::classSpec_ = {
    CreateWasmConstructor<WasmInstanceObject, WasmInstanceName>,
    GenericCreatePrototype<WasmInstanceObject>,
    WasmInstanceObject::static_methods,
    nullptr,
    WasmInstanceObject::methods,
    WasmInstanceObject::properties,
    nullptr,
    ClassSpec::DontDefineConstructor,
};

static bool IsInstance(HandleValue v) {
  return v.isObject() && v.toObject().is<WasmInstanceObject>();
}

/* static */
bool WasmInstanceObject::exportsGetterImpl(JSContext* cx,
                                           const CallArgs& args) {
  args.rval().setObject(
      args.thisv().toObject().as<WasmInstanceObject>().exportsObj());
  return true;
}

/* static */
bool WasmInstanceObject::exportsGetter(JSContext* cx, unsigned argc,
                                       Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsInstance, exportsGetterImpl>(cx, args);
}

const JSPropertySpec WasmInstanceObject::properties[] = {
    JS_PSG("exports", WasmInstanceObject::exportsGetter, JSPROP_ENUMERATE),
    JS_STRING_SYM_PS(toStringTag, "WebAssembly.Instance", JSPROP_READONLY),
    JS_PS_END,
};

const JSFunctionSpec WasmInstanceObject::methods[] = {
    JS_FS_END,
};

const JSFunctionSpec WasmInstanceObject::static_methods[] = {
    JS_FS_END,
};

bool WasmInstanceObject::isNewborn() const {
  MOZ_ASSERT(is<WasmInstanceObject>());
  return getReservedSlot(INSTANCE_SLOT).isUndefined();
}

// WeakScopeMap maps from function index to js::Scope. This maps is weak
// to avoid holding scope objects alive. The scopes are normally created
// during debugging.
//
// This is defined here in order to avoid recursive dependency between
// WasmJS.h and Scope.h.
using WasmFunctionScopeMap =
    JS::WeakCache<GCHashMap<uint32_t, WeakHeapPtr<WasmFunctionScope*>,
                            DefaultHasher<uint32_t>, CellAllocPolicy>>;
class WasmInstanceObject::UnspecifiedScopeMap {
 public:
  WasmFunctionScopeMap& asWasmFunctionScopeMap() {
    return *(WasmFunctionScopeMap*)this;
  }
};

/* static */
void WasmInstanceObject::finalize(JS::GCContext* gcx, JSObject* obj) {
  WasmInstanceObject& instance = obj->as<WasmInstanceObject>();
  gcx->delete_(obj, &instance.scopes().asWasmFunctionScopeMap(),
               MemoryUse::WasmInstanceScopes);
  gcx->delete_(obj, &instance.indirectGlobals(),
               MemoryUse::WasmInstanceGlobals);
  if (!instance.isNewborn()) {
    if (instance.instance().debugEnabled()) {
      instance.instance().debug().finalize(gcx);
    }
    Instance::destroy(&instance.instance());
    gcx->removeCellMemory(obj, sizeof(Instance),
                          MemoryUse::WasmInstanceInstance);
  }
}

/* static */
void WasmInstanceObject::trace(JSTracer* trc, JSObject* obj) {
  WasmInstanceObject& instanceObj = obj->as<WasmInstanceObject>();
  instanceObj.indirectGlobals().trace(trc);
  if (!instanceObj.isNewborn()) {
    instanceObj.instance().tracePrivate(trc);
  }
}

/* static */
WasmInstanceObject* WasmInstanceObject::create(
    JSContext* cx, const SharedCode& code,
    const DataSegmentVector& dataSegments,
    const ModuleElemSegmentVector& elemSegments, uint32_t instanceDataLength,
    Handle<WasmMemoryObjectVector> memories, SharedTableVector&& tables,
    const JSObjectVector& funcImports, const GlobalDescVector& globals,
    const ValVector& globalImportValues,
    const WasmGlobalObjectVector& globalObjs,
    const WasmTagObjectVector& tagObjs, HandleObject proto,
    UniqueDebugState maybeDebug) {
  UniquePtr<WasmFunctionScopeMap> scopes =
      js::MakeUnique<WasmFunctionScopeMap>(cx->zone(), cx->zone());
  if (!scopes) {
    ReportOutOfMemory(cx);
    return nullptr;
  }
  // Note that `scopes` is a WeakCache, auto-linked into a sweep list on the
  // Zone, and so does not require rooting.

  uint32_t indirectGlobals = 0;

  for (uint32_t i = 0; i < globalObjs.length(); i++) {
    if (globalObjs[i] && globals[i].isIndirect()) {
      indirectGlobals++;
    }
  }

  Rooted<UniquePtr<GlobalObjectVector>> indirectGlobalObjs(
      cx, js::MakeUnique<GlobalObjectVector>(cx->zone()));
  if (!indirectGlobalObjs || !indirectGlobalObjs->resize(indirectGlobals)) {
    ReportOutOfMemory(cx);
    return nullptr;
  }

  {
    uint32_t next = 0;
    for (uint32_t i = 0; i < globalObjs.length(); i++) {
      if (globalObjs[i] && globals[i].isIndirect()) {
        (*indirectGlobalObjs)[next++] = globalObjs[i];
      }
    }
  }

  Instance* instance = nullptr;
  Rooted<WasmInstanceObject*> obj(cx);

  {
    // We must delay creating metadata for this object until after all its
    // slots have been initialized. We must also create the metadata before
    // calling Instance::init as that may allocate new objects.
    AutoSetNewObjectMetadata metadata(cx);
    obj = NewObjectWithGivenProto<WasmInstanceObject>(cx, proto);
    if (!obj) {
      return nullptr;
    }

    MOZ_ASSERT(obj->isTenured(), "assumed by WasmTableObject write barriers");

    InitReservedSlot(obj, SCOPES_SLOT, scopes.release(),
                     MemoryUse::WasmInstanceScopes);

    InitReservedSlot(obj, GLOBALS_SLOT, indirectGlobalObjs.release(),
                     MemoryUse::WasmInstanceGlobals);

    obj->initReservedSlot(INSTANCE_SCOPE_SLOT, UndefinedValue());

    // The INSTANCE_SLOT may not be initialized if Instance allocation fails,
    // leading to an observable "newborn" state in tracing/finalization.
    MOZ_ASSERT(obj->isNewborn());

    // Create this just before constructing Instance to avoid rooting hazards.
    instance = Instance::create(cx, obj, code, instanceDataLength,
                                std::move(tables), std::move(maybeDebug));
    if (!instance) {
      return nullptr;
    }

    InitReservedSlot(obj, INSTANCE_SLOT, instance,
                     MemoryUse::WasmInstanceInstance);
    MOZ_ASSERT(!obj->isNewborn());
  }

  if (!instance->init(cx, funcImports, globalImportValues, memories, globalObjs,
                      tagObjs, dataSegments, elemSegments)) {
    return nullptr;
  }

  return obj;
}

void WasmInstanceObject::initExportsObj(JSObject& exportsObj) {
  MOZ_ASSERT(getReservedSlot(EXPORTS_OBJ_SLOT).isUndefined());
  setReservedSlot(EXPORTS_OBJ_SLOT, ObjectValue(exportsObj));
}

static bool GetImportArg(JSContext* cx, HandleValue importArg,
                         MutableHandleObject importObj) {
  if (!importArg.isUndefined()) {
    if (!importArg.isObject()) {
      return ThrowBadImportArg(cx);
    }
    importObj.set(&importArg.toObject());
  }
  return true;
}

/* static */
bool WasmInstanceObject::construct(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);

  Log(cx, "sync new Instance() started");

  if (!ThrowIfNotConstructing(cx, args, "Instance")) {
    return false;
  }

#if defined(__EMSCRIPTEN__)
  if (wasm::UseHostPassthrough()) {
    if (!args.requireAtLeast(cx, "WebAssembly.Instance", 1)) {
      return false;
    }
    if (!args.get(0).isObject()) {
      JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                               JSMSG_WASM_BAD_MOD_ARG);
      return false;
    }
    RootedObject modObj(cx, &args.get(0).toObject());
    int handle = HostModuleHandle(cx, modObj);
    if (handle < 0) {
      JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                               JSMSG_WASM_BAD_MOD_ARG);
      return false;
    }
    RootedObject importObj(cx);
    if (!GetImportArg(cx, args.get(1), &importObj)) {
      return false;
    }
    if (!HostBindAndInstantiate(cx, handle, importObj)) {
      return false;
    }
    RootedObject instanceObj(cx, HostBuildInstanceObject(cx, handle));
    if (!instanceObj) {
      return false;
    }
    args.rval().setObject(*instanceObj);
    return true;
  }
#endif

  if (!args.requireAtLeast(cx, "WebAssembly.Instance", 1)) {
    return false;
  }

  if (!args[0].isObject()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_MOD_ARG);
    return false;
  }

  Rooted<WasmModuleObject*> moduleObj(
      cx, args[0].toObject().maybeUnwrapIf<WasmModuleObject>());
  if (!moduleObj) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_MOD_ARG);
    return false;
  }

  RootedObject importObj(cx);
  if (!GetImportArg(cx, args.get(1), &importObj)) {
    return false;
  }

  RootedObject proto(
      cx, GetWasmConstructorPrototype(cx, args, JSProto_WasmInstance));
  if (!proto) {
    ReportOutOfMemory(cx);
    return false;
  }

  Rooted<ImportValues> imports(cx);
  if (!GetImports(cx, moduleObj->module(), importObj, imports.address())) {
    return false;
  }

  Rooted<WasmInstanceObject*> instanceObj(cx);
  if (!moduleObj->module().instantiate(cx, imports.get(), proto,
                                       &instanceObj)) {
    return false;
  }

  Log(cx, "sync new Instance() succeeded");

  args.rval().setObject(*instanceObj);
  return true;
}

Instance& WasmInstanceObject::instance() const {
  MOZ_ASSERT(!isNewborn());
  return *(Instance*)getReservedSlot(INSTANCE_SLOT).toPrivate();
}

JSObject& WasmInstanceObject::exportsObj() const {
  return getReservedSlot(EXPORTS_OBJ_SLOT).toObject();
}

WasmFunctionScope* WasmInstanceObject::getExistingFunctionScope(
    uint32_t funcIndex) const {
  if (auto p = scopes().asWasmFunctionScopeMap().lookup(funcIndex)) {
    return p->value();
  }

  return nullptr;
}

WasmInstanceObject::UnspecifiedScopeMap& WasmInstanceObject::scopes() const {
  return *(UnspecifiedScopeMap*)(getReservedSlot(SCOPES_SLOT).toPrivate());
}

WasmInstanceObject::GlobalObjectVector& WasmInstanceObject::indirectGlobals()
    const {
  return *(GlobalObjectVector*)getReservedSlot(GLOBALS_SLOT).toPrivate();
}

/* static */
bool WasmInstanceObject::getExportedFunction(
    JSContext* cx, Handle<WasmInstanceObject*> instanceObj, uint32_t funcIndex,
    MutableHandleFunction fun) {
  Instance& instance = instanceObj->instance();
  return instance.getExportedFunction(cx, funcIndex, fun);
}

/* static */
WasmInstanceScope* WasmInstanceObject::getScope(
    JSContext* cx, Handle<WasmInstanceObject*> instanceObj) {
  if (!instanceObj->getReservedSlot(INSTANCE_SCOPE_SLOT).isUndefined()) {
    return (WasmInstanceScope*)instanceObj->getReservedSlot(INSTANCE_SCOPE_SLOT)
        .toGCThing();
  }

  Rooted<WasmInstanceScope*> instanceScope(
      cx, WasmInstanceScope::create(cx, instanceObj));
  if (!instanceScope) {
    return nullptr;
  }

  instanceObj->setReservedSlot(INSTANCE_SCOPE_SLOT,
                               PrivateGCThingValue(instanceScope));

  return instanceScope;
}

/* static */
WasmFunctionScope* WasmInstanceObject::getFunctionScope(
    JSContext* cx, Handle<WasmInstanceObject*> instanceObj,
    uint32_t funcIndex) {
  if (auto p =
          instanceObj->scopes().asWasmFunctionScopeMap().lookup(funcIndex)) {
    return p->value();
  }

  Rooted<WasmInstanceScope*> instanceScope(
      cx, WasmInstanceObject::getScope(cx, instanceObj));
  if (!instanceScope) {
    return nullptr;
  }

  Rooted<WasmFunctionScope*> funcScope(
      cx, WasmFunctionScope::create(cx, instanceScope, funcIndex));
  if (!funcScope) {
    return nullptr;
  }

  if (!instanceObj->scopes().asWasmFunctionScopeMap().putNew(funcIndex,
                                                             funcScope)) {
    ReportOutOfMemory(cx);
    return nullptr;
  }

  return funcScope;
}

// ============================================================================
// WebAssembly.Memory class and methods

const JSClassOps WasmMemoryObject::classOps_ = {
    .finalize = WasmMemoryObject::finalize,
};

const JSClass WasmMemoryObject::class_ = {
    "WebAssembly.Memory",
    JSCLASS_DELAY_METADATA_BUILDER |
        JSCLASS_HAS_RESERVED_SLOTS(WasmMemoryObject::RESERVED_SLOTS) |
        JSCLASS_FOREGROUND_FINALIZE,
    &WasmMemoryObject::classOps_,
    &WasmMemoryObject::classSpec_,
};

const JSClass& WasmMemoryObject::protoClass_ = PlainObject::class_;

static constexpr char WasmMemoryName[] = "Memory";

static JSObject* CreateWasmMemoryPrototype(JSContext* cx, JSProtoKey key) {
  RootedObject proto(cx, GlobalObject::createBlankPrototype(
                             cx, cx->global(), &WasmMemoryObject::protoClass_));
  if (!proto) {
    return nullptr;
  }
  if (MemoryControlAvailable(cx)) {
    if (!JS_DefineFunctions(cx, proto,
                            WasmMemoryObject::memoryControlMethods)) {
      return nullptr;
    }
  }
  return proto;
}

const ClassSpec WasmMemoryObject::classSpec_ = {
    CreateWasmConstructor<WasmMemoryObject, WasmMemoryName>,
    CreateWasmMemoryPrototype,
    WasmMemoryObject::static_methods,
    nullptr,
    WasmMemoryObject::methods,
    WasmMemoryObject::properties,
    nullptr,
    ClassSpec::DontDefineConstructor,
};

/* static */
void WasmMemoryObject::finalize(JS::GCContext* gcx, JSObject* obj) {
  WasmMemoryObject& memory = obj->as<WasmMemoryObject>();
  if (memory.hasObservers()) {
    gcx->delete_(obj, &memory.observers(), MemoryUse::WasmMemoryObservers);
  }
}

/* static */
WasmMemoryObject* WasmMemoryObject::create(
    JSContext* cx, Handle<ArrayBufferObjectMaybeShared*> buffer, bool isHuge,
    HandleObject proto) {
  AutoSetNewObjectMetadata metadata(cx);
  auto* obj = NewObjectWithGivenProto<WasmMemoryObject>(cx, proto);
  if (!obj) {
    return nullptr;
  }

  obj->initReservedSlot(BUFFER_SLOT, ObjectValue(*buffer));
  obj->initReservedSlot(ISHUGE_SLOT, BooleanValue(isHuge));
  MOZ_ASSERT(!obj->hasObservers());

  return obj;
}

/* static */
bool WasmMemoryObject::construct(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);

  if (!ThrowIfNotConstructing(cx, args, "Memory")) {
    return false;
  }

#if defined(__EMSCRIPTEN__)
  if (wasm::UseHostPassthrough()) {
    if (!args.requireAtLeast(cx, "WebAssembly.Memory", 1) ||
        !args.get(0).isObject()) {
      JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                               JSMSG_WASM_BAD_DESC_ARG, "memory");
      return false;
    }
    RootedObject desc(cx, &args[0].toObject());
    RootedValue v(cx);
    int32_t initial = 0;
    int32_t maximum = -1;
    if (!JS_GetProperty(cx, desc, "initial", &v) || !ToInt32(cx, v, &initial)) {
      return false;
    }
    if (!JS_GetProperty(cx, desc, "maximum", &v)) {
      return false;
    }
    if (!v.isUndefined() && !ToInt32(cx, v, &maximum)) {
      return false;
    }
    if (!JS_GetProperty(cx, desc, "shared", &v)) {
      return false;
    }
    bool shared = ToBoolean(v);
    int objId = wasmhost_mem_new(initial, maximum, shared ? 1 : 0);
    RootedObject memWrap(cx);
    if (objId >= 0) {
      memWrap = HostMakeMemoryWrapper(cx, objId, shared);
    }
    if (!memWrap) {
      JS_ReportErrorASCII(
          cx, "WebAssembly host passthrough: memory creation failed");
      return false;
    }
    args.rval().setObject(*memWrap);
    return true;
  }
#endif

  if (!args.requireAtLeast(cx, "WebAssembly.Memory", 1)) {
    return false;
  }

  if (!args.get(0).isObject()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_DESC_ARG, "memory");
    return false;
  }

  RootedObject obj(cx, &args[0].toObject());
  Limits limits;
  if (!GetLimits(cx, obj, LimitsKind::Memory, &limits) ||
      !CheckLimits(
          cx, MaxMemoryPagesValidation(limits.addressType, limits.pageSize),
          LimitsKind::Memory, &limits)) {
    return false;
  }

  if (Pages::fromPageCount(limits.initial, limits.pageSize) >
      MaxMemoryPages(limits.addressType, limits.pageSize)) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_MEM_IMP_LIMIT);
    return false;
  }
  MemoryDesc memory(limits);

  Rooted<ArrayBufferObjectMaybeShared*> buffer(cx,
                                               CreateWasmBuffer(cx, memory));
  if (!buffer) {
    return false;
  }

  RootedObject proto(cx,
                     GetWasmConstructorPrototype(cx, args, JSProto_WasmMemory));
  if (!proto) {
    ReportOutOfMemory(cx);
    return false;
  }

  Rooted<WasmMemoryObject*> memoryObj(
      cx, WasmMemoryObject::create(
              cx, buffer,
              IsHugeMemoryEnabled(limits.addressType, limits.pageSize), proto));
  if (!memoryObj) {
    return false;
  }

  args.rval().setObject(*memoryObj);
  return true;
}

static bool IsMemory(HandleValue v) {
  return v.isObject() && v.toObject().is<WasmMemoryObject>();
}

/* static */
ArrayBufferObjectMaybeShared* WasmMemoryObject::refreshBuffer(
    JSContext* cx, Handle<WasmMemoryObject*> memoryObj,
    Handle<ArrayBufferObjectMaybeShared*> buffer) {
  if (memoryObj->isShared()) {
    size_t memoryLength = memoryObj->volatileMemoryLength();
    MOZ_ASSERT_IF(!buffer->is<GrowableSharedArrayBufferObject>(),
                  memoryLength >= buffer->byteLength());

    // The `length` field on a fixed length SAB cannot change even if
    // the underlying memory has grown. The spec therefore requires that
    // accessing the buffer property will create a new fixed length SAB
    // with the current length if the underlying raw buffer's length has
    // changed. We don't need to do this for growable SAB.
    if (!buffer->is<GrowableSharedArrayBufferObject>() &&
        memoryLength > buffer->byteLength()) {
      Rooted<SharedArrayBufferObject*> newBuffer(
          cx, SharedArrayBufferObject::New(
                  cx, memoryObj->sharedArrayRawBuffer(), memoryLength));
      if (!newBuffer) {
        return nullptr;
      }
      MOZ_ASSERT(newBuffer->is<FixedLengthSharedArrayBufferObject>());
      // OK to addReference after we try to allocate because the memoryObj
      // keeps the rawBuffer alive.
      if (!memoryObj->sharedArrayRawBuffer()->addReference()) {
        JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                                  JSMSG_SC_SAB_REFCNT_OFLO);
        return nullptr;
      }
      memoryObj->setReservedSlot(BUFFER_SLOT, ObjectValue(*newBuffer));
      return newBuffer;
    }
  }
  return buffer;
}

/* static */
bool WasmMemoryObject::bufferGetterImpl(JSContext* cx, const CallArgs& args) {
  Rooted<WasmMemoryObject*> memoryObj(
      cx, &args.thisv().toObject().as<WasmMemoryObject>());

  Rooted<ArrayBufferObjectMaybeShared*> buffer(cx, &memoryObj->buffer());
  MOZ_RELEASE_ASSERT(buffer->isWasm() && !buffer->isPreparedForAsmJS());

  ArrayBufferObjectMaybeShared* refreshedBuffer =
      WasmMemoryObject::refreshBuffer(cx, memoryObj, buffer);
  if (!refreshedBuffer) {
    return false;
  }

  args.rval().setObject(*refreshedBuffer);
  return true;
}

/* static */
bool WasmMemoryObject::bufferGetter(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsMemory, bufferGetterImpl>(cx, args);
}

const JSPropertySpec WasmMemoryObject::properties[] = {
    JS_PSG("buffer", WasmMemoryObject::bufferGetter, JSPROP_ENUMERATE),
    JS_STRING_SYM_PS(toStringTag, "WebAssembly.Memory", JSPROP_READONLY),
    JS_PS_END,
};

/* static */
bool WasmMemoryObject::growImpl(JSContext* cx, const CallArgs& args) {
  Rooted<WasmMemoryObject*> memory(
      cx, &args.thisv().toObject().as<WasmMemoryObject>());

  if (!args.requireAtLeast(cx, "WebAssembly.Memory.grow", 1)) {
    return false;
  }

  uint64_t delta;
  if (!EnforceAddressValue(cx, args.get(0), memory->addressType(), "Memory",
                           "grow delta", &delta)) {
    return false;
  }

  uint32_t ret = grow(memory, delta, cx);

  if (ret == uint32_t(-1)) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr, JSMSG_WASM_BAD_GROW,
                             "memory");
    return false;
  }

  RootedValue result(cx);
  if (!CreateAddressValue(cx, ret, memory->addressType(), &result)) {
    ReportOutOfMemory(cx);
    return false;
  }
  args.rval().set(result);
  return true;
}

/* static */
bool WasmMemoryObject::grow(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsMemory, growImpl>(cx, args);
}

/* static */
bool WasmMemoryObject::discardImpl(JSContext* cx, const CallArgs& args) {
  Rooted<WasmMemoryObject*> memory(
      cx, &args.thisv().toObject().as<WasmMemoryObject>());

  if (!args.requireAtLeast(cx, "WebAssembly.Memory.discard", 2)) {
    return false;
  }

  uint64_t byteOffset;
  if (!EnforceRangeU64(cx, args.get(0), "Memory", "byte offset", &byteOffset)) {
    return false;
  }

  uint64_t byteLen;
  if (!EnforceRangeU64(cx, args.get(1), "Memory", "length", &byteLen)) {
    return false;
  }

  if (byteOffset % wasm::StandardPageSizeBytes != 0 ||
      byteLen % wasm::StandardPageSizeBytes != 0) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_UNALIGNED_ACCESS);
    return false;
  }

  if (!wasm::MemoryBoundsCheck(byteOffset, byteLen,
                               memory->volatileMemoryLength())) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_OUT_OF_BOUNDS);
    return false;
  }

  discard(memory, byteOffset, byteLen, cx);

  args.rval().setUndefined();
  return true;
}

/* static */
bool WasmMemoryObject::discard(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsMemory, discardImpl>(cx, args);
}

#ifdef ENABLE_WASM_RESIZABLE_ARRAYBUFFER
/* static */
bool WasmMemoryObject::toFixedLengthBufferImpl(JSContext* cx,
                                               const CallArgs& args) {
  Rooted<WasmMemoryObject*> memory(
      cx, &args.thisv().toObject().as<WasmMemoryObject>());

  Rooted<ArrayBufferObjectMaybeShared*> buffer(cx, &memory->buffer());
  MOZ_RELEASE_ASSERT(buffer->isWasm() && !buffer->isPreparedForAsmJS());
  // If IsFixedLengthArrayBuffer(buffer) is true, return buffer.
  if (!buffer->isResizable()) {
    ArrayBufferObjectMaybeShared* refreshedBuffer =
        refreshBuffer(cx, memory, buffer);
    if (!refreshedBuffer) {
      return false;
    }
    args.rval().set(ObjectValue(*refreshedBuffer));
    return true;
  }

  if (!memory->isShared() && buffer->as<ArrayBufferObject>().isLengthPinned()) {
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_ARRAYBUFFER_LENGTH_PINNED);
    return false;
  }

  Rooted<ArrayBufferObjectMaybeShared*> fixedBuffer(cx);
  if (memory->isShared()) {
    Rooted<SharedArrayBufferObject*> oldBuffer(
        cx, &buffer->as<SharedArrayBufferObject>());
    fixedBuffer.set(SharedArrayBufferObject::createFromWasmObject<
                    FixedLengthSharedArrayBufferObject>(cx, oldBuffer));
  } else {
    Rooted<ArrayBufferObject*> oldBuffer(cx, &buffer->as<ArrayBufferObject>());
    fixedBuffer.set(
        ArrayBufferObject::createFromWasmObject<FixedLengthArrayBufferObject>(
            cx, oldBuffer));
  }

  if (!fixedBuffer) {
    return false;
  }
  memory->setReservedSlot(BUFFER_SLOT, ObjectValue(*fixedBuffer));
  args.rval().set(ObjectValue(*fixedBuffer));
  return true;
}

/* static */
bool WasmMemoryObject::toFixedLengthBuffer(JSContext* cx, unsigned argc,
                                           Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsMemory, toFixedLengthBufferImpl>(cx, args);
}

/* static */
bool WasmMemoryObject::toResizableBufferImpl(JSContext* cx,
                                             const CallArgs& args) {
  Rooted<WasmMemoryObject*> memory(
      cx, &args.thisv().toObject().as<WasmMemoryObject>());

  Rooted<ArrayBufferObjectMaybeShared*> buffer(cx, &memory->buffer());
  // If IsFixedLengthArrayBuffer(buffer) is false, return buffer.
  if (buffer->isResizable()) {
    args.rval().set(ObjectValue(*buffer));
    return true;
  }

  if (buffer->wasmSourceMaxPages().isNothing()) {
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_WASM_MEMORY_NOT_RESIZABLE);
    return false;
  }

  if (!memory->isShared() && buffer->as<ArrayBufferObject>().isLengthPinned()) {
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_ARRAYBUFFER_LENGTH_PINNED);
    return false;
  }

  Rooted<ArrayBufferObjectMaybeShared*> resizableBuffer(cx);
  if (memory->isShared()) {
    Rooted<SharedArrayBufferObject*> oldBuffer(
        cx, &buffer->as<SharedArrayBufferObject>());
    resizableBuffer.set(SharedArrayBufferObject::createFromWasmObject<
                        GrowableSharedArrayBufferObject>(cx, oldBuffer));
  } else {
    Rooted<ArrayBufferObject*> oldBuffer(cx, &buffer->as<ArrayBufferObject>());
    resizableBuffer.set(
        ArrayBufferObject::createFromWasmObject<ResizableArrayBufferObject>(
            cx, oldBuffer));
  }

  if (!resizableBuffer) {
    return false;
  }
  memory->setReservedSlot(BUFFER_SLOT, ObjectValue(*resizableBuffer));
  args.rval().set(ObjectValue(*resizableBuffer));
  return true;
}

/* static */
bool WasmMemoryObject::toResizableBuffer(JSContext* cx, unsigned argc,
                                         Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsMemory, toResizableBufferImpl>(cx, args);
}
#endif  // ENABLE_WASM_RESIZABLE_ARRAYBUFFER

const JSFunctionSpec WasmMemoryObject::methods[] = {
#ifdef ENABLE_WASM_TYPE_REFLECTIONS
    JS_FN("type", WasmMemoryObject::type, 0, JSPROP_ENUMERATE),
#endif
    JS_FN("grow", WasmMemoryObject::grow, 1, JSPROP_ENUMERATE),
#ifdef ENABLE_WASM_RESIZABLE_ARRAYBUFFER
    JS_FN("toFixedLengthBuffer", WasmMemoryObject::toFixedLengthBuffer, 0,
          JSPROP_ENUMERATE),
    JS_FN("toResizableBuffer", WasmMemoryObject::toResizableBuffer, 0,
          JSPROP_ENUMERATE),
#endif
    JS_FS_END,
};

const JSFunctionSpec WasmMemoryObject::memoryControlMethods[] = {
    JS_FN("discard", WasmMemoryObject::discard, 2, JSPROP_ENUMERATE),
    JS_FS_END,
};

const JSFunctionSpec WasmMemoryObject::static_methods[] = {
    JS_FS_END,
};

ArrayBufferObjectMaybeShared& WasmMemoryObject::buffer() const {
  return getReservedSlot(BUFFER_SLOT)
      .toObject()
      .as<ArrayBufferObjectMaybeShared>();
}

WasmSharedArrayRawBuffer* WasmMemoryObject::sharedArrayRawBuffer() const {
  MOZ_ASSERT(isShared());
  return buffer().as<SharedArrayBufferObject>().rawWasmBufferObject();
}

#ifdef ENABLE_WASM_TYPE_REFLECTIONS
bool WasmMemoryObject::typeImpl(JSContext* cx, const CallArgs& args) {
  Rooted<WasmMemoryObject*> memoryObj(
      cx, &args.thisv().toObject().as<WasmMemoryObject>());
  RootedObject typeObj(cx, MemoryTypeToObject(cx, memoryObj->isShared(),
                                              memoryObj->addressType(),
                                              memoryObj->volatilePages(),
                                              memoryObj->sourceMaxPages()));
  if (!typeObj) {
    return false;
  }
  args.rval().setObject(*typeObj);
  return true;
}

bool WasmMemoryObject::type(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsMemory, typeImpl>(cx, args);
}
#endif

size_t WasmMemoryObject::volatileMemoryLength() const {
  if (isShared()) {
    return sharedArrayRawBuffer()->volatileByteLength();
  }
  return buffer().byteLength();
}

wasm::Pages WasmMemoryObject::volatilePages() const {
  if (isShared()) {
    return sharedArrayRawBuffer()->volatileWasmPages();
  }
  return buffer().wasmPages();
}

wasm::Pages WasmMemoryObject::clampedMaxPages() const {
  if (isShared()) {
    return sharedArrayRawBuffer()->wasmClampedMaxPages();
  }
  return buffer().wasmClampedMaxPages();
}

Maybe<wasm::Pages> WasmMemoryObject::sourceMaxPages() const {
  if (isShared()) {
    return Some(sharedArrayRawBuffer()->wasmSourceMaxPages());
  }
  return buffer().wasmSourceMaxPages();
}

wasm::AddressType WasmMemoryObject::addressType() const {
  if (isShared()) {
    return sharedArrayRawBuffer()->wasmAddressType();
  }
  return buffer().wasmAddressType();
}

bool WasmMemoryObject::isShared() const {
  return buffer().is<SharedArrayBufferObject>();
}

bool WasmMemoryObject::hasObservers() const {
  return !getReservedSlot(OBSERVERS_SLOT).isUndefined();
}

WasmMemoryObject::InstanceSet& WasmMemoryObject::observers() const {
  MOZ_ASSERT(hasObservers());
  return *reinterpret_cast<InstanceSet*>(
      getReservedSlot(OBSERVERS_SLOT).toPrivate());
}

WasmMemoryObject::InstanceSet* WasmMemoryObject::getOrCreateObservers(
    JSContext* cx) {
  if (!hasObservers()) {
    auto observers = MakeUnique<InstanceSet>(cx->zone(), cx->zone());
    if (!observers) {
      ReportOutOfMemory(cx);
      return nullptr;
    }

    InitReservedSlot(this, OBSERVERS_SLOT, observers.release(),
                     MemoryUse::WasmMemoryObservers);
  }

  return &observers();
}

bool WasmMemoryObject::isHuge() const {
  return getReservedSlot(ISHUGE_SLOT).toBoolean();
}

bool WasmMemoryObject::movingGrowable() const {
  return !isHuge() && !buffer().wasmSourceMaxPages();
}

size_t WasmMemoryObject::boundsCheckLimit() const {
  if (!buffer().isWasm() || isHuge()) {
    return buffer().byteLength();
  }
#ifdef ENABLE_WASM_CUSTOM_PAGE_SIZES
  // For tiny page sizes, we need to use the actual byte length as the bounds
  // check as we cannot rely on virtual memory for accesses between the byte
  // length and the mapped size.
  if (buffer().wasmPageSize() == wasm::PageSize::Tiny) {
    size_t limit = buffer().byteLength();
    MOZ_ASSERT(limit <= MaxMemoryBoundsCheckLimit(addressType(),
                                                  buffer().wasmPageSize()));
    return limit;
  }
#endif
  size_t mappedSize = buffer().wasmMappedSize();
#if !defined(JS_64BIT)
  // See clamping performed in CreateSpecificWasmBuffer().  On 32-bit systems
  // we do not want to overflow a uint32_t.  For the other 64-bit compilers,
  // all constraints are implied by the largest accepted value for a memory's
  // max field.
  MOZ_ASSERT(mappedSize < UINT32_MAX);
#endif
  MOZ_ASSERT(buffer().wasmPageSize() == wasm::PageSize::Standard);
  MOZ_ASSERT(mappedSize % wasm::StandardPageSizeBytes == 0);
  MOZ_ASSERT(mappedSize >= wasm::GuardSize);
  size_t limit = mappedSize - wasm::GuardSize;
  MOZ_ASSERT(limit <= MaxMemoryBoundsCheckLimit(addressType(),
                                                wasm::PageSize::Standard));
  return limit;
}

wasm::PageSize WasmMemoryObject::pageSize() const {
  if (isShared()) {
    return sharedArrayRawBuffer()->wasmPageSize();
  }
  return buffer().wasmPageSize();
}

bool WasmMemoryObject::addMovingGrowObserver(JSContext* cx,
                                             WasmInstanceObject* instance) {
  MOZ_ASSERT(movingGrowable());

  InstanceSet* observers = getOrCreateObservers(cx);
  if (!observers) {
    return false;
  }

  // A memory can be imported multiple times into an instance, but we only
  // register the instance as an observer once.
  if (!observers->put(instance)) {
    ReportOutOfMemory(cx);
    return false;
  }

  return true;
}

/* static */
uint64_t WasmMemoryObject::growShared(Handle<WasmMemoryObject*> memory,
                                      uint64_t delta) {
  WasmSharedArrayRawBuffer* rawBuf = memory->sharedArrayRawBuffer();
  WasmSharedArrayRawBuffer::Lock lock(rawBuf);

  Pages oldNumPages = rawBuf->volatileWasmPages();
  Pages newPages = oldNumPages;
  if (!newPages.checkedIncrement(delta)) {
    return uint64_t(int64_t(-1));
  }

  if (!rawBuf->wasmGrowToPagesInPlace(lock, memory->addressType(), newPages)) {
    return uint64_t(int64_t(-1));
  }
  // New buffer objects will be created lazily in all agents (including in
  // this agent) by bufferGetterImpl, above, so no more work to do here.

  return oldNumPages.pageCount();
}

/* static */
uint64_t WasmMemoryObject::grow(Handle<WasmMemoryObject*> memory,
                                uint64_t delta, JSContext* cx) {
  if (memory->isShared()) {
    return growShared(memory, delta);
  }

  Rooted<ArrayBufferObject*> oldBuf(cx,
                                    &memory->buffer().as<ArrayBufferObject>());

#if !defined(JS_64BIT)
  // TODO (large ArrayBuffer): See more information at the definition of
  // MaxMemoryBytes().
  MOZ_ASSERT(
      MaxMemoryBytes(memory->addressType(), memory->pageSize()) <= UINT32_MAX,
      "Avoid 32-bit overflows");
#endif

  Pages oldNumPages = oldBuf->wasmPages();
  Pages newPages = oldNumPages;
  if (!newPages.checkedIncrement(delta)) {
    return uint64_t(int64_t(-1));
  }

  ArrayBufferObject* newBuf;
  if (memory->movingGrowable()) {
    MOZ_ASSERT(!memory->isHuge());
    newBuf = ArrayBufferObject::wasmMovingGrowToPages(memory->addressType(),
                                                      newPages, oldBuf, cx);
  } else {
    newBuf = ArrayBufferObject::wasmGrowToPagesInPlace(memory->addressType(),
                                                       newPages, oldBuf, cx);
  }
  if (!newBuf) {
    return uint64_t(int64_t(-1));
  }

  memory->setReservedSlot(BUFFER_SLOT, ObjectValue(*newBuf));

  // Only notify moving-grow-observers after the BUFFER_SLOT has been updated
  // since observers will call buffer().
  if (memory->hasObservers()) {
    for (auto iter = memory->observers().iter(); !iter.done(); iter.next()) {
      iter.get()->instance().onMovingGrowMemory(memory);
    }
  }

  return oldNumPages.pageCount();
}

/* static */
void WasmMemoryObject::discard(Handle<WasmMemoryObject*> memory,
                               uint64_t byteOffset, uint64_t byteLen,
                               JSContext* cx) {
  if (memory->isShared()) {
    Rooted<SharedArrayBufferObject*> buf(
        cx, &memory->buffer().as<SharedArrayBufferObject>());
    SharedArrayBufferObject::wasmDiscard(buf, byteOffset, byteLen);
  } else {
    Rooted<ArrayBufferObject*> buf(cx,
                                   &memory->buffer().as<ArrayBufferObject>());
    ArrayBufferObject::wasmDiscard(buf, byteOffset, byteLen);
  }
}

bool js::wasm::IsSharedWasmMemoryObject(JSObject* obj) {
  WasmMemoryObject* mobj = obj->maybeUnwrapIf<WasmMemoryObject>();
  return mobj && mobj->isShared();
}

// ============================================================================
// WebAssembly.Table class and methods

const JSClassOps WasmTableObject::classOps_ = {
    .finalize = WasmTableObject::finalize,
    .trace = WasmTableObject::trace,
};

const JSClass WasmTableObject::class_ = {
    "WebAssembly.Table",
    JSCLASS_DELAY_METADATA_BUILDER |
        JSCLASS_HAS_RESERVED_SLOTS(WasmTableObject::RESERVED_SLOTS) |
        JSCLASS_FOREGROUND_FINALIZE,
    &WasmTableObject::classOps_,
    &WasmTableObject::classSpec_,
};

const JSClass& WasmTableObject::protoClass_ = PlainObject::class_;

static constexpr char WasmTableName[] = "Table";

const ClassSpec WasmTableObject::classSpec_ = {
    CreateWasmConstructor<WasmTableObject, WasmTableName>,
    GenericCreatePrototype<WasmTableObject>,
    WasmTableObject::static_methods,
    nullptr,
    WasmTableObject::methods,
    WasmTableObject::properties,
    nullptr,
    ClassSpec::DontDefineConstructor,
};

bool WasmTableObject::isNewborn() const {
  MOZ_ASSERT(is<WasmTableObject>());
  return getReservedSlot(TABLE_SLOT).isUndefined();
}

/* static */
void WasmTableObject::finalize(JS::GCContext* gcx, JSObject* obj) {
  WasmTableObject& tableObj = obj->as<WasmTableObject>();
  if (!tableObj.isNewborn()) {
    auto& table = tableObj.table();
    gcx->release(obj, &table, table.gcMallocBytes(), MemoryUse::WasmTableTable);
  }
}

/* static */
void WasmTableObject::trace(JSTracer* trc, JSObject* obj) {
  WasmTableObject& tableObj = obj->as<WasmTableObject>();
  if (!tableObj.isNewborn()) {
    tableObj.table().tracePrivate(trc);
  }
}

// Return the JS value to use when a parameter to a function requiring a table
// value is omitted. An implementation of [1].
//
// [1]
// https://webassembly.github.io/spec/js-api/#defaultvalue
static Value RefTypeDefaultValue(wasm::RefType tableType) {
  return tableType.isExtern() ? UndefinedValue() : NullValue();
}

/* static */
WasmTableObject* WasmTableObject::create(JSContext* cx, const TableType& type,
                                         HandleObject proto) {
  AutoSetNewObjectMetadata metadata(cx);
  Rooted<WasmTableObject*> obj(
      cx, NewObjectWithGivenProto<WasmTableObject>(cx, proto));
  if (!obj) {
    return nullptr;
  }

  MOZ_ASSERT(obj->isNewborn());

  TableDesc td(type, Nothing(),
               /*isAsmJS*/ false,
               /*isImported=*/true, /*isExported=*/true);

  SharedTable table = Table::create(cx, td, obj);
  if (!table) {
    return nullptr;
  }

  size_t size = table->gcMallocBytes();
  InitReservedSlot(obj, TABLE_SLOT, table.forget().take(), size,
                   MemoryUse::WasmTableTable);

  MOZ_ASSERT(!obj->isNewborn());
  return obj;
}

/* static */
bool WasmTableObject::construct(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);

  if (!ThrowIfNotConstructing(cx, args, "Table")) {
    return false;
  }

#if defined(__EMSCRIPTEN__)
  if (wasm::UseHostPassthrough()) {
    if (!args.requireAtLeast(cx, "WebAssembly.Table", 1) ||
        !args.get(0).isObject()) {
      JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                               JSMSG_WASM_BAD_DESC_ARG, "table");
      return false;
    }
    RootedObject desc(cx, &args[0].toObject());
    RootedValue v(cx);
    int32_t initial = 0;
    int32_t maximum = -1;
    if (!JS_GetProperty(cx, desc, "initial", &v) || !ToInt32(cx, v, &initial)) {
      return false;
    }
    if (!JS_GetProperty(cx, desc, "maximum", &v)) {
      return false;
    }
    if (!v.isUndefined() && !ToInt32(cx, v, &maximum)) {
      return false;
    }
    bool externref = false;
    if (JS_GetProperty(cx, desc, "element", &v) && v.isString()) {
      bool m = false;
      if (JS_StringEqualsLiteral(cx, v.toString(), "externref", &m)) {
        externref = m;
      }
    }
    int objId = wasmhost_table_new(initial, maximum, externref ? 1 : 0);
    RootedObject w(cx);
    if (objId >= 0) {
      w = HostMakeObjIdWrapper(cx, objId);
    }
    if (!w) {
      JS_ReportErrorASCII(
          cx, "WebAssembly host passthrough: table creation failed");
      return false;
    }
    args.rval().setObject(*w);
    return true;
  }
#endif

  if (!args.requireAtLeast(cx, "WebAssembly.Table", 1)) {
    return false;
  }

  if (!args.get(0).isObject()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_DESC_ARG, "table");
    return false;
  }

  RootedObject obj(cx, &args[0].toObject());

  JSAtom* elementAtom = Atomize(cx, "element", strlen("element"));
  if (!elementAtom) {
    return false;
  }
  RootedId elementId(cx, AtomToId(elementAtom));

  RootedValue elementVal(cx);
  if (!GetProperty(cx, obj, obj, elementId, &elementVal)) {
    return false;
  }

  RefType elemType;
  if (!ToRefType(cx, elementVal, &elemType)) {
    return false;
  }

  Limits limits;
  if (!GetLimits(cx, obj, LimitsKind::Table, &limits) ||
      !CheckLimits(cx, MaxTableElemsValidation(limits.addressType),
                   LimitsKind::Table, &limits)) {
    return false;
  }

  if (limits.initial > MaxTableElemsRuntime) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_TABLE_IMP_LIMIT);
    return false;
  }

  RootedObject proto(cx,
                     GetWasmConstructorPrototype(cx, args, JSProto_WasmTable));
  if (!proto) {
    ReportOutOfMemory(cx);
    return false;
  }

  Rooted<WasmTableObject*> table(
      cx, WasmTableObject::create(cx, TableType(limits, elemType), proto));
  if (!table) {
    return false;
  }

  // Initialize the table to a default value
  RootedValue initValue(
      cx, args.length() < 2 ? RefTypeDefaultValue(elemType) : args[1]);
  if (!CheckRefType(cx, elemType, initValue)) {
    return false;
  }

  // Skip initializing the table if the fill value is null, as that is the
  // default value.
  if (!initValue.isNull() &&
      !table->fillRange(cx, 0, limits.initial, initValue)) {
    return false;
  }
#ifdef DEBUG
  // Assert that null is the default value of a new table.
  if (initValue.isNull()) {
    table->table().assertRangeNull(0, limits.initial);
  }
  if (!elemType.isNullable()) {
    table->table().assertRangeNotNull(0, limits.initial);
  }
#endif

  args.rval().setObject(*table);
  return true;
}

static bool IsTable(HandleValue v) {
  return v.isObject() && v.toObject().is<WasmTableObject>();
}

/* static */
bool WasmTableObject::lengthGetterImpl(JSContext* cx, const CallArgs& args) {
  const WasmTableObject& tableObj =
      args.thisv().toObject().as<WasmTableObject>();
  RootedValue length(cx);
  if (!CreateAddressValue(cx, tableObj.table().length(),
                          tableObj.table().addressType(), &length)) {
    ReportOutOfMemory(cx);
    return false;
  }
  args.rval().set(length);
  return true;
}

/* static */
bool WasmTableObject::lengthGetter(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsTable, lengthGetterImpl>(cx, args);
}

const JSPropertySpec WasmTableObject::properties[] = {
    JS_PSG("length", WasmTableObject::lengthGetter, JSPROP_ENUMERATE),
    JS_STRING_SYM_PS(toStringTag, "WebAssembly.Table", JSPROP_READONLY),
    JS_PS_END,
};

// Gets an AddressValue parameter for a table. This differs from our general
// EnforceAddressValue because our table implementation still uses 32-bit sizes
// internally, and this function therefore returns a uint32_t. Values outside
// the 32-bit range will be clamped to UINT32_MAX, which will always trigger
// bounds checks for all Table uses of AddressValue. See
// MacroAssembler::wasmClampTable64Address and its uses.
//
// isAddress should be true if the value is an actual address, and false if it
// is a different quantity (e.g. a grow delta).
static bool EnforceTableAddressValue(JSContext* cx, HandleValue v,
                                     const Table& table, const char* noun,
                                     uint32_t* result, bool isAddress) {
  uint64_t result64;
  if (!EnforceAddressValue(cx, v, table.addressType(), "Table", noun,
                           &result64)) {
    return false;
  }

  static_assert(MaxTableElemsRuntime < UINT32_MAX);
  *result = result64 > UINT32_MAX ? UINT32_MAX : uint32_t(result64);

  if (isAddress && *result >= table.length()) {
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_WASM_BAD_RANGE, "Table", noun);
    return false;
  }

  return true;
}

#ifdef ENABLE_WASM_TYPE_REFLECTIONS
/* static */
bool WasmTableObject::typeImpl(JSContext* cx, const CallArgs& args) {
  Table& table = args.thisv().toObject().as<WasmTableObject>().table();
  RootedObject typeObj(
      cx, TableTypeToObject(cx, table.addressType(), table.elemType(),
                            table.length(), table.maximum()));
  if (!typeObj) {
    return false;
  }
  args.rval().setObject(*typeObj);
  return true;
}

/* static */
bool WasmTableObject::type(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsTable, typeImpl>(cx, args);
}
#endif

/* static */
bool WasmTableObject::getImpl(JSContext* cx, const CallArgs& args) {
  Rooted<WasmTableObject*> tableObj(
      cx, &args.thisv().toObject().as<WasmTableObject>());
  const Table& table = tableObj->table();

  if (!args.requireAtLeast(cx, "WebAssembly.Table.get", 1)) {
    return false;
  }

  uint32_t address;
  if (!EnforceTableAddressValue(cx, args.get(0), table, "get address", &address,
                                /*isAddress=*/true)) {
    return false;
  }

  return table.getValue(cx, address, args.rval());
}

/* static */
bool WasmTableObject::get(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsTable, getImpl>(cx, args);
}

/* static */
bool WasmTableObject::setImpl(JSContext* cx, const CallArgs& args) {
  Rooted<WasmTableObject*> tableObj(
      cx, &args.thisv().toObject().as<WasmTableObject>());
  Table& table = tableObj->table();

  if (!args.requireAtLeast(cx, "WebAssembly.Table.set", 1)) {
    return false;
  }

  uint32_t address;
  if (!EnforceTableAddressValue(cx, args.get(0), table, "set address", &address,
                                /*isAddress=*/true)) {
    return false;
  }

  RootedValue fillValue(
      cx, args.length() < 2 ? RefTypeDefaultValue(table.elemType()) : args[1]);
  if (!tableObj->fillRange(cx, address, 1, fillValue)) {
    return false;
  }

  args.rval().setUndefined();
  return true;
}

/* static */
bool WasmTableObject::set(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsTable, setImpl>(cx, args);
}

/* static */
bool WasmTableObject::growImpl(JSContext* cx, const CallArgs& args) {
  Rooted<WasmTableObject*> tableObj(
      cx, &args.thisv().toObject().as<WasmTableObject>());
  Table& table = tableObj->table();

  if (!args.requireAtLeast(cx, "WebAssembly.Table.grow", 1)) {
    return false;
  }

  uint32_t delta;
  if (!EnforceTableAddressValue(cx, args.get(0), table, "grow delta", &delta,
                                /*isAddress=*/false)) {
    return false;
  }

  RootedValue fillValue(
      cx, args.length() < 2 ? RefTypeDefaultValue(table.elemType()) : args[1]);
  Rooted<wasm::AnyRef> fillRef(cx);
  if (!CheckRefType(cx, table.elemType(), fillValue, &fillRef)) {
    return false;
  }

  uint32_t oldLength = table.grow(delta);

  if (oldLength == uint32_t(-1)) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr, JSMSG_WASM_BAD_GROW,
                             "table");
    return false;
  }

  // Skip filling the grown range of the table if the fill value is null, as
  // that is the default value.
  if (!fillRef.isNull()) {
    table.fillUninitialized(oldLength, delta, fillRef, cx);
  }
#ifdef DEBUG
  // Assert that null is the default value of the grown range.
  if (fillRef.isNull()) {
    table.assertRangeNull(oldLength, delta);
  }
  if (!table.elemType().isNullable()) {
    table.assertRangeNotNull(oldLength, delta);
  }
#endif

  RootedValue result(cx);
  if (!CreateAddressValue(cx, oldLength, table.addressType(), &result)) {
    ReportOutOfMemory(cx);
    return false;
  }
  args.rval().set(result);
  return true;
}

/* static */
bool WasmTableObject::grow(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsTable, growImpl>(cx, args);
}

const JSFunctionSpec WasmTableObject::methods[] = {
#ifdef ENABLE_WASM_TYPE_REFLECTIONS
    JS_FN("type", WasmTableObject::type, 0, JSPROP_ENUMERATE),
#endif
    JS_FN("get", WasmTableObject::get, 1, JSPROP_ENUMERATE),
    JS_FN("set", WasmTableObject::set, 2, JSPROP_ENUMERATE),
    JS_FN("grow", WasmTableObject::grow, 1, JSPROP_ENUMERATE),
    JS_FS_END,
};

const JSFunctionSpec WasmTableObject::static_methods[] = {
    JS_FS_END,
};

Table& WasmTableObject::table() const {
  return *(Table*)getReservedSlot(TABLE_SLOT).toPrivate();
}

bool WasmTableObject::fillRange(JSContext* cx, uint32_t index, uint32_t length,
                                HandleValue value) const {
  Table& tab = table();

  // All consumers are required to either bounds check or statically be in
  // bounds
  MOZ_ASSERT(uint64_t(index) + uint64_t(length) <= tab.length());

  RootedAnyRef any(cx, AnyRef::null());
  if (!wasm::CheckRefType(cx, tab.elemType(), value, &any)) {
    return false;
  }
  switch (tab.repr()) {
    case TableRepr::Func:
      MOZ_RELEASE_ASSERT(!tab.isAsmJS());
      tab.fillFuncRef(index, length, FuncRef::fromAnyRefUnchecked(any.get()),
                      cx);
      break;
    case TableRepr::Ref:
      tab.fillAnyRef(index, length, any);
      break;
  }
  return true;
}

// ============================================================================
// WebAssembly.global class and methods

const JSClassOps WasmGlobalObject::classOps_ = {
    .finalize = WasmGlobalObject::finalize,
    .trace = WasmGlobalObject::trace,
};

const JSClass WasmGlobalObject::class_ = {
    "WebAssembly.Global",
    JSCLASS_HAS_RESERVED_SLOTS(WasmGlobalObject::RESERVED_SLOTS) |
        JSCLASS_BACKGROUND_FINALIZE,
    &WasmGlobalObject::classOps_,
    &WasmGlobalObject::classSpec_,
};

const JSClass& WasmGlobalObject::protoClass_ = PlainObject::class_;

static constexpr char WasmGlobalName[] = "Global";

const ClassSpec WasmGlobalObject::classSpec_ = {
    CreateWasmConstructor<WasmGlobalObject, WasmGlobalName>,
    GenericCreatePrototype<WasmGlobalObject>,
    WasmGlobalObject::static_methods,
    nullptr,
    WasmGlobalObject::methods,
    WasmGlobalObject::properties,
    nullptr,
    ClassSpec::DontDefineConstructor,
};

/* static */
void WasmGlobalObject::trace(JSTracer* trc, JSObject* obj) {
  WasmGlobalObject* global = reinterpret_cast<WasmGlobalObject*>(obj);
  if (global->isNewborn()) {
    // This can happen while we're allocating the object, in which case
    // every single slot of the object is not defined yet. In particular,
    // there's nothing to trace yet.
    return;
  }
  global->val().get().trace(trc);
}

/* static */
void WasmGlobalObject::finalize(JS::GCContext* gcx, JSObject* obj) {
  WasmGlobalObject* global = reinterpret_cast<WasmGlobalObject*>(obj);
  if (!global->isNewborn()) {
    // Release the strong reference to the type definitions this global could
    // be referencing.
    global->type().Release();
    gcx->delete_(obj, &global->mutableVal(), MemoryUse::WasmGlobalCell);
  }
}

/* static */
WasmGlobalObject* WasmGlobalObject::create(JSContext* cx, HandleVal value,
                                           bool isMutable, HandleObject proto) {
  Rooted<WasmGlobalObject*> obj(
      cx, NewObjectWithGivenProto<WasmGlobalObject>(cx, proto));
  if (!obj) {
    return nullptr;
  }

  MOZ_ASSERT(obj->isNewborn());
  MOZ_ASSERT(obj->isTenured(), "assumed by global.set post barriers");

  HeapPtrVal* val = js_new<HeapPtrVal>(Val());
  if (!val) {
    ReportOutOfMemory(cx);
    return nullptr;
  }
  obj->initReservedSlot(MUTABLE_SLOT, JS::BooleanValue(isMutable));
  InitReservedSlot(obj, VAL_SLOT, val, MemoryUse::WasmGlobalCell);

  // It's simpler to initialize the cell after the object has been created,
  // to avoid needing to root the cell before the object creation.
  // We don't use `setVal` here because the assumes the cell has already
  // been initialized.
  obj->mutableVal() = value.get();
  // Acquire a strong reference to a type definition this global could
  // be referencing.
  obj->type().AddRef();

  MOZ_ASSERT(!obj->isNewborn());

  return obj;
}

/* static */
bool WasmGlobalObject::construct(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);

  if (!ThrowIfNotConstructing(cx, args, "Global")) {
    return false;
  }

#if defined(__EMSCRIPTEN__)
  if (wasm::UseHostPassthrough()) {
    if (!args.requireAtLeast(cx, "WebAssembly.Global", 1) ||
        !args.get(0).isObject()) {
      JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                               JSMSG_WASM_BAD_DESC_ARG, "global");
      return false;
    }
    RootedObject desc(cx, &args[0].toObject());
    RootedValue v(cx);
    int kind = 3;  // f64 default
    if (JS_GetProperty(cx, desc, "value", &v) && v.isString()) {
      RootedString s(cx, v.toString());
      bool m = false;
      if (JS_StringEqualsLiteral(cx, s, "i32", &m) && m) {
        kind = 0;
      } else if (JS_StringEqualsLiteral(cx, s, "i64", &m) && m) {
        kind = 1;
      } else if (JS_StringEqualsLiteral(cx, s, "f32", &m) && m) {
        kind = 2;
      }
    }
    if (!JS_GetProperty(cx, desc, "mutable", &v)) {
      return false;
    }
    bool mut = ToBoolean(v);
    double initVal = 0;
    if (args.length() >= 2 && !ToNumber(cx, args.get(1), &initVal)) {
      return false;
    }
    int objId = wasmhost_global_new(initVal, kind, mut ? 1 : 0);
    RootedObject w(cx);
    if (objId >= 0) {
      w = HostMakeObjIdWrapper(cx, objId);
    }
    if (!w) {
      JS_ReportErrorASCII(
          cx, "WebAssembly host passthrough: global creation failed");
      return false;
    }
    args.rval().setObject(*w);
    return true;
  }
#endif

  if (!args.requireAtLeast(cx, "WebAssembly.Global", 1)) {
    return false;
  }

  if (!args.get(0).isObject()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_DESC_ARG, "global");
    return false;
  }

  RootedObject obj(cx, &args[0].toObject());

  // Extract properties in lexicographic order per spec.

  RootedValue mutableVal(cx);
  if (!JS_GetProperty(cx, obj, "mutable", &mutableVal)) {
    return false;
  }

  RootedValue typeVal(cx);
  if (!JS_GetProperty(cx, obj, "value", &typeVal)) {
    return false;
  }

  ValType globalType;
  if (!ToValType(cx, typeVal, &globalType)) {
    return false;
  }

  if (!globalType.isExposable()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_VAL_TYPE);
    return false;
  }

  bool isMutable = ToBoolean(mutableVal);

  // Extract the initial value, or provide a suitable default.
  RootedVal globalVal(cx, globalType);

  // Override with non-undefined value, if provided.
  RootedValue valueVal(cx);
  if (globalType.isRefType()) {
    valueVal.set(args.length() < 2 ? RefTypeDefaultValue(globalType.refType())
                                   : args[1]);
    if (!Val::fromJSValue(cx, globalType, valueVal, &globalVal)) {
      return false;
    }
  } else {
    valueVal.set(args.get(1));
    if (!valueVal.isUndefined() &&
        !Val::fromJSValue(cx, globalType, valueVal, &globalVal)) {
      return false;
    }
  }

  RootedObject proto(cx,
                     GetWasmConstructorPrototype(cx, args, JSProto_WasmGlobal));
  if (!proto) {
    ReportOutOfMemory(cx);
    return false;
  }

  WasmGlobalObject* global =
      WasmGlobalObject::create(cx, globalVal, isMutable, proto);
  if (!global) {
    return false;
  }

  args.rval().setObject(*global);
  return true;
}

static bool IsGlobal(HandleValue v) {
  return v.isObject() && v.toObject().is<WasmGlobalObject>();
}

/* static */
bool WasmGlobalObject::valueGetterImpl(JSContext* cx, const CallArgs& args) {
  const WasmGlobalObject& globalObj =
      args.thisv().toObject().as<WasmGlobalObject>();
  if (!globalObj.type().isExposable()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_VAL_TYPE);
    return false;
  }
  return globalObj.val().get().toJSValue(cx, args.rval());
}

/* static */
bool WasmGlobalObject::valueGetter(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsGlobal, valueGetterImpl>(cx, args);
}

/* static */
bool WasmGlobalObject::valueSetterImpl(JSContext* cx, const CallArgs& args) {
  if (!args.requireAtLeast(cx, "WebAssembly.Global setter", 1)) {
    return false;
  }

  Rooted<WasmGlobalObject*> global(
      cx, &args.thisv().toObject().as<WasmGlobalObject>());
  if (!global->isMutable()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_GLOBAL_IMMUTABLE);
    return false;
  }

  if (!global->type().isExposable()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_VAL_TYPE);
    return false;
  }

  RootedVal val(cx);
  if (!Val::fromJSValue(cx, global->type(), args.get(0), &val)) {
    return false;
  }
  global->setVal(val);

  args.rval().setUndefined();
  return true;
}

/* static */
bool WasmGlobalObject::valueSetter(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsGlobal, valueSetterImpl>(cx, args);
}

const JSPropertySpec WasmGlobalObject::properties[] = {
    JS_PSGS("value", WasmGlobalObject::valueGetter,
            WasmGlobalObject::valueSetter, JSPROP_ENUMERATE),
    JS_STRING_SYM_PS(toStringTag, "WebAssembly.Global", JSPROP_READONLY),
    JS_PS_END,
};

const JSFunctionSpec WasmGlobalObject::methods[] = {
#ifdef ENABLE_WASM_TYPE_REFLECTIONS
    JS_FN("type", WasmGlobalObject::type, 0, JSPROP_ENUMERATE),
#endif
    JS_FN("valueOf", WasmGlobalObject::valueGetter, 0, JSPROP_ENUMERATE),
    JS_FS_END,
};

const JSFunctionSpec WasmGlobalObject::static_methods[] = {
    JS_FS_END,
};

bool WasmGlobalObject::isMutable() const {
  return getReservedSlot(MUTABLE_SLOT).toBoolean();
}

ValType WasmGlobalObject::type() const { return val().get().type(); }

HeapPtrVal& WasmGlobalObject::mutableVal() {
  return *reinterpret_cast<HeapPtrVal*>(getReservedSlot(VAL_SLOT).toPrivate());
}

const HeapPtrVal& WasmGlobalObject::val() const {
  return *reinterpret_cast<HeapPtrVal*>(getReservedSlot(VAL_SLOT).toPrivate());
}

void WasmGlobalObject::setVal(wasm::HandleVal value) {
  MOZ_ASSERT(type() == value.get().type());
  mutableVal() = value;
}

void* WasmGlobalObject::addressOfCell() const {
  return (void*)&val().get().cell();
}

#ifdef ENABLE_WASM_TYPE_REFLECTIONS
/* static */
bool WasmGlobalObject::typeImpl(JSContext* cx, const CallArgs& args) {
  Rooted<WasmGlobalObject*> global(
      cx, &args.thisv().toObject().as<WasmGlobalObject>());
  RootedObject typeObj(
      cx, GlobalTypeToObject(cx, global->type(), global->isMutable()));
  if (!typeObj) {
    return false;
  }
  args.rval().setObject(*typeObj);
  return true;
}

/* static */
bool WasmGlobalObject::type(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsGlobal, typeImpl>(cx, args);
}
#endif

// ============================================================================
// WebAssembly.Tag class and methods

const JSClassOps WasmTagObject::classOps_ = {
    .finalize = WasmTagObject::finalize,
};

const JSClass WasmTagObject::class_ = {
    "WebAssembly.Tag",
    JSCLASS_HAS_RESERVED_SLOTS(WasmTagObject::RESERVED_SLOTS) |
        JSCLASS_FOREGROUND_FINALIZE,
    &WasmTagObject::classOps_,
    &WasmTagObject::classSpec_,
};

const JSClass& WasmTagObject::protoClass_ = PlainObject::class_;

static constexpr char WasmTagName[] = "Tag";

const ClassSpec WasmTagObject::classSpec_ = {
    CreateWasmConstructor<WasmTagObject, WasmTagName>,
    GenericCreatePrototype<WasmTagObject>,
    WasmTagObject::static_methods,
    nullptr,
    WasmTagObject::methods,
    WasmTagObject::properties,
    nullptr,
    ClassSpec::DontDefineConstructor,
};

/* static */
void WasmTagObject::finalize(JS::GCContext* gcx, JSObject* obj) {
  WasmTagObject& tagObj = obj->as<WasmTagObject>();
  tagObj.tagType()->Release();
}

static bool IsTag(HandleValue v) {
  return v.isObject() && v.toObject().is<WasmTagObject>();
}

bool WasmTagObject::construct(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);

  if (!ThrowIfNotConstructing(cx, args, "Tag")) {
    return false;
  }

  if (!args.requireAtLeast(cx, "WebAssembly.Tag", 1)) {
    return false;
  }

  if (!args.get(0).isObject()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_DESC_ARG, "tag");
    return false;
  }

  RootedObject obj(cx, &args[0].toObject());
  RootedValue paramsVal(cx);
  if (!JS_GetProperty(cx, obj, "parameters", &paramsVal)) {
    return false;
  }

  ValTypeVector params;
  if (!ParseValTypes(cx, paramsVal, params)) {
    return false;
  }
  if (params.length() > MaxParams) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_EXN_TAG_PARAMS);
    return false;
  }

  RefPtr<TypeContext> types = js_new<TypeContext>();
  if (!types) {
    ReportOutOfMemory(cx);
    return false;
  }
  const TypeDef* tagTypeDef =
      types->addType(FuncType(std::move(params), ValTypeVector()));
  if (!tagTypeDef) {
    ReportOutOfMemory(cx);
    return false;
  }

  wasm::MutableTagType tagType = js_new<wasm::TagType>();
  if (!tagType || !tagType->initialize(tagTypeDef)) {
    ReportOutOfMemory(cx);
    return false;
  }

  RootedObject proto(cx,
                     GetWasmConstructorPrototype(cx, args, JSProto_WasmTag));
  if (!proto) {
    ReportOutOfMemory(cx);
    return false;
  }

  Rooted<WasmTagObject*> tagObj(cx, WasmTagObject::create(cx, tagType, proto));
  if (!tagObj) {
    return false;
  }

  args.rval().setObject(*tagObj);
  return true;
}

/* static */
WasmTagObject* WasmTagObject::create(JSContext* cx,
                                     const wasm::SharedTagType& tagType,
                                     HandleObject proto) {
  Rooted<WasmTagObject*> obj(cx,
                             NewObjectWithGivenProto<WasmTagObject>(cx, proto));
  if (!obj) {
    return nullptr;
  }

  tagType.get()->AddRef();
  obj->initReservedSlot(TYPE_SLOT, PrivateValue((void*)tagType.get()));

  return obj;
}

const JSPropertySpec WasmTagObject::properties[] = {
    JS_STRING_SYM_PS(toStringTag, "WebAssembly.Tag", JSPROP_READONLY),
    JS_PS_END,
};

#ifdef ENABLE_WASM_TYPE_REFLECTIONS
/* static */
bool WasmTagObject::typeImpl(JSContext* cx, const CallArgs& args) {
  Rooted<WasmTagObject*> tag(cx, &args.thisv().toObject().as<WasmTagObject>());
  RootedObject typeObj(cx, TagTypeToObject(cx, tag->valueTypes()));
  if (!typeObj) {
    return false;
  }
  args.rval().setObject(*typeObj);
  return true;
}

/* static  */
bool WasmTagObject::type(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsTag, typeImpl>(cx, args);
}
#endif

const JSFunctionSpec WasmTagObject::methods[] = {
#ifdef ENABLE_WASM_TYPE_REFLECTIONS
    JS_FN("type", WasmTagObject::type, 0, JSPROP_ENUMERATE),
#endif
    JS_FS_END,
};

const JSFunctionSpec WasmTagObject::static_methods[] = {
    JS_FS_END,
};

const TagType* WasmTagObject::tagType() const {
  return (const TagType*)getFixedSlot(TYPE_SLOT).toPrivate();
};

const wasm::ValTypeVector& WasmTagObject::valueTypes() const {
  return tagType()->argTypes();
};

// ============================================================================
// WebAssembly.Exception class and methods

const JSClassOps WasmExceptionObject::classOps_ = {
    .finalize = WasmExceptionObject::finalize,
    .trace = WasmExceptionObject::trace,
};

const JSClass WasmExceptionObject::class_ = {
    "WebAssembly.Exception",
    JSCLASS_HAS_RESERVED_SLOTS(WasmExceptionObject::RESERVED_SLOTS) |
        JSCLASS_FOREGROUND_FINALIZE,
    &WasmExceptionObject::classOps_,
    &WasmExceptionObject::classSpec_,
};

const JSClass& WasmExceptionObject::protoClass_ = PlainObject::class_;

static constexpr char WasmExceptionName[] = "Exception";

const ClassSpec WasmExceptionObject::classSpec_ = {
    CreateWasmConstructor<WasmExceptionObject, WasmExceptionName>,
    GenericCreatePrototype<WasmExceptionObject>,
    WasmExceptionObject::static_methods,
    nullptr,
    WasmExceptionObject::methods,
    WasmExceptionObject::properties,
    nullptr,
    ClassSpec::DontDefineConstructor,
};

/* static */
void WasmExceptionObject::finalize(JS::GCContext* gcx, JSObject* obj) {
  WasmExceptionObject& exnObj = obj->as<WasmExceptionObject>();
  if (exnObj.isNewborn()) {
    return;
  }
  gcx->free_(obj, exnObj.typedMem(), exnObj.tagType()->tagSize(),
             MemoryUse::WasmExceptionData);
  exnObj.tagType()->Release();
}

/* static */
void WasmExceptionObject::trace(JSTracer* trc, JSObject* obj) {
  WasmExceptionObject& exnObj = obj->as<WasmExceptionObject>();
  if (exnObj.isNewborn()) {
    return;
  }

  wasm::SharedTagType tag = exnObj.tagType();
  const wasm::ValTypeVector& params = tag->argTypes();
  const wasm::TagOffsetVector& offsets = tag->exceptionArgOffsets();
  uint8_t* typedMem = exnObj.typedMem();
  for (size_t i = 0; i < params.length(); i++) {
    ValType paramType = params[i];
    if (paramType.isRefRepr()) {
      GCPtr<wasm::AnyRef>* paramPtr =
          reinterpret_cast<GCPtr<AnyRef>*>(typedMem + offsets[i]);
      TraceEdge(trc, paramPtr, "wasm exception param");
    }
  }
}

static bool IsException(HandleValue v) {
  return v.isObject() && v.toObject().is<WasmExceptionObject>();
}

struct ExceptionOptions {
  bool traceStack;

  ExceptionOptions() : traceStack(false) {}

  [[nodiscard]] bool init(JSContext* cx, HandleValue val);
};

bool ExceptionOptions::init(JSContext* cx, HandleValue val) {
  if (val.isNullOrUndefined()) {
    return true;
  }

  if (!val.isObject()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_EXN_OPTIONS);
    return false;
  }
  RootedObject obj(cx, &val.toObject());

  // Get `traceStack` and coerce to boolean
  RootedValue traceStackVal(cx);
  if (!JS_GetProperty(cx, obj, "traceStack", &traceStackVal)) {
    return false;
  }
  traceStack = ToBoolean(traceStackVal);

  return true;
}

bool WasmExceptionObject::construct(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);

  if (!ThrowIfNotConstructing(cx, args, "Exception")) {
    return false;
  }

  if (!args.requireAtLeast(cx, "WebAssembly.Exception", 2)) {
    return false;
  }

  if (!IsTag(args[0])) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_EXN_ARG);
    return false;
  }
  Rooted<WasmTagObject*> exnTag(cx, &args[0].toObject().as<WasmTagObject>());

  // Per spec, WebAssembly.Exception can't be constructed with
  // WebAssembly.JSTag.
  if (exnTag->tagType() == sWrappedJSValueTagType) {
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_WASM_BAD_JSTAG_WRAP);
    return false;
  }

  if (!args.get(1).isObject()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_EXN_PAYLOAD);
    return false;
  }

  JS::ForOfIterator iterator(cx);
  if (!iterator.init(args.get(1), JS::ForOfIterator::ThrowOnNonIterable)) {
    return false;
  }

  // Get the optional 'options' parameter
  ExceptionOptions options;
  if (!options.init(cx, args.get(2))) {
    return false;
  }

  // Trace the stack if requested
  RootedObject stack(cx);
  bool captureStack =
      options.traceStack || JS::Prefs::wasm_exception_force_stack_trace();
  if (captureStack && !CaptureStack(cx, &stack, MAX_REPORTED_STACK_DEPTH)) {
    ReportOutOfMemory(cx);
    return false;
  }

  RootedObject proto(
      cx, GetWasmConstructorPrototype(cx, args, JSProto_WasmException));
  if (!proto) {
    ReportOutOfMemory(cx);
    return false;
  }

  Rooted<WasmExceptionObject*> exnObj(
      cx, WasmExceptionObject::create(cx, exnTag, stack, proto));
  if (!exnObj) {
    return false;
  }

  wasm::SharedTagType tagType = exnObj->tagType();
  const wasm::ValTypeVector& params = tagType->argTypes();
  const wasm::TagOffsetVector& offsets = tagType->exceptionArgOffsets();

  RootedValue nextArg(cx);
  for (size_t i = 0; i < params.length(); i++) {
    bool done;
    if (!iterator.next(&nextArg, &done)) {
      return false;
    }
    if (done) {
      UniqueChars expected(JS_smprintf("%zu", params.length()));
      UniqueChars got(JS_smprintf("%zu", i));
      if (!expected || !got) {
        ReportOutOfMemory(cx);
        return false;
      }

      JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                               JSMSG_WASM_BAD_EXN_PAYLOAD_LEN, expected.get(),
                               got.get());
      return false;
    }

    if (!exnObj->initArg(cx, offsets[i], params[i], nextArg)) {
      return false;
    }
  }

  args.rval().setObject(*exnObj);
  return true;
}

/* static */
WasmExceptionObject* WasmExceptionObject::create(JSContext* cx,
                                                 Handle<WasmTagObject*> tag,
                                                 HandleObject stack,
                                                 HandleObject proto) {
  Rooted<WasmExceptionObject*> obj(
      cx, NewObjectWithGivenProto<WasmExceptionObject>(cx, proto));
  if (!obj) {
    return nullptr;
  }
  const TagType* tagType = tag->tagType();

  // Allocate the data buffer before initializing the object so that an OOM
  // does not result in a partially constructed object.
  uint8_t* data = (uint8_t*)js_calloc(tagType->tagSize());
  if (!data) {
    ReportOutOfMemory(cx);
    return nullptr;
  }

  MOZ_ASSERT(obj->isNewborn());
  obj->initFixedSlot(TAG_SLOT, ObjectValue(*tag));
  tagType->AddRef();
  obj->initFixedSlot(TYPE_SLOT, PrivateValue((void*)tagType));
  InitReservedSlot(obj, DATA_SLOT, data, tagType->tagSize(),
                   MemoryUse::WasmExceptionData);
  obj->initFixedSlot(STACK_SLOT, ObjectOrNullValue(stack));

  MOZ_ASSERT(!obj->isNewborn());

  return obj;
}

WasmExceptionObject* WasmExceptionObject::wrapJSValue(JSContext* cx,
                                                      HandleValue value) {
  Rooted<WasmNamespaceObject*> wasm(cx, WasmNamespaceObject::getOrCreate(cx));
  if (!wasm) {
    return nullptr;
  }

  Rooted<AnyRef> valueAnyRef(cx);
  if (!AnyRef::fromJSValue(cx, value, &valueAnyRef)) {
    return nullptr;
  }

  Rooted<WasmTagObject*> wrappedJSValueTag(cx, wasm->wrappedJSValueTag());
  WasmExceptionObject* exn =
      WasmExceptionObject::create(cx, wrappedJSValueTag, nullptr, nullptr);
  if (!exn) {
    return nullptr;
  }
  MOZ_ASSERT(exn->isWrappedJSValue());

  exn->initRefArg(WrappedJSValueTagType_ValueOffset, valueAnyRef);
  return exn;
}

bool WasmExceptionObject::isNewborn() const {
  MOZ_ASSERT(is<WasmExceptionObject>());
  return getReservedSlot(DATA_SLOT).isUndefined();
}

bool WasmExceptionObject::isWrappedJSValue() const {
  return tagType() == sWrappedJSValueTagType;
}

Value WasmExceptionObject::wrappedJSValue() const {
  MOZ_ASSERT(isWrappedJSValue());
  return loadRefArg(WrappedJSValueTagType_ValueOffset).toJSValue();
}

const JSPropertySpec WasmExceptionObject::properties[] = {
    JS_PSG("stack", WasmExceptionObject::getStack, 0),
    JS_STRING_SYM_PS(toStringTag, "WebAssembly.Exception", JSPROP_READONLY),
    JS_PS_END,
};

/* static */
bool WasmExceptionObject::isImpl(JSContext* cx, const CallArgs& args) {
  Rooted<WasmExceptionObject*> exnObj(
      cx, &args.thisv().toObject().as<WasmExceptionObject>());

  if (!args.requireAtLeast(cx, "WebAssembly.Exception.is", 1)) {
    return false;
  }

  if (!IsTag(args[0])) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_EXN_ARG);
    return false;
  }

  Rooted<WasmTagObject*> exnTag(cx,
                                &args.get(0).toObject().as<WasmTagObject>());
  args.rval().setBoolean(exnTag.get() == &exnObj->tag());

  return true;
}

/* static  */
bool WasmExceptionObject::isMethod(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsException, isImpl>(cx, args);
}

/* static */
bool WasmExceptionObject::getArgImpl(JSContext* cx, const CallArgs& args) {
  Rooted<WasmExceptionObject*> exnObj(
      cx, &args.thisv().toObject().as<WasmExceptionObject>());

  if (!args.requireAtLeast(cx, "WebAssembly.Exception.getArg", 2)) {
    return false;
  }

  if (!IsTag(args[0])) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_EXN_ARG);
    return false;
  }

  Rooted<WasmTagObject*> exnTag(cx,
                                &args.get(0).toObject().as<WasmTagObject>());
  if (exnTag.get() != &exnObj->tag()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_EXN_TAG);
    return false;
  }

  uint32_t index;
  if (!EnforceRangeU32(cx, args.get(1), "Exception", "getArg index", &index)) {
    return false;
  }

  const wasm::ValTypeVector& params = exnTag->valueTypes();
  if (index >= params.length()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr, JSMSG_WASM_BAD_RANGE,
                             "Exception", "getArg index");
    return false;
  }

  uint32_t offset = exnTag->tagType()->exceptionArgOffsets()[index];
  RootedValue result(cx);
  if (!exnObj->loadArg(cx, offset, params[index], &result)) {
    return false;
  }
  args.rval().set(result);
  return true;
}

/* static  */
bool WasmExceptionObject::getArg(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsException, getArgImpl>(cx, args);
}

/* static */
bool WasmExceptionObject::getStack_impl(JSContext* cx, const CallArgs& args) {
  Rooted<WasmExceptionObject*> exnObj(
      cx, &args.thisv().toObject().as<WasmExceptionObject>());
  RootedObject savedFrameObj(cx, exnObj->stack());
  if (!savedFrameObj) {
    args.rval().setUndefined();
    return true;
  }
  JSPrincipals* principals = exnObj->realm()->principals();
  RootedString stackString(cx);
  if (!BuildStackString(cx, principals, savedFrameObj, &stackString)) {
    return false;
  }
  args.rval().setString(stackString);
  return true;
}

/* static */
bool WasmExceptionObject::getStack(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsException, getStack_impl>(cx, args);
}

JSObject* WasmExceptionObject::stack() const {
  return getReservedSlot(STACK_SLOT).toObjectOrNull();
}

uint8_t* WasmExceptionObject::typedMem() const {
  return (uint8_t*)getReservedSlot(DATA_SLOT).toPrivate();
}

bool WasmExceptionObject::loadArg(JSContext* cx, size_t offset,
                                  wasm::ValType type,
                                  MutableHandleValue vp) const {
  if (!type.isExposable()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_VAL_TYPE);
    return false;
  }
  return ToJSValue(cx, typedMem() + offset, type, vp);
}

bool WasmExceptionObject::initArg(JSContext* cx, size_t offset,
                                  wasm::ValType type, HandleValue value) {
  // We use writeToTenuredHeapLocation below as WasmExceptionObject is always
  // tenured.
  MOZ_ASSERT(isTenured());

  if (!type.isExposable()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_VAL_TYPE);
    return false;
  }

  // Avoid rooting hazard of `this` being live across `fromJSValue`
  // which may GC.
  uint8_t* dest = typedMem() + offset;

  RootedVal val(cx);
  if (!Val::fromJSValue(cx, type, value, &val)) {
    return false;
  }
  val.get().writeToTenuredHeapLocation(dest);
  return true;
}

void WasmExceptionObject::initRefArg(size_t offset, wasm::AnyRef ref) {
  uint8_t* dest = typedMem() + offset;
  BarrieredInit(this, dest, ref);
}

wasm::AnyRef WasmExceptionObject::loadRefArg(size_t offset) const {
  uint8_t* src = typedMem() + offset;
  return *(AnyRef*)src;
}

const JSFunctionSpec WasmExceptionObject::methods[] = {
    JS_FN("is", WasmExceptionObject::isMethod, 1, JSPROP_ENUMERATE),
    JS_FN("getArg", WasmExceptionObject::getArg, 2, JSPROP_ENUMERATE),
    JS_FS_END,
};

const JSFunctionSpec WasmExceptionObject::static_methods[] = {
    JS_FS_END,
};

const TagType* WasmExceptionObject::tagType() const {
  return (const TagType*)getReservedSlot(TYPE_SLOT).toPrivate();
}

WasmTagObject& WasmExceptionObject::tag() const {
  return getReservedSlot(TAG_SLOT).toObject().as<WasmTagObject>();
}

// ============================================================================
// WebAssembly.Function and methods
#if defined(ENABLE_WASM_TYPE_REFLECTIONS) || defined(ENABLE_WASM_JSPI)
[[nodiscard]] static bool IsWasmFunction(HandleValue v) {
  if (!v.isObject()) {
    return false;
  }
  if (!v.toObject().is<JSFunction>()) {
    return false;
  }
  return v.toObject().as<JSFunction>().isWasm();
}
#endif  // ENABLE_WASM_TYPE_REFLECTIONS || ENABLE_WASM_JSPI

#ifdef ENABLE_WASM_TYPE_REFLECTIONS
static JSObject* CreateWasmFunctionPrototype(JSContext* cx, JSProtoKey key) {
  // WasmFunction's prototype should inherit from JSFunction's prototype.
  RootedObject jsProto(cx, &cx->global()->getFunctionPrototype());
  return GlobalObject::createBlankPrototypeInheriting(cx, &PlainObject::class_,
                                                      jsProto);
}

bool WasmFunctionTypeImpl(JSContext* cx, const CallArgs& args) {
  RootedFunction function(cx, &args.thisv().toObject().as<JSFunction>());
  const FuncType& funcType = function->wasmTypeDef()->funcType();
  RootedObject typeObj(cx, FuncTypeToObject(cx, funcType));
  if (!typeObj) {
    return false;
  }
  args.rval().setObject(*typeObj);
  return true;
}

bool WasmFunctionType(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  return CallNonGenericMethod<IsWasmFunction, WasmFunctionTypeImpl>(cx, args);
}

static JSFunction* WasmFunctionCreate(JSContext* cx, HandleObject func,
                                      wasm::ValTypeVector&& params,
                                      wasm::ValTypeVector&& results,
                                      HandleObject proto) {
  MOZ_ASSERT(IsCallableNonCCW(ObjectValue(*func)));

  // We want to import the function to a wasm module and then export it again so
  // that it behaves exactly like a normal wasm function and can be used like
  // one in wasm tables. We synthesize such a module below, instantiate it, and
  // then return the exported function as the result.
  FeatureOptions options;
  SharedCompileArgs compileArgs =
      CompileArgs::buildAndReport(cx, ScriptedCaller::selfHosted(cx), options);
  if (!compileArgs) {
    return nullptr;
  }

  MutableModuleMetadata moduleMeta = js_new<ModuleMetadata>();
  if (!moduleMeta || !moduleMeta->init(*compileArgs)) {
    return nullptr;
  }
  MutableCodeMetadata codeMeta = moduleMeta->codeMeta;
  CompilerEnvironment compilerEnv(CompileMode::Once, Tier::Optimized,
                                  DebugEnabled::False);
  compilerEnv.computeParameters();

  FuncType funcType = FuncType(std::move(params), std::move(results));
  if (!codeMeta->types->addType(std::move(funcType))) {
    return nullptr;
  }

  // Add an (import (func ...))
  FuncDesc funcDesc = FuncDesc(0);
  if (!codeMeta->funcs.append(funcDesc)) {
    return nullptr;
  }
  codeMeta->numFuncImports = 1;
  codeMeta->funcImportsAreJS = true;

  // Add an (export (func 0))
  codeMeta->funcs[0].declareFuncExported(/* eager */ true,
                                         /* canRefFunc */ true);

  // We will be looking up and using the function in the future by index so the
  // name doesn't matter.
  CacheableName fieldName;
  if (!moduleMeta->exports.emplaceBack(std::move(fieldName), 0,
                                       DefinitionKind::Function)) {
    return nullptr;
  }

  if (!moduleMeta->prepareForCompile(compilerEnv.mode())) {
    return nullptr;
  }

  ModuleGenerator mg(*codeMeta, compilerEnv, compilerEnv.initialState(),
                     nullptr, nullptr, nullptr);
  if (!mg.initializeCompleteTier()) {
    return nullptr;
  }
  // We're not compiling any function definitions.
  if (!mg.finishFuncDefs()) {
    return nullptr;
  }
  SharedModule module = mg.finishModule(BytecodeBufferOrSource(), *moduleMeta,
                                        /*maybeCompleteTier2Listener=*/nullptr);
  if (!module) {
    return nullptr;
  }

  // Instantiate the module.
  Rooted<ImportValues> imports(cx);
  if (!imports.get().funcs.append(func)) {
    return nullptr;
  }
  Rooted<WasmInstanceObject*> instance(cx);
  if (!module->instantiate(cx, imports.get(), nullptr, &instance)) {
    MOZ_ASSERT(cx->isThrowingOutOfMemory());
    return nullptr;
  }

  // Get the exported function which wraps the JS function to return.
  RootedFunction wasmFunc(cx);
  if (!instance->getExportedFunction(cx, instance, 0, &wasmFunc)) {
    return nullptr;
  }
  return wasmFunc;
}

bool WasmFunctionConstruct(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);

  if (!ThrowIfNotConstructing(cx, args, "WebAssembly.Function")) {
    return false;
  }

  if (!args.requireAtLeast(cx, "WebAssembly.Function", 2)) {
    return false;
  }

  if (!args[0].isObject()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_DESC_ARG, "function");
    return false;
  }
  RootedObject typeObj(cx, &args[0].toObject());

  // Extract properties in lexicographic order per spec.

  RootedValue parametersVal(cx);
  if (!JS_GetProperty(cx, typeObj, "parameters", &parametersVal)) {
    return false;
  }

  ValTypeVector params;
  if (!ParseValTypes(cx, parametersVal, params)) {
    return false;
  }
  if (params.length() > MaxParams) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_FUNCTION_TYPE, "parameters");
    return false;
  }

  RootedValue resultsVal(cx);
  if (!JS_GetProperty(cx, typeObj, "results", &resultsVal)) {
    return false;
  }

  ValTypeVector results;
  if (!ParseValTypes(cx, resultsVal, results)) {
    return false;
  }
  if (results.length() > MaxResults) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_FUNCTION_TYPE, "results");
    return false;
  }

  // Get the target function

  if (!IsCallableNonCCW(args[1])) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_FUNCTION_VALUE, "second");
    return false;
  }
  RootedObject func(cx, &args[1].toObject());

  RootedObject proto(
      cx, GetWasmConstructorPrototype(cx, args, JSProto_WasmFunction));
  if (!proto) {
    ReportOutOfMemory(cx);
    return false;
  }

  RootedFunction wasmFunc(cx, WasmFunctionCreate(cx, func, std::move(params),
                                                 std::move(results), proto));
  if (!wasmFunc) {
    ReportOutOfMemory(cx);
    return false;
  }
  args.rval().setObject(*wasmFunc);

  return true;
}

static constexpr char WasmFunctionName[] = "Function";

static JSObject* CreateWasmFunctionConstructor(JSContext* cx, JSProtoKey key) {
  RootedObject proto(cx, &cx->global()->getFunctionConstructor());

  Rooted<JSAtom*> className(
      cx, Atomize(cx, WasmFunctionName, strlen(WasmFunctionName)));
  if (!className) {
    return nullptr;
  }
  return NewFunctionWithProto(cx, WasmFunctionConstruct, 1,
                              FunctionFlags::NATIVE_CTOR, nullptr, className,
                              proto, gc::AllocKind::FUNCTION, TenuredObject);
}

const JSFunctionSpec WasmFunctionMethods[] = {
    JS_FN("type", WasmFunctionType, 0, 0),
    JS_FS_END,
};

const ClassSpec WasmFunctionClassSpec = {
    CreateWasmFunctionConstructor,
    CreateWasmFunctionPrototype,
    nullptr,
    nullptr,
    WasmFunctionMethods,
    nullptr,
    nullptr,
    ClassSpec::DontDefineConstructor,
};

const JSClass js::WasmFunctionClass = {
    "WebAssembly.Function",
    0,
    JS_NULL_CLASS_OPS,
    &WasmFunctionClassSpec,
};

#endif

// ============================================================================
// WebAssembly class and static methods

static bool WebAssembly_toSource(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  args.rval().setString(cx->names().WebAssembly);
  return true;
}

static bool RejectWithPendingException(JSContext* cx,
                                       Handle<PromiseObject*> promise) {
  if (!cx->isExceptionPending()) {
    return false;
  }

  RootedValue rejectionValue(cx);
  if (!GetAndClearException(cx, &rejectionValue)) {
    return false;
  }

  return PromiseObject::reject(cx, promise, rejectionValue);
}

static bool Reject(JSContext* cx, const CompileArgs& args,
                   Handle<PromiseObject*> promise, const UniqueChars& error) {
  if (!error) {
    ThrowCompileOutOfMemory(cx);
    return RejectWithPendingException(cx, promise);
  }

  RootedObject stack(cx, promise->allocationSite());
  RootedObject errorObj(cx);
  if (!CreateCompileError(cx, args.scriptedCaller, stack, error.get(),
                          &errorObj)) {
    return false;
  }

  RootedValue rejectionValue(cx, ObjectValue(*errorObj));
  return PromiseObject::reject(cx, promise, rejectionValue);
}

static bool RejectWithOutOfMemory(JSContext* cx,
                                  Handle<PromiseObject*> promise) {
  ReportOutOfMemory(cx);
  return RejectWithPendingException(cx, promise);
}

static void LogAsync(JSContext* cx, const char* funcName,
                     const Module& module) {
  Log(cx, "async %s succeeded%s", funcName,
      module.loggingDeserialized() ? " (loaded from cache)" : "");
}

enum class Ret { Pair, Instance };

class AsyncInstantiateTask : public OffThreadPromiseTask {
  SharedModule module_;
  PersistentRooted<ImportValues> imports_;
  Ret ret_;

 public:
  AsyncInstantiateTask(JSContext* cx, const Module& module, Ret ret,
                       Handle<PromiseObject*> promise)
      : OffThreadPromiseTask(cx, promise),
        module_(&module),
        imports_(cx),
        ret_(ret) {}

  ImportValues& imports() { return imports_.get(); }

  bool resolve(JSContext* cx, Handle<PromiseObject*> promise) override {
    RootedObject instanceProto(
        cx, &cx->global()->getPrototype(JSProto_WasmInstance));

    Rooted<WasmInstanceObject*> instanceObj(cx);
    if (!module_->instantiate(cx, imports_.get(), instanceProto,
                              &instanceObj)) {
      return RejectWithPendingException(cx, promise);
    }

    RootedValue resolutionValue(cx);
    if (ret_ == Ret::Instance) {
      resolutionValue = ObjectValue(*instanceObj);
    } else {
      RootedObject resultObj(cx, JS_NewPlainObject(cx));
      if (!resultObj) {
        return RejectWithPendingException(cx, promise);
      }

      RootedObject moduleProto(cx,
                               &cx->global()->getPrototype(JSProto_WasmModule));
      RootedObject moduleObj(
          cx, WasmModuleObject::create(cx, *module_, moduleProto));
      if (!moduleObj) {
        return RejectWithPendingException(cx, promise);
      }

      RootedValue val(cx, ObjectValue(*moduleObj));
      if (!JS_DefineProperty(cx, resultObj, "module", val, JSPROP_ENUMERATE)) {
        return RejectWithPendingException(cx, promise);
      }

      val = ObjectValue(*instanceObj);
      if (!JS_DefineProperty(cx, resultObj, "instance", val,
                             JSPROP_ENUMERATE)) {
        return RejectWithPendingException(cx, promise);
      }

      resolutionValue = ObjectValue(*resultObj);
    }

    if (!PromiseObject::resolve(cx, promise, resolutionValue)) {
      return RejectWithPendingException(cx, promise);
    }

    LogAsync(cx, "instantiate", *module_);
    return true;
  }
};

static bool AsyncInstantiate(JSContext* cx, const Module& module,
                             HandleObject importObj, Ret ret,
                             Handle<PromiseObject*> promise) {
  auto task = js::MakeUnique<AsyncInstantiateTask>(cx, module, ret, promise);
  if (!task || !task->init(cx)) {
    return RejectWithOutOfMemory(cx, promise);
  }

  if (!GetImports(cx, module, importObj, &task->imports())) {
    return RejectWithPendingException(cx, promise);
  }

  OffThreadPromiseTask::DispatchResolveAndDestroy(std::move(task));
  return true;
}

static bool ResolveCompile(JSContext* cx, const Module& module,
                           Handle<PromiseObject*> promise) {
  RootedObject proto(cx, &cx->global()->getPrototype(JSProto_WasmModule));
  RootedObject moduleObj(cx, WasmModuleObject::create(cx, module, proto));
  if (!moduleObj) {
    return RejectWithPendingException(cx, promise);
  }

  RootedValue resolutionValue(cx, ObjectValue(*moduleObj));
  if (!PromiseObject::resolve(cx, promise, resolutionValue)) {
    return RejectWithPendingException(cx, promise);
  }

  LogAsync(cx, "compile", module);
  return true;
}

struct CompileBufferTask : PromiseHelperTask {
  BytecodeBuffer bytecode;
  SharedCompileArgs compileArgs;
  UniqueChars error;
  UniqueCharsVector warnings;
  SharedModule module;
  bool instantiate;
  PersistentRootedObject importObj;

  CompileBufferTask(JSContext* cx, Handle<PromiseObject*> promise,
                    HandleObject importObj)
      : PromiseHelperTask(cx, promise),
        instantiate(true),
        importObj(cx, importObj) {}

  CompileBufferTask(JSContext* cx, Handle<PromiseObject*> promise)
      : PromiseHelperTask(cx, promise), instantiate(false) {}

  bool init(JSContext* cx, const FeatureOptions& options,
            const char* introducer) {
    compileArgs = InitCompileArgs(cx, options, introducer);
    if (!compileArgs) {
      return false;
    }
    return PromiseHelperTask::init(cx);
  }

  void execute() override {
    module =
        CompileModule(*compileArgs, BytecodeBufferOrSource(std::move(bytecode)),
                      &error, &warnings, nullptr);
  }

  bool resolve(JSContext* cx, Handle<PromiseObject*> promise) override {
    if (!ReportCompileWarnings(cx, warnings)) {
      return false;
    }
    if (!module) {
      return Reject(cx, *compileArgs, promise, error);
    }
    if (instantiate) {
      return AsyncInstantiate(cx, *module, importObj, Ret::Pair, promise);
    }
    return ResolveCompile(cx, *module, promise);
  }
};

static bool RejectWithPendingException(JSContext* cx,
                                       Handle<PromiseObject*> promise,
                                       CallArgs& callArgs) {
  if (!RejectWithPendingException(cx, promise)) {
    return false;
  }

  callArgs.rval().setObject(*promise);
  return true;
}

static bool EnsurePromiseSupport(JSContext* cx) {
  if (!cx->runtime()->offThreadPromiseState.ref().initialized()) {
    JS_ReportErrorASCII(
        cx, "WebAssembly Promise APIs not supported in this runtime.");
    return false;
  }
  return true;
}

static bool WebAssembly_compile(JSContext* cx, unsigned argc, Value* vp) {
  if (!EnsurePromiseSupport(cx)) {
    return false;
  }

  Log(cx, "async compile() started");

  Rooted<PromiseObject*> promise(cx, PromiseObject::createSkippingExecutor(cx));
  if (!promise) {
    return false;
  }

  CallArgs callArgs = CallArgsFromVp(argc, vp);

#if defined(__EMSCRIPTEN__)
  if (wasm::UseHostPassthrough()) {
    if (!callArgs.requireAtLeast(cx, "WebAssembly.compile", 1)) {
      return RejectWithPendingException(cx, promise, callArgs);
    }
    RootedValue arg0(cx, callArgs.get(0));
    if (!arg0.isObject()) {
      JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                               JSMSG_WASM_BAD_BUF_MOD_ARG);
      return RejectWithPendingException(cx, promise, callArgs);
    }
    RootedObject bytesObj(cx, &arg0.toObject());
    int handle = HostCompileBytes(cx, bytesObj);
    RootedObject moduleObj(cx);
    if (handle >= 0) {
      moduleObj = HostMakeModuleObject(cx, handle);
    }
    if (!moduleObj) {
      return RejectWithPendingException(cx, promise, callArgs);
    }
    RootedValue result(cx, ObjectValue(*moduleObj));
    if (!PromiseObject::resolve(cx, promise, result)) {
      return RejectWithPendingException(cx, promise, callArgs);
    }
    callArgs.rval().setObject(*promise);
    return true;
  }
#endif

  JS::RootedVector<JSString*> parameterStrings(cx);
  JS::RootedVector<Value> parameterArgs(cx);
  bool canCompileStrings = false;
  if (!cx->isRuntimeCodeGenEnabled(JS::RuntimeCode::WASM, nullptr,
                                   JS::CompilationType::Undefined,
                                   parameterStrings, nullptr, parameterArgs,
                                   NullHandleValue, &canCompileStrings)) {
    return RejectWithPendingException(cx, promise, callArgs);
  }
  if (!canCompileStrings) {
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_CSP_BLOCKED_WASM, "WebAssembly.compile");
    return RejectWithPendingException(cx, promise, callArgs);
  }

  auto task = cx->make_unique<CompileBufferTask>(cx, promise);
  if (!task) {
    return false;
  }

  if (!callArgs.requireAtLeast(cx, "WebAssembly.compile", 1)) {
    return RejectWithPendingException(cx, promise, callArgs);
  }

  FeatureOptions options;
  if (!options.init(cx, callArgs.get(1))) {
    return RejectWithPendingException(cx, promise, callArgs);
  }

  if (!callArgs[0].isObject()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_BUF_ARG);
    return RejectWithPendingException(cx, promise, callArgs);
  }

  Rooted<JSObject*> sourceObj(cx, &callArgs[0].toObject());
  if (!GetBytecodeBuffer(cx, sourceObj, JSMSG_WASM_BAD_BUF_ARG,
                         &task->bytecode)) {
    return RejectWithPendingException(cx, promise, callArgs);
  }

  if (!task->init(cx, options, "WebAssembly.compile")) {
    return false;
  }

  if (!StartOffThreadPromiseHelperTask(cx, std::move(task))) {
    return false;
  }

  callArgs.rval().setObject(*promise);
  return true;
}

static bool GetInstantiateArgs(JSContext* cx, const CallArgs& callArgs,
                               MutableHandleObject firstArg,
                               MutableHandleObject importObj,
                               MutableHandleValue featureOptions) {
  if (!callArgs.requireAtLeast(cx, "WebAssembly.instantiate", 1)) {
    return false;
  }

  if (!callArgs[0].isObject()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_BUF_MOD_ARG);
    return false;
  }

  firstArg.set(&callArgs[0].toObject());

  if (!GetImportArg(cx, callArgs.get(1), importObj)) {
    return false;
  }

  featureOptions.set(callArgs.get(2));
  return true;
}

#if defined(__EMSCRIPTEN__)
#  include <emscripten.h>
#  include "js/ArrayBuffer.h"        // JS::NewArrayBufferWithUserOwnedContents
#  include "js/SharedArrayBuffer.h"  // JS::NewSharedArrayBuffer
#  include "js/CallAndConstruct.h"   // JS_CallFunctionValue
// -- Host-passthrough wasm (wasm32-emscripten) ------------------------------
// There is no in-process wasm compiler (the JIT is disabled), so guest
// WebAssembly is compiled+instantiated on the HOST browser's WebAssembly engine
// via the wasm-host-bridge.js js-library; its exports are bridged back as native
// functions, its (function) imports are bridged out to guest callbacks, and its
// linear memory is mirrored into a guest buffer (see HostMakeMemoryWrapper). The
// extern "C" declarations live near WasmModuleObject::construct (used earlier).

// Called from JIT'd wasm (the "m"."call" import) to invoke a cached compiled
// callee. The caller already marshalled the callee's args into gWJScratch, so we
// just run the callee's wasm (which reads them + writes its result to
// gWJScratch[result]). Returns the callee's deopt flag; a callee miss recorded
// in gWJMissSite propagates up to the outermost WasmJitRunCall, which fills it.
extern "C" EMSCRIPTEN_KEEPALIVE double wasmjit_invoke(int site, int argc) {
  if (site < 0 || uint32_t(site) >= gWJSiteCount) return 1.0;
  if (gWJCallFn[site] == 0 || gWJCallHandle[site] < 0) return 1.0;
  if (uint32_t(argc) != gWJCallNargs[site]) return 1.0;
  double sp = double(uintptr_t(static_cast<void*>(gWJScratch)));
  return wasmhost_call(gWJCallHandle[site], 0, &sp, 1);
}

// Mode VS no-restart helper. Completes one operation the inline fast path could
// not handle (type/shape miss) using full engine semantics, so the JIT'd
// function continues instead of restarting (heap writes happen exactly once).
// Operands are read from gWJHelpA/B/C; the result is written to
// gWJScratch[kWJResultSlot]. Returns 0.0 on success, 1.0 if the op threw (the
// caller propagates the pending exception). Engine calls here may allocate/move
// objects; Mode VS keeps all live Values in the GC-traced gWJFrameMem, so the
// object pointers the running wasm holds are updated in place by a moving GC.
extern "C" EMSCRIPTEN_KEEPALIVE double wjhelp(double kindF, double siteF) {
  JSContext* cx = js::TlsContext.get();
  if (!cx) return 1.0;
  uint32_t kind = uint32_t(kindF);
  uint32_t site = uint32_t(siteF);
  gWJHelpCalls++;
  if (kind < 32) gWJHelpKind[kind]++;
  if (gWJCurEntry) gWJCurEntry->helperCalls++;  // attribute to the running Mode VS fn
  JS::RootedValue a(cx, JS::Value::fromRawBits(gWJHelpA));
  JS::RootedValue b(cx, JS::Value::fromRawBits(gWJHelpB));
  JS::RootedValue res(cx);
  if (a.isString() || b.isString()) gWJVSStrOps++;  // JIT'd Mode VS string work
  switch (kind) {
    case WJH_ADD: if (!js::AddValues(cx, &a, &b, &res)) return 1.0; break;
    case WJH_SUB: if (!js::SubValues(cx, &a, &b, &res)) return 1.0; break;
    case WJH_MUL: if (!js::MulValues(cx, &a, &b, &res)) return 1.0; break;
    case WJH_DIV: if (!js::DivValues(cx, &a, &b, &res)) return 1.0; break;
    case WJH_MOD: if (!js::ModValues(cx, &a, &b, &res)) return 1.0; break;
    case WJH_NEG: {
      b.setInt32(-1);
      if (!js::MulValues(cx, &a, &b, &res)) return 1.0;
      break;
    }
    case WJH_BITOR: if (!js::BitOr(cx, &a, &b, &res)) return 1.0; break;
    case WJH_BITAND: if (!js::BitAnd(cx, &a, &b, &res)) return 1.0; break;
    case WJH_BITXOR: if (!js::BitXor(cx, &a, &b, &res)) return 1.0; break;
    case WJH_LSH: if (!js::BitLsh(cx, &a, &b, &res)) return 1.0; break;
    case WJH_RSH: if (!js::BitRsh(cx, &a, &b, &res)) return 1.0; break;
    case WJH_URSH: if (!js::UrshValues(cx, &a, &b, &res)) return 1.0; break;
    case WJH_BITNOT: if (!js::BitNot(cx, &a, &res)) return 1.0; break;
    case WJH_INC:
    case WJH_DEC: {
      double d;
      if (!JS::ToNumber(cx, a, &d)) return 1.0;  // ToNumeric+/-1 (no BigInt here)
      res.setNumber(kind == WJH_INC ? d + 1 : d - 1);
      break;
    }
    case WJH_TONUMBER: {  // UNBOX ensureF64 slow path: ToNumber(a) -> number
      double d;
      if (!JS::ToNumber(cx, a, &d)) return 1.0;
      res.setNumber(d);
      break;
    }
    case WJH_LT:
    case WJH_LE:
    case WJH_GT:
    case WJH_GE: {
      bool r;
      bool ok = kind == WJH_LT   ? js::LessThan(cx, &a, &b, &r)
                : kind == WJH_LE ? js::LessThanOrEqual(cx, &a, &b, &r)
                : kind == WJH_GT ? js::GreaterThan(cx, &a, &b, &r)
                                 : js::GreaterThanOrEqual(cx, &a, &b, &r);
      if (!ok) return 1.0;
      res.setBoolean(r);
      break;
    }
    case WJH_EQ:
    case WJH_NE: {
      bool r;
      if (!js::LooselyEqual(cx, a, b, &r)) return 1.0;
      res.setBoolean(kind == WJH_EQ ? r : !r);
      break;
    }
    case WJH_STRICTEQ:
    case WJH_STRICTNE: {
      bool r;
      if (!js::StrictlyEqual(cx, a, b, &r)) return 1.0;
      res.setBoolean(kind == WJH_STRICTEQ ? r : !r);
      break;
    }
    case WJH_GETPROP: {
      if (getenv("GECKO_WJVS_PHCOUNT")) {
        static uint64_t n = 0;
        if ((n++ % 100000) == 0)
          fprintf(stderr, "[ph-helper-taken] n=%llu (off-shape field read via helper)\n",
                  (unsigned long long)n);
      }
      JS::RootedObject obj(cx, JS::ToObject(cx, a));
      if (!obj) return 1.0;
      JSScript* s = gWJSites[site].script;
      js::PropertyName* name = s->getName(s->offsetToPC(gWJSites[site].pcOff));
      if (!js::GetProperty(cx, obj, a, name, &res)) return 1.0;
      if (a.isObject()) {
        gWJMissObj = a.asRawBits();
        WJFillIC(site);  // monomorphic data prop -> inline next time
      }
      break;
    }
    case WJH_NEWOBJECT: {
      // ALLOC: create the object for this NewObject/NewInit site (template/literal shape).
      JSScript* sc = gWJSites[site].script;
      JS::RootedScript rs(cx, sc);
      JSObject* obj = js::NewObjectOperation(cx, rs, sc->offsetToPC(gWJSites[site].pcOff));
      if (!obj) return 1.0;
      res.setObject(*obj);
      break;
    }
    case WJH_INITPROP: {
      // ALLOC: define a data property on obj=a with value=b (constructor field-init / literal).
      // Leaves the object on the stack (InitProp's no-pop-of-obj semantics).
      if (!a.isObject()) return 1.0;
      JS::RootedObject obj(cx, &a.toObject());
      JSScript* sc = gWJSites[site].script;
      jsbytecode* ipc = sc->offsetToPC(gWJSites[site].pcOff);
      JS::RootedId id(cx, js::NameToId(sc->getName(ipc)));
      gWJInitHelperCalls++;
      if (getenv("GECKO_DEBUG_JIT") && (gWJInitHelperCalls % 1000 == 1)) {
        fprintf(stderr, "[initprop-helper] calls=%llu\n", (unsigned long long)gWJInitHelperCalls);
        fflush(stderr);
      }
      uint32_t fromShape = uint32_t(uintptr_t(obj->shape()));  // capture BEFORE the add (IC guard)
      if (!js::DefineDataProperty(cx, obj, id, b, js::GetInitDataPropAttrs(JSOp(*ipc)))) {
        return 1.0;
      }
      // Fill the inline add-property IC for a native, non-dictionary, FIXED-slot add: next time the
      // emitted code can do shape-guard + slot-store + shape-set inline (gated on no incremental GC).
      if (site < kWJMaxSites && obj->is<js::NativeObject>()) {
        js::NativeObject& nobj = obj->as<js::NativeObject>();
        if (!nobj.inDictionaryMode()) {
          mozilla::Maybe<js::PropertyInfo> prop = nobj.lookupPure(id);
          if (prop.isSome() && prop->isDataProperty() &&
              prop->slot() < nobj.numFixedSlots()) {
            gWJICTable[2 * site] = fromShape;
            gWJICTable[2 * site + 1] = 16 + prop->slot() * 8;  // fixed-slot byte offset
            gWJInitToShape[site] =
                uint32_t(uintptr_t(nobj.shape()));  // toShape; set LAST = enables the fast path
          }
        }
      }
      res.set(a);
      break;
    }
    case WJH_SETPROP: {
      JS::RootedObject obj(cx, JS::ToObject(cx, a));
      if (!obj) return 1.0;
      JSScript* s = gWJSites[site].script;
      js::PropertyName* name = s->getName(s->offsetToPC(gWJSites[site].pcOff));
      if (getenv("GECKO_WJVS_SPDBG")) {
        static int n = 0;
        if (n++ < 25)
          fprintf(stderr, "[sp] aIsObj=%d bKind=%s bNum=%g bBits=%#llx\n",
                  a.isObject(),
                  b.isNull() ? "null" : b.isObject() ? "obj" : b.isNumber() ? "num"
                  : b.isUndefined() ? "undef" : "other",
                  b.isNumber() ? b.toNumber() : -1.0,
                  (unsigned long long)b.asRawBits());
      }
      if (!js::SetProperty(cx, obj, name, b)) return 1.0;
      res.set(b);  // SetProp/SetElem leave the assigned value on the stack
      if (a.isObject()) {
        gWJMissObj = a.asRawBits();
        WJFillIC(site);
      }
      break;
    }
    case WJH_GETELEM: {
      JS::RootedObject obj(cx, JS::ToObject(cx, a));
      if (!obj) return 1.0;
      JS::RootedId id(cx);
      if (!js::ToPropertyKey(cx, b, &id)) return 1.0;
      if (!js::GetProperty(cx, obj, a, id, &res)) return 1.0;
      if (a.isObject()) {
        gWJMissObj = a.asRawBits();
        WJFillIC(site);
      }
      break;
    }
    case WJH_SETELEM: {
      JS::RootedObject obj(cx, JS::ToObject(cx, a));
      if (!obj) return 1.0;
      JS::RootedValue cval(cx, JS::Value::fromRawBits(gWJHelpC));
      bool strict = gWJSites[site].script->strict();
      if (!js::SetObjectElement(cx, obj, b, cval, strict)) return 1.0;
      res.set(cval);
      if (a.isObject()) {
        gWJMissObj = a.asRawBits();
        WJFillIC(site);
      }
      break;
    }
    case WJH_FUNCTIONTHIS: {
      // a = the raw this value; sloppy non-object boxing (strict is inlined).
      JSObject* o = js::BoxNonStrictThis(cx, a);
      if (!o) return 1.0;
      res.setObject(*o);
      break;
    }
    case WJH_GETGNAME: {
      // Resolve a global name: global LEXICAL env (let/const/class) first, then
      // the global OBJECT (var/function) -- matching the interpreter's GetGName.
      JSScript* sc = gWJSites[site].script;
      jsbytecode* sp = sc->offsetToPC(gWJSites[site].pcOff);
      JS::Rooted<js::PropertyName*> name(cx, sc->getName(sp));
      JS::RootedId id(cx, js::NameToId(name));
      js::GlobalLexicalEnvironmentObject& lex = cx->global()->lexicalEnvironment();
      mozilla::Maybe<js::PropertyInfo> lp = lex.lookupPure(id);
      if (lp.isSome()) {
        if (!lp->isDataProperty()) return 1.0;  // accessor global lexical: rare
        JS::Value v = lex.getSlot(lp->slot());
        if (v.isMagic(JS_UNINITIALIZED_LEXICAL)) {
          js::ReportRuntimeLexicalError(cx, JSMSG_UNINITIALIZED_LEXICAL, name);
          return 1.0;
        }
        res.set(v);
        break;
      }
      JS::RootedObject global(cx, cx->global());
      JS::RootedValue recv(cx, JS::ObjectValue(*global));
      if (!js::GetProperty(cx, global, recv, name, &res)) return 1.0;
      WJFillIC(site);  // cache {holder,shape,slot} so the inline VS path warms up
      break;
    }
    case WJH_CALL: {
      // Generic call (any callee, runs in the interpreter): a = callee; this in
      // gWJScratch[kWJThisSlot]; args in gWJScratch[0..argc). Used on a callee-
      // guard miss or when a cached callee could not complete in wasm.
      uint32_t argc = gWJCallArgc[site];
      JS::RootedValue thisv(cx, JS::Value::fromRawBits(gWJScratch[kWJThisSlot]));
      JS::RootedValueVector argv(cx);
      if (!argv.reserve(argc)) return 1.0;
      for (uint32_t i = 0; i < argc; i++) {
        argv.infallibleAppend(JS::Value::fromRawBits(gWJScratch[i]));
      }
      if (!JS::Call(cx, thisv, a, JS::HandleValueArray(argv), &res)) return 1.0;
      gWJMissObj = a.asRawBits();
      WJFillIC(site);  // cache the callee's table index (if it is call_indirect-able)
      break;
    }
    case WJH_IONCALL: {
      // reused-Ion non-inlined call: a = callee fn (gWJHelpA); argc = gWJHelpB
      // (raw); this in gWJScratch[kWJThisSlot]; args in gWJScratch[0..argc). Runs
      // the callee in the interpreter; result -> res (stored to kWJResultSlot).
      uint32_t argc = uint32_t(gWJHelpB);
      JS::RootedValue thisv(cx, JS::Value::fromRawBits(gWJScratch[kWJThisSlot]));
      JS::RootedValueVector argv(cx);
      if (!argv.reserve(argc)) return 1.0;
      for (uint32_t i = 0; i < argc; i++) {
        argv.infallibleAppend(JS::Value::fromRawBits(gWJScratch[i]));
      }
      if (!JS::Call(cx, thisv, a, JS::HandleValueArray(argv), &res)) return 1.0;
      break;
    }
    case WJH_CONSTRUCT: {
      // `new callee(args)`: a = callee (gWJHelpA); argc = gWJHelpB; args in
      // gWJScratch[0..argc). Construct in the interpreter; the new object -> res.
      uint32_t argc = uint32_t(gWJHelpB);
      JS::RootedValueVector argv(cx);
      if (!argv.reserve(argc)) return 1.0;
      for (uint32_t i = 0; i < argc; i++) {
        argv.infallibleAppend(JS::Value::fromRawBits(gWJScratch[i]));
      }
      JS::RootedObject obj(cx);
      if (!JS::Construct(cx, a, JS::HandleValueArray(argv), &obj)) return 1.0;
      res.setObject(*obj);
      break;
    }
    case WJH_METHCALL: {
      // Megamorphic-dispatch fallback: resolve recv[name] then call it, instead of
      // deopting the whole (possibly stateful) caller. name = gWJHelpA (a raw
      // PropertyName*); argc = gWJHelpB; recv in gWJScratch[kWJThisSlot]; args in
      // gWJScratch[0..argc). recv must be an object (the dispatch guards that).
      uint32_t argc = uint32_t(gWJHelpB);
      JS::RootedValue recv(cx, JS::Value::fromRawBits(gWJScratch[kWJThisSlot]));
      JS::RootedObject obj(cx, JS::ToObject(cx, recv));
      if (!obj) return 1.0;
      js::PropertyName* name =
          reinterpret_cast<js::PropertyName*>(uintptr_t(gWJHelpA));
      JS::RootedValue callee(cx);
      if (!js::GetProperty(cx, obj, recv, name, &callee)) return 1.0;
      JS::RootedValueVector argv(cx);
      if (!argv.reserve(argc)) return 1.0;
      for (uint32_t i = 0; i < argc; i++) {
        argv.infallibleAppend(JS::Value::fromRawBits(gWJScratch[i]));
      }
      if (!JS::Call(cx, recv, callee, JS::HandleValueArray(argv), &res)) return 1.0;
      if (getenv("GECKO_WJVS_MCDBG")) {
        static int n = 0;
        if (n++ < 30) {
          double a0 = argc > 0 && argv[0].isNumber() ? argv[0].toNumber() : -999;
          fprintf(stderr, "[mc] argc=%u arg0=%g resKind=%s resNum=%g\n", argc, a0,
                  res.isNull() ? "null" : res.isObject() ? "obj" : res.isNumber() ? "num" : "other",
                  res.isNumber() ? res.toNumber() : -1);
        }
      }
      break;
    }
    case WJH_GETALIASED: {
      // Read a closed-over var. gWJCurEnv is this function's environmentChain (set
      // by WasmJitRunCall; the function is never reached via the fast call path).
      // Walk `hops` enclosing environments, then read the aliased slot -- exactly
      // InterpreterFrame::aliasedEnvironment(ec).aliasedBinding(ec).
      JSObject* env = gWJCurEnv;
      if (!env) return 1.0;
      JSScript* sc = gWJSites[site].script;
      jsbytecode* sp = sc->offsetToPC(gWJSites[site].pcOff);
      js::EnvironmentCoordinate ec(sp);
      for (unsigned i = ec.hops(); i; i--) {
        env = &env->as<js::EnvironmentObject>().enclosingEnvironment();
      }
      res.set(env->as<js::EnvironmentObject>().aliasedBinding(ec));
      break;
    }
    case WJH_POSTBARRIER: {
      // The inline SetProp fast path already wrote `b` into `a`'s own fixed slot.
      // Run the generational post-write barrier for that obj->value edge (a is the
      // receiver object; b is the stored value). No allocation that moves cells.
      if (a.isObject() && b.isGCThing()) {
        JSObject* o = &a.toObject();
        if (o->isTenured()) {
          if (js::gc::StoreBuffer* sb = b.toGCThing()->storeBuffer()) {
            sb->putWholeCell(o);
          }
        }
      }
      return 0.0;  // never deopts; result slot is unused (value already stored)
    }
    case WJH_RESUME: {
      // PHASE F self-resume: finish the bailing fn in PBL. gWJHelpA=script low32 (raw),
      // gWJHelpB=basesp (frame slot base); args/locals live in gWJFrameMem[basesp..], this in
      // gWJScratch[kWJThisSlot], pc in gWJScratch[kWJResumePcSlot]. Result -> result slot.
      JSScript* s = reinterpret_cast<JSScript*>(uintptr_t(uint32_t(gWJHelpA)));
      uint32_t basesp = uint32_t(gWJHelpB);
      JSFunction* fun = s->function();
      if (!fun) return 1.0;
      uint32_t nargs = fun->nargs();
      uint32_t nfx = s->nfixed();
      uint32_t pcOff = uint32_t(gWJScratch[kWJResumePcSlot]);
      uint64_t thisBits = gWJScratch[kWJThisSlot];
      const JS::Value* fargs = reinterpret_cast<const JS::Value*>(&gWJFrameMem[basesp]);
      const uint64_t* flocals = &gWJFrameMem[basesp + nargs];
      uint64_t rbits = 0;
      gWJDeoptResumes++;
      if (getenv("GECKO_DEBUG_JIT")) {
        bool hasCall = false;
        for (jsbytecode* p = s->code(); p < s->codeEnd(); p += GetBytecodeLength(p)) {
          JSOp o = JSOp(*p);
          if (o == JSOp::Call || o == JSOp::CallContent || o == JSOp::CallIgnoresRv) {
            hasCall = true;
            break;
          }
        }
        static uint64_t sLeaf = 0, sNonleaf = 0, sFirstNonleaf = 0;
        if (hasCall) sNonleaf++; else sLeaf++;
        // print the first 3 resumes, the FIRST non-leaf resume, and every 20000th total
        if (gWJDeoptResumes <= 3 || (hasCall && sFirstNonleaf == 0) ||
            gWJDeoptResumes % 20000 == 0) {
          if (hasCall) sFirstNonleaf = gWJDeoptResumes;
          fprintf(stderr, "[resume-fired] n=%llu (leaf=%llu nonleaf=%llu) %s:%u pcOff=%u CALL=%d\n",
                  (unsigned long long)gWJDeoptResumes, (unsigned long long)sLeaf,
                  (unsigned long long)sNonleaf, s->filename() ? s->filename() : "?",
                  unsigned(s->lineno()), pcOff, hasCall);
          fflush(stderr);
        }
      }
      if (!js::pbl::WasmJitResumeViaPBL(cx, s, thisBits, fargs, nargs,
                                        fun->environment(), flocals, nfx, pcOff, &rbits)) {
        return 1.0;  // resumed PBL threw -> WJVSExcCheck propagates as deopt 2
      }
      gWJScratch[kWJResultSlot] = rbits;
      return 0.0;
    }
    default:
      return 1.0;  // GetGName etc.: not yet in Mode VS
  }
  gWJScratch[kWJResultSlot] = res.asRawBits();
  return 0.0;
}

// Trace the Mode VS frame stack + the call/helper scratch as GC roots so a moving
// GC triggered inside a helper updates the object pointers the running JIT code
// holds. Every slot in [0, gWJFrameSP) belongs to an active frame and was
// initialized to a valid Value on entry; gWJScratch/gWJHelp* are always valid
// Values (zero bits = DoubleValue(0)). Registered once via JS_AddExtraGCRootsTracer.
// extern "C" + KEEPALIVE (+ EXPORTED_FUNCTIONS) so it is a DEFINED export: an
// internal-linkage address-taken function gets GOT-indirected in this PIC build
// and resolves as an unresolved `env` import -> "missing function" abort when the
// GC calls it. A defined export resolves locally + has a real table slot.
extern "C" EMSCRIPTEN_KEEPALIVE void WJTraceRoots(JSTracer* trc, void*) {
  for (uint32_t i = 0; i < gWJFrameSP; i++) {
    JS::TraceRoot(trc, reinterpret_cast<JS::Value*>(&gWJFrameMem[i]), "wjframe");
  }
  for (uint32_t i = 0; i < 72; i++) {
    JS::TraceRoot(trc, reinterpret_cast<JS::Value*>(&gWJScratch[i]), "wjscratch");
  }
  JS::TraceRoot(trc, reinterpret_cast<JS::Value*>(&gWJHelpA), "wjhelpA");
  JS::TraceRoot(trc, reinterpret_cast<JS::Value*>(&gWJHelpB), "wjhelpB");
  JS::TraceRoot(trc, reinterpret_cast<JS::Value*>(&gWJHelpC), "wjhelpC");
  if (gWJCurEnv) JS::TraceRoot(trc, &gWJCurEnv, "wjcurenv");
}

// Guest JS functions supplied as wasm imports, kept alive and indexed by id so
// the host engine can call them back via wasmhost_invoke_import. Leaked for the
// process lifetime (an instance may be called at any later time).
static JS::PersistentRootedObject* gImportCallbacks = nullptr;
static uint32_t gImportCallbackCount = 0;

static int RegisterImportCallback(JSContext* cx, HandleValue fn) {
  if (!gImportCallbacks) {
    RootedObject obj(cx, JS_NewPlainObject(cx));
    if (!obj) {
      return -1;
    }
    gImportCallbacks = new JS::PersistentRootedObject(cx, obj);
  }
  RootedObject obj(cx, *gImportCallbacks);
  uint32_t id = gImportCallbackCount++;
  RootedValue v(cx, fn);
  if (!JS_SetElement(cx, obj, id, v)) {
    return -1;
  }
  return int(id);
}

// Called by a host import shim while the host wasm runs (reentrant: nested in a
// guest export call or in instantiation). Invokes guest import callback `id`
// with `argc` numeric args read from guest memory; returns its result as a
// double. i64/BigInt and reference types are not handled.
extern "C" EMSCRIPTEN_KEEPALIVE double wasmhost_invoke_import(
    int id, const double* args, int argc) {
  JSContext* cx = js::TlsContext.get();
  if (!cx || !gImportCallbacks) {
    return 0;
  }
  double local[64];
  int n = argc > 64 ? 64 : argc;
  for (int i = 0; i < n; i++) {
    local[i] = args[i];
  }

  RootedObject obj(cx, *gImportCallbacks);
  RootedValue fnVal(cx);
  if (!JS_GetElement(cx, obj, uint32_t(id), &fnVal) || !IsCallable(fnVal)) {
    return 0;
  }
  JS::RootedValueVector argv(cx);
  for (int i = 0; i < n; i++) {
    if (!argv.append(NumberValue(local[i]))) {
      return 0;
    }
  }
  RootedValue rval(cx);
  if (!JS_CallFunctionValue(cx, nullptr, fnVal, JS::HandleValueArray(argv),
                            &rval)) {
    // The guest callback threw; we cannot propagate across the host wasm frame,
    // so swallow it (best-effort) and return 0.
    if (cx->isExceptionPending()) {
      JS_ClearPendingException(cx);
    }
    return 0;
  }
  double d = 0;
  if (rval.isNumber()) {
    d = rval.toNumber();
  } else if (!rval.isUndefined()) {
    (void)ToNumber(cx, rval, &d);
  }
  return d;
}

// Native backing for a bridged wasm export function. Extended slot 0 holds the
// host instance handle, slot 1 the export index. Numeric args/results
// round-trip through double (i32/f32/f64); i64/BigInt and reference types are
// not yet handled.
static bool WasmHostExportCall(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  JSFunction& callee = args.callee().as<JSFunction>();
  int handle = callee.getExtendedSlot(0).toInt32();
  int index = callee.getExtendedSlot(1).toInt32();

  double buf[64];
  unsigned n = args.length() > 64 ? 64 : args.length();
  for (unsigned i = 0; i < n; i++) {
    double d;
    if (!ToNumber(cx, args[i], &d)) {
      return false;
    }
    buf[i] = d;
  }
  double r = wasmhost_call(handle, index, buf, int(n));
  args.rval().setNumber(r);
  return true;
}

// Bridge function imports for an already-compiled `handle`: find each import in
// `importObj`, register it -> a callback id (-1 otherwise), then build the host
// instance. Returns false (+ reports) on failure.
static bool HostBindAndInstantiate(JSContext* cx, int handle,
                                   HandleObject importObj) {
  int importCount = wasmhost_import_count(handle);
  int* callbackIds =
      js_pod_malloc<int>(size_t(importCount > 0 ? importCount : 1));
  if (!callbackIds) {
    ReportOutOfMemory(cx);
    return false;
  }
  for (int i = 0; i < importCount; i++) {
    callbackIds[i] = -1;
    int kind = wasmhost_import_kind(handle, i);
    char modbuf[256];
    char namebuf[256];
    wasmhost_import_module(handle, i, modbuf, sizeof(modbuf));
    wasmhost_import_name(handle, i, namebuf, sizeof(namebuf));
    if (!importObj) {
      continue;
    }
    RootedValue modVal(cx);
    if (!JS_GetProperty(cx, importObj, modbuf, &modVal) || !modVal.isObject()) {
      continue;
    }
    RootedObject modObj(cx, &modVal.toObject());
    RootedValue val(cx);
    if (!JS_GetProperty(cx, modObj, namebuf, &val)) {
      continue;
    }
    if (kind == 0) {  // function -> guest callback id
      if (IsCallable(val)) {
        callbackIds[i] = RegisterImportCallback(cx, val);
      }
    } else if (val.isObject()) {  // memory/table/global -> host object id
      RootedObject valObj(cx, &val.toObject());
      callbackIds[i] = HostObjId(cx, valObj);
    }
  }

  int rc = wasmhost_instantiate(handle, callbackIds, importCount);
  js_free(callbackIds);
  if (rc != 0) {
    JS_ReportErrorASCII(cx,
                        "WebAssembly host passthrough: instantiation failed");
    return false;
  }
  return true;
}

// Build an `{ exports }` instance object for an instantiated `handle`. Export
// functions are bridged; other export kinds (memory/table/global) are null.
static JSObject* HostBuildInstanceObject(JSContext* cx, int handle) {
  RootedObject exportsObj(cx, JS_NewPlainObject(cx));
  if (!exportsObj) {
    return nullptr;
  }
  int count = wasmhost_export_count(handle);
  for (int i = 0; i < count; i++) {
    char namebuf[256];
    int nameLen = wasmhost_export_name(handle, i, namebuf, sizeof(namebuf));
    Rooted<JSAtom*> nameAtom(cx, Atomize(cx, namebuf, size_t(nameLen)));
    if (!nameAtom) {
      return nullptr;
    }
    RootedValue exportVal(cx);
    int ekind = wasmhost_export_kind(handle, i);
    if (ekind == 0) {  // function
      Rooted<JSFunction*> fn(
          cx, NewNativeFunction(cx, WasmHostExportCall, 0, nameAtom,
                                gc::AllocKind::FUNCTION_EXTENDED));
      if (!fn) {
        return nullptr;
      }
      fn->setExtendedSlot(0, Int32Value(handle));
      fn->setExtendedSlot(1, Int32Value(i));
      exportVal.setObject(*fn);
    } else if (ekind == 2) {  // memory: register + build a mirrored wrapper
      int objId = wasmhost_export_register_mem(handle, i);
      if (objId >= 0) {
        RootedObject memWrap(
            cx, HostMakeMemoryWrapper(cx, objId,
                                      wasmhost_mem_is_shared(objId) != 0));
        if (!memWrap) {
          return nullptr;
        }
        exportVal.setObject(*memWrap);
      } else {
        exportVal.setNull();
      }
    } else {
      exportVal.setNull();
    }
    if (!JS_DefineProperty(cx, exportsObj, namebuf, exportVal,
                           JSPROP_ENUMERATE)) {
      return nullptr;
    }
  }
  RootedObject instanceObj(cx, JS_NewPlainObject(cx));
  if (!instanceObj) {
    return nullptr;
  }
  RootedValue val(cx, ObjectValue(*exportsObj));
  if (!JS_DefineProperty(cx, instanceObj, "exports", val, JSPROP_ENUMERATE)) {
    return nullptr;
  }
  return instanceObj;
}

// An opaque "module" object that just carries the host compile `handle` (a
// non-enumerable property). Not a real WasmModuleObject -- instanceof fails.
static const char* const kHostModuleHandleProp = "__wasmHostHandle";

static JSObject* HostMakeModuleObject(JSContext* cx, int handle) {
  RootedObject moduleObj(cx, JS_NewPlainObject(cx));
  if (!moduleObj) {
    return nullptr;
  }
  RootedValue h(cx, Int32Value(handle));
  if (!JS_DefineProperty(cx, moduleObj, kHostModuleHandleProp, h, 0)) {
    return nullptr;
  }
  return moduleObj;
}

static int HostModuleHandle(JSContext* cx, HandleObject obj) {
  RootedValue v(cx);
  if (!JS_GetProperty(cx, obj, kHostModuleHandleProp, &v) || !v.isInt32()) {
    return -1;
  }
  return v.toInt32();
}

// Guest wrappers for host Memory/Table/Global objects carry the host registry
// id in a non-enumerable property.
static const char* const kHostObjIdProp = "__wasmHostObjId";

static int HostObjId(JSContext* cx, HandleObject obj) {
  RootedValue v(cx);
  if (!JS_GetProperty(cx, obj, kHostObjIdProp, &v) || !v.isInt32()) {
    return -1;
  }
  return v.toInt32();
}

// Build a guest WebAssembly.Memory-like wrapper for an existing host memory
// `objId`: a "mirror" ArrayBuffer over a js_malloc'd, user-owned (stable, never
// GC-moved) region that content's HEAPU8 views. The mirror is kept in sync with
// the host memory around export/import calls (see whSyncMem in the js-library).
static JSObject* HostMakeMemoryWrapper(JSContext* cx, int objId, bool shared) {
  size_t len = size_t(wasmhost_mem_bytelength(objId));
  if (len == 0) {
    len = 1;
  }
  RootedObject ab(cx);
  uint8_t* mirror = nullptr;
  if (shared) {
    // Shared memory: content checks `buffer instanceof SharedArrayBuffer`
    // (emscripten pthreads), so the mirror must be a real SAB. Its data is
    // engine-owned and non-movable, so the pointer is stable for syncing.
    ab = JS::NewSharedArrayBuffer(cx, len);
    if (!ab) {
      JS_ReportErrorASCII(
          cx, "WebAssembly host passthrough: shared memory unavailable");
      return nullptr;
    }
    JS::AutoCheckCannotGC nogc;
    bool isSharedMem = false;
    mirror = JS::GetSharedArrayBufferData(ab, &isSharedMem, nogc);
  } else {
    mirror = js_pod_calloc<uint8_t>(len);
    if (!mirror) {
      ReportOutOfMemory(cx);
      return nullptr;
    }
    ab = JS::NewArrayBufferWithUserOwnedContents(cx, len, mirror);
    if (!ab) {
      js_free(mirror);
      return nullptr;
    }
  }
  if (!mirror) {
    return nullptr;
  }
  wasmhost_obj_set_mirror(objId, mirror, int(len));

  RootedObject wrapper(cx, JS_NewPlainObject(cx));
  if (!wrapper) {
    return nullptr;
  }
  RootedValue idv(cx, Int32Value(objId));
  if (!JS_DefineProperty(cx, wrapper, kHostObjIdProp, idv, 0)) {
    return nullptr;
  }
  RootedValue abv(cx, ObjectValue(*ab));
  if (!JS_DefineProperty(cx, wrapper, "buffer", abv, JSPROP_ENUMERATE)) {
    return nullptr;
  }
  return wrapper;
}

// Minimal guest wrapper carrying just the host obj id (table/global): enough to
// bind them as imports; JS-side element/value access is not bridged yet.
static JSObject* HostMakeObjIdWrapper(JSContext* cx, int objId) {
  RootedObject wrapper(cx, JS_NewPlainObject(cx));
  if (!wrapper) {
    return nullptr;
  }
  RootedValue idv(cx, Int32Value(objId));
  if (!JS_DefineProperty(cx, wrapper, kHostObjIdProp, idv, 0)) {
    return nullptr;
  }
  return wrapper;
}

// Compile `bytesObj` (a BufferSource) on the host; returns a handle or -1 (+
// reports an error).
static int HostCompileBytes(JSContext* cx, HandleObject bytesObj) {
  JSObject* unwrapped = CheckedUnwrapStatic(bytesObj);
  SharedMem<uint8_t*> dataPointer;
  size_t byteLength;
  bool isShared;
  if (!unwrapped ||
      !IsBufferSource(cx, unwrapped, /*allowShared*/ true,
                      /*allowResizable*/ true, &dataPointer, &byteLength,
                      &isShared)) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_BUF_MOD_ARG);
    return -1;
  }
  int handle = wasmhost_compile(dataPointer.unwrap(), int(byteLength));
  if (handle < 0) {
    JS_ReportErrorASCII(cx, "WebAssembly host passthrough: compile failed");
    return -1;
  }
  return handle;
}

// instantiate()'s first arg is either a BufferSource (-> { module, instance }) or
// a host module object (-> instance). Bridges imports from `importObj`.
static bool HostPassthroughInstantiate(JSContext* cx, HandleObject firstArg,
                                       HandleObject importObj,
                                       MutableHandleValue out) {
  int handle = HostModuleHandle(cx, firstArg);
  bool wasModule = handle >= 0;
  if (!wasModule) {
    handle = HostCompileBytes(cx, firstArg);
    if (handle < 0) {
      return false;
    }
  }
  if (!HostBindAndInstantiate(cx, handle, importObj)) {
    return false;
  }

  RootedObject instanceObj(cx, HostBuildInstanceObject(cx, handle));
  if (!instanceObj) {
    return false;
  }
  if (wasModule) {
    out.setObject(*instanceObj);
    return true;
  }
  RootedObject moduleObj(cx, HostMakeModuleObject(cx, handle));
  RootedObject resultObj(cx, JS_NewPlainObject(cx));
  if (!moduleObj || !resultObj) {
    return false;
  }
  RootedValue val(cx, ObjectValue(*moduleObj));
  if (!JS_DefineProperty(cx, resultObj, "module", val, JSPROP_ENUMERATE)) {
    return false;
  }
  val.setObject(*instanceObj);
  if (!JS_DefineProperty(cx, resultObj, "instance", val, JSPROP_ENUMERATE)) {
    return false;
  }
  out.setObject(*resultObj);
  return true;
}

// -- Streaming (compileStreaming / instantiateStreaming) --------------------
// Resolve the Response promise, call response.arrayBuffer(), then compile (and
// optionally instantiate) the bytes on the host. State crosses the async steps
// in a plain closure object: { promise, importObj, instantiate }.
static bool HostStream_OnBytes(JSContext* cx, unsigned argc, Value* vp);
static bool HostStream_OnResponse(JSContext* cx, unsigned argc, Value* vp);
static bool HostStream_OnReject(JSContext* cx, unsigned argc, Value* vp);

static JSObject* HostStreamClosure(HandleObject callee) {
  return &callee->as<JSFunction>().getExtendedSlot(0).toObject();
}

static bool HostStream_OnReject(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  RootedObject callee(cx, &args.callee());
  RootedObject closure(cx, HostStreamClosure(callee));
  RootedValue promiseVal(cx);
  if (!JS_GetProperty(cx, closure, "promise", &promiseVal)) {
    return false;
  }
  RootedObject promiseObj(cx, &promiseVal.toObject());
  RootedValue reason(cx, args.get(0));
  JS::RejectPromise(cx, promiseObj, reason);
  args.rval().setUndefined();
  return true;
}

static bool HostStream_OnResponse(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  RootedObject callee(cx, &args.callee());
  RootedObject closure(cx, HostStreamClosure(callee));
  RootedValue promiseVal(cx);
  if (!JS_GetProperty(cx, closure, "promise", &promiseVal)) {
    return false;
  }
  RootedObject promiseObj(cx, &promiseVal.toObject());

  RootedValue resp(cx, args.get(0));
  if (!resp.isObject()) {
    JS::RejectPromise(cx, promiseObj, resp);
    args.rval().setUndefined();
    return true;
  }
  RootedObject respObj(cx, &resp.toObject());
  RootedValue abPromise(cx);
  if (!JS_CallFunctionName(cx, respObj, "arrayBuffer",
                           JS::HandleValueArray::empty(), &abPromise)) {
    RootedValue exn(cx, UndefinedValue());
    if (JS_GetPendingException(cx, &exn)) {
      JS_ClearPendingException(cx);
    }
    JS::RejectPromise(cx, promiseObj, exn);
    args.rval().setUndefined();
    return true;
  }

  RootedFunction onBytes(
      cx, NewNativeFunction(cx, HostStream_OnBytes, 1, nullptr,
                            gc::AllocKind::FUNCTION_EXTENDED, GenericObject));
  RootedFunction onRej(
      cx, NewNativeFunction(cx, HostStream_OnReject, 1, nullptr,
                            gc::AllocKind::FUNCTION_EXTENDED, GenericObject));
  if (!onBytes || !onRej) {
    return false;
  }
  onBytes->setExtendedSlot(0, ObjectValue(*closure));
  onRej->setExtendedSlot(0, ObjectValue(*closure));
  RootedObject abResolve(cx, PromiseObject::unforgeableResolve(cx, abPromise));
  if (!abResolve || !JS::AddPromiseReactions(cx, abResolve, onBytes, onRej)) {
    return false;
  }
  args.rval().setUndefined();
  return true;
}

static bool HostStream_OnBytes(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);
  RootedObject callee(cx, &args.callee());
  RootedObject closure(cx, HostStreamClosure(callee));
  RootedValue promiseVal(cx);
  RootedValue importVal(cx);
  RootedValue instVal(cx);
  if (!JS_GetProperty(cx, closure, "promise", &promiseVal) ||
      !JS_GetProperty(cx, closure, "importObj", &importVal) ||
      !JS_GetProperty(cx, closure, "instantiate", &instVal)) {
    return false;
  }
  RootedObject promiseObj(cx, &promiseVal.toObject());

  RootedValue bytesVal(cx, args.get(0));
  if (!bytesVal.isObject()) {
    JS::RejectPromise(cx, promiseObj, bytesVal);
    args.rval().setUndefined();
    return true;
  }
  RootedObject bytesObj(cx, &bytesVal.toObject());

  RootedValue result(cx);
  bool ok;
  if (instVal.toBoolean()) {
    RootedObject importObj(
        cx, importVal.isObject() ? &importVal.toObject() : nullptr);
    ok = HostPassthroughInstantiate(cx, bytesObj, importObj, &result);
  } else {
    int handle = HostCompileBytes(cx, bytesObj);
    ok = handle >= 0;
    if (ok) {
      RootedObject m(cx, HostMakeModuleObject(cx, handle));
      ok = m != nullptr;
      if (ok) {
        result.setObject(*m);
      }
    }
  }
  if (!ok) {
    RootedValue exn(cx, UndefinedValue());
    if (JS_GetPendingException(cx, &exn)) {
      JS_ClearPendingException(cx);
    }
    JS::RejectPromise(cx, promiseObj, exn);
    args.rval().setUndefined();
    return true;
  }
  JS::ResolvePromise(cx, promiseObj, result);
  args.rval().setUndefined();
  return true;
}

static bool HostStreamingInstantiate(JSContext* cx, HandleValue responsePromise,
                                     HandleObject importObj, bool instantiate,
                                     Handle<PromiseObject*> resultPromise) {
  RootedObject closure(cx, JS_NewPlainObject(cx));
  if (!closure) {
    return false;
  }
  RootedValue v(cx, ObjectValue(*resultPromise));
  if (!JS_DefineProperty(cx, closure, "promise", v, 0)) {
    return false;
  }
  v = importObj ? ObjectValue(*importObj) : NullValue();
  if (!JS_DefineProperty(cx, closure, "importObj", v, 0)) {
    return false;
  }
  v.setBoolean(instantiate);
  if (!JS_DefineProperty(cx, closure, "instantiate", v, 0)) {
    return false;
  }

  RootedFunction onResp(
      cx, NewNativeFunction(cx, HostStream_OnResponse, 1, nullptr,
                            gc::AllocKind::FUNCTION_EXTENDED, GenericObject));
  RootedFunction onRej(
      cx, NewNativeFunction(cx, HostStream_OnReject, 1, nullptr,
                            gc::AllocKind::FUNCTION_EXTENDED, GenericObject));
  if (!onResp || !onRej) {
    return false;
  }
  onResp->setExtendedSlot(0, ObjectValue(*closure));
  onRej->setExtendedSlot(0, ObjectValue(*closure));
  RootedObject resolve(cx,
                       PromiseObject::unforgeableResolve(cx, responsePromise));
  if (!resolve) {
    return false;
  }
  return JS::AddPromiseReactions(cx, resolve, onResp, onRej);
}
#endif  // __EMSCRIPTEN__

static bool WebAssembly_instantiate(JSContext* cx, unsigned argc, Value* vp) {
  if (!EnsurePromiseSupport(cx)) {
    return false;
  }

  Log(cx, "async instantiate() started");

  Rooted<PromiseObject*> promise(cx, PromiseObject::createSkippingExecutor(cx));
  if (!promise) {
    return false;
  }

  CallArgs callArgs = CallArgsFromVp(argc, vp);

  RootedObject firstArg(cx);
  RootedObject importObj(cx);
  RootedValue featureOptions(cx);
  if (!GetInstantiateArgs(cx, callArgs, &firstArg, &importObj,
                          &featureOptions)) {
    return RejectWithPendingException(cx, promise, callArgs);
  }

#if defined(__EMSCRIPTEN__)
  if (wasm::UseHostPassthrough()) {
    RootedValue result(cx);
    if (!HostPassthroughInstantiate(cx, firstArg, importObj, &result)) {
      return RejectWithPendingException(cx, promise, callArgs);
    }
    if (!PromiseObject::resolve(cx, promise, result)) {
      return RejectWithPendingException(cx, promise, callArgs);
    }
    callArgs.rval().setObject(*promise);
    return true;
  }
#endif

  Rooted<WasmModuleObject*> moduleObj(
      cx, firstArg->maybeUnwrapIf<WasmModuleObject>());
  if (moduleObj) {
    if (!AsyncInstantiate(cx, moduleObj->module(), importObj, Ret::Instance,
                          promise)) {
      return false;
    }
  } else {
    JS::RootedVector<JSString*> parameterStrings(cx);
    JS::RootedVector<Value> parameterArgs(cx);
    bool canCompileStrings = false;
    if (!cx->isRuntimeCodeGenEnabled(JS::RuntimeCode::WASM, nullptr,
                                     JS::CompilationType::Undefined,
                                     parameterStrings, nullptr, parameterArgs,
                                     NullHandleValue, &canCompileStrings)) {
      return RejectWithPendingException(cx, promise, callArgs);
    }
    if (!canCompileStrings) {
      JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                                JSMSG_CSP_BLOCKED_WASM,
                                "WebAssembly.instantiate");
      return RejectWithPendingException(cx, promise, callArgs);
    }

    FeatureOptions options;
    if (!options.init(cx, featureOptions)) {
      return false;
    }

    auto task = cx->make_unique<CompileBufferTask>(cx, promise, importObj);
    if (!task || !task->init(cx, options, "WebAssembly.instantiate")) {
      return false;
    }

    if (!GetBytecodeBuffer(cx, firstArg, JSMSG_WASM_BAD_BUF_MOD_ARG,
                           &task->bytecode)) {
      return RejectWithPendingException(cx, promise, callArgs);
    }

    if (!StartOffThreadPromiseHelperTask(cx, std::move(task))) {
      return false;
    }
  }

  callArgs.rval().setObject(*promise);
  return true;
}

static bool WebAssembly_validate(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs callArgs = CallArgsFromVp(argc, vp);

  if (!callArgs.requireAtLeast(cx, "WebAssembly.validate", 1)) {
    return false;
  }

  FeatureOptions options;
  if (!options.init(cx, callArgs.get(1))) {
    return false;
  }

  if (!callArgs[0].isObject()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_BUF_ARG);
    return false;
  }

  UniqueChars error;
  bool validated;
  {
    // Limit the lifetime of the bytecode to just validation and ensure we pin
    // the buffer. No user code should be running here anyways, so this is very
    // conservative.
    BytecodeBufferOrSource bytecode;
    Rooted<JSObject*> sourceObj(cx, &callArgs[0].toObject());
    if (!GetBytecodeBufferOrSource(cx, sourceObj, JSMSG_WASM_BAD_BUF_ARG,
                                   &bytecode)) {
      return false;
    }
    AutoPinBufferSourceLength pin(cx, sourceObj.get());

    validated = Validate(cx, bytecode.source(), options, &error);
  }

  // If the reason for validation failure was OOM (signalled by null error
  // message), report out-of-memory so that validate's return is always
  // correct.
  if (!validated && !error) {
    ReportOutOfMemory(cx);
    return false;
  }

  if (error) {
    MOZ_ASSERT(!validated);
    Log(cx, "validate() failed with: %s", error.get());
  }

  callArgs.rval().setBoolean(validated);
  return true;
}

static bool EnsureStreamSupport(JSContext* cx) {
  // This should match wasm::StreamingCompilationAvailable().

  if (!EnsurePromiseSupport(cx)) {
    return false;
  }

  if (!CanUseExtraThreads()) {
    JS_ReportErrorASCII(
        cx, "WebAssembly.compileStreaming not supported with --no-threads");
    return false;
  }

  if (!cx->runtime()->consumeStreamCallback) {
    JS_ReportErrorASCII(cx,
                        "WebAssembly streaming not supported in this runtime");
    return false;
  }

  return true;
}

// This value is chosen and asserted to be disjoint from any host error code.
static const size_t StreamOOMCode = 0;

static bool RejectWithStreamErrorNumber(JSContext* cx, size_t errorCode,
                                        Handle<PromiseObject*> promise) {
  if (errorCode == StreamOOMCode) {
    ReportOutOfMemory(cx);
    return false;
  }

  cx->runtime()->reportStreamErrorCallback(cx, errorCode);
  return RejectWithPendingException(cx, promise);
}

class CompileStreamTask : public PromiseHelperTask, public JS::StreamConsumer {
  // The stream progresses monotonically through these states; the helper
  // thread wait()s for streamState_ to reach Closed.
  enum StreamState { Env, Code, Tail, Closed };
  ExclusiveWaitableData<StreamState> streamState_;

  // Immutable:
  const bool instantiate_;
  const PersistentRootedObject importObj_;

  // Immutable after noteResponseURLs() which is called at most once before
  // first call on stream thread:
  const MutableCompileArgs compileArgs_;

  // Immutable after Env state:
  MutableBytes envBytes_;
  BytecodeRange codeSection_;

  // The code section vector is resized once during the Env state and filled
  // in chunk by chunk during the Code state, updating the end-pointer after
  // each chunk:
  MutableBytes codeBytes_;
  uint8_t* codeBytesEnd_;
  ExclusiveBytesPtr exclusiveCodeBytesEnd_;

  // Immutable after Tail state:
  MutableBytes tailBytes_;
  ExclusiveStreamEndData exclusiveStreamEnd_;

  // Written once before Closed state and read in Closed state on main thread:
  SharedModule module_;
  Maybe<size_t> streamError_;
  UniqueChars compileError_;
  UniqueCharsVector warnings_;

  // Set on stream thread and read racily on helper thread to abort compilation:
  mozilla::Atomic<bool> streamFailed_;

  // Called on some thread before consumeChunk(), streamEnd(), streamError()):

  void noteResponseURLs(const char* url, const char* sourceMapUrl) override {
    if (url) {
      compileArgs_->scriptedCaller.source = DuplicateString(url);
      compileArgs_->scriptedCaller.kind = ScriptedCallerKind::Url;
    }
    if (sourceMapUrl) {
      compileArgs_->sourceMapURL = DuplicateString(sourceMapUrl);
    }
  }

  // Called on a stream thread:

  // Until StartOffThreadPromiseHelperTask succeeds, we are responsible for
  // dispatching ourselves back to the JS thread.
  //
  // Warning: After this function returns, 'this' can be deleted at any time, so
  // the caller must immediately return from the stream callback.
  void setClosedAndDestroyBeforeHelperThreadStarted() {
    streamState_.lock().get() = Closed;
    dispatchResolveAndDestroy();
  }

  // See setClosedAndDestroyBeforeHelperThreadStarted() comment.
  bool rejectAndDestroyBeforeHelperThreadStarted(size_t errorNumber) {
    MOZ_ASSERT(streamState_.lock() == Env);
    MOZ_ASSERT(!streamError_);
    streamError_ = Some(errorNumber);
    setClosedAndDestroyBeforeHelperThreadStarted();
    return false;
  }

  // Once StartOffThreadPromiseHelperTask succeeds, the helper thread will
  // dispatchResolveAndDestroy() after execute() returns, but execute()
  // wait()s for state to be Closed.
  //
  // Warning: After this function returns, 'this' can be deleted at any time, so
  // the caller must immediately return from the stream callback.
  void setClosedAndDestroyAfterHelperThreadStarted() {
    auto streamState = streamState_.lock();
    MOZ_ASSERT(streamState != Closed);
    streamState.get() = Closed;
    streamState.notify_one(/* stream closed */);
  }

  // See setClosedAndDestroyAfterHelperThreadStarted() comment.
  bool rejectAndDestroyAfterHelperThreadStarted(size_t errorNumber) {
    MOZ_ASSERT(!streamError_);
    streamError_ = Some(errorNumber);
    streamFailed_ = true;
    exclusiveCodeBytesEnd_.lock().notify_one();
    exclusiveStreamEnd_.lock().notify_one();
    setClosedAndDestroyAfterHelperThreadStarted();
    return false;
  }

  bool consumeChunk(const uint8_t* begin, size_t length) override {
    switch (streamState_.lock().get()) {
      case Env: {
        if (!envBytes_->append(begin, length)) {
          return rejectAndDestroyBeforeHelperThreadStarted(StreamOOMCode);
        }

        if (!StartsCodeSection(envBytes_->begin(), envBytes_->end(),
                               &codeSection_)) {
          return true;
        }

        uint32_t extraBytes = envBytes_->length() - codeSection_.start;
        if (extraBytes) {
          envBytes_->shrinkTo(codeSection_.start);
        }

        if (codeSection_.size() > MaxCodeSectionBytes) {
          return rejectAndDestroyBeforeHelperThreadStarted(StreamOOMCode);
        }

        if (!codeBytes_->vector.resize(codeSection_.size())) {
          return rejectAndDestroyBeforeHelperThreadStarted(StreamOOMCode);
        }

        codeBytesEnd_ = codeBytes_->begin();
        exclusiveCodeBytesEnd_.lock().get() = codeBytesEnd_;

        if (!StartOffThreadPromiseHelperTask(this)) {
          return rejectAndDestroyBeforeHelperThreadStarted(StreamOOMCode);
        }

        // Set the state to Code iff StartOffThreadPromiseHelperTask()
        // succeeds so that the state tells us whether we are before or
        // after the helper thread started.
        streamState_.lock().get() = Code;

        if (extraBytes) {
          return consumeChunk(begin + length - extraBytes, extraBytes);
        }

        return true;
      }
      case Code: {
        size_t copyLength =
            std::min<size_t>(length, codeBytes_->end() - codeBytesEnd_);
        memcpy(codeBytesEnd_, begin, copyLength);
        codeBytesEnd_ += copyLength;

        {
          auto codeStreamEnd = exclusiveCodeBytesEnd_.lock();
          codeStreamEnd.get() = codeBytesEnd_;
          codeStreamEnd.notify_one();
        }

        if (codeBytesEnd_ != codeBytes_->end()) {
          return true;
        }

        streamState_.lock().get() = Tail;

        if (uint32_t extraBytes = length - copyLength) {
          return consumeChunk(begin + copyLength, extraBytes);
        }

        return true;
      }
      case Tail: {
        if (!tailBytes_->append(begin, length)) {
          return rejectAndDestroyAfterHelperThreadStarted(StreamOOMCode);
        }

        return true;
      }
      case Closed:
        MOZ_CRASH("consumeChunk() in Closed state");
    }
    MOZ_CRASH("unreachable");
  }

  void streamEnd(
      JS::OptimizedEncodingListener* completeTier2Listener) override {
    switch (streamState_.lock().get()) {
      case Env: {
        BytecodeBuffer bytecode(envBytes_, nullptr, nullptr);
        module_ = CompileModule(*compileArgs_,
                                BytecodeBufferOrSource(std::move(bytecode)),
                                &compileError_, &warnings_, nullptr);
        setClosedAndDestroyBeforeHelperThreadStarted();
        return;
      }
      case Code: {
        // Stream ended mid code section: declared code section size was
        // larger than the bytes actually delivered. The helper thread is
        // blocked in StreamingDecoder::waitForBytes()
        // Report the error and unlock the thread.
        // The order of the following operations is critical:
        // - First set cancelled to true
        // - Then wake up the helper thread waiting for bytes. This thread will
        //   see the cancelled flag and returns. If we inverted this we could
        //   wake up the thread and it would go back to sleep.
        streamFailed_ = true;
        exclusiveCodeBytesEnd_.lock().notify_one();
        exclusiveStreamEnd_.lock().notify_one();
        setClosedAndDestroyAfterHelperThreadStarted();
        return;
      }
      case Tail:
        // Unlock exclusiveStreamEnd_ before locking streamState_.
        {
          auto streamEnd = exclusiveStreamEnd_.lock();
          MOZ_ASSERT(!streamEnd->reached);
          streamEnd->reached = true;
          streamEnd->tailBytes = tailBytes_;
          streamEnd->completeTier2Listener = completeTier2Listener;
          streamEnd.notify_one();
        }
        setClosedAndDestroyAfterHelperThreadStarted();
        return;
      case Closed:
        MOZ_CRASH("streamEnd() in Closed state");
    }
  }

  void streamError(size_t errorCode) override {
    MOZ_ASSERT(errorCode != StreamOOMCode);
    switch (streamState_.lock().get()) {
      case Env:
        rejectAndDestroyBeforeHelperThreadStarted(errorCode);
        return;
      case Tail:
      case Code:
        rejectAndDestroyAfterHelperThreadStarted(errorCode);
        return;
      case Closed:
        MOZ_CRASH("streamError() in Closed state");
    }
  }

  void consumeOptimizedEncoding(const uint8_t* begin, size_t length) override {
    module_ = Module::deserialize(begin, length);

    MOZ_ASSERT(streamState_.lock().get() == Env);
    setClosedAndDestroyBeforeHelperThreadStarted();
  }

  // Called on a helper thread:

  void execute() override {
    module_ = CompileStreaming(*compileArgs_, *envBytes_, *codeBytes_,
                               exclusiveCodeBytesEnd_, exclusiveStreamEnd_,
                               streamFailed_, &compileError_, &warnings_);

    // When execute() returns, the CompileStreamTask will be dispatched
    // back to its JS thread to call resolve() and then be destroyed. We
    // can't let this happen until the stream has been closed lest
    // consumeChunk() or streamEnd() be called on a dead object.
    auto streamState = streamState_.lock();
    while (streamState != Closed) {
      streamState.wait(/* stream closed */);
    }
  }

  // Called on a JS thread after streaming compilation completes/errors:

  bool resolve(JSContext* cx, Handle<PromiseObject*> promise) override {
    MOZ_ASSERT(streamState_.lock() == Closed);

    if (!ReportCompileWarnings(cx, warnings_)) {
      return false;
    }
    if (module_) {
      MOZ_ASSERT(!streamFailed_ && !streamError_ && !compileError_);
      if (instantiate_) {
        return AsyncInstantiate(cx, *module_, importObj_, Ret::Pair, promise);
      }
      return ResolveCompile(cx, *module_, promise);
    }

    if (streamError_) {
      return RejectWithStreamErrorNumber(cx, *streamError_, promise);
    }

    return Reject(cx, *compileArgs_, promise, compileError_);
  }

 public:
  CompileStreamTask(JSContext* cx, Handle<PromiseObject*> promise,
                    CompileArgs& compileArgs, bool instantiate,
                    HandleObject importObj)
      : PromiseHelperTask(cx, promise),
        streamState_(mutexid::WasmStreamStatus, Env),
        instantiate_(instantiate),
        importObj_(cx, importObj),
        compileArgs_(&compileArgs),
        codeSection_{},
        codeBytesEnd_(nullptr),
        exclusiveCodeBytesEnd_(mutexid::WasmCodeBytesEnd, nullptr),
        exclusiveStreamEnd_(mutexid::WasmStreamEnd),
        streamFailed_(false) {
    MOZ_ASSERT_IF(importObj_, instantiate_);
  }

  [[nodiscard]] bool init(JSContext* cx) {
    envBytes_ = cx->new_<ShareableBytes>();
    if (!envBytes_) {
      return false;
    }

    codeBytes_ = js_new<ShareableBytes>();
    if (!codeBytes_) {
      return false;
    }

    tailBytes_ = js_new<ShareableBytes>();
    if (!tailBytes_) {
      return false;
    }

    return PromiseHelperTask::init(cx);
  }
};

// A short-lived object that captures the arguments of a
// WebAssembly.{compileStreaming,instantiateStreaming} while waiting for
// the Promise<Response> to resolve to a (hopefully) Promise.
class ResolveResponseClosure : public NativeObject {
  static const unsigned COMPILE_ARGS_SLOT = 0;
  static const unsigned PROMISE_OBJ_SLOT = 1;
  static const unsigned INSTANTIATE_SLOT = 2;
  static const unsigned IMPORT_OBJ_SLOT = 3;
  static const JSClassOps classOps_;

  static void finalize(JS::GCContext* gcx, JSObject* obj) {
    auto& closure = obj->as<ResolveResponseClosure>();
    gcx->release(obj, &closure.compileArgs(),
                 MemoryUse::WasmResolveResponseClosure);
  }

 public:
  static const unsigned RESERVED_SLOTS = 4;
  static const JSClass class_;

  static ResolveResponseClosure* create(JSContext* cx, const CompileArgs& args,
                                        HandleObject promise, bool instantiate,
                                        HandleObject importObj) {
    MOZ_ASSERT_IF(importObj, instantiate);

    AutoSetNewObjectMetadata metadata(cx);
    auto* obj = NewObjectWithGivenProto<ResolveResponseClosure>(cx, nullptr);
    if (!obj) {
      return nullptr;
    }

    args.AddRef();
    InitReservedSlot(obj, COMPILE_ARGS_SLOT, const_cast<CompileArgs*>(&args),
                     MemoryUse::WasmResolveResponseClosure);
    obj->setReservedSlot(PROMISE_OBJ_SLOT, ObjectValue(*promise));
    obj->setReservedSlot(INSTANTIATE_SLOT, BooleanValue(instantiate));
    obj->setReservedSlot(IMPORT_OBJ_SLOT, ObjectOrNullValue(importObj));
    return obj;
  }

  CompileArgs& compileArgs() const {
    return *(CompileArgs*)getReservedSlot(COMPILE_ARGS_SLOT).toPrivate();
  }
  PromiseObject& promise() const {
    return getReservedSlot(PROMISE_OBJ_SLOT).toObject().as<PromiseObject>();
  }
  bool instantiate() const {
    return getReservedSlot(INSTANTIATE_SLOT).toBoolean();
  }
  JSObject* importObj() const {
    return getReservedSlot(IMPORT_OBJ_SLOT).toObjectOrNull();
  }
};

const JSClassOps ResolveResponseClosure::classOps_ = {
    .finalize = ResolveResponseClosure::finalize,
};

const JSClass ResolveResponseClosure::class_ = {
    "WebAssembly ResolveResponseClosure",
    JSCLASS_DELAY_METADATA_BUILDER |
        JSCLASS_HAS_RESERVED_SLOTS(ResolveResponseClosure::RESERVED_SLOTS) |
        JSCLASS_FOREGROUND_FINALIZE,
    &ResolveResponseClosure::classOps_,
};

static ResolveResponseClosure* ToResolveResponseClosure(const CallArgs& args) {
  return &args.callee()
              .as<JSFunction>()
              .getExtendedSlot(0)
              .toObject()
              .as<ResolveResponseClosure>();
}

static bool RejectWithErrorNumber(JSContext* cx, uint32_t errorNumber,
                                  Handle<PromiseObject*> promise) {
  JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr, errorNumber);
  return RejectWithPendingException(cx, promise);
}

static bool ResolveResponse_OnFulfilled(JSContext* cx, unsigned argc,
                                        Value* vp) {
  CallArgs callArgs = CallArgsFromVp(argc, vp);

  Rooted<ResolveResponseClosure*> closure(cx,
                                          ToResolveResponseClosure(callArgs));
  Rooted<PromiseObject*> promise(cx, &closure->promise());
  CompileArgs& compileArgs = closure->compileArgs();
  bool instantiate = closure->instantiate();
  Rooted<JSObject*> importObj(cx, closure->importObj());

  auto task = cx->make_unique<CompileStreamTask>(cx, promise, compileArgs,
                                                 instantiate, importObj);
  if (!task || !task->init(cx)) {
    return RejectWithOutOfMemory(cx, promise);
  }

  if (!callArgs.get(0).isObject()) {
    return RejectWithErrorNumber(cx, JSMSG_WASM_BAD_RESPONSE_VALUE, promise);
  }

  RootedObject response(cx, &callArgs.get(0).toObject());
  if (!cx->runtime()->consumeStreamCallback(cx, response, JS::MimeType::Wasm,
                                            task.get())) {
    return RejectWithPendingException(cx, promise);
  }

  (void)task.release();

  callArgs.rval().setUndefined();
  return true;
}

static bool ResolveResponse_OnRejected(JSContext* cx, unsigned argc,
                                       Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);

  Rooted<ResolveResponseClosure*> closure(cx, ToResolveResponseClosure(args));
  Rooted<PromiseObject*> promise(cx, &closure->promise());

  if (!PromiseObject::reject(cx, promise, args.get(0))) {
    return false;
  }

  args.rval().setUndefined();
  return true;
}

static bool ResolveResponse(JSContext* cx, Handle<Value> responsePromise,
                            Handle<Value> featureOptions,
                            Handle<PromiseObject*> resultPromise,
                            bool instantiate = false,
                            HandleObject importObj = nullptr) {
  MOZ_ASSERT_IF(importObj, instantiate);

  const char* introducer = instantiate ? "WebAssembly.instantiateStreaming"
                                       : "WebAssembly.compileStreaming";

  FeatureOptions options;
  if (!options.init(cx, featureOptions)) {
    return false;
  }

  SharedCompileArgs compileArgs = InitCompileArgs(cx, options, introducer);
  if (!compileArgs) {
    return false;
  }

  RootedObject closure(
      cx, ResolveResponseClosure::create(cx, *compileArgs, resultPromise,
                                         instantiate, importObj));
  if (!closure) {
    return false;
  }

  RootedFunction onResolved(
      cx, NewNativeFunction(cx, ResolveResponse_OnFulfilled, 1, nullptr,
                            gc::AllocKind::FUNCTION_EXTENDED, GenericObject));
  if (!onResolved) {
    return false;
  }

  RootedFunction onRejected(
      cx, NewNativeFunction(cx, ResolveResponse_OnRejected, 1, nullptr,
                            gc::AllocKind::FUNCTION_EXTENDED, GenericObject));
  if (!onRejected) {
    return false;
  }

  onResolved->setExtendedSlot(0, ObjectValue(*closure));
  onRejected->setExtendedSlot(0, ObjectValue(*closure));

  RootedObject resolve(cx,
                       PromiseObject::unforgeableResolve(cx, responsePromise));
  if (!resolve) {
    return false;
  }

  return JS::AddPromiseReactions(cx, resolve, onResolved, onRejected);
}

static bool WebAssembly_compileStreaming(JSContext* cx, unsigned argc,
                                         Value* vp) {
  if (!EnsureStreamSupport(cx)) {
    return false;
  }

  Log(cx, "async compileStreaming() started");

  Rooted<PromiseObject*> resultPromise(
      cx, PromiseObject::createSkippingExecutor(cx));
  if (!resultPromise) {
    return false;
  }

  CallArgs callArgs = CallArgsFromVp(argc, vp);

  JS::RootedVector<JSString*> parameterStrings(cx);
  JS::RootedVector<Value> parameterArgs(cx);
  bool canCompileStrings = false;
  if (!cx->isRuntimeCodeGenEnabled(JS::RuntimeCode::WASM, nullptr,
                                   JS::CompilationType::Undefined,
                                   parameterStrings, nullptr, parameterArgs,
                                   NullHandleValue, &canCompileStrings)) {
    return RejectWithPendingException(cx, resultPromise, callArgs);
  }
  if (!canCompileStrings) {
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_CSP_BLOCKED_WASM,
                              "WebAssembly.compileStreaming");
    return RejectWithPendingException(cx, resultPromise, callArgs);
  }

  Rooted<Value> responsePromise(cx, callArgs.get(0));
  Rooted<Value> featureOptions(cx, callArgs.get(1));
#if defined(__EMSCRIPTEN__)
  if (wasm::UseHostPassthrough()) {
    if (!HostStreamingInstantiate(cx, responsePromise, nullptr,
                                  /*instantiate*/ false, resultPromise)) {
      return RejectWithPendingException(cx, resultPromise, callArgs);
    }
    callArgs.rval().setObject(*resultPromise);
    return true;
  }
#endif
  if (!ResolveResponse(cx, responsePromise, featureOptions, resultPromise)) {
    return RejectWithPendingException(cx, resultPromise, callArgs);
  }

  callArgs.rval().setObject(*resultPromise);
  return true;
}

static bool WebAssembly_instantiateStreaming(JSContext* cx, unsigned argc,
                                             Value* vp) {
  if (!EnsureStreamSupport(cx)) {
    return false;
  }

  Log(cx, "async instantiateStreaming() started");

  Rooted<PromiseObject*> resultPromise(
      cx, PromiseObject::createSkippingExecutor(cx));
  if (!resultPromise) {
    return false;
  }

  CallArgs callArgs = CallArgsFromVp(argc, vp);

  JS::RootedVector<JSString*> parameterStrings(cx);
  JS::RootedVector<Value> parameterArgs(cx);
  bool canCompileStrings = false;
  if (!cx->isRuntimeCodeGenEnabled(JS::RuntimeCode::WASM, nullptr,
                                   JS::CompilationType::Undefined,
                                   parameterStrings, nullptr, parameterArgs,
                                   NullHandleValue, &canCompileStrings)) {
    return RejectWithPendingException(cx, resultPromise, callArgs);
  }
  if (!canCompileStrings) {
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_CSP_BLOCKED_WASM,
                              "WebAssembly.instantiateStreaming");
    return RejectWithPendingException(cx, resultPromise, callArgs);
  }

  Rooted<JSObject*> firstArg(cx);
  Rooted<JSObject*> importObj(cx);
  Rooted<Value> featureOptions(cx);
  if (!GetInstantiateArgs(cx, callArgs, &firstArg, &importObj,
                          &featureOptions)) {
    return RejectWithPendingException(cx, resultPromise, callArgs);
  }
  Rooted<Value> responsePromise(cx, ObjectValue(*firstArg.get()));

#if defined(__EMSCRIPTEN__)
  if (wasm::UseHostPassthrough()) {
    if (!HostStreamingInstantiate(cx, responsePromise, importObj,
                                  /*instantiate*/ true, resultPromise)) {
      return RejectWithPendingException(cx, resultPromise, callArgs);
    }
    callArgs.rval().setObject(*resultPromise);
    return true;
  }
#endif
  if (!ResolveResponse(cx, responsePromise, featureOptions, resultPromise, true,
                       importObj)) {
    return RejectWithPendingException(cx, resultPromise, callArgs);
  }

  callArgs.rval().setObject(*resultPromise);
  return true;
}

#ifdef ENABLE_WASM_JSPI
static constexpr char WasmSuspendingName[] = "Suspending";

const ClassSpec WasmSuspendingObject::classSpec_ = {
    CreateWasmConstructor<WasmSuspendingObject, WasmSuspendingName>,
    GenericCreatePrototype<WasmSuspendingObject>,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    ClassSpec::DontDefineConstructor,
};

const JSClass WasmSuspendingObject::class_ = {
    "Suspending",
    JSCLASS_HAS_RESERVED_SLOTS(WasmSuspendingObject::RESERVED_SLOTS),
    JS_NULL_CLASS_OPS,
    &classSpec_,
};

const JSClass& WasmSuspendingObject::protoClass_ = PlainObject::class_;

/* static */
bool WasmSuspendingObject::construct(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);

  if (!ThrowIfNotConstructing(cx, args, "WebAssembly.Suspending")) {
    return false;
  }

  if (!args.requireAtLeast(cx, "WebAssembly.Suspending", 1)) {
    return false;
  }

  if (!IsCallableNonCCW(args[0])) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_FUNCTION_VALUE, "first");
    return false;
  }

  RootedObject callable(cx, &args[0].toObject());

  RootedObject proto(
      cx, GetWasmConstructorPrototype(cx, args, JSProto_WasmSuspending));
  if (!proto) {
    ReportOutOfMemory(cx);
    return false;
  }

  Rooted<WasmSuspendingObject*> suspending(
      cx, NewObjectWithGivenProto<WasmSuspendingObject>(cx, proto));
  if (!suspending) {
    return false;
  }
  suspending->setWrappedFunction(callable);
  args.rval().setObject(*suspending);
  return true;
}

static bool WebAssembly_promising(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);

  if (!JSPromiseIntegrationAvailable(cx)) {
    JS_ReportErrorASCII(cx, "JS-PI is not enabled");
    return false;
  }

  if (!args.requireAtLeast(cx, "WebAssembly.promising", 1)) {
    return false;
  }

  if (!IsWasmFunction(args[0])) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_FUNCTION_VALUE, "first");
    return false;
  }

  RootedObject func(cx, &args[0].toObject());
  RootedFunction promise(cx, WasmPromisingFunctionCreate(cx, func));
  if (!promise) {
    return false;
  }
  args.rval().setObject(*promise);
  return true;
}

static const JSFunctionSpec WebAssembly_jspi_methods[] = {
    JS_FN("promising", WebAssembly_promising, 1, JSPROP_ENUMERATE),
    JS_FS_END,
};

bool js::IsWasmSuspendingObject(JSObject* obj) {
  return obj->is<WasmSuspendingObject>();
}

JSObject* js::MaybeUnwrapSuspendingObject(JSObject* wrapper) {
  if (!wrapper->is<WasmSuspendingObject>()) {
    return nullptr;
  }
  return wrapper->as<WasmSuspendingObject>().wrappedFunction();
}
#else
bool js::IsWasmSuspendingObject(JSObject* obj) { return false; }
#endif  // ENABLE_WASM_JSPI

#ifdef ENABLE_WASM_MOZ_INTGEMM

static bool WebAssembly_mozIntGemm(JSContext* cx, unsigned argc, Value* vp) {
  CallArgs args = CallArgsFromVp(argc, vp);

  Rooted<WasmModuleObject*> module(cx);
  if (!wasm::CompileBuiltinModule(cx, wasm::BuiltinModuleId::IntGemm, nullptr,
                                  &module)) {
    ReportOutOfMemory(cx);
    return false;
  }
  args.rval().set(ObjectValue(*module.get()));
  return true;
}

static const JSFunctionSpec WebAssembly_mozIntGemm_methods[] = {
    JS_FN("mozIntGemm", WebAssembly_mozIntGemm, 0, JSPROP_ENUMERATE),
    JS_FS_END,
};

#endif  // ENABLE_WASM_MOZ_INTGEMM

static const JSFunctionSpec WebAssembly_static_methods[] = {
    JS_FN("toSource", WebAssembly_toSource, 0, 0),
    JS_FN("compile", WebAssembly_compile, 1, JSPROP_ENUMERATE),
    JS_FN("instantiate", WebAssembly_instantiate, 1, JSPROP_ENUMERATE),
    JS_FN("validate", WebAssembly_validate, 1, JSPROP_ENUMERATE),
    JS_FN("compileStreaming", WebAssembly_compileStreaming, 1,
          JSPROP_ENUMERATE),
    JS_FN("instantiateStreaming", WebAssembly_instantiateStreaming, 1,
          JSPROP_ENUMERATE),
    JS_FS_END,
};

static const JSPropertySpec WebAssembly_static_properties[] = {
    JS_STRING_SYM_PS(toStringTag, "WebAssembly", JSPROP_READONLY),
    JS_PS_END,
};

static JSObject* CreateWebAssemblyObject(JSContext* cx, JSProtoKey key) {
  MOZ_RELEASE_ASSERT(HasSupport(cx));

  RootedObject proto(cx, &cx->global()->getObjectPrototype());
  return NewTenuredObjectWithGivenProto(cx, &WasmNamespaceObject::class_,
                                        proto);
}

struct NameAndProtoKey {
  const char* const name;
  JSProtoKey key;
};

static bool WebAssemblyDefineConstructor(JSContext* cx,
                                         Handle<WasmNamespaceObject*> wasm,
                                         NameAndProtoKey entry,
                                         MutableHandleValue ctorValue,
                                         MutableHandleId id) {
  JSObject* ctor = GlobalObject::getOrCreateConstructor(cx, entry.key);
  if (!ctor) {
    return false;
  }
  ctorValue.setObject(*ctor);

  JSAtom* className = Atomize(cx, entry.name, strlen(entry.name));
  if (!className) {
    return false;
  }
  id.set(AtomToId(className));

  return DefineDataProperty(cx, wasm, id, ctorValue, 0);
}

static bool WebAssemblyClassFinish(JSContext* cx, HandleObject object,
                                   HandleObject proto) {
  Handle<WasmNamespaceObject*> wasm = object.as<WasmNamespaceObject>();

  constexpr NameAndProtoKey entries[] = {
      {"Module", JSProto_WasmModule},
      {"Instance", JSProto_WasmInstance},
      {"Memory", JSProto_WasmMemory},
      {"Table", JSProto_WasmTable},
      {"Global", JSProto_WasmGlobal},
      {"CompileError", GetExceptionProtoKey(JSEXN_WASMCOMPILEERROR)},
      {"LinkError", GetExceptionProtoKey(JSEXN_WASMLINKERROR)},
      {"RuntimeError", GetExceptionProtoKey(JSEXN_WASMRUNTIMEERROR)},
#ifdef ENABLE_WASM_TYPE_REFLECTIONS
      {"Function", JSProto_WasmFunction},
#endif
  };
  RootedValue ctorValue(cx);
  RootedId id(cx);
  for (const auto& entry : entries) {
    if (!WebAssemblyDefineConstructor(cx, wasm, entry, &ctorValue, &id)) {
      return false;
    }
  }

  constexpr NameAndProtoKey exceptionEntries[] = {
      {"Tag", JSProto_WasmTag},
      {"Exception", JSProto_WasmException},
  };
  for (const auto& entry : exceptionEntries) {
    if (!WebAssemblyDefineConstructor(cx, wasm, entry, &ctorValue, &id)) {
      return false;
    }
  }

  RootedObject tagProto(
      cx, GlobalObject::getOrCreatePrototype(cx, JSProto_WasmTag));
  if (!tagProto) {
    ReportOutOfMemory(cx);
    return false;
  }

  SharedTagType wrappedJSValueTagType(sWrappedJSValueTagType);
  WasmTagObject* wrappedJSValueTagObject =
      WasmTagObject::create(cx, wrappedJSValueTagType, tagProto);
  if (!wrappedJSValueTagObject) {
    return false;
  }

  wasm->setWrappedJSValueTag(wrappedJSValueTagObject);

  RootedId jsTagName(cx, NameToId(cx->names().jsTag));
  RootedValue jsTagValue(cx, ObjectValue(*wrappedJSValueTagObject));
  if (!DefineDataProperty(cx, wasm, jsTagName, jsTagValue,
                          JSPROP_READONLY | JSPROP_ENUMERATE)) {
    return false;
  }

#ifdef ENABLE_WASM_COMPONENTS
  if (ComponentsAvailable(cx)) {
    constexpr NameAndProtoKey componentEntry = {"Component",
                                                JSProto_WasmComponent};
    if (!WebAssemblyDefineConstructor(cx, wasm, componentEntry, &ctorValue,
                                      &id)) {
      return false;
    }
  }
#endif

#ifdef ENABLE_WASM_JSPI
  constexpr NameAndProtoKey jspiEntries[] = {
      {"Suspending", JSProto_WasmSuspending},
      {"SuspendError", GetExceptionProtoKey(JSEXN_WASMSUSPENDERROR)},
  };
  if (JSPromiseIntegrationAvailable(cx)) {
    if (!JS_DefineFunctions(cx, wasm, WebAssembly_jspi_methods)) {
      return false;
    }
    for (const auto& entry : jspiEntries) {
      if (!WebAssemblyDefineConstructor(cx, wasm, entry, &ctorValue, &id)) {
        return false;
      }
    }

    SharedTagType jsPromiseTagType(sJSPromiseTagType);
    WasmTagObject* jsPromiseTagObject =
        WasmTagObject::create(cx, jsPromiseTagType, tagProto);
    if (!jsPromiseTagObject) {
      return false;
    }

    wasm->setJSPromiseTag(jsPromiseTagObject);
  }
#endif

#ifdef ENABLE_WASM_MOZ_INTGEMM
  if (MozIntGemmAvailable(cx) &&
      !JS_DefineFunctions(cx, wasm, WebAssembly_mozIntGemm_methods)) {
    return false;
  }
#endif

  return true;
}

WasmNamespaceObject* WasmNamespaceObject::getOrCreate(JSContext* cx) {
  JSObject* wasm =
      GlobalObject::getOrCreateConstructor(cx, JSProto_WebAssembly);
  if (!wasm) {
    return nullptr;
  }
  return &wasm->as<WasmNamespaceObject>();
}

static const ClassSpec WebAssemblyClassSpec = {
    CreateWebAssemblyObject,       nullptr, WebAssembly_static_methods,
    WebAssembly_static_properties, nullptr, nullptr,
    WebAssemblyClassFinish,
};

const JSClass js::WasmNamespaceObject::class_ = {
    "WebAssembly",
    JSCLASS_HAS_CACHED_PROTO(JSProto_WebAssembly) |
        JSCLASS_HAS_RESERVED_SLOTS(WasmNamespaceObject::RESERVED_SLOTS),
    JS_NULL_CLASS_OPS,
    &WebAssemblyClassSpec,
};

// Sundry
