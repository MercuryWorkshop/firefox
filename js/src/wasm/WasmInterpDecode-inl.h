/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * Module decoder + per-function control-flow pre-pass for the in-process wasm
 * interpreter. Included by WasmInterp.h. The module has already passed
 * wasm::Validate (Core features only), so decoding can largely trust structure
 * but still bounds-checks every read.
 */

#ifndef wasm_WasmInterpDecode_inl_h
#define wasm_WasmInterpDecode_inl_h

namespace js {
namespace wasm {
namespace interp {

// A parsed constant init expression (one value op + end).
struct InitExpr {
  uint8_t kind;   // Op byte: I32Const/I64Const/F32Const/F64Const/GlobalGet/
                  // RefNull/RefFunc
  uint64_t imm;   // const bits / global idx / func idx (UINT32_MAX = ref.null)
};

static UniqueChars ReadName(Reader& r) {
  uint32_t len = r.u32leb();
  if (!r.ok || size_t(r.end - r.p) < len) {
    r.ok = false;
    return nullptr;
  }
  UniqueChars s(js_pod_malloc<char>(len + 1));
  if (!s) {
    r.ok = false;
    return nullptr;
  }
  memcpy(s.get(), r.p, len);
  s[len] = '\0';
  r.p += len;
  return s;
}

static bool ReadVTByte(Reader& r, VT* out) {
  uint8_t b = r.u8();
  return r.ok && DecodeVT(b, out);
}

// Decode a blocktype SLEB into (#params, #results) using the module's types.
static bool DecodeBlockType(Reader& r, const Module& m, uint32_t* nParams,
                            uint32_t* nResults) {
  // Peek: a single byte 0x40 / valtype, or a (possibly multibyte) type index.
  int64_t v = r.s64leb();
  if (!r.ok) return false;
  if (v >= 0) {
    if (uint64_t(v) >= m.types.length()) return false;
    const FuncType& ft = m.types[uint32_t(v)];
    *nParams = ft.params.length();
    *nResults = ft.results.length();
    return true;
  }
  *nParams = 0;
  uint8_t code = uint8_t(v & 0x7f);
  if (code == uint8_t(TypeCode::BlockVoid)) {
    *nResults = 0;
    return true;
  }
  VT t;
  if (!DecodeVT(code, &t)) return false;
  *nResults = 1;
  return true;
}

static bool SkipBlockType(Reader& r) {
  (void)r.s64leb();
  return r.ok;
}

// Skip the immediates of one opcode (op already read). Used by the pre-pass.
static bool SkipImmediates(Reader& r, uint8_t op) {
  switch (Op(op)) {
    case Op::Block:
    case Op::Loop:
    case Op::If:
      return SkipBlockType(r);
    case Op::Br:
    case Op::BrIf:
    case Op::Call:
    case Op::LocalGet:
    case Op::LocalSet:
    case Op::LocalTee:
    case Op::GlobalGet:
    case Op::GlobalSet:
    case Op::TableGet:
    case Op::TableSet:
    case Op::MemorySize:
    case Op::MemoryGrow:
    case Op::RefFunc:
      (void)r.u32leb();
      return r.ok;
    case Op::BrTable: {
      uint32_t n = r.u32leb();
      for (uint32_t i = 0; i < n + 1 && r.ok; i++) (void)r.u32leb();
      return r.ok;
    }
    case Op::CallIndirect:
      (void)r.u32leb();
      (void)r.u32leb();
      return r.ok;
    case Op::SelectTyped: {
      uint32_t n = r.u32leb();
      for (uint32_t i = 0; i < n && r.ok; i++) (void)r.u8();
      return r.ok;
    }
    case Op::I32Load:
    case Op::I64Load:
    case Op::F32Load:
    case Op::F64Load:
    case Op::I32Load8S:
    case Op::I32Load8U:
    case Op::I32Load16S:
    case Op::I32Load16U:
    case Op::I64Load8S:
    case Op::I64Load8U:
    case Op::I64Load16S:
    case Op::I64Load16U:
    case Op::I64Load32S:
    case Op::I64Load32U:
    case Op::I32Store:
    case Op::I64Store:
    case Op::F32Store:
    case Op::F64Store:
    case Op::I32Store8:
    case Op::I32Store16:
    case Op::I64Store8:
    case Op::I64Store16:
    case Op::I64Store32:
      (void)r.u32leb();  // align
      (void)r.u32leb();  // offset
      return r.ok;
    case Op::I32Const:
      (void)r.s32leb();
      return r.ok;
    case Op::I64Const:
      (void)r.s64leb();
      return r.ok;
    case Op::F32Const:
      (void)r.f32();
      return r.ok;
    case Op::F64Const:
      (void)r.f64();
      return r.ok;
    case Op::RefNull:
      (void)r.u8();  // heaptype
      return r.ok;
    case Op::MiscPrefix: {
      uint32_t sub = r.u32leb();
      if (!r.ok) return false;
      switch (MiscOp(sub)) {
        case MiscOp::I32TruncSatF32S:
        case MiscOp::I32TruncSatF32U:
        case MiscOp::I32TruncSatF64S:
        case MiscOp::I32TruncSatF64U:
        case MiscOp::I64TruncSatF32S:
        case MiscOp::I64TruncSatF32U:
        case MiscOp::I64TruncSatF64S:
        case MiscOp::I64TruncSatF64U:
          return true;
        case MiscOp::DataDrop:
        case MiscOp::ElemDrop:
        case MiscOp::MemoryFill:
        case MiscOp::TableGrow:
        case MiscOp::TableSize:
        case MiscOp::TableFill:
          (void)r.u32leb();
          return r.ok;
        case MiscOp::MemoryInit:
        case MiscOp::MemoryCopy:
        case MiscOp::TableInit:
        case MiscOp::TableCopy:
          (void)r.u32leb();
          (void)r.u32leb();
          return r.ok;
        default:
          return false;
      }
    }
    case Op::ThreadPrefix: {
      uint32_t sub = r.u32leb();
      if (!r.ok) return false;
      if (ThreadOp(sub) == ThreadOp::Fence) {
        (void)r.u8();  // reserved 0x00
        return r.ok;
      }
      // Notify / I32Wait / I64Wait / all atomic load/store/rmw/cmpxchg take a
      // memarg (align LEB + offset LEB).
      (void)r.u32leb();  // align
      (void)r.u32leb();  // offset
      return r.ok;
    }
    case Op::SimdPrefix: {
      uint32_t sub = r.u32leb();  // SimdOp (varU32; relaxed ops are > 0xff)
      if (!r.ok) return false;
      // Memory loads/stores + splats + load{32,64}_zero: memarg (align + offset).
      // SimdOp 0x00..0x0b = V128Load..V128Store; 0x5c/0x5d = Load32/64Zero.
      if (sub <= 0x0b || sub == 0x5c || sub == 0x5d) {
        (void)r.u32leb();  // align
        (void)r.u32leb();  // offset
        return r.ok;
      }
      // Load/store lane (0x54..0x5b): memarg + 1-byte lane index.
      if (sub >= 0x54 && sub <= 0x5b) {
        (void)r.u32leb();
        (void)r.u32leb();
        (void)r.u8();
        return r.ok;
      }
      // v128.const (0x0c) + i8x16.shuffle (0x0d): 16-byte immediate.
      if (sub == 0x0c || sub == 0x0d) {
        r.skip(16);
        return r.ok;
      }
      // extract_lane / replace_lane (0x15..0x22): 1-byte lane index.
      if (sub >= 0x15 && sub <= 0x22) {
        (void)r.u8();
        return r.ok;
      }
      // Everything else (splat/arith/compare/convert/...): no immediate.
      return r.ok;
    }
    default:
      // All remaining Core ops have no immediates.
      return true;
  }
}

static bool ParseInitExpr(Reader& r, InitExpr* out) {
  uint8_t op = r.u8();
  if (!r.ok) return false;
  out->kind = op;
  switch (Op(op)) {
    case Op::I32Const:
      out->imm = uint64_t(int64_t(r.s32leb()));
      break;
    case Op::I64Const:
      out->imm = uint64_t(r.s64leb());
      break;
    case Op::F32Const: {
      float f = r.f32();
      uint32_t b;
      memcpy(&b, &f, 4);
      out->imm = b;
      break;
    }
    case Op::F64Const: {
      double d = r.f64();
      memcpy(&out->imm, &d, 8);
      break;
    }
    case Op::GlobalGet:
      out->imm = r.u32leb();
      break;
    case Op::RefFunc:
      out->imm = r.u32leb();
      break;
    case Op::RefNull:
      (void)r.u8();
      out->imm = UINT32_MAX;
      break;
    default:
      return false;
  }
  if (!r.ok) return false;
  uint8_t end = r.u8();
  return r.ok && Op(end) == Op::End;
}

static bool DecodeLimits(Reader& r, uint32_t* min, uint32_t* max, bool* shared) {
  uint8_t flags = r.u8();
  if (!r.ok) return false;
  *shared = (flags & 0x02) != 0;
  bool hasMax = (flags & 0x01) != 0;
  if (flags & 0x04) return false;  // memory64 (unsupported)
  *min = r.u32leb();
  *max = hasMax ? r.u32leb() : UINT32_MAX;
  return r.ok;
}

// Per-function control-flow pre-pass: build the branch side-table + decode
// locals + locate body/bodyEnd. The code-section entry spans [p, entryEnd).
static bool PrepareFunc(JSContext* cx, Module* m, FuncDef& fn,
                        const uint8_t* entry, const uint8_t* entryEnd) {
  Reader r(entry, entryEnd);
  // params come from the type signature.
  const FuncType& ft = m->types[fn.typeIndex];
  fn.numParams = ft.params.length();
  for (size_t i = 0; i < ft.params.length(); i++) {
    if (!fn.localTypes.append(ft.params[i])) {
      ReportOutOfMemory(cx);
      return false;
    }
  }
  uint32_t nDecls = r.u32leb();
  for (uint32_t i = 0; i < nDecls && r.ok; i++) {
    uint32_t cnt = r.u32leb();
    VT t;
    if (!ReadVTByte(r, &t)) return false;
    for (uint32_t j = 0; j < cnt; j++) {
      if (!fn.localTypes.append(t)) {
        ReportOutOfMemory(cx);
        return false;
      }
    }
  }
  if (!r.ok) return false;
  fn.body = r.p;
  fn.bodyEnd = entryEnd;

  // Build the control side-table.
  struct Open {
    uint32_t pc;
    bool isIf;
    uint32_t elsePc;
  };
  Vector<Open, 16, SystemAllocPolicy> stack;
  Reader b(fn.body, fn.bodyEnd);
  while (b.p < b.end && b.ok) {
    uint32_t pc = uint32_t(b.p - fn.body);
    uint8_t op = b.u8();
    if (!b.ok) break;
    switch (Op(op)) {
      case Op::Block:
      case Op::Loop:
      case Op::If: {
        if (!SkipBlockType(b)) return false;
        if (!stack.append(Open{pc, Op(op) == Op::If, 0})) {
          ReportOutOfMemory(cx);
          return false;
        }
        break;
      }
      case Op::Else: {
        if (stack.empty() || !stack.back().isIf) return false;
        stack.back().elsePc = uint32_t(b.p - fn.body);
        break;
      }
      case Op::End: {
        if (stack.empty()) {
          // Function-level end: stop the run loop before this opcode.
          fn.bodyEnd = fn.body + pc;
          fn.prepared = true;
          return true;
        }
        Open o = stack.back();
        stack.popBack();
        CtrlMeta meta;
        meta.endOpPc = pc;
        meta.elsePc = o.elsePc;
        fn.ctrl[o.pc] = meta;
        break;
      }
      default:
        if (!SkipImmediates(b, op)) return false;
        break;
    }
  }
  fn.prepared = true;
  return b.ok;
}

static Module* DecodeModule(JSContext* cx, const uint8_t* bytes, size_t len) {
  Module* m = js_new<Module>();
  if (!m) {
    ReportOutOfMemory(cx);
    return nullptr;
  }
  auto guard = mozilla::MakeScopeExit([&] {
    if (m) js_delete(m);
  });
  if (!m->bytecode.append(bytes, len)) {
    ReportOutOfMemory(cx);
    return nullptr;
  }
  const uint8_t* base = m->bytecode.begin();
  Reader r(base, base + len);

  // Header.
  if (r.u32leb(), false) {
  }
  r.p = base;
  uint8_t magic[4];
  for (int i = 0; i < 4; i++) magic[i] = r.u8();
  if (!r.ok || magic[0] != 0 || magic[1] != 'a' || magic[2] != 's' ||
      magic[3] != 'm') {
    JS_ReportErrorASCII(cx, "wasm interp: bad magic");
    return nullptr;
  }
  r.skip(4);  // version

  uint32_t definedFuncCursor = 0;

  while (r.p < r.end && r.ok) {
    uint8_t id = r.u8();
    uint32_t size = r.u32leb();
    if (!r.ok || size_t(r.end - r.p) < size) {
      JS_ReportErrorASCII(cx, "wasm interp: truncated section");
      return nullptr;
    }
    const uint8_t* secEnd = r.p + size;
    Reader s(r.p, secEnd);

    switch (SectionId(id)) {
      case SectionId::Type: {
        uint32_t n = s.u32leb();
        for (uint32_t i = 0; i < n && s.ok; i++) {
          uint8_t form = s.u8();
          if (form != uint8_t(TypeCode::Func)) {
            JS_ReportErrorASCII(cx, "wasm interp: non-func type");
            return nullptr;
          }
          FuncType ft;
          uint32_t np = s.u32leb();
          for (uint32_t j = 0; j < np && s.ok; j++) {
            VT t;
            if (!ReadVTByte(s, &t) || !ft.params.append(t)) goto fail;
          }
          uint32_t nr = s.u32leb();
          for (uint32_t j = 0; j < nr && s.ok; j++) {
            VT t;
            if (!ReadVTByte(s, &t) || !ft.results.append(t)) goto fail;
          }
          if (!m->types.append(std::move(ft))) goto fail;
        }
        break;
      }
      case SectionId::Import: {
        uint32_t n = s.u32leb();
        for (uint32_t i = 0; i < n && s.ok; i++) {
          Import imp;
          imp.module = ReadName(s);
          imp.field = ReadName(s);
          uint8_t k = s.u8();
          if (!s.ok) goto fail;
          imp.kind = ExternKind(k);
          switch (imp.kind) {
            case ExternKind::Func: {
              uint32_t ti = s.u32leb();
              imp.index = ti;
              if (!m->funcTypes.append(ti)) goto fail;
              m->numImportFuncs++;
              break;
            }
            case ExternKind::Table: {
              VT et;
              if (!ReadVTByte(s, &et)) goto fail;
              uint32_t mn, mx; bool sh;
              if (!DecodeLimits(s, &mn, &mx, &sh)) goto fail;
              TableDesc td{et, mn, mx, true};
              imp.index = m->tables.length();
              if (!m->tables.append(td)) goto fail;
              m->numImportTables++;
              break;
            }
            case ExternKind::Memory: {
              uint32_t mn, mx; bool sh;
              if (!DecodeLimits(s, &mn, &mx, &sh)) goto fail;
              MemoryDesc md{mn, mx, sh, true};
              imp.index = m->memories.length();
              if (!m->memories.append(md)) goto fail;
              m->numImportMemories++;
              break;
            }
            case ExternKind::Global: {
              VT gt;
              if (!ReadVTByte(s, &gt)) goto fail;
              uint8_t mut = s.u8();
              GlobalDesc gd{gt, mut != 0, true, 0, 0};
              imp.index = m->globals.length();
              if (!m->globals.append(gd)) goto fail;
              m->numImportGlobals++;
              break;
            }
            default:
              goto fail;
          }
          if (!m->imports.append(std::move(imp))) goto fail;
        }
        break;
      }
      case SectionId::Function: {
        uint32_t n = s.u32leb();
        for (uint32_t i = 0; i < n && s.ok; i++) {
          uint32_t ti = s.u32leb();
          if (!m->funcTypes.append(ti)) goto fail;
          FuncDef fn;
          fn.typeIndex = ti;
          if (!m->defined.append(std::move(fn))) goto fail;
        }
        break;
      }
      case SectionId::Table: {
        uint32_t n = s.u32leb();
        for (uint32_t i = 0; i < n && s.ok; i++) {
          VT et;
          if (!ReadVTByte(s, &et)) goto fail;
          uint32_t mn, mx; bool sh;
          if (!DecodeLimits(s, &mn, &mx, &sh)) goto fail;
          TableDesc td{et, mn, mx, false};
          if (!m->tables.append(td)) goto fail;
        }
        break;
      }
      case SectionId::Memory: {
        uint32_t n = s.u32leb();
        for (uint32_t i = 0; i < n && s.ok; i++) {
          uint32_t mn, mx; bool sh;
          if (!DecodeLimits(s, &mn, &mx, &sh)) goto fail;
          MemoryDesc md{mn, mx, sh, false};
          if (!m->memories.append(md)) goto fail;
        }
        break;
      }
      case SectionId::Global: {
        uint32_t n = s.u32leb();
        for (uint32_t i = 0; i < n && s.ok; i++) {
          VT gt;
          if (!ReadVTByte(s, &gt)) goto fail;
          uint8_t mut = s.u8();
          InitExpr ie;
          if (!ParseInitExpr(s, &ie)) goto fail;
          GlobalDesc gd{gt, mut != 0, false, ie.kind, ie.imm};
          if (!m->globals.append(gd)) goto fail;
        }
        break;
      }
      case SectionId::Export: {
        uint32_t n = s.u32leb();
        for (uint32_t i = 0; i < n && s.ok; i++) {
          Export ex;
          ex.name = ReadName(s);
          uint8_t k = s.u8();
          ex.kind = ExternKind(k);
          ex.index = s.u32leb();
          if (!s.ok || !m->exports.append(std::move(ex))) goto fail;
        }
        break;
      }
      case SectionId::Start: {
        m->startFunc = s.u32leb();
        m->hasStart = true;
        break;
      }
      case SectionId::Elem: {
        uint32_t n = s.u32leb();
        for (uint32_t i = 0; i < n && s.ok; i++) {
          uint32_t flags = s.u32leb();
          ElemSeg seg;
          seg.active = (flags & 0x01) == 0;
          seg.declared = (flags & 0x03) == 0x03;
          seg.tableIndex = 0;
          seg.elem = VT::FuncRef;
          seg.offKind = uint8_t(Op::I32Const);
          seg.offImm = 0;
          bool useExprs = (flags & 0x04) != 0;
          if (flags & 0x02) {  // explicit table index (active) / passive marker
            if (seg.active) seg.tableIndex = s.u32leb();
          }
          if (seg.active && (flags & 0x02)) {
            // handled below for offset
          }
          if (seg.active) {
            InitExpr off;
            if (!ParseInitExpr(s, &off)) goto fail;
            seg.offKind = off.kind;
            seg.offImm = off.imm;
          }
          if (flags & 0x03) {
            // passive/declared: an elemkind or reftype byte precedes the vec
            // (except flag 0 active which has none).
            if (useExprs) {
              VT et;
              if (!ReadVTByte(s, &et)) goto fail;
              seg.elem = et;
            } else {
              (void)s.u8();  // elemkind 0x00
            }
          } else if ((flags & 0x02) && !useExprs) {
            (void)s.u8();  // active w/ table index: elemkind byte
          } else if ((flags & 0x02) && useExprs) {
            VT et;
            if (!ReadVTByte(s, &et)) goto fail;
            seg.elem = et;
          }
          uint32_t cnt = s.u32leb();
          for (uint32_t j = 0; j < cnt && s.ok; j++) {
            uint32_t fi;
            if (useExprs) {
              InitExpr e;
              if (!ParseInitExpr(s, &e)) goto fail;
              fi = (Op(e.kind) == Op::RefFunc) ? uint32_t(e.imm) : UINT32_MAX;
            } else {
              fi = s.u32leb();
            }
            if (!seg.funcIndices.append(fi)) goto fail;
          }
          if (!m->elems.append(std::move(seg))) goto fail;
        }
        break;
      }
      case SectionId::Code: {
        uint32_t n = s.u32leb();
        for (uint32_t i = 0; i < n && s.ok; i++) {
          uint32_t bodySize = s.u32leb();
          if (!s.ok || size_t(s.end - s.p) < bodySize) goto fail;
          const uint8_t* entry = s.p;
          const uint8_t* entryEnd = s.p + bodySize;
          s.p = entryEnd;
          if (definedFuncCursor >= m->defined.length()) goto fail;
          // Lazy: just record the body range; PrepareFunc runs on first call
          // (see Instance::invoke). Avoids building 362K side-tables up front.
          m->defined[definedFuncCursor].codeStart = entry;
          m->defined[definedFuncCursor].codeEnd = entryEnd;
          definedFuncCursor++;
        }
        break;
      }
      case SectionId::Data: {
        uint32_t n = s.u32leb();
        for (uint32_t i = 0; i < n && s.ok; i++) {
          uint32_t flags = s.u32leb();
          DataSeg seg;
          seg.active = (flags & 0x01) == 0;
          seg.memIndex = 0;
          seg.offKind = uint8_t(Op::I32Const);
          seg.offImm = 0;
          if (flags == 0x02) seg.memIndex = s.u32leb();
          if (seg.active) {
            InitExpr off;
            if (!ParseInitExpr(s, &off)) goto fail;
            seg.offKind = off.kind;
            seg.offImm = off.imm;
          }
          uint32_t dlen = s.u32leb();
          if (!s.ok || size_t(s.end - s.p) < dlen) goto fail;
          if (!seg.bytes.append(s.p, dlen)) goto fail;
          s.p += dlen;
          if (!m->datas.append(std::move(seg))) goto fail;
        }
        break;
      }
      case SectionId::DataCount:
      case SectionId::Custom:
      case SectionId::Tag:
      default:
        break;  // ignore
    }

    if (!s.ok) {
      JS_ReportErrorASCII(cx, "wasm interp: section decode error");
      return nullptr;
    }
    r.p = secEnd;
  }

  if (!r.ok) {
    JS_ReportErrorASCII(cx, "wasm interp: decode error");
    return nullptr;
  }

  // Per-defined-function prepared flags (thread-safe lazy PrepareFunc). Stable
  // address (atomics are non-movable); allocated once here, zero-initialized.
  if (!m->defined.empty()) {
    m->preparedFlags =
        std::make_unique<std::atomic<uint8_t>[]>(m->defined.length());
    for (size_t i = 0; i < m->defined.length(); i++) {
      m->preparedFlags[i].store(0, std::memory_order_relaxed);
    }
  }

  guard.release();
  return m;

fail:
  JS_ReportErrorASCII(cx, "wasm interp: section payload decode error");
  return nullptr;
}

}  // namespace interp
}  // namespace wasm
}  // namespace js

#endif  // wasm_WasmInterpDecode_inl_h
