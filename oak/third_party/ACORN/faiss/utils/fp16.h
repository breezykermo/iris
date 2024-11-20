#pragma once

#include <cstdint>

#include "oak/third_party/ACORN/faiss/impl/platform_macros.h"

#if defined(__F16C__)
#include "oak/third_party/ACORN/faiss/utils/fp16-fp16c.h"
#else
#include "oak/third_party/ACORN/faiss/utils/fp16-inl.h"
#endif
