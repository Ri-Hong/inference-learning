# Phase 1 — CUDA fundamentals
#
#   make            build everything into bin/
#   make vector_add build one target
#   make run        build and run all of them in order
#   make clean
#
# ARCH defaults to `native`, which asks nvcc to compile for the GPU present at
# build time. Override it if you are cross-compiling:
#
#   make ARCH=sm_75    # T4
#   make ARCH=sm_80    # A100
#   make ARCH=sm_86    # A4000 / A5000 / A6000 / A40 / 3090
#   make ARCH=sm_89    # L4 / L40S / 4090
#   make ARCH=sm_90    # H100 / H200

NVCC    ?= nvcc
ARCH    ?= native
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo
LDFLAGS   :=

BIN := bin

TARGETS := vector_add transpose reduction matmul streams

vector_add_SRC := 01_vector_add/vector_add.cu
transpose_SRC  := 02_transpose/transpose.cu
reduction_SRC  := 03_reduction/reduction.cu
matmul_SRC     := 04_matmul/matmul.cu
streams_SRC    := 05_streams/streams.cu

matmul_LIBS := -lcublas

.PHONY: all run clean $(TARGETS)

all: $(addprefix $(BIN)/,$(TARGETS))

$(BIN):
	mkdir -p $(BIN)

define BUILD_RULE
$(BIN)/$(1): $$($(1)_SRC) common/cuda_utils.cuh | $(BIN)
	$$(NVCC) $$(NVCCFLAGS) $$< -o $$@ $$($(1)_LIBS) $$(LDFLAGS)

$(1): $(BIN)/$(1)
endef

$(foreach t,$(TARGETS),$(eval $(call BUILD_RULE,$(t))))

run: all
	@for t in $(TARGETS); do \
		echo ""; \
		echo "############################################################"; \
		echo "# $$t"; \
		echo "############################################################"; \
		./$(BIN)/$$t || exit 1; \
	done

clean:
	rm -rf $(BIN)
