# GPU-accelerated Monte Carlo options pricing engine (CUDA C).
#
#   make                 build bin/mc_pricer and bin/test_pricer (needs nvcc)
#   make ARCH=sm_75      build for a specific GPU (default: the GPU in this machine)
#                        RTX 2080 Ti = sm_75, A30/A100 = sm_80, RTX 30xx = sm_86, RTX 40xx = sm_89
#   make FAST_MATH=1     build the kernels with -use_fast_math (__expf, __logf, ...)
#   make MARCH=          portable CPU code (use when compiling on a different machine
#                        than the one that runs it, e.g. a cluster login node)
#   make test            build and run the test suite
#   make regs            per-kernel register / shared memory / spill usage (ptxas -v)
#   make debug           -G -g build for cuda-gdb / compute-sanitizer
#   make cpu             CPU-only build, no CUDA needed: bin/mc_pricer_cpu, bin/test_pricer_cpu
#   make clean

NVCC      ?= nvcc
CC        ?= gcc
ARCH      ?= native

MARCH     ?= -march=native
CFLAGS    ?= -O3
CFLAGS    += $(MARCH) -std=gnu11 -fopenmp -Wall -Wextra -Iinclude
NVCCFLAGS ?= -O3 -lineinfo
NVCCFLAGS += -arch=$(ARCH) -Iinclude
ifeq ($(FAST_MATH),1)
NVCCFLAGS += -use_fast_math
endif
LDLIBS    := -lcurand -lgomp -lm

BUILD     := build
BIN       := bin

HOST_SRC  := src/black_scholes.c src/cpu_mc.c
HOST_OBJ  := $(HOST_SRC:src/%.c=$(BUILD)/%.o)
GPU_OBJ   := $(BUILD)/gpu_mc.o
HEADERS   := $(wildcard include/*.h)

.PHONY: all test regs debug cpu test-cpu clean

all: $(BIN)/mc_pricer $(BIN)/test_pricer

$(BUILD) $(BIN):
	mkdir -p $@

$(BUILD)/%.o: src/%.c $(HEADERS) | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/test_pricer.o: tests/test_pricer.c $(HEADERS) | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(GPU_OBJ): src/gpu_mc.cu $(HEADERS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN)/mc_pricer: $(BUILD)/main.o $(HOST_OBJ) $(GPU_OBJ) | $(BIN)
	$(NVCC) -arch=$(ARCH) $^ -o $@ $(LDLIBS)

$(BIN)/test_pricer: $(BUILD)/test_pricer.o $(HOST_OBJ) $(GPU_OBJ) | $(BIN)
	$(NVCC) -arch=$(ARCH) $^ -o $@ $(LDLIBS)

test: $(BIN)/test_pricer
	./$(BIN)/test_pricer

regs: | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -Xptxas -v -c src/gpu_mc.cu -o $(BUILD)/gpu_mc_regs.o 2>&1 \
	  | grep -E "Compiling entry|Used|spill" | sed -e 's/ptxas info *: //'

debug: | $(BUILD) $(BIN)
	$(CC) $(CFLAGS) -O0 -g -c src/main.c -o $(BUILD)/main_dbg.o
	$(CC) $(CFLAGS) -O0 -g -c src/black_scholes.c -o $(BUILD)/black_scholes_dbg.o
	$(CC) $(CFLAGS) -O0 -g -c src/cpu_mc.c -o $(BUILD)/cpu_mc_dbg.o
	$(NVCC) -G -g -arch=$(ARCH) -Iinclude -c src/gpu_mc.cu -o $(BUILD)/gpu_mc_dbg.o
	$(NVCC) -G -g -arch=$(ARCH) $(BUILD)/main_dbg.o $(BUILD)/black_scholes_dbg.o $(BUILD)/cpu_mc_dbg.o \
	  $(BUILD)/gpu_mc_dbg.o -o $(BIN)/mc_pricer_debug $(LDLIBS)

# ---- CPU-only build (no CUDA toolkit required) ----
CPU_BUILD := $(BUILD)/cpu

$(CPU_BUILD):
	mkdir -p $@

$(CPU_BUILD)/%.o: src/%.c $(HEADERS) | $(CPU_BUILD)
	$(CC) $(CFLAGS) -DMC_NO_CUDA -c $< -o $@

$(CPU_BUILD)/test_pricer.o: tests/test_pricer.c $(HEADERS) | $(CPU_BUILD)
	$(CC) $(CFLAGS) -DMC_NO_CUDA -c $< -o $@

$(BIN)/mc_pricer_cpu: $(CPU_BUILD)/main.o $(HOST_OBJ:$(BUILD)/%=$(CPU_BUILD)/%) | $(BIN)
	$(CC) $(CFLAGS) $^ -o $@ -lm

$(BIN)/test_pricer_cpu: $(CPU_BUILD)/test_pricer.o $(HOST_OBJ:$(BUILD)/%=$(CPU_BUILD)/%) | $(BIN)
	$(CC) $(CFLAGS) $^ -o $@ -lm

cpu: $(BIN)/mc_pricer_cpu $(BIN)/test_pricer_cpu

test-cpu: $(BIN)/test_pricer_cpu
	./$(BIN)/test_pricer_cpu

clean:
	rm -rf $(BUILD) $(BIN)
