# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SGLang is a high-performance serving framework for large language models (LLMs) and vision-language models (VLMs). It provides both a backend runtime engine for efficient inference and a frontend language for structured generation.

## Repository Structure

The repository consists of three main packages:

- **`python/sglang/`** - Main Python package
  - `srt/` - SGLang Runtime (SRT) - the backend inference engine
  - `lang/` - Frontend language for structured generation
  - `cli/` - Command-line interface
  - `bench_*.py` - Benchmarking scripts

- **`sgl-kernel/`** - Separate Python package for optimized CUDA/C++ kernels

- **`test/`** - Test suites organized by type

## Key Architecture Components

### Backend Runtime (srt/)

The inference engine uses a multi-process architecture:

- **entrypoints/** - Server entry points (HTTP via FastAPI, gRPC, engine API)
- **managers/** - Core runtime managers
  - `scheduler.py` - Main scheduler with continuous batching
  - `tokenizer_manager.py` - Handles tokenization
  - `detokenizer_manager.py` - Handles detokenization
  - `data_parallel_controller.py` - Data parallelism coordination
- **models/** - Model implementations (100+ models supported)
  - `registry.py` - Model registration system
- **layers/** - Neural network layers (attention, MoE, quantization)
- **mem_cache/** - KV cache management with RadixAttention

### Frontend Language (lang/)

- `api.py` - Public API (`gen`, `select`, `function`, etc.)
- `backend/` - Backend connectors (RuntimeEndpoint, OpenAI, Anthropic)
- `interpreter.py` - Executes structured generation programs

## Common Commands

### Installation from Source

```bash
pip install -e "python"
```

### Running the Server

```bash
# Via CLI
sglang serve meta-llama/Llama-3.1-8B-Instruct

# Via Python module
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct
```

### Running Tests

```bash
# Run a single test file
python test/srt/test_srt_endpoint.py

# Run a single test method
python test/srt/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# Run a test suite
python test/run_suite.py --hw cuda --suite stage-b-test-small-1-gpu
```

Tests use Python's `unittest` framework. Add `unittest.main()` for unittest or `sys.exit(pytest.main([__file__]))` for pytest at the end of test files.

### Code Formatting

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Uses isort, black, ruff (F401, F821 only), and clang-format for C++/CUDA.

### Building sgl-kernel

```bash
cd sgl-kernel
make build

# Limit parallelism (for memory-constrained builds)
make build MAX_JOBS=2
```

## Development Guidelines

### Performance-Critical Code

- Minimize CPU-GPU synchronization (`tensor.item()`, `tensor.cpu()`)
- Cache runtime check results as booleans when they're repeated across layers
- Use vectorized operations
- Keep functions pure when possible

### Code Organization

- Files should not exceed 2,000 lines; split into mixins if needed (e.g., `scheduler.py` + `scheduler_*_mixin.py`)
- Test files taking >500 seconds should be split
- Avoid code duplication; extract shared snippets (>5 lines) into functions

### Adding New Models

Models are registered via the `EntryClass` attribute. Add implementation in `srt/models/` following existing patterns. The `TransformersForCausalLM` backend serves as a fallback.

### Adding New Kernels to sgl-kernel

1. Implement kernel in `sgl-kernel/csrc/`
2. Expose interface in `sgl-kernel/include/sgl_kernel_ops.h`
3. Create torch extension in `csrc/common_extension.cc`
4. Update `CMakeLists.txt`
5. Add Python bindings in `sgl-kernel/python/sgl_kernel/`
6. Add tests in `sgl-kernel/tests/`

Since sglang and sgl-kernel are separate packages, kernel changes require multiple PRs: first to update sgl-kernel, then to bump the version and use it in sglang.

### Test Organization

- `test/srt/` - Backend runtime tests
- `test/unit/` - Unit tests
- `test/registered/` - Registry-based CI tests with `register_cuda_ci()`, `register_amd_ci()`, etc.
- `test/manual/` - Manual tests not in CI

CI suites: `stage-b-test-small-1-gpu` (5090), `stage-b-test-large-1-gpu` (H100), `stage-b-test-large-2-gpu`, nightly suites for larger configurations.

## Environment Variables

Key environment variables are defined in `srt/environ.py`. Use `envs.VARIABLE_NAME.get()` pattern to access them.
