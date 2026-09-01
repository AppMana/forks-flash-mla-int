import os
from pathlib import Path
from datetime import datetime
import subprocess
import shutil

from setuptools import setup, find_packages

import torch


def _enable_sccache() -> None:
    """Use sccache automatically for local CUDA extension builds when present."""
    if os.getenv("FLASH_MLA_DISABLE_SCCACHE", "FALSE") == "TRUE":
        return
    sccache = shutil.which("sccache")
    if not sccache:
        return

    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    nvcc = str(Path(cuda_home) / "bin" / "nvcc")
    if Path(nvcc).exists():
        os.environ.setdefault("PYTORCH_NVCC", f"{sccache} {nvcc}")
        # sccache does not support nvcc's combined compile/dependency mode.
        os.environ.setdefault("TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES", "1")


_enable_sccache()

# ---------------------------------------------------------------- target architectures
#
# There are two kernel families in csrc/:
#   *_sm80.cu  -- the Ampere family (sparse MLA decode/prefill, fp8 + int8, dense bf16).
#                 Uses cp.async / ldmatrix / mma.m16n8k16, so sm_80 is the hard floor.
#                 Compiled for FLASH_MLA_CUDA_ARCHS.
#   *_sm90.cu  -- the Hopper family (cutlass/WGMMA). Always compiled for sm_90a.
# Everything else is compiled for sm_90a plus FLASH_MLA_CUDA_ARCHS.
#
# FLASH_MLA_CUDA_ARCHS is a comma-separated list of nvcc `code=sm_XX` names, e.g.
#   FLASH_MLA_CUDA_ARCHS=86              -> native RTX A5000 / RTX 3090 build
#   FLASH_MLA_CUDA_ARCHS=80,86,89        -> Ampere/Ada fat binary
CUDA_ARCHS = [
    a.strip()
    for a in os.getenv("FLASH_MLA_CUDA_ARCHS", "80").replace(";", ",").split(",")
    if a.strip()
]


def _gencode(archs):
    flags = []
    for a in archs:
        flags += ["-gencode", f"arch=compute_{a},code=sm_{a}"]
    return flags


def _retarget(cuda_post_cflags, archs):
    """Strip every `-gencode arch=...` pair from `cuda_post_cflags`, then add `archs`."""
    kept, i = [], 0
    while i < len(cuda_post_cflags):
        flag = cuda_post_cflags[i]
        if flag == "-gencode" and i + 1 < len(cuda_post_cflags) \
                and cuda_post_cflags[i + 1].startswith("arch="):
            i += 2
            continue
        kept.append(flag)
        i += 1
    return kept + _gencode(archs)


# Copied from https://github.com/Dao-AILab/flash-attention/blob/main/hopper/setup.py
# HACK: we monkey patch pytorch's _write_ninja_file to pass
# "-gencode arch=compute_sm90a,code=sm_90a" to files ending in '_sm90.cu',
# and pass "-gencode arch=compute_sm80,code=sm_80" to files ending in '_sm80.cu'
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_HIP_EXTENSION,
    COMMON_HIP_FLAGS,
    SUBPROCESS_DECODE_ARGS,
    IS_WINDOWS,
    get_cxx_compiler,
    _join_rocm_home,
    _join_cuda_home,
    _is_cuda_file,
    _maybe_write,
)

def _write_ninja_file(path,
                      cflags,
                      post_cflags,
                      cuda_cflags,
                      cuda_post_cflags,
                      cuda_dlink_post_cflags,
                      sources,
                      objects,
                      ldflags,
                      library_target,
                      with_cuda,
                      **kwargs  # kwargs (ignored) to absorb new flags in torch.utils.cpp_extension.
) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
    `cuda_postflags`: list of flags to append to the $nvcc invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_cuda`: If we should be compiling with CUDA.
    """
    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = sanitize_flags(cuda_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    assert len(sources) == len(objects)
    assert len(sources) > 0

    compiler = get_cxx_compiler()

    # Version 1.3 is required for the `deps` directive.
    config = ['ninja_required_version = 1.3']
    config.append(f'cxx = {compiler}')
    if with_cuda or cuda_dlink_post_cflags:
        if IS_HIP_EXTENSION:
            nvcc = _join_rocm_home('bin', 'hipcc')
        else:
            nvcc = _join_cuda_home('bin', 'nvcc')
        if "PYTORCH_NVCC" in os.environ:
            nvcc_from_env = os.getenv("PYTORCH_NVCC") # user can set nvcc compiler with ccache using the environment variable here.
        else:
            nvcc_from_env = nvcc
        config.append(f'nvcc_from_env = {nvcc_from_env}')
        config.append(f'nvcc = {nvcc}')

    if IS_HIP_EXTENSION:
        post_cflags = COMMON_HIP_FLAGS + post_cflags
    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
        # *_sm80.cu -> the requested architectures only.
        cuda_post_cflags_sm80 = _retarget(cuda_post_cflags, CUDA_ARCHS)
        flags.append(f'cuda_post_cflags_sm80 = {" ".join(cuda_post_cflags_sm80)}')
        # architecture-agnostic sources -> sm_90a plus the requested architectures.
        # 9.x entries are dropped here: nvcc rejects sm_90 and sm_90a in one invocation.
        cuda_post_cflags_sm80_sm90 = _retarget(
            cuda_post_cflags, ['90a'] + [a for a in CUDA_ARCHS if not a.startswith('9')])
        flags.append(f'cuda_post_cflags_sm80_sm90 = {" ".join(cuda_post_cflags_sm80_sm90)}')
        cuda_post_cflags_sm100 = _retarget(cuda_post_cflags, ['100a'])
        flags.append(f'cuda_post_cflags_sm100 = {" ".join(cuda_post_cflags_sm100)}')
    flags.append(f'cuda_dlink_post_cflags = {" ".join(cuda_dlink_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ['rule compile']
    if IS_WINDOWS:
        compile_rule.append(
            '  command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags')
        compile_rule.append('  deps = msvc')
    else:
        compile_rule.append(
            '  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags')
        compile_rule.append('  depfile = $out.d')
        compile_rule.append('  deps = gcc')

    if with_cuda:
        cuda_compile_rule = ['rule cuda_compile']
        nvcc_gendeps = ''
        # --generate-dependencies-with-compile is not supported by ROCm
        # Nvcc flag `--generate-dependencies-with-compile` is not supported by sccache, which may increase build time.
        if torch.version.cuda is not None and os.getenv('TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES', '0') != '1':
            cuda_compile_rule.append('  depfile = $out.d')
            cuda_compile_rule.append('  deps = gcc')
            # Note: non-system deps with nvcc are only supported
            # on Linux so use --generate-dependencies-with-compile
            # to make this work on Windows too.
            nvcc_gendeps = '--generate-dependencies-with-compile --dependency-output $out.d'
        cuda_compile_rule_sm80 = ['rule cuda_compile_sm80'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm80'
        ]
        cuda_compile_rule_sm80_sm90 = ['rule cuda_compile_sm80_sm90'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm80_sm90'
        ]
        cuda_compile_rule_sm100 = ['rule cuda_compile_sm100'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm100'
        ]
        cuda_compile_rule.append(
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags')

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        if is_cuda_source:
            if source_file.endswith('_sm90.cu'):
                rule = 'cuda_compile'
            elif source_file.endswith('_sm80.cu'):
                rule = 'cuda_compile_sm80'
            elif source_file.endswith('_sm100.cu'):
                rule = 'cuda_compile_sm100'
            else:
                rule = 'cuda_compile_sm80_sm90'
        else:
            rule = 'compile'
        if IS_WINDOWS:
            source_file = source_file.replace(':', '$:')
            object_file = object_file.replace(':', '$:')
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f'build {object_file}: {rule} {source_file}')

    if cuda_dlink_post_cflags:
        devlink_out = os.path.join(os.path.dirname(objects[0]), 'dlink.o')
        devlink_rule = ['rule cuda_devlink']
        devlink_rule.append('  command = $nvcc $in -o $out $cuda_dlink_post_cflags')
        devlink = [f'build {devlink_out}: cuda_devlink {" ".join(objects)}']
        objects += [devlink_out]
    else:
        devlink_rule, devlink = [], []

    if library_target is not None:
        link_rule = ['rule link']
        if IS_WINDOWS:
            cl_paths = subprocess.check_output(['where',
                                                'cl']).decode(*SUBPROCESS_DECODE_ARGS).split('\r\n')
            if len(cl_paths) >= 1:
                cl_path = os.path.dirname(cl_paths[0]).replace(':', '$:')
            else:
                raise RuntimeError("MSVC is required to load C++ extensions")
            link_rule.append(f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out')
        else:
            link_rule.append('  command = $cxx $in $ldflags -o $out')

        link = [f'build {library_target}: link {" ".join(objects)}']

        default = [f'default {library_target}']
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm80)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm80_sm90)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm100)  # type: ignore[possibly-undefined]
    blocks += [devlink_rule, link_rule, build, devlink, link, default]
    content = "\n\n".join("\n".join(b) for b in blocks)
    # Ninja requires a new line at the end of the .ninja file.
    content += "\n"
    _maybe_write(path, content)

# Monkey patching.
torch.utils.cpp_extension._write_ninja_file = _write_ninja_file

DISABLE_FP16 = os.getenv("FLASH_MLA_DISABLE_FP16", "FALSE") == "TRUE"
DEBUG_BUILD = os.getenv("FLASH_MLA_DEBUG", "FALSE") == "TRUE"


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return nvcc_extra_args + ["--threads", nvcc_threads]


def get_sources():
    sources = [
        "csrc/flash_api.cpp",
        "csrc/flash_api_dispatch.cu",
        "csrc/debug_imma_sm80.cu",
        "csrc/flash_sparse_mla_decode_sm80.cu",
        "csrc/flash_sparse_mla_prefill_fused_sm80.cu",
        "csrc/flash_fwd_mla_bf16_sm80.cu",
        "csrc/flash_fwd_mla_bf16_ws_sm80.cu",
        "csrc/flash_fwd_mla_bf16_sm90.cu",
        "csrc/flash_fwd_mla_metadata.cu",
    ]

    if not DISABLE_FP16:
        sources.append("csrc/flash_fwd_mla_fp16_sm90.cu")

    return sources


def get_features_args():
    features_args = []
    if DISABLE_FP16:
        features_args.append("-DFLASH_MLA_DISABLE_FP16")
    return features_args


def get_cuda_include_dirs():
    include_dirs = []
    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home:
        # CUDA lays this out per target triple, and the arm64 one is
        # sbsa-linux, not aarch64-linux. Hardcoding x86_64 silently dropped the
        # cccl include on the DGX Spark builds. Glob rather than name the
        # triple, so a future target needs no edit here.
        for cccl_include in sorted((Path(cuda_home) / "targets").glob("*/include/cccl")):
            include_dirs.append(cccl_include)
    return include_dirs


if shutil.which("git"):
    subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])
elif not (Path(__file__).parent / "csrc" / "cutlass" / "include").exists():
    raise RuntimeError("csrc/cutlass is missing and git is not available to initialize it")

cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_90a,code=sm_90a")

this_dir = os.path.dirname(os.path.abspath(__file__))

if DEBUG_BUILD:
    if IS_WINDOWS:
        cxx_args = ["/Od", "/Zi", "/std:c++17", "/W0"]
    else:
        cxx_args = ["-O0", "-g", "-std=c++17", "-Wno-deprecated-declarations"]
else:
    if IS_WINDOWS:
        cxx_args = ["/O2", "/std:c++17", "/DNDEBUG", "/W0"]
    else:
        cxx_args = ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"]

ext_modules = []
# Torch + Python stable ABI (abi3 / cp39). The binding (csrc/flash_api.cpp) uses
# only torch::stable + the AOTI shim (no ATen/c10/pybind), so the resulting wheel
# is cp39-abi3 and works on any torch >= 2.9 (the stable-ABI floor). Disable with
# FLASH_MLA_DISABLE_STABLE_ABI=TRUE to fall back to a version-specific pybind build.
STABLE_ABI = os.getenv("FLASH_MLA_DISABLE_STABLE_ABI", "FALSE") != "TRUE"
# -DUSE_CUDA exposes the CUDA-specific AOTI shim decls (aoti_torch_get_current_cuda_stream),
# which sit behind #ifdef USE_CUDA in shim.h; the symbol is present in the CUDA-built libtorch.
stable_abi_flags = ["-DPy_LIMITED_API=0x03090000", "-DTORCH_STABLE_ONLY", "-DUSE_CUDA"] if STABLE_ABI else []
ext_modules.append(
    CUDAExtension(
        name="flash_mla_cuda",
        sources=get_sources(),
        extra_compile_args={
            "cxx": cxx_args + get_features_args() + stable_abi_flags,
            "nvcc": append_nvcc_threads(
                [
                    "-O0" if DEBUG_BUILD else "-O3",
                    "-std=c++17",
                    "-D_USE_MATH_DEFINES",
                    "-Wno-deprecated-declarations",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ]
                + ([] if DEBUG_BUILD else ["-DNDEBUG", "--use_fast_math"])
                + (["-g", "-G", "-lineinfo"] if DEBUG_BUILD else [])
                + ["--ptxas-options=-v,--register-usage-level=10"]
                + cc_flag
            ) + get_features_args() + stable_abi_flags,
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "cutlass" / "include",
        ] + get_cuda_include_dirs(),
        py_limited_api=STABLE_ABI,
    )
)


# Release builds default to "2.0.0". Tagged automation can provide a PEP 440
# version through FLASH_MLA_VERSION without rewriting source files. Set
# FLASH_MLA_LOCAL_VERSION=TRUE to append a local-version segment ("+<git sha>",
# or a timestamp outside a git checkout) so throwaway development wheels stay
# distinguishable from the release.
VERSION = os.getenv("FLASH_MLA_VERSION", "2.0.0")
if os.getenv("FLASH_MLA_LOCAL_VERSION", "FALSE") == "TRUE":
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        VERSION += '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception:
        VERSION += '+' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


setup(
    name="flash_mla",
    version=VERSION,
    packages=find_packages(include=['flash_mla']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    options=({"bdist_wheel": {"py_limited_api": "cp39"}} if STABLE_ABI else {}),
)
