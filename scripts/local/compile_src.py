#!/usr/bin/env python3
"""
Compile src/ into platform-native extension modules (.pyd on Windows, .so on
Linux/macOS) via Cython, for shipping a closed-source build of the package
(e.g. into a public repo/Colab image) while keeping the .py source private.

Requires (not installed by this script):
    pip install cython
    A C compiler:
      - Windows: MSVC Build Tools ("Desktop development with C++" workload,
        via https://visualstudio.microsoft.com/visual-cpp-build-tools/)
      - Linux:   gcc/build-essential
      - macOS:   Xcode Command Line Tools (xcode-select --install)

Usage (from the project root):
    python scripts/local/compile_src.py
    python scripts/local/compile_src.py --out dist/compiled_src
    python scripts/local/compile_src.py --clean   # remove build artifacts after building

Output:
    A full copy of src/ under --out (default: build/compiled_src/src), with
    every .py file replaced by its compiled extension module. __init__.py
    files are copied as plain Python (they're empty package markers; Cython
    compiles them fine too, but there's no benefit and it's one less thing
    to rebuild). src/Data_tools/dataset_loader.py and
    src/env/streaming_env_optuna.py used to contain match/case statements,
    which Cython does not support (a hard compiler crash, not a graceful
    error) — already rewritten to if/elif. If match/case reappears in future
    edits to those files (or any other src/ file), it needs converting back
    before compiling.

The annotation_typing=False Cython directive is set deliberately: this
codebase's type hints aren't reliable contracts everywhere (found during the
initial feasibility test — e.g. a parameter typed as `float` that's always
actually passed a `list`, since fixed). Plain Python never enforces
annotations at runtime; Cython does by default. Without this flag, calls
that work fine against the uncompiled source can raise spurious TypeErrors
against the compiled one.

Platform note: the artifacts this produces are tied to the OS, CPU
architecture, and Python minor version they were built on. A .pyd built here
on Windows/Python 3.11 will NOT load on Linux, macOS, or a different Python
minor version. For a Colab-only distribution, build this on a matching
Linux/x86_64/Python image (e.g. via a Colab notebook cell, or a matching
Docker image / GitHub Actions Linux runner) rather than on a local Windows
machine.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Known-unsupported constructs as of Cython 3.2.9 — match/case triggers a
# hard compiler crash (not a graceful error). If a file appears here, either
# rewrite it to avoid match/case, or drop it from compilation (it'll need to
# ship as plain .py alongside the compiled modules, which weakens the "no
# readable source" goal for that file specifically).
KNOWN_UNSUPPORTED = set()


def _check_prereqs():
    try:
        import Cython  # noqa: F401
    except ImportError:
        sys.exit("Cython is not installed. Run: pip install cython")

    try:
        import setuptools.dist
        import distutils.ccompiler
        compiler = distutils.ccompiler.new_compiler()
        compiler.initialize()
    except Exception as e:
        sys.exit(
            "No working C compiler found (needed to build the compiled "
            f"extension modules). Underlying error: {e}\n"
            "Install a C compiler first (see this script's docstring)."
        )


def _collect_source_files(src_dir: str):
    py_files = []
    init_files = []
    for f in glob.glob(os.path.join(src_dir, "**", "*.py"), recursive=True):
        if os.path.basename(f) == "__init__.py":
            init_files.append(f)
        else:
            py_files.append(f)
    return py_files, init_files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "build", "compiled_src"),
                        help="Output directory (default: build/compiled_src)")
    parser.add_argument("--clean", action="store_true",
                        help="Remove intermediate build artifacts (.c files, build/temp.*) after compiling, "
                             "keeping only the final .pyd/.so files")
    args = parser.parse_args()

    _check_prereqs()
    from Cython.Build import cythonize

    src_dir = os.path.join(REPO_ROOT, "src")
    if not os.path.isdir(src_dir):
        sys.exit(f"src/ not found at {src_dir}")

    out_dir = os.path.abspath(args.out)
    out_src_dir = os.path.join(out_dir, "src")

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(src_dir, out_src_dir)

    py_files, init_files = _collect_source_files(out_src_dir)
    py_files = [f for f in py_files if os.path.relpath(f, out_dir) not in KNOWN_UNSUPPORTED]

    print(f"Compiling {len(py_files)} files ({len(init_files)} __init__.py left as plain Python):")
    for f in sorted(py_files):
        print("  ", os.path.relpath(f, out_dir))

    cwd = os.getcwd()
    os.chdir(out_dir)
    try:
        rel_py_files = [os.path.relpath(f, out_dir) for f in py_files]
        ext_modules = cythonize(
            rel_py_files,
            language_level="3",
            compiler_directives={
                "binding": True,
                # See module docstring: this codebase's type hints aren't
                # reliable contracts everywhere, so don't enforce them.
                "annotation_typing": False,
            },
        )

        from setuptools import setup
        # setup() reads argv; force build_ext --inplace regardless of how
        # this script itself was invoked.
        sys.argv = [sys.argv[0], "build_ext", "--inplace"]
        setup(ext_modules=ext_modules)
    finally:
        os.chdir(cwd)

    if args.clean:
        for f in glob.glob(os.path.join(out_src_dir, "**", "*.c"), recursive=True):
            os.remove(f)
        build_tmp = os.path.join(out_dir, "build")
        if os.path.isdir(build_tmp):
            shutil.rmtree(build_tmp)

    compiled = glob.glob(os.path.join(out_src_dir, "**", "*.pyd"), recursive=True) + \
               glob.glob(os.path.join(out_src_dir, "**", "*.so"), recursive=True)
    print(f"\nDone. {len(compiled)} compiled modules written under {out_src_dir}")
    print("Original .py files for the compiled modules are still present alongside them")
    print("(Python prefers .py over .pyd/.so) — delete them before shipping this tree,")
    print("or the compiled modules will be silently ignored in favor of the source.")


if __name__ == "__main__":
    main()
