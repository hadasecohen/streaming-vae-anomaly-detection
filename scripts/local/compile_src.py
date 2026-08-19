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
    python scripts/local/compile_src.py --no-zip     # skip the final .zip
    python scripts/local/compile_src.py --keep-py    # don't delete original .py after compiling
    python scripts/local/compile_src.py --keep-c     # keep Cython's intermediate .c files

Runs the full pipeline in one call: copy src/ -> compile every .py file with
Cython -> verify every file that should have compiled has a .pyd/.so ->
delete the original .py source (compiled modules would otherwise be
silently ignored in favor of it) -> delete intermediate .c files and the
build/temp.*, build/lib.* scratch dirs -> zip the result. This is meant to
be the single command you run to produce a distributable compiled_src.zip
(e.g. from scripts/compile_test_colab.ipynb on Colab, to build for Colab's
own Linux/Python target).

Output:
    A full copy of src/ under --out (default: build/compiled_src/src), with
    every .py file replaced by its compiled extension module, plus
    <out>.zip (default: build/compiled_src.zip) containing that src/ tree.
    __init__.py files are copied as plain Python (they're empty package
    markers; Cython compiles them fine too, but there's no benefit and it's
    one less thing to rebuild). src/Data_tools/dataset_loader.py and
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

    import distutils.ccompiler
    compiler = distutils.ccompiler.new_compiler()
    if hasattr(compiler, "initialize"):
        # MSVC-specific: actually probes for a usable install (vs. just
        # picking the class for this platform), so failures here are real.
        try:
            compiler.initialize()
        except Exception as e:
            sys.exit(
                "No working C compiler found (needed to build the compiled "
                f"extension modules). Underlying error: {e}\n"
                "Install a C compiler first (see this script's docstring)."
            )
    else:
        # Unix compilers (gcc/clang) have no such probe — new_compiler()
        # picks the class unconditionally, so check the actual binary exists.
        import shutil
        exe = (compiler.compiler_cxx or compiler.compiler)[0] if (compiler.compiler_cxx or compiler.compiler) else None
        if not exe or not shutil.which(exe):
            sys.exit(
                f"No working C compiler found on PATH (looked for {exe!r}).\n"
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


def _step1_copy_src(src_dir: str, out_src_dir: str, out_dir: str):
    print("[1/6] Copying src/ ...")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(src_dir, out_src_dir)


def _step2_compile(out_dir: str, out_src_dir: str):
    from Cython.Build import cythonize

    py_files, init_files = _collect_source_files(out_src_dir)
    py_files = [f for f in py_files if os.path.relpath(f, out_dir) not in KNOWN_UNSUPPORTED]

    print(f"[2/6] Compiling {len(py_files)} files ({len(init_files)} __init__.py left as plain Python):")
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

    return py_files


def _step3_verify(out_dir: str, out_src_dir: str, expected_py_files):
    compiled = glob.glob(os.path.join(out_src_dir, "**", "*.pyd"), recursive=True) + \
               glob.glob(os.path.join(out_src_dir, "**", "*.so"), recursive=True)
    print(f"[3/6] Verifying: {len(compiled)} compiled modules for {len(expected_py_files)} source files")
    if len(compiled) != len(expected_py_files):
        sys.exit(
            f"Compiled module count ({len(compiled)}) doesn't match the number of "
            f"files fed to Cython ({len(expected_py_files)}). Check the build output "
            "above for a file that silently failed to produce a .pyd/.so."
        )
    return compiled


def _step4_delete_py(out_src_dir: str, keep_py: bool):
    if keep_py:
        print("[4/6] --keep-py set: leaving original .py source in place")
        return
    removed = 0
    for f in glob.glob(os.path.join(out_src_dir, "**", "*.py"), recursive=True):
        if os.path.basename(f) != "__init__.py":
            os.remove(f)
            removed += 1
    print(f"[4/6] Deleted {removed} original .py files (kept __init__.py package markers) "
          "so the compiled modules aren't shadowed by source on import")


def _step5_clean(out_dir: str, out_src_dir: str, keep_c: bool):
    if keep_c:
        print("[5/6] --keep-c set: leaving intermediate .c files and build/ scratch dirs")
        return
    for f in glob.glob(os.path.join(out_src_dir, "**", "*.c"), recursive=True):
        os.remove(f)
    build_tmp = os.path.join(out_dir, "build")
    if os.path.isdir(build_tmp):
        shutil.rmtree(build_tmp)
    print("[5/6] Removed intermediate .c files and build/ scratch directories")


def _step6_zip(out_dir: str, make_zip: bool):
    if not make_zip:
        print("[6/6] --no-zip set: skipping archive")
        return None
    zip_base = out_dir  # shutil.make_archive appends .zip
    archive_path = shutil.make_archive(zip_base, "zip", out_dir)
    print(f"[6/6] Wrote {archive_path}")
    return archive_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "build", "compiled_src"),
                        help="Output directory (default: build/compiled_src)")
    parser.add_argument("--no-zip", action="store_true", help="Skip producing <out>.zip")
    parser.add_argument("--keep-py", action="store_true",
                        help="Don't delete the original .py files after compiling "
                             "(they'd otherwise shadow the compiled modules on import)")
    parser.add_argument("--keep-c", action="store_true",
                        help="Don't delete Cython's intermediate .c files / build/ scratch dirs")
    args = parser.parse_args()

    _check_prereqs()

    src_dir = os.path.join(REPO_ROOT, "src")
    if not os.path.isdir(src_dir):
        sys.exit(f"src/ not found at {src_dir}")

    out_dir = os.path.abspath(args.out)
    out_src_dir = os.path.join(out_dir, "src")

    _step1_copy_src(src_dir, out_src_dir, out_dir)
    py_files = _step2_compile(out_dir, out_src_dir)
    _step3_verify(out_dir, out_src_dir, py_files)
    _step4_delete_py(out_src_dir, args.keep_py)
    _step5_clean(out_dir, out_src_dir, args.keep_c)
    archive_path = _step6_zip(out_dir, not args.no_zip)

    print(f"\nDone. Compiled src/ ready under {out_src_dir}")
    if archive_path:
        print(f"Download/ship: {archive_path}")


if __name__ == "__main__":
    main()
