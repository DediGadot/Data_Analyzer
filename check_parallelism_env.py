#!/usr/bin/env python3
import os

# Check if any environment variables are limiting parallelism
print("=== Environment Variables That Limit Parallelism ===")
limiting_vars = {
    "OMP_NUM_THREADS": "OpenMP threads",
    "MKL_NUM_THREADS": "Intel MKL threads", 
    "NUMBA_NUM_THREADS": "Numba JIT threads",
    "OPENBLAS_NUM_THREADS": "OpenBLAS threads",
    "BLAS_NUM_THREADS": "BLAS threads",
    "VECLIB_MAXIMUM_THREADS": "Apple vecLib threads",
    "NUMEXPR_NUM_THREADS": "NumExpr threads"
}

issues_found = []
for var, desc in limiting_vars.items():
    value = os.environ.get(var, "Not set")
    if value != "Not set":
        print(f"⚠️  {var}={value} ({desc}) - MAY LIMIT PARALLELISM")
        if value == "1":
            issues_found.append(var)
    else:
        print(f"✓ {var} not set ({desc})")

print()
if issues_found:
    print("=== ISSUES FOUND ===")
    print("The following variables are set to 1, which limits parallelism:")
    for var in issues_found:
        print(f"  - {var}")
    print()
    print("=== FIXES ===")
    print("Set these environment variables to enable parallelism:")
    for var in issues_found:
        print(f"export {var}=4")
    print()
    print("Or add this to your ~/.bashrc:")
    for var in issues_found:
        print(f"echo 'export {var}=4' >> ~/.bashrc")
else:
    print("=== NO ISSUES FOUND ===")
    print("Environment variables are not limiting parallelism.")

# Also check if we need to explicitly set parallelism
print()
print("=== PARALLELISM ENABLEMENT ===")
print("You can also explicitly enable parallelism by setting:")
print("export OMP_NUM_THREADS=4")
print("export MKL_NUM_THREADS=4") 
print("export NUMBA_NUM_THREADS=4")