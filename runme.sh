#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="${ROOT_DIR}/artifacts"
mkdir -p "${ARTIFACT_DIR}"

if [[ -z "${SIMD_BENCHMARK_CSV:-}" ]]; then
    SIMD_BENCHMARK_CSV="${ARTIFACT_DIR}/benchmark_results.csv"
else
    csv_path="${SIMD_BENCHMARK_CSV}"
    if [[ "${csv_path}" != /* ]]; then
        csv_path="${ROOT_DIR}/${csv_path}"
    fi
    SIMD_BENCHMARK_CSV="${csv_path}"
fi
rm -f "${SIMD_BENCHMARK_CSV}"
export SIMD_BENCHMARK_CSV

mapfile -t examples < <(cd "$ROOT_DIR" && find src -type f -name 'main.cpp' -printf '%h\n' | sort)

if (( ${#examples[@]} == 0 )); then
    echo "No example directories with main.cpp found." >&2
    exit 1
fi

failures=()

for example in "${examples[@]}"; do
    echo
    echo "=== Building and running ${example} ==="
    pushd "$ROOT_DIR/$example" > /dev/null

    if [ ! -f Makefile ]; then
        echo "Skipping ${example}: Makefile not found." >&2
        failures+=("${example}: missing Makefile")
        popd > /dev/null
        continue
    fi

    if ! make clean >/dev/null 2>&1; then
        echo "make clean failed for ${example}" >&2
        failures+=("${example}: make clean failed")
        popd > /dev/null
        continue
    fi

    if ! make; then
        echo "make failed for ${example}" >&2
        failures+=("${example}: make failed")
        popd > /dev/null
        continue
    fi

    if [ ! -x ./simd_program ]; then
        echo "Executable simd_program not produced in ${example}" >&2
        failures+=("${example}: missing simd_program")
        popd > /dev/null
        continue
    fi

    if ! ./simd_program; then
        echo "Execution failed for ${example}" >&2
        failures+=("${example}: execution failed")
        popd > /dev/null
        continue
    fi

    popd > /dev/null
    echo "--- Completed ${example} ---"

    if [[ -n "${KEEP_BUILD_ARTIFACTS:-}" ]]; then
        continue
    fi

    pushd "$ROOT_DIR/$example" > /dev/null
    make clean >/dev/null 2>&1 || true
    popd > /dev/null
    echo "Cleaned ${example} artifacts."

done

if (( ${#failures[@]} )); then
    echo
    echo "Failures detected:" >&2
    for entry in "${failures[@]}"; do
        echo " - ${entry}" >&2
    done
    exit 1
fi

echo
echo "All SIMD examples built and ran successfully."
if [[ -n "${SIMD_BENCHMARK_CSV:-}" && -f "${SIMD_BENCHMARK_CSV}" ]]; then
    echo "Benchmark CSV saved to ${SIMD_BENCHMARK_CSV}"
fi

echo
echo "Generating plots via Python scripts..."
python3 "${ROOT_DIR}/scripts/plot_results.py" \
    --benchmarks-csv "${SIMD_BENCHMARK_CSV}" \
    --benchmarks-output "${ARTIFACT_DIR}/benchmark_speedups.png" \
    --attention-csv "${ARTIFACT_DIR}/attention_components.csv" \
    --attention-output "${ARTIFACT_DIR}/attention_speedups.png"
echo "Plots saved to ${ARTIFACT_DIR}"
