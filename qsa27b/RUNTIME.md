# Qwen3.8-27B QSA native runtime snapshot

This archive preserves the exact running llama-server and eight shared libraries,
plus the complete patched source. SHA256SUMS includes every regular file.
The launcher places this archive's libraries first in LD_LIBRARY_PATH because
the captured ELF RUNPATH still refers to the original build location.

## Start

Put the selected GGUF model and shared mmproj from the sibling HF repository in one directory. The default model is Q8_0; set QSA_MODEL to select another variant:

```sh
QSA_MODEL=Qwen3.8-27B-QSA-Unsloth-UD-Q5_K_M.gguf QSA_MODEL_DIR=/path/to/gguf ./launch.sh
```

The captured reference configuration uses Q8_0, localhost:8091, two sm70 32GB GPUs, layer split 1:1, context 262144,
FP16 KV, batch2048/ubatch512, one slot, MTP draft2, lag0/async0.
QSA_MODEL, QSA_HOST, QSA_PORT, QSA_CONTEXT, QSA_BIN and CUDA_HOME can be overridden.
Extra server arguments are passed through. No daemon or system configuration is changed.
Do not run this model alongside another model that occupies the same GPUs.

The NVIDIA driver and CUDA 12.9 libraries are external dependencies. Captured
environment details are in environment.json. Other CPU/GPU/runtime combinations
are not validated by this binary snapshot.

## Build from the included source

`source/` is upstream base 6209237 plus the complete QSA patch and deployment files.
It is a full source snapshot; no earlier CUDA object cache or shallow git bundle
is needed for a fresh full build. The following is a source-build recipe, not a
claim that a fresh build will have the same binary bytes as the captured runtime.

```sh
CUDA_HOME=/usr/local/cuda-12.9 bash source/qsa27b/build.sh
```

To start the source-built binary with the same settings:

```sh
QSA_BIN="$PWD/source/build-qsa/bin/llama-server" QSA_MODEL_DIR=/path/to/gguf ./launch.sh
```

For the exact validated executable, use the archived bin/ and lib/ files.
The historical build compiled a changed CUDA object against unchanged base
objects, then linked the new C++ server against that GGML library. That cache is
not required to run this archive and is not part of the fresh-build recipe.

## Convert from BF16

Install the BF16 repository's requirements and the source converter requirements.
The custom converter preserves indexers as FP32 and folds the indexer norm as
`1 + weight`; Q8_0 is applied to eligible main-model matrices.

```sh
python source/convert_hf_to_gguf.py /path/to/Qwen3.8-27B-QSA-BF16 \
  --outtype q8_0 --outfile /path/to/new-main.gguf
python source/convert_hf_to_gguf.py /path/to/Qwen3.8-27B-QSA-BF16 \
  --mmproj --outtype f16 --outfile /path/to/new-mmproj.gguf
```

The two outputs are validated artifacts. Re-conversion across package versions
can change metadata/serialization bytes; compare tensor payloads as well as file hashes.

## Provenance and verification

The included full patch applies once to base 6209237; do not layer an older QSA
patch on top of it. qsa27b/validation contains the original small fixtures and
artifact checks. Some research drivers retain their historical filesystem layout
and must be pointed at your copied assets before use. No new 27B quality run was
performed during publication. Captured validation summaries and the measured
text-throughput result are included separately from the source fixtures.

## Indexer versions

Since 2026-11-20 the GGUF repository's model files carry indexer v2. Only the 80 FP32
indexer tensors changed; tensor names, shapes, types and all other bytes match v1, so
this runtime and source were not changed. The v1 files remain at the Hugging Face tag `v1`.
qsa27b/validation/validate-mm2-artifacts.py checks the v1 (MM2) Q8_0 file and its fixed
SHA256; it does not apply to v2 files. The v2 release checks are recorded in the GGUF
repository's validation/artifacts.json. On 2026-11-20 the v2 UD-Q4_K_M file was started
with this captured sm70 runtime on two V100 32GB GPUs (launch.sh) and on one V100: text and
image prompts used the sparse route, and two 128k recall questions were answered correctly
on the two-GPU setup. This was a startup and response check, not a quality run.

Source is MIT, with upstream third-party notices preserved. GGUF model weights
are Apache-2.0. Host CUDA SDK and NVIDIA driver binaries are not bundled.
