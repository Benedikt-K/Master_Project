# Standalone Direction Prediction Tool

This folder contains a CLI for predicting CRISPR array direction. It is based on a finetuned version of the Carbon-500m model.

It is intended for inference only.

## Quick Start

Recommended: create a dedicated Python environment first.

Conda:

```bash
conda create -n crispr-standalone python=3.12 -y
conda activate crispr-standalone
pip install -r Standalone/requirements.txt
```

First run: 

```bash
python Standalone/predict_direction.py \
  --input_file Standalone/test.jsonl \
  --allow_downloads
```

After the first run it can be run without the --allow_downloads flag, as it downloads base model weights to `Standalone/model_params/base_model_cache` and resuses them from there.

The tool will:

- Print a summary to stdout.
- Write a JSON file named `prediction_result.json` next to the input file.

## Device Behavior

Default behavior:

- Uses GPU if CUDA is available, otherwise CPU.

Overrides:

- `--cpu` forces CPU.
- `--gpu` forces GPU and errors if CUDA is unavailable.
- `--allow_downloads` allows first-run download of base model weights when local full weights are
  missing.

Important for first run:

- Run once with `--allow_downloads` if `model_params/base_model_cache` does not already contain full base weights.
- Later runs can be offline and reuse cached files.

## Supported Input Types

## 1) Standard JSON / JSONL

Accepted formats:

1. Direct example object:

```json
{
  "array_name": "my_array",
  "group_name": "my_group",
  "cas_subtype": "I-E",
  "repeats": ["CGGTTTATCCCCGCTGGCGCGGGGAACTC", "CGGTTTATCCCCGCTGGCGCGGGGAACTC"],
  "spacers": ["CAGCGTCAGGCGTGAAATCTCACCGTCGTTGC"]
}
```

2. Wrapped example object:

```json
{
  "example": {
    "repeats": ["..."],
    "spacers": ["..."]
  }
}
```

Notes:

- Multi-line JSON files are supported.
- For JSONL files, the first non-empty line is used.

## 2) CRISPRCasFinder Result JSON (`--ccf`)

You can pass Results from CRISPRCasFinder directly, for that only the `result.json` file is used.

Example:

```bash
python Standalone/predict_direction.py \
  --input_file test_out/Result_XXX/result.json \
  --ccf
```

If multiple entries exist, select which one to predict:

```bash
python Standalone/predict_direction.py \
  --input_file test_out/Result_XXX/result.json \
  --ccf \
  --ccf_sequence_index 0 \
  --ccf_crispr_index 0
```

CCF extraction mapping used by the script:

- `Sequences[i].Crisprs[j].Regions` entries with `Type=DR` -> repeats
- `Type=Spacer` -> spacers
- `Type=LeftFLANK` / `Type=RightFLANK` -> flanks
- If DR list is missing, `DR_Consensus` is used as fallback

## Output

## Console output

Always text summary, for example:

```text
Predicted direction is "Forward" with probability 0.998 (99.8%).
Alternative direction "Reverse" has probability 0.002 (0.2%).
Input summary: repeats=31, spacers=30, tokens=367, sequence_mode=interleaved, include_flanks=False.
Array name: NZ_CP123870_1
Input file: <project-root>/test_out/Result_XXX/result.json
Saved result JSON to: <project-root>/test_out/Result_XXX/prediction_result.json
```

## Result JSON file

Location:

- Default: `<input_file_directory>/prediction_result.json`
- Custom: `--result_file path/to/result.json`

Core fields in output JSON:

- `input_mode` (`standard` or `ccf`)
- `input_file`
- `model_dir`
- `base_model`
- `device`
- `array_name`, `group_name`, `cas_subtype`
- `predicted_label_id`, `predicted_label`
- `prob_reverse`, `prob_forward`
- `token_count`
- `ccf` metadata block (present only with `--ccf`)

Label mapping:

- `1 = Forward`
- `0 = Reverse`

## CLI Reference

```bash
python Standalone/predict_direction.py --input_file PATH [options]
```

Options:

- `--input_file PATH` (required): Input JSON or JSONL.
- `--ccf`: Interpret input as CRISPRCasFinder `result.json`.
- `--ccf_sequence_index N`: CCF `Sequences` index (default `0`).
- `--ccf_crispr_index M`: CCF `Crisprs` index within the selected sequence (default `0`).
- `--model_dir PATH`: Model directory (default: `Standalone/model_params`).
- `--sequence_mode interleaved|spacers_only`: Sequence construction mode (default `interleaved`).
- `--include_flanks`: Include left/right flanks in constructed sequence.
- `--max_length N`: Tokenizer truncation length (default `256`).
- `--cpu`: Force CPU.
- `--gpu`: Force GPU (fails if CUDA is unavailable).
- `--result_file PATH`: Custom output JSON path.
- `--allow_downloads`: Allow online fallback if local assets are missing.

LoRA runtime notes:

- If `adapter_config.json` and `adapter_model.safetensors` exist in `model_params`, the adapter is
  loaded automatically.
- If adapter files are present but `peft` is not installed, install it with:

```bash
pip install peft
```

## Practical Examples

Custom JSON:

```bash
python Standalone/predict_direction.py \
  --input_file Standalone/my_array.json
```

Force CPU and custom result path:

```bash
python Standalone/predict_direction.py \
  --input_file Standalone/my_array.json \
  --cpu \
  --result_file Standalone/my_array_prediction.json
```

## Included Assets

- Predictor script: [predict_direction.py](predict_direction.py)
- LoRA adapter + tokenizer assets: [model_params](model_params)

- If full base weights are already cached in `Standalone/model_params/base_model_cache`, they are reused.
- If full weights are missing, the script downloads it from Hugging Face on first run,
  and stores the weights in `Standalone/model_params/base_model_cache`.
- If adapter files are present (`adapter_config.json` + `adapter_model.safetensors`), LoRA is
  applied automatically on top of the cached base model.

## Troubleshooting

`--gpu was requested, but CUDA is not available`:

- Retry with `--cpu`.

Input validation errors:

- Standard mode: verify `repeats` and `spacers` are lists of strings.
- CCF mode: verify file contains `Sequences -> Crisprs -> Regions`.

Unexpected downloads:

- Do not pass `--allow_downloads` when offline-only behavior is required.