# Standalone Direction Prediction Tool

This folder contains a CLI for predicting CRISPR array direction. It is based on a finetuned version of the Carbon-500m model, 
which was finetuned on the predictions of 

It is intended for inference only.

## Quick Start

Recommended: create a dedicated Python environment first.

Conda:

```bash
conda create -n crispr-standalone python=3.12 -y
conda activate crispr-standalone
pip install -r Standalone/requirements.txt
```

For the first run you can test, if everything works on your end with this example array:

```bash
python Standalone/predict_direction.py \
  --input_file Standalone/example_array.json \
  --allow_downloads
```

After the first run it can be run without the --allow_downloads flag, as it downloads base model weights to `Standalone/model_params/base_model_cache` and resuses them from there.

The tool will:

- Print a summary to stdout.
- Write a JSON file named `prediction_result.json` next to the input file.

## Train/Val Lookup (Trust Signal)

Lookup is optional and runs only when you pass `--lookup`.
When enabled, the standalone CLI looks up your query array in the bundled train/val DB and reports:

- exact presence in train/val (if the same array is present)
- similarity to nearest train/val arrays (if not present)

The lookup uses spacers + repeats for exact match and spacer similarity for nearest-neighbor reporting.
The lookup DB path is `Standalone/lookup/array_lookup_db.json`.

Enable lookup explicitly:

```bash
python Standalone/predict_direction.py \
  --input_file Standalone/example_array.json \
  --lookup
```

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

## 1) Standard JSON

Example object:

```json
{
  "array_name":"my_array",
  "repeats": ["..."],
  "spacers": ["..."]
}
```

Here spacers and repeats have to be in order and there have to be multiple repeats. If only one repeat occurs, still provide that #spacers+1 times,
like in the provided example_array.json. If any are missing, that degrades the quality of the prediction.

Notes:

- Multi-line JSON files are supported.
- For JSONL files, the first non-empty line is used.

## 2) CRISPRCasFinder Result JSON (`--ccf`)

You can pass Results from CRISPRCasFinder directly, for that only the `result.json` file is used.

Example:

```bash
python Standalone/predict_direction.py \
  --input_file Result_XXX/result.json \
  --ccf
```

If multiple entries exist, select which one to predict:

```bash
python Standalone/predict_direction.py \
  --input_file Result_XXX/result.json \
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
Predicted direction is "Forward" with probability 0.991 (99.1%).
Alternative direction "Reverse" has probability 0.009 (0.9%).
Input summary: repeats=31, spacers=30, tokens=367.
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
- `array_name`, `cas_subtype` (if provided in input)
- `predicted_label_id`, `predicted_label`
- `prob_reverse`, `prob_forward`
- `token_count`
- `lookup` block (only when `--lookup` is used), containing exact train/val presence or nearest similarities
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
- `--max_length N`: Tokenizer truncation length (default `256`).
- `--cpu`: Force CPU.
- `--gpu`: Force GPU (fails if CUDA is unavailable).
- `--result_file PATH`: Custom output JSON path.
- `--allow_downloads`: Allow online fallback if local assets are missing.
- `--lookup`: Enable lookup against bundled train/val DB (off by default for faster inference).

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
- Bundled train/val lookup DB: [lookup/array_lookup_db.json](lookup/array_lookup_db.json)

- If full base weights are already cached in `Standalone/model_params/base_model_cache`, they are reused.
- If full weights are missing, the script downloads it from Hugging Face on first run,
  and stores the weights in `Standalone/model_params/base_model_cache`.
- If adapter files are present (`adapter_config.json` + `adapter_model.safetensors`), LoRA is
  applied automatically on top of the cached base model.

## Troubleshooting

`--gpu was requested, but CUDA is not available`:

- Retry with `--cpu`.

FileNotFoundError: No full model weights found:

- You dont have the base model saved in your directory, run again with --allow_downloads.

Input validation errors:

- Standard mode: verify `repeats` and `spacers` are lists of strings and the file has enough repeats in the list.
- CCF mode: verify file contains `Sequences -> Crisprs -> Regions`.

Unexpected downloads:

- Do not pass `--allow_downloads` when offline-only behavior is required.

Tranformers warning:

Key          | Status  | 
-------------+---------+-
score.weight | MISSING | 

- You can safely ignore these, as we are applying a LoRa adapter, we fill them in.