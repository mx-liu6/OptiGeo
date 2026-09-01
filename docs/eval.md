# Evaluation

We provide a unified evaluation script that runs baselines on multiple benchmarks. It takes a baseline model and evaluation configurations, evaluates on-the-fly, and reports results instantly in a JSON file.

## Benchmarks

Download the processed datasets from [Hugging Face Datasets](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation) and put them in the `data/eval` directory, using `huggingface-cli`:

```bash
mkdir -p data/eval
huggingface-cli download Ruicheng/monocular-geometry-evaluation --repo-type dataset --local-dir data/eval --local-dir-use-symlinks False
```

Then unzip the downloaded files:

```bash
cd data/eval  
unzip '*.zip'
# rm *.zip # if you don't keep the zip files
```

## Configuration

See [`configs/eval/all_benchmarks.json`](../configs/eval/all_benchmarks.json) for the default evaluation configuration. You can modify this file to evaluate on different benchmarks or different baselines.

## Baseline

Some examples of baselines are provided in [`baselines/`](../baselines/). Pass the path to the baseline model Python code to the `--baseline` argument of the evaluation script.

## Run Evaluation

Run the script [`optigeo/scripts/eval_baseline.py`](../optigeo/scripts/eval_baseline.py). 
For example, 

```bash
# Evaluate OptiGeo on the configured benchmarks
python optigeo/scripts/eval_baseline.py --baseline baselines/optigeo.py --config configs/eval/all_benchmarks.json --output eval_output/optigeo.json --pretrained mxliu-hku/OptiGeo --resolution_level 9

# Evaluate VGGT on the configured benchmarks.
python optigeo/scripts/eval_baseline.py --baseline baselines/vggt.py --config configs/eval/all_benchmarks.json --output eval_output/vggt.json
```

The `--baseline`, `--config`, and `--output` arguments are for the evaluation script. The rest arguments, e.g. `--pretrained` and `--resolution_level`, are customized for loading the baseline model.

Details of the arguments:

```
Usage: eval_baseline.py [OPTIONS]

  Evaluation script.

Options:
  --baseline PATH  Path to the baseline model python code.
  --config PATH    Path to the evaluation configurations. Defaults to
                   "configs/eval/all_benchmarks.json".
  --output PATH    Path to the output json file.
  --oracle         Use oracle mode for evaluation, i.e., use the GT intrinsics
                   input.
  --dump_pred      Dump prediction results.
  --dump_gt        Dump ground truth.
  --help           Show this message and exit.
```



## Wrap a Customized Baseline

Wrap any baseline method with [`optigeo.test.baseline.OptiGeoBaselineInterface`](../optigeo/test/baseline.py).
See [`baselines/`](../baselines/) for more examples.

It is a good idea to check the correctness of the baseline implementation by running inference on a small set of images via [`optigeo/scripts/infer_baseline.py`](../optigeo/scripts/infer_baseline.py):

```bash
python optigeo/scripts/infer_baseline.py --baseline baselines/optigeo.py --input example_images/ --output infer_output/optigeo --pretrained mxliu-hku/OptiGeo --maps --ply
```
