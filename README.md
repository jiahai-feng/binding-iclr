# How do language models bind entities in context?

This repository contains code that generates the data and plots for the ICLR 2024 paper [How do language models bind entities in context?](https://arxiv.org/abs/2310.17191).

This is built off the [TransformerLens](https://github.com/neelnanda-io/TransformerLens) library, and contains code from [Danny Halawi](https://github.com/dannyallover/overthinking_the_truth) and [Evan Hernandez](https://github.com/evandez/REMEDI).

## Setup
Use python 3.10. Install the `requirements.txt` file.

You will need to obtain a local copy of LLaMA weights compatible with Hugging Face. Follow the instructions from the [HF documentation](https://huggingface.co/docs/transformers/main/en/model_doc/llama). Once you have a path to LLaMA weights, save it in the shell environment variable `LLAMA_WEIGHTS` (refer to `models.py`).

Experiments were conducted on 4 A100 GPUs.


## Running experiments
Run

```
python -m scripts.run_experiment \
    --model "llama-30b-hf" \
    --force_time 'results' \
    --num_devices 4 \
    --cyclic \
    --factorizability \
    --position \
    --means_intervention \
    --means_intervention_baseline \
    --all_tasks_means \
    --cross_tasks_means \
    --cross_tasks_subspace \
    --baseline \
    --local_position \
```

Here's a breakdown of the flags:
- factorizability: Fig 3a, 3b
- position: Fig 4
- means_intervention, means_intervention_baseline: Table 1
- all_tasks_means: Fig 6 right
- cross_tasks_subspace: Table 2
- baseline: baselines for Fig 6 right
- local_position: Position plots for all tasks (appendix)

To generate Fig 5, run:
```
python -m scripts.run_experiment \
    --model "llama-13b-hf" \
    --force_time "results" \
    --num_devices 3 \
    --subspace
```

To generate Fig 6 left, run:
```
models="EleutherAI/pythia-70m EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1b EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b EleutherAI/pythia-6.9b EleutherAI/pythia-12b llama-7b-hf llama-13b-hf"

for model in $models; do
    echo $model
    python -m scripts.run_experiment \
        --model $model \
        --force_time "results" \
        --num_devices 3 \
        --capitals_mean \
        --cyclic \
        --capitals_baseline
done
```
You may add the flags `--factorizability --position` if you want to run those experiments for all models.


To generate one-hop:
```
python -m scripts.run_experiment \
    --model "llama-65b-hf" \
    --force_time 'results' \
    --country_width 2\
    --num_devices 4 \
    --onehop
```

To generate MCQ:
```
python -m scripts.run_experiment \
    --model "tulu-13b" \
    --force_time 'results' \
    --country_width 2\
    --num_devices 4 \
    --mcq
```