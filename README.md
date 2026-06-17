# InnerQ: Hardware-Aware Tuning-Free Quantization of KV Cache for Large Language Models

This is the official code for the paper [InnerQ](https://arxiv.org/abs/2602.23200).

We propose a hardware-aware KV cache quantization method which spans quantization groups oveer the inner dimension to achieve higher memory reuse and a better inference latency.

## Installation

The following environment is what we have used in our experiments and is merely a suggestion.

```
conda create -n innerq python=3.10
conda activate innerq
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

For GSM8k, Minerva Math, MBPP, and HumanEval tasks we use [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) which we install from source to enable easier patching.

```
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout 5d7dc4c0bbaac6f1186a8086bbb9c1da43be24cf
cp ../lm_eval_patch/lm_eval/models/* lm_eval/models/
pip install -e .[math]
cd ..
```

We also included a lightly modified version of [LongBench](https://github.com/THUDM/LongBench) (from the commit `2e00731f8d0bff23dc4325161044d0ed8af94c1e`) in our repo for long context evaluation.

## Usage

The core cache functionality is in [cache.py](./src/innerq/cache.py) and [quant.py](src/innerq/quant.py). You can add support for new models by adding new files in ```./src/innerq/models```.

You can also run/modify the following scripts to reproduce our results:
```
scripts/run_lmeval.sh
scripts/run_longbench.sh
scripts/run_speedtest.sh
```

## Correspondence

For any correspondence, please contact [mohammadreza.tayaranian@mail.mcgill.ca](mailto:mohammadreza.tayaranian@mail.mcgill.ca).

## Citation

```
@misc{hosseini2026innerq,
      title={InnerQ: Hardware-Aware Tuning-Free Quantization of KV Cache for Large Language Models}, 
      author={Sayed Mohammadreza Tayaranian Hosseini and Amir Ardakani and Warren J. Gross},
      year={2026},
      eprint={2602.23200},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.23200}, 
}
```
