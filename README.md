# SMoE
> [Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging](https://arxiv.org/abs/2506.23266)
以下为自用
## Venv
```bash
apt-get update
apt install python3.10-venv
python3 -m venv .venv

source .venv/bin/activate
```

## Install
```bash
bash init.sh
pip install -r requirements.txt
# pip install git+https://github.com/huggingface/transformers
cd lm-evaluation-harness
pip install -e .
```

## Model and Data
```bash
bash script/download_model.sh

# bash script/download_data.sh

bash script/prepare_data.sh # it takes 1-2 mins
```

## Run
### Qwen1.5-MoE-A2.7B
```bash
python -m src.qwen.analysis.run_qwen_analysis
```