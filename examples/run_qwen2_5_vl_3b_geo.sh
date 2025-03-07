set -x

source ~/.bashrc
source ~/miniconda3/bin/activate easyr1
cd /home/aiops/liuxy/MultimodalR1/EasyR1

export VLLM_ATTENTION_BACKEND=XFORMERS

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY="454cd689208c89f63deeb877aed3fba714fce4c6"
export VLLM_USE_V1=0

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=examples/grpo_example.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_3b_geo \
    trainer.n_gpus_per_node=4
