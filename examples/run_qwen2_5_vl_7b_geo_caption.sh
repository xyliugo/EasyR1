set -x

source ~/.bashrc
source ~/miniconda3/bin/activate easyr1
cd /home/aiops/liuxy/MultimodalR1/EasyR1

export VLLM_ATTENTION_BACKEND=XFORMERS

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY="454cd689208c89f63deeb877aed3fba714fce4c6"
export VLLM_USE_V1=0

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

SYSTEM_PROMPT="""You First extract ALL useful conditions from the given visual and textual context that can be used to solve the given problem. Then, think through the reasoning process as an internal monologue before providing the final answer.
 The conditions MUST BE enclosed within <conditions> </conditions> tags. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=examples/grpo_example.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.system_prompt="${SYSTEM_PROMPT}" \
    data.max_response_length=2048 \
    data.max_pixels=1000000 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    worker.reward.compute_score=caption_math \
    worker.rollout.gpu_memory_utilization=0.35 \
    worker.rollout.tensor_parallel_size=4 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_7b_geo_cap_len2048 \
    trainer.n_gpus_per_node=8 \
    trainer.save_freq=20 \

