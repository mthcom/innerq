set -ex
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

lm_eval_run() {
	EXTRA_ARGS=()
	case "$TASK_NAME" in
	  mbpp_instruct|humaneval_instruct)
	    EXTRA_ARGS+=(--apply_chat_template)
	    ;;
	esac

	mkdir -p $OUTPUT_DIR

	lm-eval \
		--model $METHOD \
		--model_args pretrained=$MODEL_NAME,k_bits=$K_BIT,v_bits=$V_BIT,key_group_size=$K_GROUP,value_group_size=$V_GROUP,key_w_recent=$K_RECENT,value_w_recent=$V_RECENT,key_w_sink=$K_SINKS,value_w_sink=$V_SINKS,key_flush_size=$K_FLUSH,value_flush_size=$V_FLUSH,normalize_key=$K_NORMALIZE,key_quantize_over_token=$K_OVERTOKEN,value_quantize_over_token=$V_OVERTOKEN,key_quantize_mode=$K_MODE,value_quantize_mode=$V_MODE \
		--tasks $TASK_NAME \
		--output_path $OUTPUT_DIR \
		--device cuda:0 \
		--seed $SEED \
		--confirm_run_unsafe_code \
		--batch_size 1 \
		"${EXTRA_ARGS[@]}"

}
baseline() {
	K_BIT=16
	V_BIT=16

	OUTPUT_DIR="output/lm_eval/$TASK_NAME/baseline/seed_$SEED"

	EXTRA_ARGS=()
	case "$TASK_NAME" in
	  mbpp_instruct|humaneval_instruct)
	    EXTRA_ARGS+=(--apply_chat_template)
	    ;;
	esac
	mkdir -p $OUTPUT_DIR

	lm-eval \
		--model $METHOD \
		--model_args pretrained=$MODEL_NAME,k_bits=$K_BIT,v_bits=$V_BIT \
		--tasks $TASK_NAME \
		--output_path $OUTPUT_DIR \
		--device cuda:0 \
		--seed $SEED \
		--confirm_run_unsafe_code \
		--batch_size 1 \
		"${EXTRA_ARGS[@]}"

}
kivi() {
	K_FLUSH=32
	V_FLUSH=1
	K_GROUP=32
	V_GROUP=32
	K_RECENT=128
	V_RECENT=128
	K_SINKS=0
	V_SINKS=0
	K_BIT=2
	V_BIT=2
	K_NORMALIZE=False
	K_OVERTOKEN=True
	V_OVERTOKEN=False
	K_MODE='asymmetric'
	V_MODE='asymmetric'

	OUTPUT_DIR="output/lm_eval/$TASK_NAME/kivi/seed_$SEED"
	lm_eval_run
}
kivi_sink() {
	K_BIT=2
	V_BIT=2
	K_FLUSH=32
	V_FLUSH=1
	K_GROUP=32
	V_GROUP=32
	K_RECENT=96
	V_RECENT=96
	K_SINKS=32
	V_SINKS=32
	K_NORMALIZE=False
	K_OVERTOKEN=True
	V_OVERTOKEN=False
	K_MODE='asymmetric'
	V_MODE='asymmetric'

	OUTPUT_DIR="output/lm_eval/$TASK_NAME/kivi_sink/seed_$SEED"
	lm_eval_run
}
innerq_base() {
	K_BIT=3
	V_BIT=3
	K_FLUSH=1
	V_FLUSH=32
	K_GROUP=32
	V_GROUP=32
	K_RECENT=96
	V_RECENT=96
	K_SINKS=32
	V_SINKS=32
	K_NORMALIZE=True
	K_OVERTOKEN=False
	V_OVERTOKEN=True
	K_MODE='symmetric'
	V_MODE='symmetric'

	OUTPUT_DIR="output/lm_eval/$TASK_NAME/innerq_base/seed_$SEED"
	lm_eval_run
}
innerq_small() {
	K_BIT=3
	V_BIT=2
	K_FLUSH=1
	V_FLUSH=32
	K_GROUP=32
	V_GROUP=32
	K_RECENT=96
	V_RECENT=96
	K_SINKS=32
	V_SINKS=32
	K_NORMALIZE=True
	K_OVERTOKEN=False
	V_OVERTOKEN=True
	K_MODE='symmetric'
	V_MODE='symmetric'

	OUTPUT_DIR="output/lm_eval/$TASK_NAME/innerq_small/seed_$SEED"
	lm_eval_run
}
innerq_hybrid() {
	K_BIT=3
	V_BIT=2
	K_FLUSH=1
	V_FLUSH=32
	K_GROUP=32
	V_GROUP=32
	K_RECENT=96
	V_RECENT=96
	K_SINKS=32
	V_SINKS=32
	K_NORMALIZE=True
	K_OVERTOKEN=False
	V_OVERTOKEN=True
	K_MODE='symmetric'
	V_MODE='hybrid'

	OUTPUT_DIR="output/lm_eval/$TASK_NAME/innerq_hybrid/seed_$SEED"
	lm_eval_run
}

export HF_ALLOW_CODE_EVAL=1
METHOD="hfinnerq"
SEED=89

model_names=("meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.3" "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-3B")
model_folder_names=("meta-llama__Llama-3.1-8B" "mistralai__Mistral-7B-v0.3" "meta-llama__Llama-2-7b-hf" "meta-llama__Llama-3.2-1B" "meta-llama__Llama-3.2-3B")
tasks=(gsm8k minerva_math500)

for TASK_NAME in "${tasks[@]}"; do
for i in "${!model_names[@]}"; do
	MODEL_NAME="${model_names[i]}"
	MODEL_FOLDER_NAME="${model_folder_names[i]}"

	baseline
	kivi
	kivi_sink
	innerq_base
	innerq_small
	innerq_hybrid
done
done

model_names=("meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3")
model_folder_names=("meta-llama__Llama-3.2-1B-Instruct" "meta-llama__Llama-3.2-3B-Instruct" "meta-llama__Llama-3.1-8B-Instruct" "mistralai__Mistral-7B-Instruct-v0.3")
tasks=(mbpp_instruct humaneval_instruct)

for TASK_NAME in "${tasks[@]}"; do
for i in "${!model_names[@]}"; do
	MODEL_NAME="${model_names[i]}"
	MODEL_FOLDER_NAME="${model_folder_names[i]}"

	baseline
	kivi
	kivi_sink
	innerq_base
	innerq_small
	innerq_hybrid
done
done
