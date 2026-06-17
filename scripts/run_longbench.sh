set -ex
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

cd LongBench

eval_function() {
	python eval.py --output_dir $OUTPUT_DIR
}
pred_function_quantized() {
	mkdir -p $OUTPUT_DIR

	python pred.py \
		--output_dir $OUTPUT_DIR \
		--model $MODEL_NAME \
		--dataset $DATASET \
		--k_bits $K_BIT \
		--v_bits $V_BIT \
		--key_group_size $K_GROUP \
		--value_group_size $V_GROUP \
		--key_w_recent $K_RECENT \
		--value_w_recent $V_RECENT \
		--key_w_sink $K_SINKS \
		--value_w_sink $V_SINKS \
		--key_flush_size $K_FLUSH \
		--value_flush_size $V_FLUSH \
		--key_normalize $K_NORMALIZE \
		--key_quantize_over_token $K_OVERTOKEN \
		--value_quantize_over_token $V_OVERTOKEN \
		--key_quantize_mode $K_MODE \
		--value_quantize_mode $V_MODE

	eval_function
}


baseline() {
	OUTPUT_DIR="output/baseline/$MODEL_NAME"

	mkdir -p $OUTPUT_DIR

	python pred.py \
		--output_dir $OUTPUT_DIR \
		--model $MODEL_NAME \
		--dataset $DATASET

	eval_function
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

	OUTPUT_DIR="output/kivi/$MODEL_NAME"
	pred_function_quantized
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

	OUTPUT_DIR="output/kivi_sink/$MODEL_NAME"
	pred_function_quantized
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

	OUTPUT_DIR="output/innerq_base/$MODEL_NAME"
	pred_function_quantized
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

	OUTPUT_DIR="output/innerq_small/$MODEL_NAME"
	pred_function_quantized
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

	OUTPUT_DIR="output/innerq_hybrid/$MODEL_NAME"
	pred_function_quantized
}

model_names=("llama3.1-8b-instruct" "llama3.2-1b-instruct" "llama3.2-3b-instruct" "llama2-7b-chat-4k" "llama2-13b-chat-4k")
datasets=("qasper" "gov_report" "multi_news" "trec" "triviaqa" "samsum" "lcc" "repobench-p")

SEED=89

for i in "${!model_names[@]}"; do
	MODEL_NAME="${model_names[i]}"
for DATASET in "${datasets[@]}"; do
	baseline
	kivi
	kivi_sink
	innerq_base
	innerq_small
	innerq_hybrid
done
done
