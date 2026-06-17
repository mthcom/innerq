set -ex
export CUDA_VISIBLE_DEVICES=0

cd speedtest

# Quantization
Ns=(1 32)
OUTPUT_FILE="output_quant.txt"
touch $OUTPUT_FILE

for N in "${Ns[@]}"
do
nvcc -DN=$N -O3 -arch=sm_72 benchmark_quantization.cu -o benchmark
if [ $? -eq 0 ]; then
    ./benchmark >> $OUTPUT_FILE
else
    rm benchmark
    exit
fi
done
rm benchmark

# Dequantization
Ns=(512 1024 2048 4096 8192 16384 32768)

OUTPUT_FILE="output_dequant.txt"
touch $OUTPUT_FILE

for N in "${Ns[@]}"
do
nvcc -DN=$N -O3 -arch=sm_72 benchmark_qkt.cu -o benchmark
if [ $? -eq 0 ]; then
    ./benchmark >> $OUTPUT_FILE
else
    rm benchmark
    exit
fi
done
rm benchmark

for N in "${Ns[@]}"
do
nvcc -DN=$N -O3 -arch=sm_72 benchmark_attn_v.cu -o benchmark
if [ $? -eq 0 ]; then
    ./benchmark >> $OUTPUT_FILE
else
    rm benchmark
    exit
fi
done
rm benchmark

