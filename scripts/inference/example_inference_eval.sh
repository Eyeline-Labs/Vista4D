RESOLUTION=${RESOLUTION:-384p}  # 384p, 720p
NUM_SHARDS=${NUM_SHARDS:-0}
SHARD_ID=${SHARD_ID:-0}
USE_USP=${USE_USP:-false}
NUM_GPUS=${NUM_GPUS:-8}

echo "Script kwargs:"
echo "    RESOLUTION=$RESOLUTION"
echo "    NUM_SHARDS=$NUM_SHARDS"
echo "    SHARD_ID=$SHARD_ID"
echo "    USE_USP=$USE_USP"
echo "    NUM_GPUS=$NUM_GPUS"

LOCAL_WAN_FOLDER=./checkpoints/wan
WAN_NAME=Wan2.1-T2V-14B
WAN_PATHS="${WAN_NAME}:diffusion_pytorch_model*.safetensors,${WAN_NAME}:models_t5_umt5-xxl-enc-bf16.pth,${WAN_NAME}:Wan2.1_VAE.pth"
TOKENIZER_PATHS="${WAN_NAME}:google/*"

if [ $RESOLUTION == 384p ]; then
    VISTA4D_FOLDER="./checkpoints/vista4d/384p49_step=30000"
    HEIGHT=384
    WIDTH=672
elif [ $RESOLUTION == 720p ]; then
    VISTA4D_FOLDER="./checkpoints/vista4d/720p49_step=3000"
    HEIGHT=720
    WIDTH=1280
else
    echo "Unrecognized RESOLUTION=$RESOLUTION, exiting script."
    exit 1
fi
NUM_FRAMES=49

EVAL_DATA_FOLDER=./eval_data/render_$RESOLUTION
EVAL_DATA_CSV=./eval_data/metadata.csv
OUTPUT_FOLDER=./results/eval/vista4d_$RESOLUTION

ARGS=""
if [ $USE_USP == true ]; then
    ARGS="$ARGS --use_usp"
    RUN="torchrun --standalone --nproc_per_node $NUM_GPUS"
else
    RUN=python3
fi

$RUN -m scripts.inference.inference_eval \
    --model_id_with_origin_paths "$WAN_PATHS" \
    --tokenizer_id_with_origin_path "$TOKENIZER_PATHS" \
    --local_model_folder $LOCAL_WAN_FOLDER \
    --vista4d_checkpoint $VISTA4D_FOLDER/dit.pth \
    --vista4d_config_path $VISTA4D_FOLDER/config.yaml \
    --eval_data_folder $EVAL_DATA_FOLDER \
    --eval_data_csv $EVAL_DATA_CSV \
    --output_folder $OUTPUT_FOLDER \
    --height $HEIGHT --width $WIDTH --num_frames $NUM_FRAMES \
    --save_gif \
    --num_shards $NUM_SHARDS --shard_id $SHARD_ID \
    $ARGS
