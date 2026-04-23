EXAMPLE=${EXAMPLE:-lounge-cup}  # We provide some dynamic scene expansion (DSE) examples to demo Vista4D!
RESOLUTION=${RESOLUTION:-720p}  # 384p, 720p
USE_USP=${USE_USP:-false}
NUM_GPUS=${NUM_GPUS:-8}
TRIAL_INFERENCE_TIME=${TRIAL_INFERENCE_TIME:-false}

echo "Script kwargs:"
echo "    EXAMPLE=$EXAMPLE"
echo "    RESOLUTION=$RESOLUTION"
echo "    USE_USP=$USE_USP"
echo "    NUM_GPUS=$NUM_GPUS"
echo "    TRIAL_INFERENCE_TIME=$TRIAL_INFERENCE_TIME"

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

INPUT_FOLDER=./results/dse/$EXAMPLE/render_$RESOLUTION
OUTPUT_FOLDER=./results/dse/$EXAMPLE/vista4d_$RESOLUTION

# Since dynamic scene expansion (DSE) is an experimental application of Vista4D, we also provide the seed for each of
# the following examples. We picked the best result from ~2-3 random seeds that we tried.
if [ $EXAMPLE == conference-punch ]; then
    PROMPT="Two people share a goofy, lighthearted moment behind a row of chairs. The young woman, wearing a white beanie and a black puffer jacket, throws a playful superhero punch toward the man. He reacts with comedic exaggeration, pretending to be knocked backward by the force. The woman laughs as he continues the dramatic pose, stumbling away."
    SEED=(46718)
elif [ $EXAMPLE == conference-study ]; then
    PROMPT="In a conference room, a young woman wearing a white beanie and a black puffer jacket person stands and leans over a table, pointing to something in an open notebook. A young man sitting at the table looks at the book, then gestures with their hand as they look up. They then put their hand to their head, appearing to be thinking hard or feeling puzzled by the topic being discussed."
    SEED=(20112)
elif [ $EXAMPLE == hall-cartwheel ]; then
    PROMPT="A man stands in a brightly lit, modern hall with large windows in the background. He begins by raising his arms out to his sides, as if stretching. He then performs a quick, athletic cartwheel across a blue mat on the floor, moving from left to right. After completing the acrobatic move, he lands on his feet and straightens up."
    SEED=(86966)
elif [ $EXAMPLE == lounge-cup ]; then
    PROMPT="A young man with dark, wavy hair and a beard in a grey sweater and blue jeans sits at a table in a room with several plants and a large window overlooking a modern building bathed in a cool blue daylight. He picks up a white cup, shakes it a couple of times, and places it back on the table. He then picks the cup up again and takes a drink from it."
    SEED=(69377)
elif [ $EXAMPLE == lounge-drink ]; then
    PROMPT="A young man with dark, wavy hair and a beard in a grey sweater and blue jeans sits at a table in a room with several plants and a large window overlooking a modern building bathed in a cool blue daylight. He slides a white cup across the table, picks it up, and inspects it carefully. Then, he drinks from the white cup."
    SEED=(70023)
elif [ $EXAMPLE == plaza-point ]; then
    PROMPT="A young woman wearing a white beanie and black puffer jacket stands outside in a plaza in front of a modern, multi-storied building with large windows. She smiles and looks off to the side as if interacting with someone just out of frame. She makes a few small, playful hand gestures toward the ground."
    SEED=(25491)
elif [ $EXAMPLE == room-lift ]; then
    PROMPT="A young man with dark, wavy hair and a beard sits at a round wooden table in a room with large cardboard boxes stacked next to the window behind him. He smiles and briefly raises his arms in a small celebratory gesture. He then lowers his hands to the table and begins to speak, gesturing as if he is explaining something. The setting appears to be an office or storage area with a large window overlooking other buildings."
    SEED=(34028)
elif [ $EXAMPLE == room-walk ]; then
    PROMPT="A young man with dark wavy hair and a beard, wearing a grey sweatshirt, is sitting at a wooden conference table. He appears to be in the middle of a conversation as he speaks. He gestures actively with his hands, tapping his fingers on the table and 'walking' them across the surface as if explaining an idea. The scene takes place in a meeting room or office, with a white whiteboard and other office chairs visible in the background."
    SEED=(10027)
else
    echo "Unrecognized EXAMPLE=$EXAMPLE, exiting script."
    exit 1
fi

ARGS=""
if [ $USE_USP == true ]; then
    ARGS="$ARGS --use_usp"
    RUN="torchrun --standalone --nproc_per_node $NUM_GPUS"
else
    RUN=python3
fi
if [ $TRIAL_INFERENCE_TIME == true ]; then
    ARGS="$ARGS --num_inference_time_trials 3"
fi

$RUN -m scripts.inference.inference \
    --model_id_with_origin_paths "$WAN_PATHS" \
    --tokenizer_id_with_origin_path "$TOKENIZER_PATHS" \
    --local_model_folder $LOCAL_WAN_FOLDER \
    --vista4d_checkpoint $VISTA4D_FOLDER/dit.pth \
    --vista4d_config_path $VISTA4D_FOLDER/config.yaml \
    --input_folder $INPUT_FOLDER \
    --output_folder $OUTPUT_FOLDER \
    --prompt "$PROMPT" \
    --height $HEIGHT --width $WIDTH --num_frames $NUM_FRAMES \
    --seed "${SEED[@]}" \
    --save_gif \
    $ARGS
