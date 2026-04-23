EXAMPLE=${EXAMPLE:-hike_cow}  # We provide some 4D scene recomposition examples to demo Vista4D!
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

INPUT_FOLDER=./results/edit/$EXAMPLE/render_$RESOLUTION
OUTPUT_FOLDER=./results/edit/$EXAMPLE/vista4d_$RESOLUTION

# Since 4D scene recomposition is an experimental application of Vista4D, we also provide the seed for each of the
# following examples. We picked the best result from ~2-3 random seeds that we tried
if [ $EXAMPLE == couple-hug_duplicate-car ]; then
    PROMPT="A couple shares a series of intimate moments in a snowy forest, seated in the trunk of a black car. The woman, dressed in a cream coat and pink beanie, and the man, in a green jacket and dark pants, embrace and kiss, surrounded by the serene beauty of snow-covered trees. The car's open trunk, adorned with festive tassels, adds a celebratory touch to the scene. The couple's affectionate interactions, including tender embraces and a kiss, convey a deep connection and contentment amidst the tranquil winter landscape. There is also an identical black car to the left of the couple's black car, also with its trunk open, and turned slightly to the side."
    SEED=(85171)
elif [ $EXAMPLE == couple-hug_couple-newspaper ]; then
    PROMPT="A couple shares a series of intimate moments in a snowy forest, seated in the trunk of a black car. The woman, dressed in a cream coat and pink beanie, and the man, in a green jacket and dark pants, embrace and kiss, surrounded by the serene beauty of snow-covered trees. The car's open trunk, adorned with festive tassels, adds a celebratory touch to the scene. The couple's affectionate interactions, including tender embraces and a kiss, convey a deep connection and contentment amidst the tranquil winter landscape. To the left of the couple is an elderly couple who are seated on a wooden bench, reading a newspaper and then glancing over at the hugging couple. The wooden bench has its legs slightly buried in the snow, and the elderly couple's shoes are on the snow as well. The elderly man, wearing a white polo shirt and dark trousers, has sunglasses on his head, while the elderly woman, in a striped top and beige pants, leans on him."
    SEED=(15058)
elif [ $EXAMPLE == funeral-procession_remove-priest ]; then
    PROMPT="Four young men in formal suits carry a polished wooden casket adorned with white floral tributes along a cobblestone path, surrounded by lush greenery and headstones. They are the first person in the line of people in this funeral procession. The scene, set under an overcast sky, reflects a mood of reverence and final farewell. As the procession continues, the number of pallbearers varies slightly, but the somber atmosphere remain constant, underscoring the gravity of the funeral service."
    SEED=(35968)
elif [ $EXAMPLE == funeral-procession_rhino ]; then
    PROMPT="An elderly priest with a white beard, dressed in a black shirt and white ceremonial robe, stands solemnly in a cemetery, observing a funeral procession. Four young men in formal suits carry a polished wooden casket adorned with white floral tributes along a cobblestone path, surrounded by lush greenery and headstones. The scene, set under an overcast sky, reflects a mood of reverence and final farewell. As the procession continues, the number of pallbearers varies slightly, but the somber atmosphere and the priest's contemplative gaze remain constant, underscoring the gravity of the funeral service. In front of the procession is a large white rhinoceros walking steadily across the path, doing its slow and deliberate stroll."
    SEED=(96370)
elif [ $EXAMPLE == hike_enlarge-backpack ]; then
    PROMPT="A hiker, equipped with a huge backpack and a cap, walks along a wide, rocky trail from left to right. The backpack he is wearing is extremely large, significantly larger than the standard hiking backpack and almost the height of his body. He is trekking through a stark, high-altitude alpine environment, which appears to be above the treeline. The background is dominated by a massive, steep scree slope rising to a dramatic, jagged mountain ridge. The rugged, gravelly terrain and sparse vegetation emphasize the vastness of the mountain landscape."
    SEED=(70251)
elif [ $EXAMPLE == hike_cow ]; then
    PROMPT="A hiker, equipped with a large backpack and a cap, walks along a wide, rocky trail from left to right. He is trekking through a stark, high-altitude alpine environment, which appears to be above the treeline. The background is dominated by a massive, steep scree slope rising to a dramatic, jagged mountain ridge. The rugged, gravelly terrain and sparse vegetation emphasize the vastness of the mountain landscape. To the left of hiker is a large white cow with brown spots wearing a collar walking in roughly the same direction as the hiker, following him."
    SEED=(81032)
elif [ $EXAMPLE == swing_shrink-person ]; then
    PROMPT="A blonde woman wearing a white dress shirt, blue jeans, and white shoes, with hair tied into a small bun, is sitting on a swing and swinging back and forth. She is a very small person, to the point where the swing looks like it is for giants as she sits on it. She looks around while she swings, and her hands are holding on to the metal chains of the swing. Her right hand is also holding onto a folded up black jump rope. The yellow/light brown colored swing is in the middle of a small park or apartment complex common area, being on top of black playground mats, which are on top of grass and in front of a bush with an apartment building behind. The scene is overcast during the day."
    SEED=(66038)
elif [ $EXAMPLE == swing_couple-walk ]; then
    PROMPT="A blonde woman wearing a white dress shirt, blue jeans, and white shoes, with hair tied into a small bun, is sitting on a swing and swinging back and forth. She looks around while she swings, and her hands are holding on to the metal chains of the swing. Her right hand is also holding onto a folded up black jump rope. To the left of the swing is a man in a blue hoodie and green pants, holding a bouquet of yellow flowers. He is walking alongside a woman in a beige trench coat and white pants, accompanied by a small white dog on a red leash. The man, woman, and dog are all steadily strolling forwards. The yellow/light brown colored swing is in the middle of a small park or apartment complex common area, being on top of black playground mats, which are on top of grass and in front of a bush with an apartment building behind. The scene is overcast during the day."
    SEED=(77062)
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
