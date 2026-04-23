EXAMPLE=${EXAMPLE:-lounge-cup}  # We provide some dynamic scene expansion (DSE) examples to demo Vista4D!
RECON_METHOD=${RECON_METHOD:-pi3}  # pi3, da3
echo "Script kwargs:"
echo "    EXAMPLE=$EXAMPLE"
echo "    RECON_METHOD=$RECON_METHOD"

# Source video = ./media/dse/$EXAMPLE.mp4; casual scene capture = ./media/dse/$SCENE.mp4 (strip last "-<suffix>")
SCENE=${EXAMPLE%-*}

if [ $EXAMPLE == conference-punch ]; then
    SEG_KEYWORDS=("person")
elif [ $EXAMPLE == conference-study ]; then
    SEG_KEYWORDS=("person" "notebook")
elif [ $EXAMPLE == hall-cartwheel ]; then
    SEG_KEYWORDS=("person")
elif [ $EXAMPLE == lounge-cup ]; then
    SEG_KEYWORDS=("person" "white cup")
elif [ $EXAMPLE == lounge-drink ]; then
    SEG_KEYWORDS=("person" "white cup")
elif [ $EXAMPLE == plaza-point ]; then
    SEG_KEYWORDS=("person")
elif [ $EXAMPLE == room-lift ]; then
    SEG_KEYWORDS=("person")
elif [ $EXAMPLE == room-walk ]; then
    SEG_KEYWORDS=("person")
else
    echo "Unrecognized EXAMPLE=$EXAMPLE, exiting script."
    exit 1
fi

HEIGHT=720  # The included videos in ./media/ are all 720p, so we run and save results at 720p
WIDTH=1280  # Also, the highest resolution Vista4D checkpoints are trained on is 720p
NUM_FRAMES=49
NUM_DSE_FRAMES=49  # Subsample DSE evenly, safety cap for long scene captures

PI3_PIXEL_LIMIT=255000  # Pixel limit as given by Pi3(X)
DA3_PROCESS_RES=896  # Can be lowered if encountering OOM with DA3, DA3 max training width is 896

python3 -m scripts.preprocess.recon_and_seg_single \
    --video_path ./media/dse/$EXAMPLE.mp4 \
    --dse_video_path ./media/dse/$SCENE.mp4 \
    --output_folder "./results/dse/$EXAMPLE/recon_and_seg" \
    --seg_keywords "${SEG_KEYWORDS[@]}" \
    --recon_method $RECON_METHOD \
    --da3_model_id depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
    --pi3_model_id yyfz233/Pi3X \
    --height $HEIGHT --width $WIDTH --num_frames $NUM_FRAMES --num_dse_frames $NUM_DSE_FRAMES \
    --pi3_pixel_limit $PI3_PIXEL_LIMIT --da3_process_res $DA3_PROCESS_RES \
    --save_vis
