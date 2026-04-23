EXAMPLE=${EXAMPLE:-hike}  # We provide some 4D scene recomposition examples to demo Vista4D!
RECON_METHOD=${RECON_METHOD:-pi3}  # pi3, da3
echo "Script kwargs:"
echo "    EXAMPLE=$EXAMPLE"
echo "    RECON_METHOD=$RECON_METHOD"

HEIGHT=720  # The included videos in ./media/ are all 720p, so we run and save results at 720p
WIDTH=1280  # Also, the highest resolution Vista4D checkpoints are trained on is 720p
NUM_FRAMES=49

PI3_PIXEL_LIMIT=255000  # Pixel limit as given by Pi3(X)
DA3_PROCESS_RES=896  # Can be lowered if encountering OOM with DA3, DA3 max training width is 896

run_recon_and_seg() {
    local EXAMPLE_NAME=$1
    shift
    local SEG_KEYWORDS=("$@")
    local VIDEO_PATH
    if [ -f "./media/edit/$EXAMPLE_NAME.mp4" ]; then  # Video is auto-located in ./media/edit or ./media/single
        VIDEO_PATH="./media/edit/$EXAMPLE_NAME.mp4"
    elif [ -f "./media/single/$EXAMPLE_NAME.mp4" ]; then
        VIDEO_PATH="./media/single/$EXAMPLE_NAME.mp4"
    else
        echo "Video not found for $EXAMPLE_NAME in ./media/edit or ./media/single, exiting."
        exit 1
    fi
    echo ""
    echo "=== recon_and_seg: $EXAMPLE_NAME (from $VIDEO_PATH, keywords: ${SEG_KEYWORDS[*]}) ==="
    python3 -m scripts.preprocess.recon_and_seg_single \
        --video_path "$VIDEO_PATH" \
        --output_folder "./results/edit/$EXAMPLE_NAME/recon_and_seg" \
        --seg_keywords "${SEG_KEYWORDS[@]}" \
        --recon_method $RECON_METHOD \
        --da3_model_id depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
        --pi3_model_id yyfz233/Pi3X \
        --height $HEIGHT --width $WIDTH --num_frames $NUM_FRAMES \
        --pi3_pixel_limit $PI3_PIXEL_LIMIT --da3_process_res $DA3_PROCESS_RES \
        --save_vis
}

# Each edit example pairs a main scene with an insert scene (one of the two provided edits include insert)
# Here we reconstruct both the main and insert scenes
if [ $EXAMPLE == couple-hug ]; then
    run_recon_and_seg couple-hug         "man" "woman"
    run_recon_and_seg couple-newspaper   "man" "woman" "newspaper" "blue car"
elif [ $EXAMPLE == funeral-procession ]; then
    run_recon_and_seg funeral-procession "person" "coffin" "flowers"
    run_recon_and_seg rhino              "rhino"
elif [ $EXAMPLE == hike ]; then
    run_recon_and_seg hike               "person" "backpack"
    run_recon_and_seg cow                "cow" "collar"
elif [ $EXAMPLE == swing ]; then
    run_recon_and_seg swing              "person" "swing" "swing seat" "swing chains" "chains" "jump rope"
    run_recon_and_seg couple-walk        "man" "woman" "dog" "leash" "flower bouquet"
else
    echo "Unrecognized EXAMPLE=$EXAMPLE, exiting script."
    exit 1
fi
