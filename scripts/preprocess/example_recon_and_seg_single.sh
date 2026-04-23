EXAMPLE=${EXAMPLE:-couple-newspaper}  # We provide some examples to demo Vista4D!
RECON_METHOD=${RECON_METHOD:-pi3}  # pi3, da3
echo "Script kwargs:"
echo "    EXAMPLE=$EXAMPLE"
echo "    RECON_METHOD=$RECON_METHOD"

if [ $EXAMPLE == couple-newspaper ]; then
    SEG_KEYWORDS=("man" "woman" "newspaper" "blue car")
elif [ $EXAMPLE == couple-walk ]; then
    SEG_KEYWORDS=("man" "woman" "dog" "leash" "flower bouquet")
elif [ $EXAMPLE == elderly-tennis ]; then
    SEG_KEYWORDS=("man" "tennis ball" "tennis racket" "shoes")
elif [ $EXAMPLE == mountain-hike ]; then
    SEG_KEYWORDS=("person" "backpack")
elif [ $EXAMPLE == park-selfie ]; then
    SEG_KEYWORDS=("man" "woman" "phone" "hand")
elif [ $EXAMPLE == parkour ]; then
    SEG_KEYWORDS=("man")
elif [ $EXAMPLE == snowboard ]; then
    SEG_KEYWORDS=("person" "helmet" "goggles" "snow suit" "snowboard")
elif [ $EXAMPLE == soapbox ]; then
    SEG_KEYWORDS=("person" "helmet" "blue cart" "blue trolley" "wheel" "barrel")
else
    echo "Unrecognized EXAMPLE=$EXAMPLE, exiting script."
    exit 1
fi

HEIGHT=720  # The included videos in ./media/ are all 720p, so we run and save results at 720p
WIDTH=1280  # Also, the highest resolution Vista4D checkpoints are trained on is 720p
NUM_FRAMES=49

PI3_PIXEL_LIMIT=255000  # Pixel limit as given by Pi3(X)
DA3_PROCESS_RES=896  # Can be lowered if encountering OOM with DA3, DA3 max training width is 896

python3 -m scripts.preprocess.recon_and_seg_single \
    --video_path ./media/single/$EXAMPLE.mp4 \
    --output_folder "./results/single/$EXAMPLE/recon_and_seg" \
    --seg_keywords "${SEG_KEYWORDS[@]}" \
    --recon_method $RECON_METHOD \
    --da3_model_id depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
    --pi3_model_id yyfz233/Pi3X \
    --height $HEIGHT --width $WIDTH --num_frames $NUM_FRAMES \
    --pi3_pixel_limit $PI3_PIXEL_LIMIT --da3_process_res $DA3_PROCESS_RES \
    --save_vis
