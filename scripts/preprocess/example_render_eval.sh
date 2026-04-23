RESOLUTION=${RESOLUTION:-384p}  # 384p, 720p
RENDER_ONLY_NECESSARY=${RENDER_ONLY_NECESSARY:-true}
FORCE_SAME_FIRST_FRAME=${FORCE_SAME_FIRST_FRAME:-false}
echo "Script kwargs:"
echo "    RESOLUTION=$RESOLUTION"
echo "    RENDER_ONLY_NECESSARY=$RENDER_ONLY_NECESSARY"
echo "    FORCE_SAME_FIRST_FRAME=$FORCE_SAME_FIRST_FRAME"

if [ $RESOLUTION == 384p ]; then
    HEIGHT=384
    WIDTH=672
elif [ $RESOLUTION == 720p ]; then
    HEIGHT=720
    WIDTH=1280
fi

OUTPUT_FOLDER=./eval_data/render

ARGS=""
if [ $RENDER_ONLY_NECESSARY == true ]; then
    ARGS="$ARGS --render_only_necessary"
fi
if [ $FORCE_SAME_FIRST_FRAME == true ]; then
    ARGS="$ARGS --force_same_first_frame"
    OUTPUT_FOLDER=${OUTPUT_FOLDER}_fsff
fi

python3 -m scripts.preprocess.render_eval \
    --metadata_path ./eval_data/metadata.csv \
    --recon_and_seg_folder ./eval_data/recon_and_seg \
    --cam_folder ./eval_data/cameras \
    --output_folder ${OUTPUT_FOLDER}_$RESOLUTION \
    --height $HEIGHT --width $WIDTH \
    --save_vis \
    $ARGS
