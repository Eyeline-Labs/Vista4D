EXAMPLE=${EXAMPLE:-hike_cow}  # We provide some 4D scene recomposition examples to demo Vista4D!
RESOLUTION=${RESOLUTION:-720p}  # 384p, 720p
RENDER_ONLY_NECESSARY=${RENDER_ONLY_NECESSARY:-true}
echo "Script kwargs:"
echo "    EXAMPLE=$EXAMPLE"
echo "    RESOLUTION=$RESOLUTION"
echo "    RENDER_ONLY_NECESSARY=$RENDER_ONLY_NECESSARY"

EXAMPLE_NAME="${EXAMPLE%_*}"

if [ $EXAMPLE == couple-hug_duplicate-car ]; then
    CAM_PATH=./media/edit/couple-hug_duplicate-car_crane-above.npz
elif [ $EXAMPLE == couple-hug_couple-newspaper ]; then
    CAM_PATH=./media/edit/couple-hug_couple-newspaper_arc-right.npz
elif [ $EXAMPLE == funeral-procession_remove-priest ]; then
    CAM_PATH=./media/edit/funeral-procession_remove-priest_side-truck.npz
elif [ $EXAMPLE == funeral-procession_rhino ]; then
    CAM_PATH=./media/edit/funeral-procession_rhino_crane-below.npz
elif [ $EXAMPLE == hike_enlarge-backpack ]; then
    CAM_PATH=./media/edit/hike_enlarge-backpack_arc-left.npz
elif [ $EXAMPLE == hike_cow ]; then
    CAM_PATH=./media/edit/hike_cow_crane-above-right.npz
elif [ $EXAMPLE == swing_shrink-person ]; then
    CAM_PATH=./media/edit/swing_shrink-person_crane-below-right.npz
elif [ $EXAMPLE == swing_couple-walk ]; then
    CAM_PATH=./media/edit/swing_couple-walk_arc-left.npz
else
    echo "Unrecognized EXAMPLE=$EXAMPLE, exiting script."
    exit 1
fi

if [ $RESOLUTION == 384p ]; then
    HEIGHT=384
    WIDTH=672
elif [ $RESOLUTION == 720p ]; then
    HEIGHT=720
    WIDTH=1280
fi
NUM_FRAMES=49

ARGS=""
if [ $RENDER_ONLY_NECESSARY == true ]; then
    ARGS="$ARGS --render_only_necessary"
fi

python3 -m scripts.preprocess.render_single \
    --recon_and_seg_folder ./results/edit/$EXAMPLE_NAME/recon_and_seg \
    --cam_path $CAM_PATH \
    --edits_path ./media/edit/$EXAMPLE.json \
    --output_folder ./results/edit/$EXAMPLE/render_$RESOLUTION \
    --height $HEIGHT --width $WIDTH --num_frames $NUM_FRAMES \
    --save_vis \
    $ARGS
