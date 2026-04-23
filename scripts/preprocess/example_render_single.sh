EXAMPLE=${EXAMPLE:-couple-newspaper}  # We provide some examples to demo Vista4D!
RESOLUTION=${RESOLUTION:-720p}  # 384p, 720p
RENDER_ONLY_NECESSARY=${RENDER_ONLY_NECESSARY:-true}
echo "Script kwargs:"
echo "    EXAMPLE=$EXAMPLE"
echo "    RESOLUTION=$RESOLUTION"
echo "    RENDER_ONLY_NECESSARY=$RENDER_ONLY_NECESSARY"

# These provided cameras are designed by the authors based on the Pi3X 4D reconstruction. If you are using DA3 or Pi3,
# these cameras may not be in their intended positions. Use --scene_scale when running recon_and_seg_single.py (to match
# your 4D reconstruction to that of Pi3X's) or design your own cameras if you are not using Pi3X reconstruction!
if [ $EXAMPLE == couple-newspaper ]; then
    CAM_PATH=./media/single/couple-newspaper_dolly-zoom.npz
elif [ $EXAMPLE == couple-walk ]; then
    CAM_PATH=./media/single/couple-walk_front-follow.npz
elif [ $EXAMPLE == elderly-tennis ]; then
    CAM_PATH=./media/single/elderly-tennis_arc-right.npz
elif [ $EXAMPLE == mountain-hike ]; then
    CAM_PATH=./media/single/mountain-hike_crane-below.npz
elif [ $EXAMPLE == park-selfie ]; then
    CAM_PATH=./media/single/park-selfie_dolly-in.npz
elif [ $EXAMPLE == parkour ]; then
    CAM_PATH=./media/single/parkour_truck-left.npz
elif [ $EXAMPLE == snowboard ]; then
    CAM_PATH=./media/single/snowboard_back-follow.npz
elif [ $EXAMPLE == soapbox ]; then
    CAM_PATH=./media/single/soapbox_crane-above-right.npz
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
    --recon_and_seg_folder ./results/single/$EXAMPLE/recon_and_seg \
    --cam_path $CAM_PATH \
    --output_folder ./results/single/$EXAMPLE/render_$RESOLUTION \
    --height $HEIGHT --width $WIDTH --num_frames $NUM_FRAMES \
    --save_vis \
    $ARGS
