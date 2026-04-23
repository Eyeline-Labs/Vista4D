EXAMPLE=${EXAMPLE:-lounge-cup}  # We provide some dynamic scene expansion (DSE) examples to demo Vista4D!
RESOLUTION=${RESOLUTION:-720p}  # 384p, 720p
RENDER_ONLY_NECESSARY=${RENDER_ONLY_NECESSARY:-true}
DSE_FRAME_INTERVAL=${DSE_FRAME_INTERVAL:-4}  # Stride through stored DSE frames during unprojection (1 = no skip)
echo "Script kwargs:"
echo "    EXAMPLE=$EXAMPLE"
echo "    RESOLUTION=$RESOLUTION"
echo "    RENDER_ONLY_NECESSARY=$RENDER_ONLY_NECESSARY"
echo "    DSE_FRAME_INTERVAL=$DSE_FRAME_INTERVAL"

if [ $EXAMPLE == conference-punch ]; then
    CAM_PATH=./media/dse/conference-punch_arc-right.npz
elif [ $EXAMPLE == conference-study ]; then
    CAM_PATH=./media/dse/conference-study_arc-left.npz
elif [ $EXAMPLE == hall-cartwheel ]; then
    CAM_PATH=./media/dse/hall-cartwheel_side-follow.npz
elif [ $EXAMPLE == lounge-cup ]; then
    CAM_PATH=./media/dse/lounge-cup_dolly-in.npz
elif [ $EXAMPLE == lounge-drink ]; then
    CAM_PATH=./media/dse/lounge-drink_arc-right-out.npz
elif [ $EXAMPLE == plaza-point ]; then
    CAM_PATH=./media/dse/plaza-point_arc-right.npz
elif [ $EXAMPLE == room-lift ]; then
    CAM_PATH=./media/dse/room-lift_arc-left.npz
elif [ $EXAMPLE == room-walk ]; then
    CAM_PATH=./media/dse/room-walk_zoom-out.npz
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
    --recon_and_seg_folder ./results/dse/$EXAMPLE/recon_and_seg \
    --cam_path $CAM_PATH \
    --output_folder ./results/dse/$EXAMPLE/render_$RESOLUTION \
    --height $HEIGHT --width $WIDTH --num_frames $NUM_FRAMES \
    --dse_frame_interval $DSE_FRAME_INTERVAL \
    --save_vis \
    $ARGS
