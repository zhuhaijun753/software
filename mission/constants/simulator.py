from mission.constants.missions import Navigate, Recovery

BUOY_BACKUP_DIST = 2.0
BUOY_BACKUP_FOR_OVER_DIST = 0.2
BUOY_OVER_DEPTH = 0.4
BUOY_SEARCH_DEPTH = 0.5
BUOY_TO_PIPE_DIST = 2.5

PIPE_SEARCH_DEPTH = 0.2
PIPE_FOLLOW_DEPTH = 0.5

BINS_CAM_TO_ARM_OFFSET = 0.12
BINS_HEIGHT = 0.5
BINS_DROP_ALTITUDE = BINS_HEIGHT + 0.65
BINS_PICKUP_ALTITUDE = BINS_HEIGHT + 0.2
BINS_SEARCH_DEPTH = 0.5

HYDROPHONES_SEARCH_DEPTH = 0.5
HYDROPHONES_PINGER_DEPTH = 3.0

navigate = Navigate(
    max_distance=10,
    debugging=True,
    thresh_c=12,
    kernel_size=1,
    min_area=40,
    block_size=311,
)

recovery = Recovery(
    stack_dive_altitude=1.7,
    vstack_grab_altitude=1.33,
    hstack_grab_altitude=1.25,

    debugging=True,
    table_c=-83,
    table_block_size=2656,
    table_min_area=5000,
    table_min_fill_ratio=0.5,
    red_stack_c=-15,
    green_stack_c=12,
    red_mark_c=-30,
    green_mark_c=20,
    min_stack_area=40,
    min_stack_width=10,
)
