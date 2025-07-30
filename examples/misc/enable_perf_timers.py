import numpy as np
import genesis as gs

from genesis.utils.tools import Timer2

gs.init(backend=gs.cpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=50,
        run_in_thread=False,
        enable_interaction=True,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.02,
    ),
    profiling_options=gs.options.ProfilingOptions(
        show_FPS=False,
    ),
    show_viewer=True,
)

plane = scene.add_entity(gs.morphs.Plane(pos=(0, 0, 0)))

# create a stack of boxes
box_size = np.array([0.25, 0.25, 0.25])
for i in range(5):
    scene.add_entity(gs.morphs.Box(size=box_size, pos=(0, 0, (i + 0.5) * box_size[2])))

scene.build()

Timer2.enable()
for i in range(500):
    Timer2.begin("demo")
    scene.step()
    Timer2.end()
    Timer2.print_timers_and_reset()
