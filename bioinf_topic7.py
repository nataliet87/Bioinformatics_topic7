from functools import wraps
import numpy as np
from numba import jit  ## "just in time" -- speeds shit up
from numba import prange  ## for parallelization
import matplotlib
matplotlib.use('TkAgg')  ## renderer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Constants
# ========
FPS = 60
GRID_W = 200
GRID_H = 200
ZEROS = np.zeros((GRID_W, GRID_H))
# ========


def timefn(fn):
    """wrapper to time the enclosed function"""

    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print("@timefn: {} took {} seconds".format(fn.__name__, t2 - t1))
        return result

    return measure_time


def initialize_grid():
    """
Generate start grid, with species 1, 2, and 3 populating 1% of the space
Return this grid to main function
    """
    grid = ZEROS.copy()
    g_len = grid.shape[0]
    for x in range(g_len):
        for y in range(g_len):
            state = np.random.random()
            if state <= 0.01:
                grid[y][x] = np.random.randint(1, 4)
    return grid


## can use numpy convolution to flatten some of these loops!?!
## also; try to hoist logic above loops or leave it all inside to make parallelization easier (how about sorting first??)
# @jit(parallel=True, nopython=True, fastmath=True)
@jit

def update_grid(grid):
    """Calculate the next iteration of the grid using the previous iteration

    :param grid: 2D grid of dead/alive cells
    :returns: 2D grid of dead/alive cells

    """
    g_len = grid.shape[0]
    grid_new = ZEROS.copy()

    for y in prange(g_len):
        for x in range(g_len):
            total = np.zeros((1,3))  # accumulator for neighboring cells; next loop sums neighboring cells
            # grid_new[y][x] = grid[y][x]  # would it be faster to have a conditional statement here?
            if not grid[y][x]:
                ### store
            if grid[y][x]:  ## if cell exists; check adjacent populations
                for j in range(-2, 3, 1):
                    for i in range(-2, 3, 1):  ## cycle through surrounding 5x5 grid
                        if (np.abs(i + j) > 2):  # if np.sum(sp_present) == 3:  ## early out -- (test later)
                            continue
                        if grid[(y + j) % g_len][(x + i) % g_len] == 1:
                            total[0][0] += 1
                        elif grid[(y + j) % g_len][(x + i) % g_len] == 2:
                            total[0][1] += 1
                        elif grid[(y + j) % g_len][(x + i) % g_len] == 3:
                            total[0][2] += 1

### list of cell counts has been generated
## next: determine outcome for cell:
                if grid[y][x] == 1:
                    if total[0][1] >= total[0][2]:
                        grid_new[y][x] = 1
                    else:
                        grid_new[y][x] = 4  ## 4 = placeholder for dead cells
                elif grid[y][x] == 2:
                    if total[0][2] >= total[0][0]:
                        grid_new[y][x] = 2
                    else:
                        grid_new[y][x] = 4
                elif grid[y][x] == 3:
                    if total[0][0] >= total[0][1]:
                        grid_new[y][x] = 3
                    else:
                        grid_new[y][x] = 4

    grid = grid_new.copy()

    # cell propagation next:
    for y in prange(g_len):
        for x in range(g_len):
            sp_present = np.zeros((1, 3))
            if grid[y][x] == 0:
                for j in range(-2, 3, 1):
                    for i in range(-2, 3, 1):
                        if (np.abs(i + j) > 2):  # if np.sum(sp_present) == 3:  ## early out -- (test later)
                            continue
                        if grid[(y + j) % g_len][(x + i) % g_len] == 1:
                            sp_present[0][0] = 1
                        elif grid[(y + j) % g_len][(x + i) % g_len] == 2:
                            sp_present[0][1] = 1
                        elif grid[(y + j) % g_len][(x + i) % g_len] == 3:
                            sp_present[0][2] = 1
                        ## would it be possible to switch the below logic to call unique values from the array of interest
                        ## and then randomly select from among them?; look at using np.where()

                if sp_present.any():
                    grid_new[y][x] = np.random.choice(np.nonzero(sp_present)[1]) + 1
                    ## populate space with a randomly selected species w/in r-disp range

            elif grid[y][x] == 4: # handle dead cells
                grid_new[y][x] = 0

    return grid_new


## (an alternate method for grid update might have coordinates in a dictionary and run through that dictionary once
## rather than cycling through a whole bunch of for loops)


def update(frame, im, grid):
    """function which is called each tick of the animation

    :param frame: The current frame index
    :type frame: int
    :param im: The image being updated
    :type im: matplotlib imshow
    :param grid: 2D grid of dead/alive cells
    :type grid: np.array
    :returns: updated image
    """
    new_grid = update_grid(grid)
    im.set_array(new_grid)
    grid[:] = new_grid[:]
    return (im, )  ## returns a tuple; these get put into a list so don't return it as an object


@timefn
def draw_N(grid, N):  ## this just runs the grid update without rendering an image; useful for optimization
    for i in range(N):
        grid[:] = update_grid(grid)[:]


@timefn
def main():
    grid = initialize_grid()
    # draw_N(grid, 200)
    # update_grid.parallel_diagnostics(level=1)
    fig = plt.figure()
    im = plt.imshow(
        grid,
        cmap="tab10",
        aspect="equal",
        interpolation="none"  # this sets grid resolution at screen resolution
    )
    ani = FuncAnimation(
        fig,
        update,
        fargs=(im, grid),
        frames=FPS * 5,
        interval=1000 / FPS,
        repeat=False,  # stop rendering once you've hit your max number of frames
        blit=True,  # stops redrawing pixels that haven't changed
    )
    plt.show()
    return ani


if __name__ == "__main__":
    ani = main()
    ## this will be super slow; saving video takes ages
    ## ani.save("./test.mpg")

# next up you're going to check how long it takes the simulation to run without doing any of the rendering
