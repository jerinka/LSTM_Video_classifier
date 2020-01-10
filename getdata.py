import numpy as np
import random


# generate the next frame in the sequence
def next_frame(last_step, last_frame, column):
    # define the scope of the next step
    lower = max(0, last_step-1)
    upper = min(last_frame.shape[0]-1, last_step+1)
    # choose the row index for the next step
    step = random.randint(lower, upper)
    # copy the prior frame
    frame = last_frame.copy()
    # add the new step
    frame[step, column] = 1
    return frame, step
    
# generate a sequence of frames of a dot moving across an image
def build_frames(timesteps,width,height, ch=1):
    frames = list()
    # create the first frame
    frame = np.zeros((width,height))
    step = random.randint(0, timesteps-1)
    # decide if we are heading left or right
    right = 1 if random.random() < 0.5 else 0
    col = 0 if right else timesteps-1
    frame[step, col] = 1
    frames.append(frame)
    # create all remaining frames
    for i in range(1, timesteps):
        col = i if right else timesteps-1-i
        frame, step = next_frame(step, frame, col)
        frames.append(frame)
    return frames, right
        
# generate multiple sequences of frames and reshape for network input
def generate_examples(n_patterns,timesteps,width,height,channels):
    X, y = list(), list()
    for _ in range(n_patterns):
        frames, right = build_frames(timesteps,width,height, ch=channels)
        X.append(frames)
        y.append(right)
    import pdb;pdb.set_trace()
    # resize as [samples, timesteps, width, height, channels]
    X = np.array(X).reshape(n_patterns, timesteps,width,height, channels)
    y = np.array(y).reshape(n_patterns, 1)
    return X, y
    
timesteps =5
width=100
height=100
channels=1
samples = 50

X, y = generate_examples(samples, timesteps,width,height,channels)

print('X',X.shape())
print('y',y.shape())


