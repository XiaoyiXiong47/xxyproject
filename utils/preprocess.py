"""
Created by: Xiaoyi Xiong
Date: 04/03/2025
"""

import numpy as np
from scipy.interpolate import interp1d




def process_frame(left_hand, right_hand):
    """
    The dimension of each frame is 2 × 21 × 3 = 126, fill in zeros even if hand is missing.
    Concatenate 3D (x3) coordinates for all 21 keypoints of both hands (x2) .
    """
    # left_hand/right_hand is in shape (21, 3) or None
    # left = left_hand if left_hand is not None else np.zeros((1, 3))
    left = left_hand if left_hand is not None else np.zeros((21, 3))
    # right = right_hand if right_hand is not None else np.zeros((1, 3))
    right = right_hand if right_hand is not None else np.zeros((21, 3))
    out = np.concatenate([left.flatten(), right.flatten()])
    # print("process_frame output shape:", out.shape)
    return out




def interpolate_sequence(sequence):
    """Perform interpolation for each keypoint's time sequence"""
    # sequence: (n_frames, 126)
    seq_interp = sequence.copy()
    for i in range(sequence.shape[1]):
        col = sequence[:, i] # i-th column
        mask = col != 0 # mask non-zero column
        if mask.sum() < 2: # less than two valid columns
            continue  # do not interpolate
        #
        f = interp1d(np.where(mask)[0], col[mask], kind='linear', fill_value='extrapolate')
        seq_interp[:, i] = f(np.arange(len(col)))
    return seq_interp



