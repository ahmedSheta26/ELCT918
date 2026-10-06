# preprocessing.py -- turns a camera region (ROI) into a tensor ready for LeNet-5

import cv2             # OpenCV: image-processing library
import numpy as np     # numpy: arrays of numbers (images are arrays in OpenCV)
import torch           # PyTorch: we need to output a tensor

# EXACT values used in training (see lenet_mnist_info.json). Never change these
# without retraining, or the model will receive inputs it was not trained on.
MEAN, STD = 0.1307, 0.3081

MIN_FRAC = 0.005   # a digit must cover at least 0.5% of the box (rejects specks)
MAX_FRAC = 0.5     # ...and at most 50% (rejects faces, shadows, big dark areas)
KERNEL = np.ones((3, 3), np.uint8)   # 3x3 block of ones, used to thicken strokes (step 2)


def preprocess(roi, return_steps=False):
    """
    roi: a BGR image (numpy array, height x width x 3) cropped from a camera frame.
    Returns a float tensor of shape (1, 1, 32, 32) = (batch, channels, height, width).
    If return_steps=True, also returns a dict of every intermediate image
    (used for the Task 2 figure and the 'model's view' window in Task 3).
    """
    steps = {"original": roi}

    # ---- Step 1: grayscale + blur ----
    # OpenCV loads colour as Blue-Green-Red, NOT Red-Green-Blue, so we use BGR2GRAY.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Gaussian blur: each pixel becomes a weighted average of its 5x5 neighbourhood.
    # The 0 lets OpenCV choose the blur strength from the kernel size.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    steps["gray_blur"] = blur

    # ---- Step 2: threshold + invert (+ thicken) ----
    # THRESH_BINARY_INV: pixels darker than the threshold become 255 (white), others 0.
    #   That flips dark ink on light paper into a white digit on black.
    # THRESH_OTSU: automatically finds the best threshold (so we pass 0 as a dummy value).
    # The function returns (threshold_used, image); we only need the image, hence `_`.
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Dilate: every white pixel grows into its 3x3 neighbours = thicker strokes.
    # Useful because thin pen lines would almost vanish when we shrink to 20 px.
    # If your strokes are already thick, this is harmless.
    binary = cv2.dilate(binary, KERNEL, iterations=1)
    steps["binary"] = binary

    # ---- Step 3: find and crop the digit ----
    # findContours finds the outlines of white blobs.
    # RETR_EXTERNAL = only outer outlines (ignores holes, e.g. inside a 0 or 8).
    # CHAIN_APPROX_SIMPLE = stores only corner points of each outline (saves memory).
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Keep only outlines big enough to be a digit (removes specks of noise).
    H, W = binary.shape           # box height and width in pixels
    box_area = H * W              # total pixels in the box; sizes are measured relative to this
    valid = []                    # blobs that look like a digit
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)   # rectangle around this blob
        area = cv2.contourArea(c)          # blob area in pixels
        # True if the blob touches any edge of the box (1-pixel tolerance)
        touches_border = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
        if MIN_FRAC * box_area <= area <= MAX_FRAC * box_area and not touches_border:
            valid.append(c)
    contours = valid              # the rest of the code works unchanged

    if len(contours) == 0:
        # Nothing in the box: return an all-background image so the program doesn't crash.
        canvas32 = np.zeros((32, 32), np.uint8)
        steps.update(bbox=None, crop=canvas32, canvas28=canvas32[2:30, 2:30], padded32=canvas32)
    else:
        largest = max(contours, key=cv2.contourArea)       # the digit = the biggest blob
        x, y, w, h = cv2.boundingRect(largest)             # smallest upright rectangle around it
        crop = binary[y:y + h, x:x + w]                    # numpy slicing: rows y..y+h, columns x..x+w
        steps["bbox"] = (x, y, w, h)
        # --- Adaptive thickening: make the stroke a fixed FRACTION of the digit size ---
        # The stroke should be about 10% of the digit's height after shrinking to 20 px
        # (like MNIST). k is how many pixels we grow the stroke by, before shrinking.
        k = max(3, int(round(max(w, h) * 0.07)))   # 7% of the larger side; at least 3
        # Add k black pixels on every side first. Without this, the thickened
        # stroke would be cut off at the edge of the crop.
        crop = cv2.copyMakeBorder(crop, k, k, k, k, cv2.BORDER_CONSTANT, value=0)
        # An ellipse-shaped kernel thickens evenly in all directions (round pen tip).
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        crop = cv2.dilate(crop, kernel)            # grow every white pixel by about k/2
        # Cut off the extra black border again, so the crop hugs the thickened digit.
        ys, xs = np.where(crop > 0)                # row and column of every white pixel
        crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = crop.shape                          # new height and width (used by the resize below)
        steps["crop"] = crop

        # ---- Step 4: resize (keep aspect ratio) and centre ----
        scale = 20.0 / max(w, h)                  # makes the LARGER side exactly 20 px
        new_w = max(1, round(w * scale))          # max(1, ...) avoids a 0-pixel side
        new_h = max(1, round(h * scale))
        # INTER_AREA is the best interpolation method for shrinking images.
        # NOTE: cv2.resize wants (width, height), the opposite order of numpy's (rows, cols).
        small = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas28 = np.zeros((28, 28), np.uint8)   # black 28x28 image
        x_off = (28 - new_w) // 2                 # left margin that centres the digit horizontally
        y_off = (28 - new_h) // 2                 # top margin that centres it vertically
        canvas28[y_off:y_off + new_h, x_off:x_off + new_w] = small   # paste digit in the middle
        steps["canvas28"] = canvas28

        # Pad 2 black pixels on every side: 28x28 -> 32x32 (same as T.Pad(2) in training).
        canvas32 = cv2.copyMakeBorder(canvas28, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        steps["padded32"] = canvas32

    # ---- Step 5: convert to a tensor ----
    arr = canvas32.astype(np.float32) / 255.0     # 0-255 integers -> 0.0-1.0 decimals (like ToTensor)
    arr = (arr - MEAN) / STD                      # same normalisation as training
    # torch.from_numpy makes a tensor; the two unsqueeze(0) calls add the "batch"
    # and "channel" dimensions: (32,32) -> (1,32,32) -> (1,1,32,32).
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    steps["tensor"] = tensor

    return (tensor, steps) if return_steps else tensor