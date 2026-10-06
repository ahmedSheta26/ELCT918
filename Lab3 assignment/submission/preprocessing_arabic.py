# preprocessing_arabic.py -- turns a camera region (ROI) into a tensor for the MADBase LeNet-5

import cv2             # OpenCV: image processing
import numpy as np     # numpy: arrays of numbers
import torch           # PyTorch: the output must be a tensor

# ---- Normalisation: EXACT values measured on the MADBase training images ----
# (they are saved in lenet_madbase_info.json). Never change them without retraining.
MEAN, STD = 0.0777, 0.2438

# ---- Size limits for a normal digit blob, as fractions of the box area (same as the English pipeline) ----
MIN_FRAC = 0.005    # below 0.5% of the box a blob is too small to be a normal digit
MAX_FRAC = 0.5      # above 50% it is a face, a shadow or something else big

# ---- Dot-zero rule: used ONLY when the box holds no normal digit ----
DOT_MIN_FRAC = 0.0005   # a deliberate dot covers at least 0.05% of the box (smaller = camera speck, ignored)
DOT_MIN_FILL = 0.50     # ink pixels / bounding-box pixels: a round dot is about 0.79, MADBase zeros: median 0.61
DOT_MIN_ASPECT = 0.55   # shorter side / longer side of the bounding box: 1.0 = perfectly round

# ---- Stroke thickness target (measured in Colab) ----
# A MADBase stroke is about 9% of the digit size. Measured at camera resolution this reads 0.086.
TARGET_FRAC = 0.086

KERNEL = np.ones((3, 3), np.uint8)   # 3x3 block of ones: keeps thin strokes connected (step 2)


def stroke_width(binary):
    """Average pen-stroke width in pixels = 2 x area / perimeter.
    A ribbon of width s and length L has area s*L and outline length about 2*L, so the ratio gives s.
    This is the SAME measuring method we used on MADBase in Colab."""
    area = cv2.countNonZero(binary)                 # number of white (ink) pixels
    # Outlines of every white shape, including the inner edge of loops (like the hole of a 6).
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)   # total outline length
    return 2.0 * area / perimeter if perimeter > 0 else 0.0     # avoid dividing by zero


def preprocess(roi, return_steps=False):
    """
    roi: a BGR image (height x width x 3) cropped from a camera frame.
    Returns a float tensor of shape (1, 1, 32, 32). With return_steps=True it also returns
    a dict of every intermediate image plus a text note saying what was detected.
    """
    steps = {"original": roi}

    # ---- Step 1: grayscale + blur (same as English; OpenCV colours are BGR) ----
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)         # smooths camera speckle
    steps["gray_blur"] = blur

    # ---- Step 2: threshold + invert (MADBase is also white digit on black, so the same call works) ----
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # One light 3x3 thickening so thin strokes do not break into pieces; the real thickness is set in step 3.
    binary = cv2.dilate(binary, KERNEL, iterations=1)
    steps["binary"] = binary

    # ---- Step 3a: find all white blobs and sort them into "normal digit" and "small" ----
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = binary.shape
    box_area = H * W
    big, small = [], []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        touches_border = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
        if touches_border:
            continue                                 # a held digit sits inside the box; edge-touching blobs are not digits
        if MIN_FRAC * box_area <= area <= MAX_FRAC * box_area:
            big.append(c)                            # normal-sized blob: a digit candidate
        elif DOT_MIN_FRAC * box_area <= area < MIN_FRAC * box_area:
            small.append(c)                          # small but above the speck floor: a dot candidate
        # anything smaller than the floor is treated as camera noise and ignored completely

    # ---- Step 3b: decide what to use ----
    chosen, note = None, "no digit found"
    if big:                                          # a normal digit exists: it always wins
        chosen = max(big, key=cv2.contourArea)       # the digit = the biggest blob
        note = "digit"
    elif len(small) == 1:                            # no digit, exactly ONE small blob: maybe a dot-zero
        x, y, w, h = cv2.boundingRect(small[0])
        # Same definitions as in the Colab measurement: ink pixels / box pixels, and short side / long side.
        fill = cv2.countNonZero(binary[y:y + h, x:x + w]) / (w * h)
        aspect = min(w, h) / max(w, h)
        if fill >= DOT_MIN_FILL and aspect >= DOT_MIN_ASPECT:
            chosen, note = small[0], "dot"
        else:
            note = f"small blob rejected (fill {fill:.2f}, aspect {aspect:.2f})"
    elif len(small) > 1:                             # several small blobs: cannot tell which is the dot
        note = f"{len(small)} small blobs, ambiguous: no dot accepted"
    steps["note"] = note
    steps["stroke_frac"] = None                      # filled in below if a digit is found

    if chosen is None:
        # Nothing usable: return an all-background image so the program does not crash.
        canvas32 = np.zeros((32, 32), np.uint8)
        steps.update(bbox=None, crop=canvas32, canvas28=canvas32[2:30, 2:30], padded32=canvas32)
    else:
        x, y, w, h = cv2.boundingRect(chosen)
        crop = binary[y:y + h, x:x + w]
        steps["bbox"] = (x, y, w, h)

        # ---- Step 3c: thicken ONLY as much as needed to reach MADBase's stroke thickness ----
        L = max(w, h)                                # larger side of the digit, in pixels
        # 1 black pixel around the crop so the shape does not touch the image edge when measured
        # (the situation in the Colab measurement, where digits sit inside the frame).
        padded = cv2.copyMakeBorder(crop, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        width = stroke_width(padded)                 # measure what we actually have
        steps["stroke_frac"] = width / L
        # Adding e pixels to BOTH the stroke and the size gives (width+e)/(L+e). Setting this equal to
        # TARGET_FRAC and solving for e gives the formula below. If the stroke is already thick
        # enough, e is negative and we clamp it to 0 (a thick marker is left alone).
        extra = max(0.0, (TARGET_FRAC * L - width) / (1 - TARGET_FRAC))
        half = int(round(extra / 2))                 # pixels to grow on EACH side of the stroke
        if half >= 1:
            crop = cv2.copyMakeBorder(crop, half, half, half, half, cv2.BORDER_CONSTANT, value=0)  # room to grow
            kd = 2 * half + 1                        # an odd kernel of diameter 2*half+1 grows every edge by `half`
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))   # round pen tip
            crop = cv2.dilate(crop, kernel)
            ys, xs = np.where(crop > 0)              # cut the extra black border off again
            crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            h, w = crop.shape                        # new size, used by the resize below
        steps["crop"] = crop

        # ---- Step 4: resize so the LARGER side is 20 px (MADBase does this for every digit, the zero included) ----
        scale = 20.0 / max(w, h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        # INTER_AREA is the best choice when SHRINKING. When ENLARGING (a small dot) it copies pixels
        # into blocks, so we use INTER_LINEAR, which blends neighbours and keeps the edges smooth.
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

        canvas28 = np.zeros((28, 28), np.uint8)
        x_off = (28 - new_w) // 2
        y_off = (28 - new_h) // 2
        canvas28[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        steps["canvas28"] = canvas28

        canvas32 = cv2.copyMakeBorder(canvas28, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)   # 28x28 -> 32x32
        steps["padded32"] = canvas32

    # ---- Step 5: tensor, normalised with the MADBase values ----
    arr = canvas32.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)    # (32,32) -> (1,1,32,32)
    steps["tensor"] = tensor

    return (tensor, steps) if return_steps else tensor