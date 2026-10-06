# capture.py -- shows the webcam with a centre box and saves ONLY the boxed region
import cv2   # OpenCV: opens the camera, draws on images, saves files

# CAP_DSHOW = DirectShow, a Windows camera driver. The default driver often returns
# empty frames on Windows, so we ask for this one explicitly.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # 0 = the default laptop camera
if not cap.isOpened():   # stop with a clear message if the camera could not be opened
    raise SystemExit("Could not open camera 0. Try VideoCapture(1, cv2.CAP_DSHOW) or close apps using the camera.")

count = 0   # how many images saved so far; gives each file a different name
failed = 0  # how many frames in a row came back empty

while True:                 # keep reading frames until the user quits
    ok, frame = cap.read()  # ok = did it work?, frame = the current image (BGR numpy array)
    if not ok:
        # Cameras often give a few empty frames while warming up, so we tolerate some.
        failed += 1
        print("no frame from camera, retrying...", failed)
        if failed > 30:     # give up after 30 failures in a row, and say why
            raise SystemExit("Camera opened but gives no images. Check Windows camera privacy settings.")
        cv2.waitKey(100)    # wait 100 ms before trying again
        continue            # skip the rest of the loop and read the next frame
    failed = 0              # a good frame arrived, so reset the failure counter

    h, w = frame.shape[:2]            # frame height and width in pixels
    side = int(min(h, w) * 0.6)       # box side = 60% of the shorter dimension (a square that fits)
    x1 = (w - side) // 2              # left edge that centres the box horizontally
    y1 = (h - side) // 2              # top edge that centres the box vertically
    x2, y2 = x1 + side, y1 + side     # right and bottom edges

    # Draw on a COPY so the green lines are not saved inside the image
    # (they would be mistaken for part of the digit).
    display = frame.copy()
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)   # (0,255,0) = green in BGR
    cv2.putText(display, "SPACE = save   q = quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("capture", display)    # show the frame with the box

    key = cv2.waitKey(1) & 0xFF       # wait 1 ms for a key press; & 0xFF keeps only the key code
    if key == ord(" "):               # SPACE: save the clean region inside the box
        roi = frame[y1:y2, x1:x2]     # numpy slicing: rows first (y), then columns (x)
        count += 1
        name = f"digit_{count}.jpg"   # digit_1.jpg, digit_2.jpg, ...
        cv2.imwrite(name, roi)        # write the image to the current folder
        print("saved", name)
    elif key == ord("q"):             # q: leave the loop
        break

cap.release()            # give the camera back to the system
cv2.destroyAllWindows()  # close the preview window