import cv2
import numpy as np

def main():
    """
    Motion detection using Lucas-Kanade Optical Flow.
    """
    cap = cv2.VideoCapture(0)

    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    mask = np.zeros_like(first_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_points, None, **lk_params
        )

        good_new = next_points[status == 1]
        good_old = prev_points[status == 1]

        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 4, (0, 0, 255), -1)

        output = cv2.add(frame, mask)
        cv2.imshow("Optical Flow Motion Detection", output)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        prev_gray = gray.copy()
        prev_points = good_new.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
