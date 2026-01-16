# Motion Detection Using Optical Flow

This project implements motion detection using the classical Lucas–Kanade optical flow algorithm. The objective is to track feature points across consecutive video frames and analyze motion patterns.

## Methodology
- Feature points are detected using Shi–Tomasi corner detection.
- Motion vectors are estimated using the Lucas–Kanade optical flow method.
- Motion trajectories are visualized using vector lines.

## Technologies Used
- Python 3
- OpenCV
- NumPy

## How to Run
```bash
pip install -r requirements.txt
python src/optical_flow.py
