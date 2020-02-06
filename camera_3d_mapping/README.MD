## OpenCV Camera Calibration  
1. print the following check board marker on A4/Letter size at 1:1 scale and stick it on a firm flat board:  
Camera Data Analysis/camera 3d mapping/00_Camera Calibration/camera-calibration-checker-board_8x7.pdf  
  
2. use the camera to take 10 photos of the check board from different angles and position. Make sure the check board is completely seen, and occupies the center, each corner and edge of the picture. Here's a set of example photos:  
Camera Data Analysis/camera 3d mapping/00_Camera Calibration/input images/  
  
3. Run the following python script. Click "space" key to advance each picture. When success, "data/calib.npz" will be saved with camera parameter matix for later use.  
Camera Data Analysis/camera 3d mapping/00_Camera Calibration/OpenCV_Python_Camera_Calibration.py  
