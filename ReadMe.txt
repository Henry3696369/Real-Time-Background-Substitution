
A computer vision project exploring multi-modal spatiotemporal post-processing techniques to improve neural confidence maps for real-time background substitution.


python -m venv venv

.\venv\Scripts\activate

pip install -r requirements.txt

python main.py



Manual:
"-" : confidence map 
"=": depth map
"1": original camera frame
"0": binary mask substition
"2": Version1 -- Hysteresis && topological
"3": Version2 -- + Depth model
"4": Version3 -- + Optical flow
	{It can only work unerder version 3:{
		Press"8": Switch between 
			Basic mode, 
			Motion blur mode and,
			Wild mode
		}
	}
"9": Glowing eyes effect (Don't use it when you are in confidence map or depth map mode.
"r": Reset the background image


Important Notes

Compatibility: I cannot guarantee the app will run perfectly on all hardware.
Optimization: For the midas_small model, I am using a version optimized specifically for my AMD CPU.
Code Logic: If the application fails to run on your machine, the implementation logic should still be clear through my code comments and documentation.
