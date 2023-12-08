import XYBehavior
import os 

root = input("Folder to look in?")

files = os.listdir(root)

nfiles = [x for x in files if x[-4:] == ".mp4"]

for file in nfiles:
    print(file)
    xyDetector = XYBehavior.XY_Behavior_Detection(show=True, show_detects_only=True)
    xyDetector.run(os.path.join(root, file))