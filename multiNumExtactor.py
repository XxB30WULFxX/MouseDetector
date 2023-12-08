from contNumExtractor import contNumExtractor as CNE
import os 

root = input("Folder to look in?")

files = os.listdir(root)

nfiles = [x for x in files if x[-4:] == ".mp4"]

for file in nfiles:
    print(file)
    cne = CNE()
    cne.run(os.path.join(root,file))