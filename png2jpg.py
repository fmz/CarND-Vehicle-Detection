import sys, os
import numpy as np
import cv2

# Search the specified directory for .png images and convert them to .jpg
def _png2jpg(dirname, outdir):
    out_file_num = 0
    for name in os.listdir(dirname):
        if name == ".DS_Store":
            # Eradicate this pest.
            os.remove(os.path.join(dirname, name))
            continue

        if os.path.isfile(os.path.join(dirname, name)):
            print("Converting: " + name)
            fname, fext = os.path.splitext(name)
            if (fext == '.png'):
                img = cv2.imread(os.path.join(dirname, name))
                cv2.imwrite(os.path.join(outdir, str(out_file_num) \
                    + ".jpg"), img)
                out_file_num += 1
        else:
            newdir = os.path.join(dirname, name)
            print("Going into directory: ", newdir)
            _png2jpg(newdir, outdir)
#            newoutdir = os.path.join(outdir, name)
#            if not os.path.exists(newoutdir):
#                os.makedirs(newoutdir)
#            else:
#                print("Error: directory " + newoutdir + " already exists")
        
def png2jpg(dirname):
    outdir = dirname + "_out"
    if not os.path.exists(outdir):
        os.makedirs(outdir) 
        _png2jpg(dirname, outdir)
        print("Done!")
    else:
        print("Error: directory " + outdir + " already exists")


if __name__ == "__main__":
    from sys import argv
    if len(argv) != 2:
        print("USAGE: png2jpg <dirname>")

    png2jpg(argv[1])


