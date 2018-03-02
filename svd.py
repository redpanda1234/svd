import numpy as np
import os
from PIL import Image
import sys
import time

def to_gray(rgb):
    """
    Takes as input an rgb tuple and returns a scalar representing the
    grayscale value
    """
    return .21*rgb[0] + .72*rgb[1] + .07*rgb[2]

def scoop(sigma, u, v):
    return sigma * np.outer(u,v)


def main():
    pics = [name for name in os.listdir() if ".png" in name]
    print(pics)
    og_dir = os.getcwd()
    for name in pics:
        rest = name[:-4]
        try:
            os.mkdir(rest+"/")
        except FileExistsError:
            continue

        start = time.time()
        print("began processing ", name)
        Im = Image.open(name).getdata()
        # extract the png image data and store it in a numpy array
        im_arr = np.array(Im)
        print("successfully created array, in just ", time.time() -
              start, ". reshaping...")

        num_slots = im_arr.shape[-1]

        if num_slots == 4:
            print(im_arr.shape)
            im_arr = im_arr[:,:3]
        rs_start = time.time()
        image = np.reshape(im_arr, (Im.size[1], Im.size[0], 3))
        print("reshaped in just ", time.time() - rs_start,
              ". Beginning setup")

        os.chdir(rest)


        gs_start = time.time()
        image = np.apply_along_axis(to_gray, 2, image)

        print("succeeded with grayscale application, in just", time.time()-gs_start)

        i = 1
        j = 0

        svd_start = time.time()
        u, s, vh = np.linalg.svd(image, full_matrices=True)
        print("successfully factored w/ SVD, in ", time.time() - svd_start)

        mat = np.zeros(image.shape)
        k = 0

        while i < image.shape[0] and j < 14:
            mat_start = time.time()

            s1 = np.zeros(s.shape) + s
            u1 = np.zeros(u.shape) + u
            vh1 = np.zeros(vh.shape) + vh

            while k <= i:
                max_index = np.argmax(s1)
                max_s = s1[max_index]

                s1 = np.delete(s1, max_index, 0)
                k += 1
                u_in = u1[:,max_index]
                vh_in = vh1[max_index]

                mat += scoop(max_s, u_in, vh_in)
                u1 = np.delete(u1, max_index, 1)
                vh1 = np.delete(vh1, max_index, 0)

            mat -= mat.min()
            mat /= mat.max()
            mat *= 255

            print("finished processing the matrix in", time.time() - mat_start)

            write_start = time.time()
            im = Image.fromarray(np.uint8(mat))
            im.save(rest + "_" + str(i)+".png", optimize=True)
            print("wrote out in just ", time.time() - write_start)
            j += 1
            i *= 2



        print("processed", name)

        os.chdir(og_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        os.chdir("..")
        print("Unexpected error:", sys.exc_info()[0])
        raise Exception("ahahah")
