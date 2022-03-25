#!/usr/bin/python

import subprocess
import os
import sys

cwd = os.getcwd()

def write_result(file, output):
    if (output[1] is not None and len(output[1]) > 0):
        print(output[1].decode("utf-8"))
        exit()
    file.write("%s" % (output[0].decode("utf-8")))
    file.flush()


def run(prog, path):
    os.makedirs(path, exist_ok=True)
    tpb = range(1, 33)
    blocks = [(1, 0), (1, 1), (2, 1)]
    n = 10000000

    for block, mode in blocks:
        f = open(os.path.join(path, "warps%d_%d.csv" % (block, mode)), "w")

        proc = subprocess.Popen([prog,
                                "-b", str(block),
                                "-B", str(block),
                                "-n", str(n),
                                "-N", str(n),
                                "-M", str(n),
                                "-K", str(n),
                                "-m", str(mode),
                                "-w", str(tpb[0]),
                                "-W", str(tpb[-1])],
                                env=os.environ, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out = proc.communicate()
        write_result(f, out)
        f.close()
    # 1080Ti
    tpb = [32]
    blocks = range(1, 56 * 3 + 1)
    # 2080Ti
    # tpb = [32]
    # blocks = range(1, 68 * 3 + 1)
    # A100
    # tpb = [32]
    # blocks = range(1, 108 * 3 + 1)

    f = open(os.path.join(path, "blocks.csv"), "w")
    proc = subprocess.Popen([prog,
                            "-b", str(blocks[0]),
                            "-B", str(blocks[-1]),
                            "-n", str(n),
                            "-N", str(n),
                            "-M", str(n),
                            "-K", str(n),
                            "-w", str(tpb[0]),
                            "-W", str(tpb[-1])],
                            env=os.environ, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out = proc.communicate()
    write_result(f, out)
    f.close()

    tpbs = [32, 16, 8, 4]
    for tpb in tpbs:
        blocks = range(1, 4 * 32 // tpb + 1)
        f = open(os.path.join(path, "kerneltimes%d.csv" % (tpb)), "w")
        os.makedirs(os.path.join(path, "kerneltimes"), exist_ok=True)
        proc = subprocess.Popen([prog,
                                "-b", str(blocks[0]),
                                "-B", str(blocks[-1]),
                                "-n", str(n),
                                "-N", str(n),
                                "-M", str(n),
                                "-K", str(n),
                                "-w", str(tpb),
                                "-W", str(tpb),
                                "-e", str(1),
                                "-m", str(1),
                                "-d", os.path.join(path, "kerneltimes")],
                                env=os.environ, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out = proc.communicate()
        write_result(f, out)
        f.close()


if __name__ == "__main__":
    outfolder = "output"
    prog = "./arithmetic" # "./matmult"
    flags = ["-DINSTR=0", "-DINSTR=0 -DTIMING", "-DINSTR=1", "-DINSTR=1 -DTIMING"]

    if (len(sys.argv) > 1):
        outfolder = sys.argv[1]

    for i, flag in enumerate(flags):
        # i Type                          Size (MxNxK)
        #  0 Tensor half * half     = float  16 x 16 x 16
        #  1 Tensor half * half     = float  32 x  8 x 16
        #  2 Tensor half * half     = float   8 x 32 x 16
        #  3 Tensor half * half     = half   16 x 16 x 16
        #  4 Tensor half * half     = half   32 x  8 x 16
        #  5 Tensor half * half     = half    8 x 32 x 16
        #  6 Tensor uint8 * uint8   = int    16 x 16 x 16
        #  7 Tensor uint8 * uint8   = int    32 x  8 x 16
        #  8 Tensor uint8 * uint8   = int     8 x 32 x 16
        #  9 Tensor int8 * int8     = int    16 x 16 x 16
        # 10 Tensor int8 * int8     = int    32 x  8 x 16
        # 11 Tensor int8 * int8     = int     8 x 32 x 16
        # 12 Tensor bf16 * bf16     = float  16 x 16 x 16
        # 13 Tensor bf16 * bf16     = float  32 x  8 x 16
        # 14 Tensor bf16 * bf16     = float   8 x 32 x 16
        # 15 Tensor tf32 * tf32     = float   8 x 32 x 16
        # 16 Tensor double * double = double  8 x  8 x  4
        # 17 CUDA   half * half     = float   4 x  8 x 16
        # 18 CUDA   half * half     = half    4 x  8 x 16
        # 19 CUDA   uint8 * uint8   = int     4 x  8 x 16
        # 20 CUDA   double * double = double  4 x  8 x  4
        # 21 CUDA   float * float   = float   4 x  8 x 16
        for config in range(17, 22):
            subprocess.run(["make", "clean"])
            envcpy = os.environ.copy()
            if (not "NVCCFLAGS" in envcpy):
                envcpy["NVCCFLAGS"] = ""
            envcpy["NVCCFLAGS"] += "%s -DCONFIG=%d" % (flag, config)
            try:
                subprocess.run(["make"], env=envcpy, check=True)
            except:
                continue

            run(prog, os.path.join(outfolder, str(i) + "_" + str(config)))

