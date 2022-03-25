#!/usr/bin/python

import subprocess
import os
import sys

def write_result(file, output):
    if (output[1] is not None and len(output[1]) > 0):
        print(output[1].decode("utf-8"))
        exit()
    file.write("%s" % (output[0].decode("utf-8")))
    file.flush()


def run(path, policy):
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "default", policy), exist_ok=True)

    ###########################################################################
    # PART 1: Broad exploration
    ###########################################################################

    # CONFIG: different numbers of blocks to try
    blocks = [1, 14, 28, 56, 112]        # used for 1080 Ti in paper
    # blocks = [1, 17, 34, 68, 136]      # used for 2080 Ti in paper
    # blocks = [1, 14, 27, 54, 108, 216] # used for A100 in paper

    # CONFIG: different numbers of threads per block
    tpb = [32, 64, 128, 256, 512, 1024]

    # CONFIG: different matrix sizes
    size = [(40, 40), (200, 200), (1000, 1000), (5000, 5000), (20000, 20000)]

    # CONFIG: number of repetitions
    # NOTE:   The length of this list must equal the length of the `size`
    # parameter, and the position in this list indicates the number of
    # repetitions for the experiment with the matrix size at the same index.
    repeats = [3, 3, 3, 3, 3]

    for i in range(4):
        # i mapping
        # 0 Linear copy
        # 1 Row major read
        # 2 Column major read
        # 3 Cache optimised
        f = open(os.path.join(path, "default", policy, "res_%d.csv" % (i)), "w")
        for r, (n, m) in zip(repeats, size):
            for block in blocks:
                for t in tpb:
                    print("%d (%d, %d) %d %d" % (i, n, m, block, t))
                    proc = subprocess.Popen(["./transpose",
                                            "-b", str(block),
                                            "-n", str(n),
                                            "-m", str(m),
                                            "-t", str(t),
                                            "-r", str(r),
                                            "-v", str(i)],
                                            env=os.environ, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
                    out = proc.communicate()
                    write_result(f, out)
        f.close()

    ###########################################################################
    # PART 2: Matrix size exploration
    ###########################################################################

    # CONFIG: different numbers of blocks to try
    blocks = [14]    # used for 1080 ti in paper
    # blocks = [136] # used for 2080 ti in paper
    # blocks = [108] # used for A100 in paper

    # CONFIG: different numbers of threads per block
    tpb = [256]      # used for 1080 ti in paper
    # tpb = [128]    # used for 2080 ti in paper
    # tpb = [512]    # used for A100 in paper

    # CONFIG: different matrix sizes
    size = [(i * 100, i * 100) for i in range(50, 200)]

    # CONFIG: number of repetitions
    repeats = [3 for _ in range(50, 200)]

    os.makedirs(os.path.join(path, "mat_size", policy), exist_ok=True)
    for i in range(4):
        f = open(os.path.join(path, "mat_size", policy, "res_%d.csv" % (i)), "w")
        for r, (n, m) in zip(repeats, size):
            for block in blocks:
                for t in tpb:
                    print("%d (%d, %d) %d %d" % (i, n, m, block, t))
                    proc = subprocess.Popen(["./transpose",
                                            "-b", str(block),
                                            "-n", str(n),
                                            "-m", str(m),
                                            "-t", str(t),
                                            "-r", str(r),
                                            "-v", str(i)],
                                            env=os.environ, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
                    out = proc.communicate()
                    write_result(f, out)
        f.close()

    ###########################################################################
    # PART 3: Block count exploration
    ###########################################################################

    # CONFIG
    blocks = range(14, 225) # used for 1080 Ti in paper
    # blocks = range(14, 273) # used for 2080 Ti in paper
    # blocks = range(14, 217) # used for A100 in paper

    # CONFIG
    tpb = [256] # used for 1080 Ti in paper
    # tpb = [128] # used for 2080 Ti in paper
    # tpb = [512] # used for A100 in paper

    # CONFIG
    size = [(5000, 5000), (20000, 20000)]

    # CONFIG
    repeats = [3, 3]

    os.makedirs(os.path.join(path, "blocks", policy), exist_ok=True)
    for i in range(4):
        f = open(os.path.join(path, "blocks", policy, "res_%d.csv" % (i)), "w")
        for r, (n, m) in zip(repeats, size):
            for block in blocks:
                for t in tpb:
                    print("%d (%d, %d) %d %d" % (i, n, m, block, t))
                    proc = subprocess.Popen(["./transpose",
                                            "-b", str(block),
                                            "-n", str(n),
                                            "-m", str(m),
                                            "-t", str(t),
                                            "-r", str(r),
                                            "-v", str(i)],
                                            env=os.environ, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
                    out = proc.communicate()
                    write_result(f, out)
        f.close()


if __name__ == "__main__":
    load_policies = [".ca", ".cg", ".cs", ".lu", ".cv"]
    store_policies = [".wb", ".cg", ".cs", ".wt"]
    outfolder = "output"

    if (len(sys.argv) > 1):
        outfolder = sys.argv[1]

    for loadpolicy in load_policies:
        for storepolicy in store_policies:
            subprocess.run(["make", "clean"])
            envcpy = os.environ.copy()
            if (not "NVCCFLAGS" in envcpy):
                envcpy["NVCCFLAGS"] = ""
            envcpy["NVCCFLAGS"] += " -DLOAD_POLICY=%s -DSTORE_POLICY=%s " % (loadpolicy, storepolicy)
            subprocess.run(["make"], env=envcpy)

            run(outfolder, "%s_%s" % (loadpolicy[1:], storepolicy[1:]))
