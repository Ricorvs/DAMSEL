#!/usr/bin/python

import subprocess
import os
import sys

iterations = 5000

# Used values for different GPUs
# 1080Ti: 6144
# 2080Ti: 6144
# A100: 12288
# Values are equal for Bandwidth measurements
cache_size = 6144 # words
min_size = 128 # words
max_size = 4 * cache_size

def write_result(file, output):
    if (output[1] is not None and len(output[1]) > 0):
        print(output[1].decode("utf-8"))
        exit()
    file.write("%s" % (output[0].decode("utf-8")))
    file.flush()

def run(path):
    os.makedirs(path, exist_ok=True)

    # Regular Pchase
    print("Regular pchase")
    os.makedirs(os.path.join(path, "default"), exist_ok=True)
    for preheat in range(2):
        for s in range(0, 6):
            print("%d/%d %d/%d" % (preheat, 1, s, 5))
            stride = 2**s
            f = open(os.path.join(path, "default", "preheat%d_stride%d.csv" % (preheat, stride)), "w")
            proc = subprocess.Popen([
                "./mem",
                "-s", str(stride),
                "-i", str(iterations),
                "-n", str(min_size),
                "-N", str(max_size),
                "-p", str(preheat),
                "-r", "3"
            ], env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out = proc.communicate()
            write_result(f, out)
            f.flush()

    # Texture memory
    print("Texture")
    os.makedirs(os.path.join(path, "texture"), exist_ok=True)
    for preheat in range(2):
        for s in range(0, 6):
            print("%d/%d %d/%d" % (preheat, 1, s, 5))
            stride = 2**s
            f = open(os.path.join(path, "texture", "preheat%d_stride%d.csv" % (preheat, stride)), "w")
            proc = subprocess.Popen([
                "./mem",
                "-s", str(stride),
                "-i", str(iterations),
                "-n", str(min_size),
                "-N", str(max_size),
                "-p", str(preheat),
                "-2",
                "-r", "3"
            ], env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out = proc.communicate()
            write_result(f, out)

    print("single_sm")
    os.makedirs(os.path.join(path, "single_sm"), exist_ok=True)
    for b in range(1, 33):
        print("%d/%d" % (b, 32))
        proc = subprocess.Popen([
            "./mem",
            "-s", "8",
            "-b", str(b),
            "-i", "192",
            "-n", str(cache_size),
            "-l", "-1",
            "-h", "0",
            "-f", os.path.join(path, "single_sm", "%d_warps.csv" % (b)),
            "-3"
        ], env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = proc.communicate()

    print("multi_sm")
    os.makedirs(os.path.join(path, "multi_sm"), exist_ok=True)
    for b in range(1, 33):
        print("%d/%d" % (b, 32))
        proc = subprocess.Popen([
            "./mem",
            "-s", "8",
            "-b", str(b),
            "-i", "192",
            "-n", str(cache_size),
            "-l", "-1",
            "-h", "0",
            "-f", os.path.join(path, "multi_sm", "%d_warps.csv" % (b)),
            "-4"
        ], env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = proc.communicate()

def run_bw(path):
    print("bw")
    os.makedirs(path, exist_ok=True)
    max_blockwarps = 7
    f = open(os.path.join(path, "bw.csv"), "w")
    for blocks in range(0, 6):
        ablocks = 2**blocks
        for warps in range(0, 6):
            awarps = 2**warps
            if (max_blockwarps < blocks + warps):
                continue
            for active in range(1, awarps + 1):
                proc = subprocess.Popen(["./bw", "-b", str(ablocks),
                                         "-w", str(awarps),
                                         "-a", str(active),
                                         "-f", os.path.join(path, "bw_%db_%dw_%da.csv" % (ablocks, awarps, active))],
                                        env=os.environ, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
                out = proc.communicate()
                write_result(f, out)
    f.close()


if __name__ == "__main__":
    cache_policies = [".ca", ".cg", ".cs", ".lu", ".cv"]
    bw_bitsize = [32, 64]
    bw_instrn = ["ADD", "NEG", "MOV", "NONE"]
    bw_instr = range(4)
    outfolder = "output"

    if (len(sys.argv) > 1):
        outfolder = sys.argv[1]

    for policy in cache_policies:
        subprocess.run(["make", "clean"])
        envcpy = os.environ.copy()
        if (not "NVCCFLAGS" in envcpy):
            envcpy["NVCCFLAGS"] = ""
        envcpy["NVCCFLAGS"] += " -DCACHE_POLICY=" + policy
        subprocess.run(["make"], env=envcpy)

        run(os.path.join(outfolder, policy[1:]))

        flags = envcpy["NVCCFLAGS"]
        for bitsize in bw_bitsize:
            for instr in bw_instr:
                envcpy["NVCCFLAGS"] = flags + " -DINSTR=%d -DBIT_SIZE=%d" % (instr, bitsize)
                subprocess.run(["make", "clean", "all"], env=envcpy)
                run_bw(os.path.join(outfolder, policy[1:], "bw", str(bitsize), bw_instrn[instr]))
