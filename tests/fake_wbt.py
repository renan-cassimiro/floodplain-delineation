#!/usr/bin/env python3
"""Fake WhiteboxTools executable used to test failure propagation (behaviour set by FAKE_WBT_MODE).

Modes: ok (default, exits 0 without outputs), exit1, error_text, no_output, garbage,
slow (prints lines, sleeps FAKE_WBT_SLEEP seconds), copy (copies --dem/--input to --output).
If FAKE_WBT_ARGS_FILE is set, the arguments of every tool run are appended to that file.
"""
import os, shutil, sys, time
args = sys.argv[1:]
mode = os.environ.get("FAKE_WBT_MODE", "ok")
if any(a == "--version" for a in args):
    print("WhiteboxTools vFAKE"); sys.exit(0)
for a in args:
    if a.startswith("--toolhelp="):
        print("Flag Description\n-i, --dem Input\n-o, --output Output\n--input In\n--pntr P\n--out_type T\n--streams S\n--flow_accum F\n--threshold T\n--zero_background Z\n--fix_flats X")
        sys.exit(0)
if os.environ.get("FAKE_WBT_ARGS_FILE"):
    with open(os.environ["FAKE_WBT_ARGS_FILE"], "a") as fh:
        fh.write(" ".join(args) + "\n")
out = [a.split("=", 1)[1] for a in args if a.startswith("--output=")]
if mode == "exit1":
    sys.stderr.write("thread 'main' panicked at 'boom'\n"); sys.exit(101)
if mode == "error_text":
    print("Error: something went wrong"); sys.exit(0)
if mode == "no_output":
    sys.exit(0)
if mode == "garbage":
    for o in out: open(o, "w").write("not a tiff")
    sys.exit(0)
if mode == "slow":
    print("starting", flush=True)
    time.sleep(float(os.environ.get("FAKE_WBT_SLEEP", "1.5")))
    print("done", flush=True)
    for o in out: open(o, "w").write("x")
    sys.exit(0)
if mode == "copy":
    src = [a.split("=", 1)[1] for a in args if a.startswith(("--dem=", "--input="))][0]
    for o in out: shutil.copyfile(src, o)
    sys.exit(0)
sys.exit(0)
