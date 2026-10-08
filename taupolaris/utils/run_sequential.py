"""Run several commands one after the other inside a single batch job.

batch_submission.py submits one python script per job; to bundle several short jobs (e.g. a set of
evaluations) into one slot, list the commands in a text file, one per line ('#' starts a comment),
and submit this script with it:

    python taupolaris/utils/batch_submission.py -c "python taupolaris/utils/run_sequential.py jobs/<list>.txt" \\
        --fast_gpu --runtime 86400 --job_name <name>

Each command runs from the current directory with its output streamed to the job's .out file. A
failing command is reported and the next one still runs; the exit status is non-zero if any failed.
"""
import shlex
import subprocess
import sys
import time


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cmds = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.strip().startswith('#')]
    results = []
    for i, cmd in enumerate(cmds, 1):
        print(f'\n===== [{i}/{len(cmds)}] {time.strftime("%Y-%m-%d %H:%M:%S")}  {cmd}', flush=True)
        t0 = time.time()
        rc = subprocess.call(shlex.split(cmd))
        results.append((cmd, rc, time.time() - t0))
        print(f'===== [{i}/{len(cmds)}] exit {rc} after {(time.time() - t0) / 3600:.2f} h', flush=True)
    print('\n===== summary')
    for cmd, rc, dt in results:
        print(f'  {"OK  " if rc == 0 else "FAIL"} {dt / 3600:5.2f} h  {cmd}')
    sys.exit(1 if any(rc for _, rc, _ in results) else 0)


if __name__ == '__main__':
    main()
