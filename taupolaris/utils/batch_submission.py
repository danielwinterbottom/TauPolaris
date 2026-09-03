import os

def create_job(
    python_script: str,
    job_name: str,
    python_arguments: str = "",
    use_gpu: bool = False,
    use_fast_gpu: bool = False,
    runtime_seconds: int = 3600,
    memory_mb: int = 0,
    cpus: int = 0,
):
    jobs_dir = "jobs"
    os.makedirs(jobs_dir, exist_ok=True)

    sh_filename = f"{job_name}.sh"
    sh_path = os.path.join(jobs_dir, sh_filename)

    with open(sh_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"python -u {python_script} \"$@\"\n")

    os.chmod(sh_path, 0o755)

    sub_filename = f"{jobs_dir}/{job_name}.sub"

    lines = [
        f"executable = {sh_path}",
        "getenv = True",
        f'arguments = "{python_arguments}"',
        f"output = {jobs_dir}/{job_name}_$(CLUSTER).out",
        f"error = {jobs_dir}/{job_name}_$(CLUSTER).err",
        f"log = {jobs_dir}/{job_name}_$(CLUSTER).log",
    ]

    if use_gpu:
        lines.append("request_gpus = 1")
        if use_fast_gpu:
            raise ValueError("Use only --fast_gpu option instead")

    if use_fast_gpu:
        lines.append("request_gpus = 1")
        lines.append(
            'requirements = (GPUs_DeviceName == "NVIDIA RTX A6000") || '
            '(GPUs_DeviceName == "Tesla V100-PCIE-32GB")'
        )

    # Without an explicit request HTCondor hands out the slot default, which on
    # these nodes came with 40 CPUs and ~87 GB -- and a training job that exceeds
    # it is evicted and held ("gone over cgroup memory limit"), not slowed down.
    if memory_mb:
        lines.append(f"request_memory = {memory_mb}")
    if cpus:
        lines.append(f"request_cpus = {cpus}")
    lines.append(f"+MaxRuntime = {runtime_seconds}")
    lines.append("queue")

    with open(sub_filename, "w") as f:
        f.write("\n".join(lines))

    return sub_filename, sh_path

def submit_job(submission_file: str):
    os.system(f"condor_submit {submission_file}")

if __name__ == "__main__":

    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--command', '-c', help='command to run', type=str, required=True)
    argparser.add_argument('--job_name', '-n', help='name of the job', type=str, required=True)
    argparser.add_argument('--runtime', help='maximum runtime of the job in seconds', type=int, required=True) 
    argparser.add_argument('--gpu', help='whether to request a gpu', action='store_true')
    argparser.add_argument('--fast_gpu', help='whether to request a fast gpu (A6000 or V100)', action='store_true')
    argparser.add_argument('--memory', help='request_memory in MB. Unset uses the slot default, which '
                                            'is what a job exceeds when it is evicted with "gone over '
                                            'cgroup memory limit".', type=int, default=0)
    argparser.add_argument('--cpus', help='request_cpus. Unset uses the slot default.', type=int, default=0)
    args = argparser.parse_args()

    # parse command to get python script and its arguments
    # first check if it starts with "python" and if so remove it
    command_split = args.command.split()
    if command_split[0] == "python":
        command_split = command_split[1:]
    python_script = command_split[0]
    python_arguments = " ".join(command_split[1:])

    python_script = command_split[0]
    python_args=""
    if len(command_split) > 1:
        python_args = " ".join(command_split[1:])

    print(f'>>{python_script}')
    print(f'>>{python_args}')

    submission_file, sh_file = create_job(
        python_script=python_script,
        python_arguments=python_args,
        job_name=args.job_name,
        use_gpu=args.gpu,
        use_fast_gpu=args.fast_gpu,
        runtime_seconds=args.runtime,
        memory_mb=args.memory,
        cpus=args.cpus,
    )
    print(f"Created submission file: {submission_file} and shell file: {sh_file}")
    submit_job(submission_file)
    print(f"Submitted job {args.job_name} to HTCondor.")