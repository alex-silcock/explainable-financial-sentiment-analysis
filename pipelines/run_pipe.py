import os
import sys
import subprocess
import argparse
import yaml

def main():
    token_path = os.path.join(os.getcwd(), ".token")
    if not os.path.isfile(token_path):
        print(f"Error: .token file not found in {os.getcwd()}")
        sys.exit(1)
    with open(token_path, "r") as f:
        token = f.read().strip()

    yaml_file = os.path.join(os.getcwd(), YAML_FILE)
    if not os.path.isfile(yaml_file):
        print(f"Error: YAML file not found at {yaml_file}")
        sys.exit(1)
    try:
        with open(yaml_file, "r") as yf:
            pipe_data = yaml.safe_load(yf)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        sys.exit(1)

    subprocess.run(["hare", "reserve", "/mnt/faster0/as4387"])

    pwd = os.getcwd()
    uid = str(os.getuid())
    gid = str(os.getgid())

    steps = pipe_data.get("steps", [])
    for step in steps:
        step_name = step.get("name", "Unnamed Step")
        print(f"Running step: {step_name}")

        docker_image = step.get("docker_image")
        container_command = step.get("command")
        args_list = step.get("args", [])

        env_flags = []
        env_vars = step.get("environment", {})
        for key, val in env_vars.items():
            if val in ["$TOKEN", '"$TOKEN"']:
                val = token
            env_flags.extend(["-e", f"{key}={val}"])

        # hare run --rm {hare_run_flags} --gpus {gpu_device} -v {pwd}:/app -u {uid}:{gid}
        #     (env flags) (docker_image) (container_command) (args...)
        command = (
            ["hare", "run", "--rm", hare_run_flags, "--gpus", GPU_DEVICE, "-v", f"{pwd}:/app", "-u", f"{uid}:{gid}"] +
            env_flags +
            [docker_image, container_command] +
            args_list
        )

        subprocess.run(command, check=True)

    subprocess.run(["hare", "release", "/mnt/faster0/as4387"], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run pipeline steps from YAML using hare."
    )
    parser.add_argument(
        "--gpus", default="device=6", help="GPU device to use; default: device=6"
    )
    parser.add_argument(
        "-d", action="store_true",
        help="Run in detached mode (uses '-dit' instead of '-it')"
    )
    parser.add_argument(
        "-y", "--yaml", default="pipe.yaml",
        help="Path to YAML file containing pipeline steps; default: pipe.yaml"
    )
    args = parser.parse_args()

    GPU_DEVICE = args.gpus
    DETACHED_MODE = args.d
    YAML_FILE = args.yaml

    hare_run_flags = "-dit" if DETACHED_MODE else "-it"
    main()
    print("Exiting Script.")