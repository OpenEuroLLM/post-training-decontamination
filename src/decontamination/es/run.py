import logging
import os
from pathlib import Path

from decontamination.util import run_command_streaming

from .util import MOUNTS, resolve_image_arch, resolve_jvm_cpus, select_image

log = logging.getLogger(__name__)

def run(
    image: str | None = None,
    mounts: dict[Path, Path] = MOUNTS,
    jvm_cpus: int | None = None,
) -> None:
    """Runs an Apptainer container from the specified image with the given mounts and command.

    Args:
        image: The path to the Apptainer image to run.
        mounts: A dictionary mapping local paths to container paths for mounting.
        jvm_cpus: JVM processor count. Defaults to the process CPU affinity.
    """
    if image is None:
        log.info("No image specified; prompting user to select an image...")
        image = select_image(Path("images"))

    mount_args = []
    for local_path, container_path in mounts.items():
        mount_args.extend(["--bind", f"{local_path}:{container_path}"])

    java_opts = os.environ.get("ES_JAVA_OPTS", "")
    java_opts = f"{java_opts} -XX:ActiveProcessorCount={resolve_jvm_cpus(jvm_cpus)}".strip()
    command = ["apptainer", "run", "--env", f"ES_JAVA_OPTS={java_opts}"] + mount_args + [
        resolve_image_arch(image)
    ]
    log.info(f"Running Apptainer container with command: {' '.join(command)}")
    returncode, output = run_command_streaming(command)
    if returncode != 0:
        log.error(f"Failed to run Apptainer container: {output}")
        raise RuntimeError(f"Failed to run Apptainer container: {output}")
