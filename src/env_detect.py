import os


def get_env_value(key):
    value = os.getenv(key)
    return value if value else None


def detect_slurm_context():
    job_id = get_env_value("SLURM_JOB_ID")
    if not job_id:
        return None

    slurm_fields = {
        "job_id": job_id,
        "nodelist": get_env_value("SLURM_NODELIST"),
        "nnodes": get_env_value("SLURM_NNODES"),
        "procid": get_env_value("SLURM_PROCID"),
        "localid": get_env_value("SLURM_LOCALID"),
        "gpus": get_env_value("SLURM_GPUS"),
        "partition": get_env_value("SLURM_JOB_PARTITION"),
    }
    return {k: v for k, v in slurm_fields.items() if v is not None}


def detect_instance_type():
    return get_env_value("EC2_INSTANCE_TYPE")
