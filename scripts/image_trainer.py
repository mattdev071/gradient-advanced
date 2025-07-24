#!/usr/bin/env python3
"""
Standalone script for image model training (SDXL or Flux)
"""

import argparse
import asyncio
import os
import subprocess
import sys

import toml
import optuna
import re


# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
from core.config.config_handler import save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType


def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path

def create_config(task_id, model, model_type, expected_repo_name):
    """Create the diffusion config file"""
    # In Docker environment, adjust paths
    if os.path.exists("/workspace/core/config"):
        config_path = "/workspace/core/config"
        sdxl_path = f"{config_path}/base_diffusion_sdxl.toml"
        flux_path = f"{config_path}/base_diffusion_flux.toml"
    else:
        sdxl_path = cst.CONFIG_TEMPLATE_PATH_DIFFUSION_SDXL
        flux_path = cst.CONFIG_TEMPLATE_PATH_DIFFUSION_FLUX

    # Load appropriate config template
    if model_type == ImageModelType.SDXL.value:
        with open(sdxl_path, "r") as file:
            config = toml.load(file)
    elif model_type == ImageModelType.FLUX.value:
        with open(flux_path, "r") as file:
            config = toml.load(file)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Update config
    config["pretrained_model_name_or_path"] = model
    config["train_data_dir"] = f"/dataset/images/{task_id}/img/"
    output_dir = f"{train_cst.IMAGE_CONTAINER_SAVE_PATH}{task_id}/{expected_repo_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir

    # Save config to file
    config_path = os.path.join("/dataset/configs", f"{task_id}.toml")
    save_config_toml(config, config_path)
    print(f"Created config at {config_path}", flush=True)
    return config_path


def run_training(model_type, config_path):
    print(f"Starting training with config: {config_path}", flush=True)

    training_command = [
        "accelerate", "launch",
        "--dynamo_backend", "no",
        "--dynamo_mode", "default",
        "--mixed_precision", "bf16",
        "--num_processes", "1",
        "--num_machines", "1",
        "--num_cpu_threads_per_process", "2",
        f"/app/sd-scripts/{model_type}_train_network.py",
        "--config_file", config_path
    ]

    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        val_loss = None
        for line in process.stdout:
            print(line, end="", flush=True)
            # Parse validation loss from logs (customize regex as needed)
            match = re.search(r"eval_loss: ([0-9.]+)", line)
            if match:
                val_loss = float(match.group(1))

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)
        return val_loss

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")


def objective(trial, args, model_path, output_dir):
    # Define search space
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-4, 0.2)
    train_batch_size = trial.suggest_categorical("train_batch_size", [16, 32, 64])
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [4, 8, 16])
    epoch = trial.suggest_int("epoch", 10, 30)
    # Load and modify config
    if args.model_type == "sdxl":
        config_path = "/workspace/core/config/base_diffusion_sdxl.toml"
    else:
        config_path = "/workspace/core/config/base_diffusion_flux.toml"
    import toml
    with open(config_path, "r") as file:
        config = toml.load(file)
    config["pretrained_model_name_or_path"] = model_path
    config["train_data_dir"] = f"/dataset/images/{args.task_id}/img/"
    config["output_dir"] = output_dir
    config["learning_rate"] = learning_rate
    config["weight_decay"] = weight_decay
    config["train_batch_size"] = train_batch_size
    config["gradient_accumulation_steps"] = gradient_accumulation_steps
    config["epoch"] = epoch
    config["gradient_checkpointing"] = True
    # Save trial config
    trial_config_path = os.path.join(output_dir, f"trial_{trial.number}.toml")
    with open(trial_config_path, "w") as f:
        toml.dump(config, f)
    # Run training and return validation loss
    val_loss = run_training(args.model_type, trial_config_path)
    return val_loss if val_loss is not None else float("inf")


async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux"], help="Model type")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--tune", action="store_true", help="Enable Optuna hyperparameter search")
    args = parser.parse_args()

    # Create required directories
    os.makedirs("/dataset/configs", exist_ok=True)
    os.makedirs("/dataset/outputs", exist_ok=True)
    os.makedirs("/dataset/images", exist_ok=True)

    model_folder = args.model.replace("/", "--")
    model_path = get_model_path(f"{train_cst.CACHE_PATH}/models/{model_folder}")

    # Create config file
    config_path = create_config(
        args.task_id,
        model_path,
        args.model_type,
        args.expected_repo_name,
    )

    # Prepare dataset
    print("Preparing dataset...", flush=True)

    # Set DIFFUSION_DATASET_DIR to environment variable if available
    original_dataset_dir = cst.DIFFUSION_DATASET_DIR
    if os.environ.get("DATASET_DIR"):
        cst.DIFFUSION_DATASET_DIR = os.environ.get("DATASET_DIR")

    prepare_dataset(
        training_images_zip_path=f"{train_cst.CACHE_PATH}/datasets/{args.task_id}.zip",
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
    )

    # Restore original value
    cst.DIFFUSION_DATASET_DIR = original_dataset_dir

    if args.tune:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, args, model_path, output_dir), n_trials=20)
        print("Best trial:", study.best_trial.params)
        # Save best config
        best_config_path = os.path.join(output_dir, "best_config.toml")
        import toml
        with open(best_config_path, "w") as f:
            toml.dump(study.best_trial.params, f)
        print(f"Best config saved to {best_config_path}")
        return

    # Run training
    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())
