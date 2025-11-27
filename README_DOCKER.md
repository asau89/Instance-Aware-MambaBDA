# Docker Instructions for MambaCD

This project can be run inside a Docker container to avoid setting up the environment manually.

## Prerequisites

1.  **Docker**: Install Docker Desktop or Docker Engine.
2.  **NVIDIA Container Toolkit**: Required for GPU support inside Docker.
    *   Follow instructions here: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

## Setup

1.  **Build the Docker image:**

    ```bash
    docker-compose build
    ```

    Or using plain docker:
    ```bash
    docker build -t mambacd .
    ```

2.  **Prepare Data:**
    *   Place your datasets in a folder (e.g., `data/`) on your host machine.
    *   Update `docker-compose.yml` to mount your dataset folder to `/data` inside the container.
        ```yaml
        volumes:
          - ./your_local_data_folder:/data
        ```
    *   Ensure `pretrained_weight` folder contains the necessary model weights.

## Running Inference

1.  **Start the container:**

    ```bash
    docker-compose run --rm mambacd
    ```

    This will open a shell inside the container.

2.  **Run the inference script:**

    Inside the container, you can run the inference scripts. For example:

    ```bash
    python changedetection/script/infer_MambaBDA.py \
        --dataset 'xBD' \
        --test_dataset_path '/data/xBD/test' \
        --test_data_list_path '/data/xBD/test_list.txt' \
        --pretrained_weight_path 'pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth' \
        --result_saved_path 'results'
    ```

    *Note: Adjust paths and arguments according to your setup.*

## Troubleshooting

*   **CUDA Errors**: Ensure your host machine has NVIDIA drivers installed and the NVIDIA Container Toolkit is configured.
*   **Shared Memory**: If you see errors related to DataLoader worker processes (e.g., "Bus error"), ensure `ipc: host` is set in `docker-compose.yml` or use `--shm-size=8g` with `docker run`.
