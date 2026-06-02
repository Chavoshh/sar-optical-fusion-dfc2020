"""Resolve and print a Hydra config without doing anything else.

Useful for verifying that the composition works before kicking off training.
Run as e.g.:
    uv run python scripts/inspect_config.py experiment=s2_only
"""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()