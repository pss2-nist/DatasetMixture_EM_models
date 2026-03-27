"""Configuration system for DatasetMixture EM models using OmegaConf."""

from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf, DictConfig


def load_config(config_path: Optional[Path] = None) -> DictConfig:
    """
    Load configuration from YAML file with OmegaConf. Nested values can be accessed with dot notation. and custom config can override defaults.
    """
    # Load default config from package
    default_config_path = Path(__file__).parent / "defaults" / "config.yaml"
    cfg = OmegaConf.load(default_config_path)

    # Merge with custom config if provided
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        custom_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(cfg, custom_cfg)

    # Make config immutable to prevent accidental changes (optional, can comment out)
    # OmegaConf.set_struct(cfg, True)

    return cfg


def save_config(cfg: DictConfig, output_path: Path) -> None:
    """
    Save configuration to YAML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_path)
