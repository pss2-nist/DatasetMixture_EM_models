"""Path resolution and management utilities."""

from pathlib import Path
from typing import Optional, List


class PathManager:
    """Centralized path management to eliminate hardcoded paths."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize with project root directory."""
        self.project_root = Path(project_root or Path.cwd()).resolve()

    def get_data_dir(self, dataset_name: Optional[str] = None) -> Path:
        """Get data directory, optionally for a specific dataset."""
        data_dir = self.project_root / "data"
        if dataset_name:
            data_dir = data_dir / dataset_name
            self._ensure_exists(data_dir, create=True)
        return data_dir

    def get_output_dir(self, subdirs: Optional[List[str]] = None) -> Path:
        """Get output directory, optionally with subdirectories."""
        output_dir = self.project_root / "output"
        if subdirs:
            for subdir in subdirs:
                output_dir = output_dir / subdir
        self._ensure_exists(output_dir, create=True)
        return output_dir

    def get_checkpoint_dir(self, model_name: Optional[str] = None) -> Path:
        """Get checkpoint directory for models."""
        checkpoint_dir = self.project_root / "checkpoints"
        if model_name:
            checkpoint_dir = checkpoint_dir / model_name
        self._ensure_exists(checkpoint_dir, create=True)
        return checkpoint_dir

    def get_log_dir(self, experiment_name: Optional[str] = None) -> Path:
        """Get logging directory, optionally with experiment subdirectory."""
        log_dir = self.project_root / "logs"
        if experiment_name:
            log_dir = log_dir / experiment_name
        self._ensure_exists(log_dir, create=True)
        return log_dir

    def get_tile_dir(self, dataset_name: Optional[str] = None) -> Path:
        """Get tiles directory for preprocessing."""
        tile_dir = self.project_root / "tiles"
        if dataset_name:
            tile_dir = tile_dir / dataset_name
        self._ensure_exists(tile_dir, create=True)
        return tile_dir

    def get_result_dir(self, experiment_name: Optional[str] = None) -> Path:
        """Get results directory (metrics, plots, etc.)."""
        result_dir = self.project_root / "results"
        if experiment_name:
            result_dir = result_dir / experiment_name
        self._ensure_exists(result_dir, create=True)
        return result_dir

    def get_metric_file(self, experiment_name: str, metric_type: str = "metrics") -> Path:
        """Get path to a metrics CSV file."""
        result_dir = self.get_result_dir(experiment_name)
        return result_dir / f"{metric_type}.csv"

    def get_checkpoint_file(self, model_name: str, epoch: int, best: bool = False) -> Path:
        """Get path to a model checkpoint file."""
        checkpoint_dir = self.get_checkpoint_dir(model_name)
        return checkpoint_dir / ("best_model.pt" if best else f"checkpoint_epoch_{epoch:03d}.pt")

    def validate_dataset_path(self, dataset_path: Path) -> bool:
        """Validate that a dataset path exists and is a directory."""
        dataset_path = Path(dataset_path)
        return dataset_path.exists() and dataset_path.is_dir()

    def resolve_dataset_paths(self, dataset_paths: List[str]) -> List[Path]:
        """Resolve dataset paths, supporting relative and absolute paths."""
        resolved_paths = []
        for path_str in dataset_paths:
            path = Path(path_str)
            if not path.is_absolute():
                path = self.project_root / path
            path = path.resolve()
            if not self.validate_dataset_path(path):
                raise ValueError(f"Invalid dataset path: {path}")
            resolved_paths.append(path)
        return resolved_paths

    @staticmethod
    def _ensure_exists(path: Path, create: bool = False) -> None:
        """Ensure path exists, optionally creating it."""
        if not path.exists():
            if create:
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Path does not exist: {path}")

    def __repr__(self) -> str:
        return f"PathManager(project_root={self.project_root})"


# Singleton instance
_path_manager: Optional[PathManager] = None


def get_path_manager(project_root: Optional[Path] = None) -> PathManager:
    """Get or create global PathManager instance."""
    global _path_manager
    if _path_manager is None:
        _path_manager = PathManager(project_root)
    return _path_manager


# Singleton instance for convenience
_path_manager: Optional[PathManager] = None


def get_path_manager(project_root: Optional[Path] = None) -> PathManager:
    """
    Get or create a global PathManager instance.
    """
    global _path_manager
    if _path_manager is None:
        _path_manager = PathManager(project_root)
    return _path_manager
