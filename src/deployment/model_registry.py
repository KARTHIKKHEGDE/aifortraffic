"""
Model Registry and Versioning System

Provides:
- Model versioning with metadata
- Checkpoint management
- Model comparison
- Rollback support
- Production model promotion
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum


class ModelStage(Enum):
    """Model lifecycle stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelMetadata:
    """Metadata for a model version"""
    version: str
    name: str
    description: str = ""
    stage: ModelStage = ModelStage.DEVELOPMENT
    
    # Training info
    algorithm: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_episodes: int = 0
    training_steps: int = 0
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Timestamps
    created_at: str = ""
    updated_at: str = ""
    
    # File info
    file_path: str = ""
    file_hash: str = ""
    file_size_bytes: int = 0
    
    # Tags and notes
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['stage'] = self.stage.value
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        data = data.copy()
        if 'stage' in data:
            data['stage'] = ModelStage(data['stage'])
        return cls(**data)


class ModelRegistry:
    """
    Registry for managing model versions
    
    Directory structure:
    registry_path/
      models/
        model_v1/
          model.pt
          metadata.json
        model_v2/
          ...
      registry.json
    """
    
    def __init__(self, registry_path: str = "model_registry"):
        self.registry_path = Path(registry_path)
        self.models_path = self.registry_path / "models"
        self.registry_file = self.registry_path / "registry.json"
        
        # Create directories
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize registry
        self._registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        
        return {
            'name': 'TrafficControlRegistry',
            'created_at': datetime.now().isoformat(),
            'models': {},
            'production_version': None,
            'staging_version': None
        }
    
    def _save_registry(self):
        """Save registry to file"""
        with open(self.registry_file, 'w') as f:
            json.dump(self._registry, f, indent=2, default=str)
    
    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _generate_version(self) -> str:
        """Generate new version number"""
        versions = list(self._registry['models'].keys())
        
        if not versions:
            return "v1.0.0"
        
        # Parse latest version and increment
        latest = sorted(versions)[-1]
        
        try:
            parts = latest.replace('v', '').split('.')
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            return f"v{major}.{minor}.{patch + 1}"
        except:
            return f"v{len(versions) + 1}.0.0"
    
    def register_model(
        self,
        model_path: str,
        name: str,
        description: str = "",
        algorithm: str = "",
        hyperparameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        training_info: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        version: Optional[str] = None
    ) -> str:
        """
        Register a new model version
        
        Args:
            model_path: Path to model file
            name: Model name
            description: Description
            algorithm: Training algorithm
            hyperparameters: Training hyperparameters
            metrics: Performance metrics
            training_info: Training episode/step info
            tags: Tags for categorization
            version: Specific version (auto-generated if None)
            
        Returns:
            Version string of registered model
        """
        source_path = Path(model_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Generate version
        version = version or self._generate_version()
        
        # Create version directory
        version_dir = self.models_path / version
        version_dir.mkdir(exist_ok=True)
        
        # Copy model file
        dest_path = version_dir / f"model{source_path.suffix}"
        shutil.copy2(source_path, dest_path)
        
        # Create metadata
        training_info = training_info or {}
        
        metadata = ModelMetadata(
            version=version,
            name=name,
            description=description,
            algorithm=algorithm,
            hyperparameters=hyperparameters or {},
            training_episodes=training_info.get('episodes', 0),
            training_steps=training_info.get('steps', 0),
            metrics=metrics or {},
            file_path=str(dest_path),
            file_hash=self._compute_hash(dest_path),
            file_size_bytes=dest_path.stat().st_size,
            tags=tags or []
        )
        
        # Save metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Update registry
        self._registry['models'][version] = {
            'name': name,
            'stage': ModelStage.DEVELOPMENT.value,
            'created_at': metadata.created_at,
            'metrics': metrics or {}
        }
        
        self._save_registry()
        
        print(f"Registered model: {name} as {version}")
        return version
    
    def get_model_metadata(self, version: str) -> Optional[ModelMetadata]:
        """Get metadata for a model version"""
        if version not in self._registry['models']:
            return None
        
        metadata_path = self.models_path / version / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return ModelMetadata.from_dict(json.load(f))
    
    def get_model_path(self, version: str) -> Optional[Path]:
        """Get path to model file"""
        metadata = self.get_model_metadata(version)
        
        if metadata is None:
            return None
        
        return Path(metadata.file_path)
    
    def list_models(
        self,
        stage: Optional[ModelStage] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List registered models
        
        Args:
            stage: Filter by stage
            tags: Filter by tags
            
        Returns:
            List of model info dicts
        """
        models = []
        
        for version, info in self._registry['models'].items():
            # Stage filter
            if stage and info['stage'] != stage.value:
                continue
            
            # Get full metadata
            metadata = self.get_model_metadata(version)
            
            # Tag filter
            if tags and metadata:
                if not any(tag in metadata.tags for tag in tags):
                    continue
            
            models.append({
                'version': version,
                'name': info['name'],
                'stage': info['stage'],
                'created_at': info['created_at'],
                'metrics': info.get('metrics', {})
            })
        
        # Sort by version
        models.sort(key=lambda x: x['version'], reverse=True)
        
        return models
    
    def promote_model(
        self,
        version: str,
        target_stage: ModelStage
    ) -> bool:
        """
        Promote model to a new stage
        
        Args:
            version: Model version
            target_stage: Target stage
            
        Returns:
            True if successful
        """
        if version not in self._registry['models']:
            print(f"Version not found: {version}")
            return False
        
        old_stage = self._registry['models'][version]['stage']
        self._registry['models'][version]['stage'] = target_stage.value
        
        # Update special pointers
        if target_stage == ModelStage.PRODUCTION:
            # Demote current production
            if self._registry['production_version']:
                old_prod = self._registry['production_version']
                self._registry['models'][old_prod]['stage'] = ModelStage.ARCHIVED.value
            
            self._registry['production_version'] = version
        
        elif target_stage == ModelStage.STAGING:
            self._registry['staging_version'] = version
        
        # Update metadata
        metadata = self.get_model_metadata(version)
        if metadata:
            metadata.stage = target_stage
            metadata.updated_at = datetime.now().isoformat()
            
            metadata_path = self.models_path / version / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
        
        self._save_registry()
        
        print(f"Promoted {version}: {old_stage} -> {target_stage.value}")
        return True
    
    def get_production_model(self) -> Optional[Tuple[str, Path]]:
        """
        Get the current production model
        
        Returns:
            Tuple of (version, path) or None
        """
        version = self._registry.get('production_version')
        
        if not version:
            return None
        
        path = self.get_model_path(version)
        
        if path is None:
            return None
        
        return version, path
    
    def compare_models(
        self,
        version_a: str,
        version_b: str
    ) -> Dict[str, Any]:
        """
        Compare two model versions
        
        Returns:
            Comparison dict
        """
        meta_a = self.get_model_metadata(version_a)
        meta_b = self.get_model_metadata(version_b)
        
        if meta_a is None or meta_b is None:
            raise ValueError("One or both versions not found")
        
        # Compare metrics
        all_metrics = set(meta_a.metrics.keys()) | set(meta_b.metrics.keys())
        
        metric_comparison = {}
        for metric in all_metrics:
            val_a = meta_a.metrics.get(metric, 0)
            val_b = meta_b.metrics.get(metric, 0)
            
            diff = val_b - val_a
            pct_change = (diff / val_a * 100) if val_a != 0 else 0
            
            metric_comparison[metric] = {
                f'{version_a}': val_a,
                f'{version_b}': val_b,
                'difference': diff,
                'percent_change': pct_change,
                'improved': diff > 0 if 'reward' in metric.lower() else diff < 0
            }
        
        return {
            'version_a': version_a,
            'version_b': version_b,
            'metrics': metric_comparison,
            'algorithm_changed': meta_a.algorithm != meta_b.algorithm,
            'hyperparameter_diffs': self._diff_dicts(
                meta_a.hyperparameters, 
                meta_b.hyperparameters
            )
        }
    
    def _diff_dicts(
        self, 
        dict_a: Dict[str, Any], 
        dict_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Find differences between two dicts"""
        all_keys = set(dict_a.keys()) | set(dict_b.keys())
        
        diffs = {}
        for key in all_keys:
            val_a = dict_a.get(key)
            val_b = dict_b.get(key)
            
            if val_a != val_b:
                diffs[key] = {'old': val_a, 'new': val_b}
        
        return diffs
    
    def rollback_to_version(self, version: str) -> bool:
        """
        Rollback production to a specific version
        
        Args:
            version: Version to rollback to
            
        Returns:
            True if successful
        """
        if version not in self._registry['models']:
            print(f"Version not found: {version}")
            return False
        
        return self.promote_model(version, ModelStage.PRODUCTION)
    
    def delete_model(self, version: str) -> bool:
        """
        Delete a model version
        
        Args:
            version: Version to delete
            
        Returns:
            True if successful
        """
        if version not in self._registry['models']:
            return False
        
        # Don't delete production model
        if version == self._registry.get('production_version'):
            print("Cannot delete production model")
            return False
        
        # Remove files
        version_dir = self.models_path / version
        if version_dir.exists():
            shutil.rmtree(version_dir)
        
        # Remove from registry
        del self._registry['models'][version]
        self._save_registry()
        
        print(f"Deleted model version: {version}")
        return True
    
    def export_model(
        self,
        version: str,
        export_path: str,
        include_metadata: bool = True
    ) -> bool:
        """
        Export model to a directory
        
        Args:
            version: Model version
            export_path: Export directory
            include_metadata: Include metadata file
            
        Returns:
            True if successful
        """
        metadata = self.get_model_metadata(version)
        
        if metadata is None:
            return False
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        source = Path(metadata.file_path)
        shutil.copy2(source, export_dir / source.name)
        
        # Copy metadata
        if include_metadata:
            with open(export_dir / "metadata.json", 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
        
        print(f"Exported {version} to {export_path}")
        return True


class CheckpointManager:
    """
    Manager for training checkpoints
    
    Handles:
    - Periodic checkpointing
    - Best model tracking
    - Checkpoint pruning
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 10,
        keep_best: int = 3
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        
        # Track best checkpoints
        self.best_checkpoints = []  # List of (metric, path)
        
        # State file
        self.state_file = self.checkpoint_dir / "checkpoint_state.json"
        self._state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load checkpoint state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        
        return {
            'checkpoints': [],
            'best_checkpoints': [],
            'latest': None
        }
    
    def _save_state(self):
        """Save checkpoint state"""
        with open(self.state_file, 'w') as f:
            json.dump(self._state, f, indent=2)
    
    def save_checkpoint(
        self,
        model: Any,
        step: int,
        metrics: Dict[str, float],
        additional_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a checkpoint
        
        Args:
            model: Model to save
            step: Training step
            metrics: Current metrics
            additional_state: Additional state to save
            
        Returns:
            Path to checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_step{step}_{timestamp}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model
        model_path = checkpoint_path / "model.pt"
        
        try:
            import torch
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model, model_path)
        except ImportError:
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'step': step,
            'timestamp': timestamp,
            'metrics': metrics,
            'additional_state': additional_state or {}
        }
        
        with open(checkpoint_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update state
        self._state['checkpoints'].append(str(checkpoint_path))
        self._state['latest'] = str(checkpoint_path)
        
        # Update best tracking
        primary_metric = metrics.get('reward', metrics.get('score', 0))
        self._update_best(primary_metric, str(checkpoint_path))
        
        # Prune old checkpoints
        self._prune_checkpoints()
        
        self._save_state()
        
        print(f"Saved checkpoint: {checkpoint_name}")
        return str(checkpoint_path)
    
    def _update_best(self, metric: float, path: str):
        """Update best checkpoints list"""
        self.best_checkpoints.append((metric, path))
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)
        self.best_checkpoints = self.best_checkpoints[:self.keep_best]
        
        self._state['best_checkpoints'] = [p for _, p in self.best_checkpoints]
    
    def _prune_checkpoints(self):
        """Remove old checkpoints, keeping best ones"""
        checkpoints = self._state['checkpoints']
        best_paths = set(self._state['best_checkpoints'])
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Find checkpoints to remove
        to_remove = []
        
        for cp in checkpoints[:-self.max_checkpoints]:
            if cp not in best_paths:
                to_remove.append(cp)
        
        # Remove
        for cp in to_remove:
            cp_path = Path(cp)
            if cp_path.exists():
                shutil.rmtree(cp_path)
            self._state['checkpoints'].remove(cp)
        
        print(f"Pruned {len(to_remove)} old checkpoints")
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint (None for latest)
            
        Returns:
            Tuple of (model_state, metadata)
        """
        if checkpoint_path is None:
            checkpoint_path = self._state.get('latest')
        
        if checkpoint_path is None:
            raise ValueError("No checkpoints available")
        
        cp_path = Path(checkpoint_path)
        
        # Load model
        model_path = cp_path / "model.pt"
        
        try:
            import torch
            model_state = torch.load(model_path)
        except ImportError:
            import pickle
            with open(model_path, 'rb') as f:
                model_state = pickle.load(f)
        
        # Load metadata
        with open(cp_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        return model_state, metadata
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint"""
        if self._state['best_checkpoints']:
            return self._state['best_checkpoints'][0]
        return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints with metadata"""
        result = []
        
        for cp in self._state['checkpoints']:
            cp_path = Path(cp)
            
            if not cp_path.exists():
                continue
            
            with open(cp_path / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            is_best = cp in self._state['best_checkpoints']
            is_latest = cp == self._state['latest']
            
            result.append({
                'path': cp,
                'step': metadata['step'],
                'timestamp': metadata['timestamp'],
                'metrics': metadata['metrics'],
                'is_best': is_best,
                'is_latest': is_latest
            })
        
        return result


if __name__ == '__main__':
    # Demo
    registry = ModelRegistry("demo_registry")
    
    # Create a dummy model file
    dummy_model_path = Path("demo_model.pt")
    dummy_model_path.write_text("dummy model content")
    
    try:
        # Register model
        version = registry.register_model(
            model_path=str(dummy_model_path),
            name="DQN-Traffic-v1",
            description="DQN agent for traffic control",
            algorithm="DQN",
            hyperparameters={'lr': 0.001, 'gamma': 0.99},
            metrics={'mean_reward': 150.5, 'avg_wait_time': 25.3},
            training_info={'episodes': 1000, 'steps': 100000},
            tags=['dqn', 'bangalore']
        )
        
        print(f"\nRegistered: {version}")
        
        # List models
        print("\nAll models:")
        for model in registry.list_models():
            print(f"  {model['version']}: {model['name']} ({model['stage']})")
        
        # Promote to production
        registry.promote_model(version, ModelStage.PRODUCTION)
        
        # Get production model
        prod = registry.get_production_model()
        print(f"\nProduction model: {prod}")
        
    finally:
        # Cleanup
        dummy_model_path.unlink(missing_ok=True)
        shutil.rmtree("demo_registry", ignore_errors=True)
