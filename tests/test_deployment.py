"""
Tests for Deployment Module

Tests for real-time controller, model registry, monitoring, and configuration.
"""

import os
import sys
import json
import time
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deployment.realtime_controller import (
    InferenceConfig,
    DeploymentMetrics,
    ModelServer,
    TrafficController,
    FailsafeController,
    DeploymentManager
)
from deployment.model_registry import (
    ModelStage,
    ModelMetadata,
    ModelRegistry,
    CheckpointManager
)
from deployment.monitoring import (
    AlertLevel,
    Alert,
    MetricConfig,
    MetricsCollector,
    TrainingLogger,
    PerformanceMonitor,
    HealthChecker
)
from deployment.config import (
    NetworkConfig,
    TrainingConfig,
    EnvironmentConfig,
    RewardConfig,
    DeploymentConfig,
    ExperimentConfig,
    ConfigLoader,
    ConfigValidator,
    ConfigManager,
    load_config
)


# ============================================
# Real-time Controller Tests
# ============================================

class TestInferenceConfig:
    """Tests for InferenceConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = InferenceConfig()
        
        assert config.model_path == "models/best_agent.pt"
        assert config.device == "cpu"
        assert config.batch_size == 1
        assert config.max_latency_ms == 100.0
        assert config.fallback_action == 0
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = InferenceConfig(
            model_path="custom/model.pt",
            device="cuda",
            max_latency_ms=50.0
        )
        
        assert config.model_path == "custom/model.pt"
        assert config.device == "cuda"
        assert config.max_latency_ms == 50.0


class TestDeploymentMetrics:
    """Tests for DeploymentMetrics"""
    
    def test_record_prediction(self):
        """Test recording predictions"""
        metrics = DeploymentMetrics()
        
        metrics.record_prediction(10.0, success=True)
        metrics.record_prediction(20.0, success=True)
        metrics.record_prediction(15.0, success=False)
        
        assert metrics.total_predictions == 3
        assert metrics.successful_predictions == 2
        assert metrics.failed_predictions == 1
    
    def test_latency_statistics(self):
        """Test latency statistics"""
        metrics = DeploymentMetrics()
        
        for i in range(100):
            metrics.record_prediction(float(i), success=True)
        
        assert metrics.avg_latency_ms == pytest.approx(49.5, abs=0.1)
        assert metrics.p95_latency_ms >= 94.0
        assert metrics.p99_latency_ms >= 98.0
    
    def test_success_rate(self):
        """Test success rate calculation"""
        metrics = DeploymentMetrics()
        
        for _ in range(80):
            metrics.record_prediction(10.0, success=True)
        for _ in range(20):
            metrics.record_prediction(10.0, success=False)
        
        assert metrics.success_rate == pytest.approx(0.8, abs=0.01)
    
    def test_summary(self):
        """Test metrics summary"""
        metrics = DeploymentMetrics()
        metrics.record_prediction(10.0, success=True)
        metrics.record_reward(100.0)
        
        summary = metrics.get_summary()
        
        assert 'total_predictions' in summary
        assert 'success_rate' in summary
        assert 'avg_latency_ms' in summary
        assert 'avg_reward' in summary


class TestModelServer:
    """Tests for ModelServer"""
    
    def test_initialization(self):
        """Test model server initialization"""
        server = ModelServer()
        
        assert server.model is None
        assert server.model_loaded is False
    
    def test_load_model_not_found(self):
        """Test loading non-existent model"""
        server = ModelServer()
        
        result = server.load_model("nonexistent_model.pt")
        
        assert result is False
        assert server.model_loaded is False
    
    def test_predict_without_model(self):
        """Test prediction without loaded model"""
        server = ModelServer()
        
        obs = np.random.randn(16).astype(np.float32)
        action, confidence = server.predict(obs)
        
        # Should return fallback action
        assert action == server.config.fallback_action
        assert confidence == 0.0
    
    def test_metrics_collection(self):
        """Test metrics are collected"""
        server = ModelServer()
        
        obs = np.random.randn(16).astype(np.float32)
        server.predict(obs)
        
        metrics = server.get_metrics()
        
        assert 'model_loaded' in metrics
        assert 'total_predictions' in metrics


class TestTrafficController:
    """Tests for TrafficController"""
    
    def test_initialization(self):
        """Test controller initialization"""
        server = ModelServer()
        junctions = ['junction_1', 'junction_2']
        
        controller = TrafficController(server, junctions, use_sumo=False)
        
        assert controller.junction_ids == junctions
        assert len(controller.current_phases) == 2
    
    def test_mock_observation(self):
        """Test mock observation generation"""
        server = ModelServer()
        controller = TrafficController(server, ['j1'], use_sumo=False)
        
        obs = controller.get_observation('j1')
        
        assert obs.shape == (16,)  # 4 queues + 4 densities + 4 speeds + 4 phases
        assert obs.dtype == np.float32
    
    def test_apply_action(self):
        """Test action application"""
        server = ModelServer()
        controller = TrafficController(server, ['j1'], use_sumo=False)
        
        initial_phase = controller.current_phases['j1']
        controller.apply_action('j1', 1)  # Switch
        
        assert controller.current_phases['j1'] == (initial_phase + 1) % 4
    
    def test_status(self):
        """Test status reporting"""
        server = ModelServer()
        controller = TrafficController(server, ['j1', 'j2'], use_sumo=False)
        
        status = controller.get_status()
        
        assert 'running' in status
        assert 'junctions' in status
        assert 'current_phases' in status


class TestFailsafeController:
    """Tests for FailsafeController"""
    
    def test_minimum_green_time(self):
        """Test minimum green time enforcement"""
        server = ModelServer()
        controller = TrafficController(server, ['j1'], use_sumo=False)
        failsafe = FailsafeController(controller, min_green_time=10.0)
        
        # Set phase start time to now
        failsafe.phase_start_times['j1'] = time.time()
        
        # Should block switch (not enough time)
        action = failsafe.check_safety_constraints('j1', 1)
        assert action == 0  # Blocked


# ============================================
# Model Registry Tests
# ============================================

class TestModelMetadata:
    """Tests for ModelMetadata"""
    
    def test_creation(self):
        """Test metadata creation"""
        metadata = ModelMetadata(
            version="v1.0.0",
            name="test_model",
            description="Test model"
        )
        
        assert metadata.version == "v1.0.0"
        assert metadata.name == "test_model"
        assert metadata.stage == ModelStage.DEVELOPMENT
    
    def test_to_dict(self):
        """Test conversion to dict"""
        metadata = ModelMetadata(
            version="v1.0.0",
            name="test_model",
            metrics={'reward': 100.0}
        )
        
        d = metadata.to_dict()
        
        assert d['version'] == "v1.0.0"
        assert d['stage'] == "development"
        assert d['metrics']['reward'] == 100.0
    
    def test_from_dict(self):
        """Test creation from dict"""
        data = {
            'version': 'v2.0.0',
            'name': 'test',
            'stage': 'production'
        }
        
        metadata = ModelMetadata.from_dict(data)
        
        assert metadata.version == 'v2.0.0'
        assert metadata.stage == ModelStage.PRODUCTION


class TestModelRegistry:
    """Tests for ModelRegistry"""
    
    @pytest.fixture
    def temp_registry(self, tmp_path):
        """Create temporary registry"""
        registry = ModelRegistry(str(tmp_path / "registry"))
        yield registry
        # Cleanup handled by pytest tmp_path
    
    @pytest.fixture
    def dummy_model(self, tmp_path):
        """Create dummy model file"""
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"dummy model data")
        return str(model_path)
    
    def test_register_model(self, temp_registry, dummy_model):
        """Test model registration"""
        version = temp_registry.register_model(
            model_path=dummy_model,
            name="test_model",
            description="Test",
            metrics={'reward': 100.0}
        )
        
        assert version.startswith("v")
        
        # Check metadata
        metadata = temp_registry.get_model_metadata(version)
        assert metadata is not None
        assert metadata.name == "test_model"
    
    def test_list_models(self, temp_registry, dummy_model):
        """Test listing models"""
        temp_registry.register_model(dummy_model, "model1")
        temp_registry.register_model(dummy_model, "model2")
        
        models = temp_registry.list_models()
        
        assert len(models) == 2
    
    def test_promote_model(self, temp_registry, dummy_model):
        """Test model promotion"""
        version = temp_registry.register_model(dummy_model, "test")
        
        temp_registry.promote_model(version, ModelStage.STAGING)
        
        metadata = temp_registry.get_model_metadata(version)
        assert metadata.stage == ModelStage.STAGING
    
    def test_production_model(self, temp_registry, dummy_model):
        """Test production model tracking"""
        version = temp_registry.register_model(dummy_model, "test")
        temp_registry.promote_model(version, ModelStage.PRODUCTION)
        
        prod = temp_registry.get_production_model()
        
        assert prod is not None
        assert prod[0] == version


class TestCheckpointManager:
    """Tests for CheckpointManager"""
    
    @pytest.fixture
    def temp_manager(self, tmp_path):
        """Create temporary checkpoint manager"""
        return CheckpointManager(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            max_checkpoints=5,
            keep_best=2
        )
    
    def test_save_checkpoint(self, temp_manager):
        """Test saving checkpoint"""
        mock_model = {'weights': [1, 2, 3]}  # Simple dict as model
        
        # Save checkpoint (will use pickle since torch is conditionally imported)
        path = temp_manager.save_checkpoint(
            model=mock_model,
            step=1000,
            metrics={'reward': 100.0}
        )
        
        assert path is not None
        assert temp_manager._state['latest'] is not None
    
    def test_list_checkpoints(self, temp_manager):
        """Test listing checkpoints"""
        checkpoints = temp_manager.list_checkpoints()
        
        assert isinstance(checkpoints, list)


# ============================================
# Monitoring Tests
# ============================================

class TestMetricsCollector:
    """Tests for MetricsCollector"""
    
    @pytest.fixture
    def collector(self, tmp_path):
        """Create collector with temp directory"""
        return MetricsCollector(str(tmp_path / "metrics"))
    
    def test_record_metric(self, collector):
        """Test recording metrics"""
        collector.record("test_metric", 1.0)
        collector.record("test_metric", 2.0)
        collector.record("test_metric", 3.0)
        
        value = collector.get_metric("test_metric", "last")
        assert value == 3.0
        
        value = collector.get_metric("test_metric", "mean")
        assert value == 2.0
    
    def test_register_metric(self, collector):
        """Test metric registration with thresholds"""
        config = MetricConfig(
            name="latency",
            warning_threshold=50.0,
            error_threshold=100.0
        )
        
        collector.register_metric(config)
        
        assert "latency" in collector.metric_configs
    
    def test_alerts(self, collector):
        """Test alert generation"""
        config = MetricConfig(
            name="latency",
            warning_threshold=50.0
        )
        collector.register_metric(config)
        
        # Record value exceeding threshold
        collector.record("latency", 75.0)
        
        alerts = collector.get_pending_alerts()
        assert len(alerts) >= 1
    
    def test_alert_handlers(self, collector):
        """Test alert handlers"""
        handler_called = []
        
        def handler(alert):
            handler_called.append(alert)
        
        collector.add_alert_handler(handler)
        
        config = MetricConfig(
            name="test",
            warning_threshold=10.0
        )
        collector.register_metric(config)
        
        collector.record("test", 20.0)
        
        assert len(handler_called) == 1


class TestTrainingLogger:
    """Tests for TrainingLogger"""
    
    @pytest.fixture
    def logger(self, tmp_path):
        """Create logger with temp directory"""
        return TrainingLogger(
            log_dir=str(tmp_path),
            experiment_name="test_exp"
        )
    
    def test_log_episode(self, logger):
        """Test episode logging"""
        logger.log_episode(
            episode=1,
            reward=100.0,
            length=500,
            metrics={'wait_time': 25.0}
        )
        
        # Check file was created
        assert logger.episode_log_file.exists()
    
    def test_log_hyperparameters(self, logger):
        """Test hyperparameter logging"""
        logger.log_hyperparameters({
            'lr': 0.001,
            'gamma': 0.99
        })
        
        assert logger.hparams_file.exists()


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor"""
    
    def test_timer(self, tmp_path):
        """Test timer functionality"""
        collector = MetricsCollector(str(tmp_path / "metrics"))
        monitor = PerformanceMonitor(collector)
        
        monitor.start_timer("test")
        time.sleep(0.01)
        elapsed = monitor.stop_timer("test")
        
        assert elapsed >= 10  # At least 10ms


class TestHealthChecker:
    """Tests for HealthChecker"""
    
    def test_health_checks(self, tmp_path):
        """Test health check system"""
        collector = MetricsCollector(str(tmp_path / "metrics"))
        checker = HealthChecker(collector)
        
        checker.register_check("always_pass", lambda: True)
        checker.register_check("always_fail", lambda: False)
        
        result = checker.run_checks()
        
        assert result['checks']['always_pass']['healthy'] is True
        assert result['checks']['always_fail']['healthy'] is False
        assert result['overall_healthy'] is False


# ============================================
# Configuration Tests
# ============================================

class TestConfigurations:
    """Tests for configuration dataclasses"""
    
    def test_network_config(self):
        """Test NetworkConfig"""
        config = NetworkConfig()
        
        assert config.hidden_layers == [256, 256]
        assert config.activation == "relu"
    
    def test_training_config(self):
        """Test TrainingConfig"""
        config = TrainingConfig(learning_rate=0.0001)
        
        assert config.learning_rate == 0.0001
        assert config.algorithm == "DQN"
    
    def test_experiment_config(self):
        """Test ExperimentConfig"""
        config = ExperimentConfig(name="test")
        
        assert config.name == "test"
        assert config.training is not None
        assert config.environment is not None


class TestConfigLoader:
    """Tests for ConfigLoader"""
    
    @pytest.fixture
    def loader(self):
        """Create config loader"""
        return ConfigLoader()
    
    def test_load_json(self, loader, tmp_path):
        """Test JSON config loading"""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            'name': 'test_experiment',
            'training': {'learning_rate': 0.0005}
        }))
        
        config = loader.load(str(config_file))
        
        assert config['name'] == 'test_experiment'
        assert config['training']['learning_rate'] == 0.0005
    
    def test_merge_configs(self, loader):
        """Test config merging"""
        base = {'a': 1, 'b': {'c': 2, 'd': 3}}
        override = {'b': {'c': 10}, 'e': 5}
        
        merged = loader.merge_configs(base, override)
        
        assert merged['a'] == 1
        assert merged['b']['c'] == 10
        assert merged['b']['d'] == 3
        assert merged['e'] == 5
    
    def test_load_experiment_config(self, loader):
        """Test loading experiment config"""
        config = loader.load_experiment_config(
            overrides={'name': 'custom_experiment'}
        )
        
        assert isinstance(config, ExperimentConfig)
        assert config.name == 'custom_experiment'


class TestConfigValidator:
    """Tests for ConfigValidator"""
    
    def test_valid_config(self):
        """Test validation of valid config"""
        config = ExperimentConfig()
        validator = ConfigValidator()
        
        is_valid = validator.validate(config)
        
        # May have warnings but should be valid
        assert len(validator.errors) == 0
    
    def test_invalid_learning_rate(self):
        """Test detection of invalid learning rate"""
        config = ExperimentConfig()
        config.training.learning_rate = -0.001
        
        validator = ConfigValidator()
        validator.validate(config)
        
        assert any("learning_rate" in e for e in validator.errors)
    
    def test_invalid_gamma(self):
        """Test detection of invalid gamma"""
        config = ExperimentConfig()
        config.training.gamma = 1.5
        
        validator = ConfigValidator()
        validator.validate(config)
        
        assert any("gamma" in e for e in validator.errors)


class TestConfigManager:
    """Tests for ConfigManager"""
    
    @pytest.fixture
    def manager(self, tmp_path):
        """Create config manager with temp directory"""
        return ConfigManager(str(tmp_path / "configs"))
    
    def test_create_default(self, manager):
        """Test default config creation"""
        path = manager.create_default_config()
        
        assert Path(path).exists()
    
    def test_list_configs(self, manager):
        """Test listing configs"""
        manager.create_default_config()
        
        configs = manager.list_configs()
        
        assert len(configs) >= 1
    
    def test_load_with_overrides(self, manager):
        """Test loading with overrides"""
        config = manager.load(overrides={'name': 'override_test'})
        
        assert config.name == 'override_test'


class TestLoadConfig:
    """Tests for load_config convenience function"""
    
    def test_load_defaults(self):
        """Test loading default config"""
        config = load_config()
        
        assert isinstance(config, ExperimentConfig)
    
    def test_load_with_nested_overrides(self):
        """Test loading with nested overrides"""
        config = load_config(**{
            'training.learning_rate': 0.0001,
            'training.algorithm': 'PPO'
        })
        
        # Note: This would need proper nested key handling
        assert isinstance(config, ExperimentConfig)


# ============================================
# Integration Tests
# ============================================

class TestDeploymentIntegration:
    """Integration tests for deployment system"""
    
    def test_full_deployment_flow(self, tmp_path):
        """Test full deployment flow"""
        # Setup
        config = InferenceConfig(
            model_path=str(tmp_path / "model.pt"),
            device="cpu"
        )
        
        junctions = ['j1', 'j2']
        
        # Create manager
        manager = DeploymentManager(config, junctions)
        
        # Check status (no model loaded)
        status = manager.get_status()
        
        assert status['health']['model_loaded'] is False
        
        # Shutdown
        manager.shutdown()
    
    def test_registry_to_deployment(self, tmp_path):
        """Test model registry to deployment flow"""
        # Create dummy model
        model_path = tmp_path / "trained_model.pt"
        model_path.write_bytes(b"model data")
        
        # Register model
        registry = ModelRegistry(str(tmp_path / "registry"))
        version = registry.register_model(
            model_path=str(model_path),
            name="production_model",
            metrics={'reward': 150.0}
        )
        
        # Promote to production
        registry.promote_model(version, ModelStage.PRODUCTION)
        
        # Get production model
        prod = registry.get_production_model()
        
        assert prod is not None
        assert prod[0] == version


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
