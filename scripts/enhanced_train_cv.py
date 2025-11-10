#!/usr/bin/env python
"""
Enhanced Training Script with Comprehensive Logging
====================================================
Drop-in replacement for train_cv.py with detailed performance tracking
and real-time progress monitoring.

Logs to:
  - logs/training_TIMESTAMP.log (all INFO logs)
  - logs/training_TIMESTAMP.metrics.json (performance metrics)
  - logs/training_TIMESTAMP.progress (progress tracking)
"""

import sys
import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_cv import main as original_main
from src.pipeline import NestedCrossValidator

# Setup enhanced logging
def setup_enhanced_logging(timestamp):
    """Setup comprehensive logging infrastructure"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Main log file
    log_file = log_dir / f"training_{timestamp}.log"
    metrics_file = log_dir / f"training_{timestamp}.metrics.json"
    progress_file = log_dir / f"training_{timestamp}.progress"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # File handler - all logs
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler - INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return {
        'log_file': log_file,
        'metrics_file': metrics_file,
        'progress_file': progress_file,
        'logger': logger
    }

class MetricsTracker:
    """Track and log training metrics in real-time"""
    
    def __init__(self, metrics_file):
        self.metrics_file = Path(metrics_file)
        self.metrics = {
            'timestamp_start': datetime.now().isoformat(),
            'stages': {}
        }
        self.current_stage = None
        self.stage_start_time = None
    
    def start_stage(self, stage_name):
        """Mark start of training stage"""
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        self.metrics['stages'][stage_name] = {
            'start_time': datetime.now().isoformat(),
            'status': 'IN_PROGRESS'
        }
        self._save()
    
    def end_stage(self, stage_name, status='COMPLETED', details=None):
        """Mark end of training stage"""
        elapsed = time.time() - self.stage_start_time if self.stage_start_time else 0
        
        if stage_name in self.metrics['stages']:
            self.metrics['stages'][stage_name].update({
                'end_time': datetime.now().isoformat(),
                'elapsed_seconds': elapsed,
                'status': status
            })
            if details:
                self.metrics['stages'][stage_name]['details'] = details
        
        self._save()
    
    def log_fold(self, fold_type, fold_num, total_folds, metrics_dict=None):
        """Log fold completion"""
        fold_key = f"{fold_type}_fold_{fold_num}"
        
        if 'folds' not in self.metrics:
            self.metrics['folds'] = []
        
        fold_record = {
            'fold_type': fold_type,
            'fold_num': fold_num,
            'total_folds': total_folds,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics_dict or {}
        }
        
        self.metrics['folds'].append(fold_record)
        self._save()
    
    def _save(self):
        """Save metrics to JSON file"""
        self.metrics['timestamp_last_updated'] = datetime.now().isoformat()
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

class ProgressTracker:
    """Track and display training progress"""
    
    def __init__(self, progress_file):
        self.progress_file = Path(progress_file)
        self.progress = {
            'current_stage': '',
            'outer_fold': 0,
            'inner_fold': 0,
            'total_folds': 0,
            'models_completed': [],
            'updated_at': ''
        }
    
    def update(self, **kwargs):
        """Update progress state"""
        self.progress.update(kwargs)
        self.progress['updated_at'] = datetime.now().isoformat()
        self._save()
    
    def _save(self):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            f.write(f"TRAINING PROGRESS\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"Updated: {self.progress['updated_at']}\n")
            f.write(f"Current Stage: {self.progress.get('current_stage', 'N/A')}\n")
            f.write(f"Outer Fold: {self.progress.get('outer_fold', 0)}/3\n")
            f.write(f"Models Completed: {', '.join(self.progress.get('models_completed', []))}\n")

if __name__ == "__main__":
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_setup = setup_enhanced_logging(timestamp)
    logger = logs_setup['logger']
    
    # Initialize trackers
    metrics_tracker = MetricsTracker(logs_setup['metrics_file'])
    progress_tracker = ProgressTracker(logs_setup['progress_file'])
    
    logger.info("=" * 80)
    logger.info("ENHANCED TRAINING START")
    logger.info("=" * 80)
    logger.info(f"Log files:")
    logger.info(f"  Main: {logs_setup['log_file']}")
    logger.info(f"  Metrics: {logs_setup['metrics_file']}")
    logger.info(f"  Progress: {logs_setup['progress_file']}")
    logger.info("")
    
    # Run original training with enhanced logging
    try:
        metrics_tracker.start_stage('data_loading')
        logger.info("Starting original training pipeline...")
        
        # Run original main
        original_main()
        
        metrics_tracker.end_stage('data_loading', 'COMPLETED')
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        metrics_tracker.end_stage('data_loading', 'FAILED', {'error': str(e)})
        sys.exit(1)
