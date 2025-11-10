#!/usr/bin/env python3
"""
Real-time Training Log Viewer
==============================
Follow training progress in real-time with live updates and metrics extraction.

Usage:
    python scripts/monitor_log.py                 # Follow latest training
    python scripts/monitor_log.py -f training_log.log  # Follow specific log
"""

import sys
import argparse
import re
from pathlib import Path
from datetime import datetime
import time
from collections import defaultdict
from typing import Optional


class LogMonitor:
    """Monitor and parse training logs in real-time."""
    
    def __init__(self, log_file: str, follow: bool = True):
        self.log_file = Path(log_file)
        self.follow = follow
        self.last_position = 0
        self.metrics_cache = defaultdict(dict)
        
        # Regex patterns for metrics extraction
        self.patterns = {
            'fold': r'Outer fold (\d+)/(\d+)',
            'roc_auc': r'ROC-AUC:\s+([\d.]+)',
            'f1': r'F1 Score:\s+([\d.]+)',
            'accuracy': r'Accuracy:\s+([\d.]+)',
            'precision': r'Precision:\s+([\d.]+)',
            'recall': r'Recall:\s+([\d.]+)',
            'specificity': r'Specificity:\s+([\d.]+)',
            'train_score': r'Train:\s+([\d.]+)',
            'val_score': r'Val:\s+([\d.]+)',
            'elapsed': r'Elapsed:\s+([\d.]+)s',
            'model_combo': r'\[(\d+)/(\d+)\]\s+Training:\s+(\w+)\s+\+\s+(\w+)',
        }
    
    def read_new_lines(self):
        """Read new lines from log file since last position."""
        if not self.log_file.exists():
            return []
        
        with open(self.log_file, 'r') as f:
            f.seek(self.last_position)
            lines = f.readlines()
            self.last_position = f.tell()
        
        return lines
    
    def extract_metrics(self, line: str) -> dict:
        """Extract metrics from a log line."""
        metrics = {}
        
        for metric_name, pattern in self.patterns.items():
            match = re.search(pattern, line)
            if match:
                if metric_name in ['fold', 'model_combo']:
                    metrics[metric_name] = match.groups()
                else:
                    try:
                        metrics[metric_name] = float(match.group(1))
                    except (ValueError, IndexError):
                        pass
        
        return metrics
    
    def display_metrics(self, metrics: dict):
        """Display extracted metrics in a formatted way."""
        if 'model_combo' in metrics:
            current, total, feat_method, model = metrics['model_combo']
            print(f"\n{'='*70}")
            print(f"[{current}/{total}] {feat_method} + {model}")
            print(f"{'='*70}")
        
        if 'fold' in metrics:
            fold, total = metrics['fold']
            print(f"  \u250c Fold {fold}/{total}")
        
        # Performance metrics
        perf_metrics = {
            'roc_auc': 'ROC-AUC',
            'f1': 'F1 Score',
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'specificity': 'Specificity',
        }
        
        has_perf = False
        for key, label in perf_metrics.items():
            if key in metrics:
                if not has_perf:
                    print(f"  \u2502 Performance:")
                    has_perf = True
                print(f"  \u2502   {label:.<25} {metrics[key]:.3f}")
        
        # CV scores
        if 'train_score' in metrics or 'val_score' in metrics:
            print(f"  \u2502 CV Scores:")
            if 'train_score' in metrics:
                print(f"  \u2502   Train:.<25 {metrics['train_score']:.3f}")
            if 'val_score' in metrics:
                print(f"  \u2502   Val:.<25 {metrics['val_score']:.3f}")
        
        if 'elapsed' in metrics:
            print(f"  \u2514 Elapsed: {metrics['elapsed']:.1f}s")
    
    def monitor(self, update_interval: float = 1.0):
        """Monitor log file and display updates."""
        print(f"Monitoring: {self.log_file}")
        print(f"Press Ctrl+C to stop\n")
        
        try:
            while True:
                lines = self.read_new_lines()
                
                if lines:
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Check for section headers
                        if '=' * 20 in line and ('STEP' in line or 'TRAINING' in line):
                            print(f"\n{line}")
                        elif '✓' in line or '✗' in line:
                            print(f"\n{line}")
                        elif 'ERROR' in line or 'WARNING' in line:
                            print(f"\n\033[31m{line}\033[0m")  # Red
                        
                        # Extract and display metrics
                        metrics = self.extract_metrics(line)
                        if metrics:
                            self.display_metrics(metrics)
                
                if not self.follow:
                    break
                
                time.sleep(update_interval)
        
        except KeyboardInterrupt:
            print(f"\n\nMonitoring stopped at {datetime.now().strftime('%H:%M:%S')}")


def find_latest_log(log_dir: str = 'logs') -> Optional[Path]:
    """Find the most recently modified training log."""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        return None
    
    log_files = list(log_path.glob('training_*.log'))
    
    if not log_files:
        return None
    
    return max(log_files, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description='Real-time training log monitor'
    )
    parser.add_argument(
        '-f', '--file',
        type=str,
        help='Log file to monitor (default: latest training log)'
    )
    parser.add_argument(
        '--no-follow',
        action='store_true',
        help='Read log file once without following'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Update interval in seconds (default: 1.0)'
    )
    parser.add_argument(
        '-d', '--log-dir',
        type=str,
        default='logs',
        help='Directory containing logs (default: logs)'
    )
    
    args = parser.parse_args()
    
    # Determine log file
    if args.file:
        log_file = Path(args.file)
    else:
        log_file = find_latest_log(args.log_dir)
        if not log_file:
            print(f"Error: No log files found in {args.log_dir}/")
            sys.exit(1)
    
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    # Start monitoring
    monitor = LogMonitor(str(log_file), follow=not args.no_follow)
    monitor.monitor(update_interval=args.interval)


if __name__ == '__main__':
    main()
