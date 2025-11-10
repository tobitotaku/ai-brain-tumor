#!/usr/bin/env python
"""
Training Performance Diagnostic Tool
=====================================
Analyzes training progress, bottlenecks, and provides actionable recommendations.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import subprocess

class TrainingDiagnostics:
    def __init__(self, main_pid=66681, logfile="logs/training_20251110_225121.log", errfile="logs/training_20251110_225121.err"):
        self.main_pid = main_pid
        self.logfile = Path(logfile)
        self.errfile = Path(errfile)
        self.start_time = None
        self.metrics = {}
        
    def check_process_status(self):
        """Check if training process is alive and get resource usage"""
        try:
            result = subprocess.run(
                ["ps", "-p", str(self.main_pid), "-o", "pid,etime,%cpu,%mem,rss"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    return {
                        "status": "RUNNING",
                        "output": lines[1].strip()
                    }
        except:
            pass
        return {"status": "NOT_RUNNING"}
    
    def analyze_logs(self):
        """Parse logs to determine training progress"""
        if not self.errfile.exists():
            return {"error": "Error log not found"}
        
        with open(self.errfile) as f:
            content = f.read()
        
        # Extract key metrics
        metrics = {}
        
        # Look for fold progress
        if "Outer fold 1/3" in content:
            metrics["current_fold"] = "Outer fold 1/3"
        elif "Outer fold 2/3" in content:
            metrics["current_fold"] = "Outer fold 2/3"
        elif "Outer fold 3/3" in content:
            metrics["current_fold"] = "Outer fold 3/3"
        else:
            metrics["current_fold"] = "Unknown"
        
        # Look for convergence warnings (indicate heavy computation)
        convergence_warnings = content.count("Liblinear failed to converge")
        metrics["convergence_warnings"] = convergence_warnings
        
        # Look for batch correction messages
        batch_msgs = content.count("No batch information")
        metrics["batch_skip_count"] = batch_msgs
        
        # Get file sizes
        metrics["log_size_bytes"] = self.logfile.stat().st_size if self.logfile.exists() else 0
        metrics["err_size_bytes"] = self.errfile.stat().st_size if self.errfile.exists() else 0
        
        return metrics
    
    def check_worker_processes(self):
        """Analyze worker process status"""
        try:
            result = subprocess.run(
                ["pgrep", "-P", str(self.main_pid), "-f", "LokyProcess"],
                capture_output=True,
                text=True
            )
            worker_pids = result.stdout.strip().split('\n')
            worker_pids = [p for p in worker_pids if p]
            
            workers = []
            for pid in worker_pids:
                ps_result = subprocess.run(
                    ["ps", "-p", pid, "-o", "%cpu,%mem,rss"],
                    capture_output=True,
                    text=True
                )
                if ps_result.returncode == 0:
                    lines = ps_result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        workers.append({
                            "pid": pid,
                            "stats": lines[1].strip()
                        })
            
            return {"count": len(workers), "workers": workers}
        except:
            return {"count": 0}
    
    def check_io_patterns(self):
        """Check disk I/O patterns and memory mapping"""
        try:
            # Check for joblib temp files (indicates heavy data movement)
            result = subprocess.run(
                ["lsof", "-p", str(self.main_pid), "-a", "-d", "cwd"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if "joblib" in result.stdout:
                return {"pattern": "Heavy memory-mapped I/O (joblib temp files)", "severity": "HIGH"}
            elif "pkl" in result.stdout or "csv" in result.stdout:
                return {"pattern": "File-based I/O active", "severity": "MEDIUM"}
            else:
                return {"pattern": "Minimal file I/O", "severity": "LOW"}
        except:
            return {"pattern": "Could not determine"}
    
    def estimate_remaining_time(self):
        """Estimate remaining training time"""
        process_info = self.check_process_status()
        if process_info.get("status") != "RUNNING":
            return {"estimate": "N/A", "reason": "Process not running"}
        
        # Parse elapsed time from process info
        output = process_info.get("output", "")
        parts = output.split()
        if len(parts) > 1:
            etime = parts[1]  # Format: HH:MM:SS
            try:
                time_parts = etime.split(':')
                if len(time_parts) == 3:
                    hours, mins, secs = map(int, time_parts)
                    elapsed_mins = hours * 60 + mins + secs / 60
                    
                    # Expected total: 45-60 minutes for full training
                    # But Filter_L1 adds significant I/O overhead
                    expected_total = 75  # Conservative estimate
                    remaining = expected_total - elapsed_mins
                    
                    if remaining < 0:
                        return {
                            "estimate": f"{abs(remaining):.1f} min OVER",
                            "elapsed": f"{elapsed_mins:.1f} min",
                            "expected": f"{expected_total} min",
                            "status": "May be slower than expected"
                        }
                    else:
                        return {
                            "estimate": f"~{remaining:.1f} min",
                            "elapsed": f"{elapsed_mins:.1f} min",
                            "expected": f"{expected_total} min",
                            "status": "On track (with I/O overhead)"
                        }
            except:
                pass
        
        return {"estimate": "Could not calculate"}
    
    def generate_report(self):
        """Generate comprehensive diagnostic report"""
        print("\n" + "="*70)
        print("TRAINING PERFORMANCE DIAGNOSTICS")
        print("="*70 + "\n")
        
        print("1. PROCESS STATUS")
        print("-" * 70)
        process_status = self.check_process_status()
        if process_status.get("status") == "RUNNING":
            print(f"✓ Training process (PID {self.main_pid}) is RUNNING")
            print(f"  {process_status.get('output', 'N/A')}")
        else:
            print(f"✗ Training process is NOT RUNNING")
        
        print("\n2. TRAINING PROGRESS")
        print("-" * 70)
        logs = self.analyze_logs()
        print(f"Current Stage: {logs.get('current_fold', 'Unknown')}")
        print(f"Convergence Warnings: {logs.get('convergence_warnings', 0)} (L1 hyperparameter tuning)")
        print(f"Batch Correction Skips: {logs.get('batch_skip_count', 0)} (Expected - no batch info)")
        print(f"Log File Size: {logs.get('log_size_bytes', 0)} bytes (stdout)")
        print(f"Error File Size: {logs.get('err_size_bytes', 0)} bytes (all output)")
        
        print("\n3. WORKER PROCESSES")
        print("-" * 70)
        workers = self.check_worker_processes()
        print(f"Active Workers: {workers.get('count', 0)} / 5")
        for w in workers.get('workers', []):
            print(f"  PID {w['pid']}: {w['stats']}")
        
        print("\n4. I/O PATTERNS")
        print("-" * 70)
        io_info = self.check_io_patterns()
        print(f"Pattern: {io_info.get('pattern', 'Unknown')}")
        print(f"Severity: {io_info.get('severity', 'Unknown')}")
        
        print("\n5. TIME ESTIMATE")
        print("-" * 70)
        time_est = self.estimate_remaining_time()
        print(f"Status: {time_est.get('status', 'Unknown')}")
        if 'elapsed' in time_est:
            print(f"Elapsed: {time_est['elapsed']}")
            print(f"Expected Total: {time_est['expected']}")
            print(f"Estimated Remaining: {time_est['estimate']}")
        
        print("\n6. ANALYSIS & RECOMMENDATIONS")
        print("-" * 70)
        
        if workers.get('count', 0) < 5:
            print("⚠  WARNING: Not all workers are active")
            print("  Recommendation: Check for hung processes or resource constraints")
        else:
            print("✓ All 5 workers are active")
        
        if io_info.get('severity') == 'HIGH':
            print("⚠  WARNING: Heavy I/O detected (joblib memory-mapped files)")
            print("  Root Cause: OptimizedFilterL1 feature selection involves:")
            print("    - Variance filtering (18,752 → 15,000 genes)")
            print("    - ANOVA F-test pre-filtering (15,000 → 2,000 genes)")
            print("    - Correlation matrix computation (2,000² = 4M operations)")
            print("    - L1 regularization per fold")
            print("  This is computationally intensive and I/O bound.")
            print("  Expected Impact: 75-90 min total (vs. 12-15 min originally estimated)")
        
        if logs.get('convergence_warnings', 0) > 5:
            print("ℹ  Note: Multiple convergence warnings are NORMAL for L1 hyperparameter tuning")
            print("  L1 regularization with liblinear solver is iterative and may not")
            print("  converge with default max_iter in hyperparameter search.")
        
        print("\n7. DECISION MATRIX")
        print("-" * 70)
        elapsed_mins = self.estimate_remaining_time().get('elapsed', '?')
        print(f"Elapsed Time: {elapsed_mins}")
        print(f"Current Stage: {logs.get('current_fold', 'Unknown')}")
        print("\nOptions:")
        print("  A) CONTINUE - Let it finish (may take 75-90 min total)")
        print("  B) ABORT & SWITCH - Kill and use PCA-only approach (faster)")
        print("  C) OPTIMIZE - Kill and rerun with reduced CV folds or features")
        
        print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    diag = TrainingDiagnostics()
    diag.generate_report()
