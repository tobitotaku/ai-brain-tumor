#!/usr/bin/env python
"""
Training Performance Report Generator
======================================
Generates detailed analysis of training runs for documentation and debugging.

Output:
  - Markdown report with timeline, metrics, and insights
  - JSON data for programmatic access
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def parse_log_file(log_file):
    """Extract structured data from log file"""
    data = {
        'stages': defaultdict(list),
        'warnings': [],
        'errors': [],
        'line_count': 0
    }
    
    if not log_file.exists():
        return data
    
    with open(log_file) as f:
        for line in f:
            data['line_count'] += 1
            
            if 'WARNING' in line:
                data['warnings'].append(line.strip())
            elif 'ERROR' in line:
                data['errors'].append(line.strip())
            
            # Extract stage markers
            if 'STEP' in line or 'fold' in line.lower():
                data['stages']['progress'].append(line.strip())
    
    return data

def generate_markdown_report(log_file, metrics_file, output_file=None):
    """Generate comprehensive markdown report"""
    
    log_data = parse_log_file(log_file)
    
    metrics_data = {}
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics_data = json.load(f)
    
    # Build report
    report = []
    report.append("# Training Performance Report")
    report.append("")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    
    if metrics_data:
        start_time = metrics_data.get('timestamp_start', 'N/A')
        report.append(f"- **Start Time**: {start_time}")
        
        if 'stages' in metrics_data:
            stages = metrics_data['stages']
            report.append(f"- **Stages Executed**: {len(stages)}")
            
            total_elapsed = sum(
                s.get('elapsed_seconds', 0) for s in stages.values()
            )
            report.append(f"- **Total Elapsed Time**: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    
    report.append("")
    
    # Log Statistics
    report.append("## Log Statistics")
    report.append("")
    report.append(f"- **Total Lines**: {log_data['line_count']}")
    report.append(f"- **Warnings**: {len(log_data['warnings'])}")
    report.append(f"- **Errors**: {len(log_data['errors'])}")
    report.append("")
    
    # Detailed Metrics
    if metrics_data and 'stages' in metrics_data:
        report.append("## Training Stages")
        report.append("")
        
        for stage_name, stage_info in metrics_data['stages'].items():
            status = stage_info.get('status', 'UNKNOWN')
            elapsed = stage_info.get('elapsed_seconds', 0)
            
            report.append(f"### {stage_name}")
            report.append(f"- **Status**: {status}")
            report.append(f"- **Duration**: {elapsed:.1f}s")
            
            if 'details' in stage_info:
                report.append("- **Details**:")
                for key, value in stage_info['details'].items():
                    report.append(f"  - {key}: {value}")
            
            report.append("")
    
    # Warnings and Errors
    if log_data['warnings']:
        report.append("## Warnings")
        report.append("")
        for warn in log_data['warnings'][:10]:  # First 10
            report.append(f"- {warn}")
        report.append("")
    
    if log_data['errors']:
        report.append("## Errors")
        report.append("")
        for err in log_data['errors']:
            report.append(f"- {err}")
        report.append("")
    
    # Output
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {output_file}")
    else:
        print(report_text)
    
    return report_text

if __name__ == "__main__":
    # Find latest training files
    log_dir = Path("logs")
    log_files = sorted(log_dir.glob("training_*.log"), reverse=True)
    
    if not log_files:
        print("No training logs found")
        sys.exit(1)
    
    latest_log = log_files[0]
    latest_metrics = log_dir / latest_log.name.replace('.log', '.metrics.json')
    
    print(f"Analyzing: {latest_log}")
    print(f"Metrics: {latest_metrics}")
    print("")
    
    # Generate report
    report_file = log_dir / latest_log.name.replace('.log', '.report.md')
    generate_markdown_report(latest_log, latest_metrics, report_file)
