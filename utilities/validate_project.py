#!/usr/bin/env python
"""
Project Validation Script
==========================
Checks project integrity, data availability, configuration, and readiness.

Validates:
  - File existence and structure
  - Data availability (raw, interim, processed)
  - Configuration files
  - Environment setup
  - Code quality and imports
"""

import sys
import json
from pathlib import Path
from importlib import util

class ProjectValidator:
    def __init__(self, project_root=None):
        self.root = Path(project_root or ".")
        self.checks = {}
        self.results = {}
    
    def check_directory_structure(self):
        """Validate directory structure"""
        required_dirs = [
            "src", "data", "scripts", "logs", "reports", "notebooks", "models", "docs"
        ]
        
        results = {}
        for dir_name in required_dirs:
            dir_path = self.root / dir_name
            exists = dir_path.exists() and dir_path.is_dir()
            results[dir_name] = {"exists": exists}
            
            if exists:
                file_count = len(list(dir_path.glob("**/*")))
                results[dir_name]["files"] = file_count
        
        self.results['directories'] = results
        return results
    
    def check_data_files(self):
        """Validate data availability"""
        data_checks = {
            'raw': [
                'data/raw/combined_labeled_standardized.csv',
                'data/raw/gene_expression.csv',
                'data/raw/metadata.csv'
            ],
            'processed': [
                'data/processed/expression_processed.csv',
                'data/processed/metadata_processed.csv'
            ]
        }
        
        results = {}
        for category, files in data_checks.items():
            results[category] = {}
            for file_path in files:
                full_path = self.root / file_path
                exists = full_path.exists()
                size_mb = full_path.stat().st_size / (1024**2) if exists else 0
                
                results[category][file_path] = {
                    "exists": exists,
                    "size_mb": round(size_mb, 2)
                }
        
        self.results['data'] = results
        return results
    
    def check_config_files(self):
        """Validate configuration files"""
        config_files = [
            'config.yaml',
            'config_academic_filter_l1_optimized_v2.yaml',
            'config_academic_filter_l1_optimized.yaml',
            'environment.yml'
        ]
        
        results = {}
        for config_file in config_files:
            full_path = self.root / config_file
            exists = full_path.exists()
            results[config_file] = {"exists": exists}
            
            if exists and config_file.endswith('.yaml'):
                try:
                    import yaml
                    with open(full_path) as f:
                        config = yaml.safe_load(f)
                    results[config_file]["valid"] = True
                    results[config_file]["keys"] = list(config.keys()) if config else []
                except Exception as e:
                    results[config_file]["valid"] = False
                    results[config_file]["error"] = str(e)
        
        self.results['configs'] = results
        return results
    
    def check_python_modules(self):
        """Check Python module availability"""
        modules = ['sklearn', 'pandas', 'numpy', 'joblib', 'yaml', 'scipy']
        
        results = {}
        for module_name in modules:
            spec = util.find_spec(module_name)
            available = spec is not None
            results[module_name] = {"available": available}
            
            if available and hasattr(__import__(module_name), '__version__'):
                mod = __import__(module_name)
                results[module_name]["version"] = mod.__version__
        
        self.results['modules'] = results
        return results
    
    def check_source_code(self):
        """Validate source code structure"""
        src_dir = self.root / "src"
        
        required_files = [
            'src/__init__.py',
            'src/data.py',
            'src/features.py',
            'src/models.py',
            'src/pipeline.py',
            'src/eval.py'
        ]
        
        results = {}
        for file_path in required_files:
            full_path = self.root / file_path
            exists = full_path.exists()
            size_kb = full_path.stat().st_size / 1024 if exists else 0
            
            results[file_path] = {
                "exists": exists,
                "size_kb": round(size_kb, 2)
            }
            
            # Check for imports
            if exists:
                try:
                    with open(full_path) as f:
                        content = f.read()
                    has_imports = 'import' in content
                    has_functions = 'def ' in content
                    results[file_path]["has_imports"] = has_imports
                    results[file_path]["has_functions"] = has_functions
                except Exception as e:
                    results[file_path]["error"] = str(e)
        
        self.results['source'] = results
        return results
    
    def generate_report(self):
        """Generate validation report"""
        report = []
        report.append("# Project Validation Report")
        report.append(f"\n**Date**: {json.dumps({})}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append("")
        report.append(f"- **Root**: {self.root}")
        report.append(f"- **Total Checks**: {len(self.results)}")
        report.append("")
        
        # Directory Structure
        if 'directories' in self.results:
            report.append("## Directory Structure")
            report.append("")
            all_exist = all(d['exists'] for d in self.results['directories'].values())
            report.append(f"- **Status**: {'✓ Valid' if all_exist else '✗ Missing directories'}")
            for dir_name, info in self.results['directories'].items():
                status = "✓" if info['exists'] else "✗"
                report.append(f"  - {status} {dir_name} ({info.get('files', 0)} files)")
            report.append("")
        
        # Data Files
        if 'data' in self.results:
            report.append("## Data Files")
            report.append("")
            data = self.results['data']
            for category, files in data.items():
                all_exist = all(f['exists'] for f in files.values())
                status = "✓" if all_exist else "✗"
                report.append(f"### {status} {category.title()}")
                for file_path, info in files.items():
                    status = "✓" if info['exists'] else "✗"
                    size = f" ({info['size_mb']:.1f}MB)" if info['size_mb'] > 0 else ""
                    report.append(f"  - {status} {file_path}{size}")
            report.append("")
        
        # Configuration
        if 'configs' in self.results:
            report.append("## Configuration Files")
            report.append("")
            for config_file, info in self.results['configs'].items():
                status = "✓" if info['exists'] else "✗"
                valid = "" if not info['exists'] else f" (Valid: {info.get('valid', '?')})"
                report.append(f"  - {status} {config_file}{valid}")
            report.append("")
        
        # Python Environment
        if 'modules' in self.results:
            report.append("## Python Environment")
            report.append("")
            all_available = all(m['available'] for m in self.results['modules'].values())
            status = "✓" if all_available else "⚠"
            report.append(f"- **Status**: {status}")
            for module_name, info in self.results['modules'].items():
                status = "✓" if info['available'] else "✗"
                version = f" v{info.get('version', '?')}" if info['available'] else ""
                report.append(f"  - {status} {module_name}{version}")
            report.append("")
        
        # Source Code
        if 'source' in self.results:
            report.append("## Source Code")
            report.append("")
            all_exist = all(f['exists'] for f in self.results['source'].values())
            status = "✓" if all_exist else "✗"
            report.append(f"- **Status**: {status}")
            for file_path, info in self.results['source'].items():
                status = "✓" if info['exists'] else "✗"
                report.append(f"  - {status} {file_path}")
            report.append("")
        
        return "\n".join(report)
    
    def validate_all(self):
        """Run all validation checks"""
        print("Validating project structure...")
        self.check_directory_structure()
        self.check_data_files()
        self.check_config_files()
        self.check_python_modules()
        self.check_source_code()
        
        report = self.generate_report()
        print(report)
        
        return self.results

if __name__ == "__main__":
    validator = ProjectValidator()
    results = validator.validate_all()
