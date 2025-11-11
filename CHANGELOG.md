# Changelog

All notable changes to the GBM Classification project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-11-12

### Added
- Comprehensive `.gitignore` for production-ready repository structure
- `.gitattributes` configuration for Git LFS support
- `CHANGELOG.md` for tracking project changes
- `CITATION.cff` for academic citation metadata
- Professional `LICENSE` file (MIT License)

### Changed
- Updated `README.md` to remove emojis and improve professional tone
- Enhanced `.gitignore` to cover all Python artifacts, virtual environments, and generated files
- Standardized markdown formatting across all documentation

### Fixed
- Repository structure cleaned for academic collaboration
- Large file handling configured via Git LFS

## [1.0.0] - 2025-11-11

### Added
- Final academic report with comprehensive corrections (12 categories)
- Cohen's d verification and update (65.17)
- PCA explained variance estimation (38-40% for 100 PCs)
- Complete TODO verification and completion

### Changed
- Updated `reports/Final_Academic_Report_FIXED.md` to version 1.1
- ChangeLog updated with all verification statuses

### Fixed
- Class distribution percentages (GBM: 93.3%, Healthy: 6.7%)
- PR-AUC baseline calculation (0.067)
- Permutation test p-value (0.0476)
- Cohen's d effect size calculations (3 locations)
- PCA explained variance percentages

## [0.9.0] - 2025-11-05

### Added
- Complete nested cross-validation pipeline (5x3 and 3x3 configurations)
- Multiple feature selection routes (filter+L1, PCA)
- Model comparison framework (Logistic Regression, Random Forest, LightGBM)
- Comprehensive evaluation metrics (ROC-AUC, PR-AUC, calibration, decision curves)
- Bootstrap confidence intervals (n=1000)
- Feature stability analysis via bootstrap resampling
- SHAP-based explainability framework
- Academic documentation (`docs/Protocol.md`, `metadata/model_card.md`, `metadata/data_card.md`)

### Changed
- Reorganized configuration files into `config/` directory
- Moved utility scripts to `utilities/` directory
- Split documentation into `docs/` (active) and `docs_old/` (archived)

### Fixed
- Data leakage prevention in preprocessing pipeline
- Stratified cross-validation for class imbalance
- Random seed management for reproducibility

## [0.5.0] - 2025-10-15

### Added
- Initial project structure
- Basic preprocessing pipeline
- Exploratory data analysis notebooks
- Initial model training scripts

### Changed
- N/A (initial release)

### Removed
- N/A (initial release)

---

## Version Numbering

- **Major version (X.0.0)**: Breaking changes, major architectural updates
- **Minor version (0.X.0)**: New features, enhancements, backward-compatible changes
- **Patch version (0.0.X)**: Bug fixes, documentation updates, minor improvements

## Categories

- **Added**: New features or capabilities
- **Changed**: Changes to existing functionality
- **Deprecated**: Features that will be removed in future versions
- **Removed**: Features that have been removed
- **Fixed**: Bug fixes
- **Security**: Security-related changes
