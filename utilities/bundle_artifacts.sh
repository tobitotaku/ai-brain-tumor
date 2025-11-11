#!/bin/bash
# Bundle all artifacts from compact full run into a timestamped archive

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="compact_full_${TIMESTAMP}.zip"
ARTIFACT_DIR="artifacts"

echo "=================================================="
echo "Artifact Bundler - Compact Full Run"
echo "=================================================="
echo "Archive: ${ARCHIVE_NAME}"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Create artifacts directory
mkdir -p "$ARTIFACT_DIR"

# Create temporary staging directory
STAGING_DIR=$(mktemp -d)
echo "Staging directory: $STAGING_DIR"
echo ""

# Copy files to staging
echo "Collecting artifacts..."

# Models
if [ -d "models" ]; then
    mkdir -p "$STAGING_DIR/models"
    cp models/*.pkl "$STAGING_DIR/models/" 2>/dev/null || true
    MODEL_COUNT=$(ls "$STAGING_DIR/models" 2>/dev/null | wc -l)
    echo "  ✓ Models: $MODEL_COUNT files"
fi

# Tables
if [ -d "reports/tables" ]; then
    mkdir -p "$STAGING_DIR/reports/tables"
    cp reports/tables/*.csv "$STAGING_DIR/reports/tables/" 2>/dev/null || true
    cp reports/tables/*.tex "$STAGING_DIR/reports/tables/" 2>/dev/null || true
    TABLE_COUNT=$(ls "$STAGING_DIR/reports/tables" 2>/dev/null | wc -l)
    echo "  ✓ Tables: $TABLE_COUNT files"
fi

# Figures
if [ -d "figures" ]; then
    mkdir -p "$STAGING_DIR/figures"
    cp -r figures/modeling "$STAGING_DIR/figures/" 2>/dev/null || true
    cp -r figures/calibration "$STAGING_DIR/figures/" 2>/dev/null || true
    cp -r figures/shap "$STAGING_DIR/figures/" 2>/dev/null || true
    FIG_COUNT=$(find "$STAGING_DIR/figures" -name "*.png" 2>/dev/null | wc -l)
    echo "  ✓ Figures: $FIG_COUNT PNG files"
fi

# Outputs (predictions, probabilities)
if [ -d "outputs" ]; then
    mkdir -p "$STAGING_DIR/outputs"
    cp outputs/*.csv "$STAGING_DIR/outputs/" 2>/dev/null || true
    cp outputs/*.npy "$STAGING_DIR/outputs/" 2>/dev/null || true
    OUTPUT_COUNT=$(ls "$STAGING_DIR/outputs" 2>/dev/null | wc -l)
    echo "  ✓ Outputs: $OUTPUT_COUNT files"
fi

# Config
if [ -f "config/config_compact_full.yaml" ]; then
    mkdir -p "$STAGING_DIR/config"
    cp config/config_compact_full.yaml "$STAGING_DIR/config/"
    echo "  ✓ Configuration file"
fi

# Report
if [ -f "reports/COMPACT_FULL_REPORT.md" ]; then
    cp reports/COMPACT_FULL_REPORT.md "$STAGING_DIR/"
    echo "  ✓ Analysis report"
fi

# Logs (last 3 training logs)
if [ -d "logs" ]; then
    mkdir -p "$STAGING_DIR/logs"
    ls -t logs/training_*.log 2>/dev/null | head -3 | while read log; do
        cp "$log" "$STAGING_DIR/logs/"
    done
    LOG_COUNT=$(ls "$STAGING_DIR/logs" 2>/dev/null | wc -l)
    echo "  ✓ Training logs: $LOG_COUNT files"
fi

# Create manifest
echo ""
echo "Creating manifest..."
cat > "$STAGING_DIR/MANIFEST.txt" << EOF
Compact Full GBM Training Run - Artifact Archive
=================================================

Generated: $(date '+%Y-%m-%d %H:%M:%S')
Archive: ${ARCHIVE_NAME}

Contents:
---------
EOF

# Add directory tree to manifest
if command -v tree &> /dev/null; then
    tree -L 2 "$STAGING_DIR" >> "$STAGING_DIR/MANIFEST.txt"
else
    find "$STAGING_DIR" -type f | sort >> "$STAGING_DIR/MANIFEST.txt"
fi

echo "  ✓ Manifest created"
echo ""

# Create archive
echo "Creating archive..."
cd "$STAGING_DIR"
zip -r -q "$ARCHIVE_NAME" .
mv "$ARCHIVE_NAME" "../$ARTIFACT_DIR/"
cd - > /dev/null

# Cleanup staging
rm -rf "$STAGING_DIR"

# Report
ARCHIVE_PATH="$ARTIFACT_DIR/$ARCHIVE_NAME"
ARCHIVE_SIZE=$(du -h "$ARCHIVE_PATH" | cut -f1)

echo ""
echo "=================================================="
echo "Archive created successfully!"
echo "=================================================="
echo "Location: $ARCHIVE_PATH"
echo "Size: $ARCHIVE_SIZE"
echo ""
echo "Contents summary:"
ls -lh "$ARTIFACT_DIR/$ARCHIVE_NAME"
echo ""
echo "To extract:"
echo "  unzip $ARCHIVE_PATH -d <destination>"
echo "=================================================="
