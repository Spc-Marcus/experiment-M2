#!/bin/bash
#SBATCH --job-name=Test-ILP-Time
#SBATCH --output=test.txt
#SBATCH --ntasks=1
#SBATCH --mem=16G

. /local/env/envconda.sh
conda activate strainminer


echo "Test run max e wr (avec warm start)"
echo "==================================="

if python No_error/main.py; then
    echo "✅ ILP time completed successfully!"
else
    echo "❌ ILP time failed"
    exit 1
fi