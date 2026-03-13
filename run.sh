#!/bin/bash
#SBATCH --job-name=T1-gamma  
#SBATCH --output=toto.txt
#SBATCH --ntasks=6
#SBATCH --mem=24G

. /local/env/envconda.sh
conda activate strainminer


echo "Testing ILP time..."
echo "==================================="

if python T1/run_experiment.py; then
    echo "✅ completed successfully!"
else
    echo "❌ failed"
    exit 1
fi