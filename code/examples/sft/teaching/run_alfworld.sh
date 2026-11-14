#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../../.."
python -m scripts.teaching_experiment train --group "${1:?Pass high or low}" --config "${2:-examples/teaching/config.yaml}"
