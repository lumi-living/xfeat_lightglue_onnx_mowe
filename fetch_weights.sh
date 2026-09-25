#!/usr/bin/env bash
# Fetch + pin the XFeat / LighterGlue weights (Apache-2.0, verlab/accelerated_features;
# mowe-nav-kb 09 licence table). Idempotent: verifies sha256 if the file already exists.
set -euo pipefail
cd "$(dirname "$0")/weights"
BASE=https://github.com/verlab/accelerated_features/raw/main/weights
declare -A SHA=(
  [xfeat.pt]=0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b
  [xfeat-lighterglue.pt]=766102df37f11189efe5b0811d1f47c72b22629b79bfabfcfff9d2a2f84654b8
)
for f in "${!SHA[@]}"; do
  [ -s "$f" ] || curl -fsSL -o "$f" "$BASE/$f"
  echo "${SHA[$f]}  $f" | sha256sum -c -
done
