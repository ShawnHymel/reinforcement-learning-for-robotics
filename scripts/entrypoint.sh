#!/bin/bash

# Clear the xfdesktop icon layout cache before the desktop starts.
# Without this, stale entries in icons.screen0.yaml from a previous
# container run cause duplicate desktop icons on subsequent starts.
rm -f /config/.config/xfce4/desktop/icons.screen0.yaml

# Activates the Python virtual environment for all shell sessions, then
# hands off to the linuxserver/webtop init process (s6-overlay), which
# starts the XFCE desktop, noVNC, and all other Webtop services.
source /opt/rl-env/bin/activate

# Start TensorBoard in the background so it's available as soon as the
# container starts. Logs are written to /tmp/tensorboard.log.
# Access at http://localhost:6006 in a browser.
mkdir -p /workspace/software/runs
nohup /opt/rl-env/bin/tensorboard \
    --logdir /workspace/software \
    --host 0.0.0.0 \
    --port 6006 \
    --reload_interval 5 \
    &> /tmp/tensorboard.log &

echo "TensorBoard started on port 6006 (logs: /tmp/tensorboard.log)"

exec /init