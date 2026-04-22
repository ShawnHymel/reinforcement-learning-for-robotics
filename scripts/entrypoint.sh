#!/bin/bash

# Clear the xfdesktop icon layout cache before the desktop starts.
# Without this, stale entries in icons.screen0.yaml from a previous
# container run cause duplicate desktop icons on subsequent starts.
rm -f /config/.config/xfce4/desktop/icons.screen0.yaml

# Activates the Python virtual environment for all shell sessions, then
# hands off to the linuxserver/webtop init process (s6-overlay), which
# starts the XFCE desktop, noVNC, and all other Webtop services.
source /opt/rl-env/bin/activate

exec /init