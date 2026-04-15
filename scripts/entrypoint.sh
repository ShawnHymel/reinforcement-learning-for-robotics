#!/bin/bash

# Activates the Python virtual environment for all shell sessions, then
# hands off to the linuxserver/webtop init process (s6-overlay), which
# starts the XFCE desktop, noVNC, and all other Webtop services.

source /opt/rl-env/bin/activate

exec /init