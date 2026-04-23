#!/bin/bash
set -e

LOG=/tmp/xfce-config.log
BASE="/backdrop/screen0/monitorselkies-primary/workspace0"

#-------------------------------------------------------------------------------
# Set desktop background to a solid color

{
  echo "starting xfce-config.sh"
  echo "DISPLAY=$DISPLAY"
  echo "DBUS_SESSION_BUS_ADDRESS=$DBUS_SESSION_BUS_ADDRESS"

  xfconf-query -c xfce4-desktop -p /backdrop/single-workspace-mode -n -t bool -s true || true
  xfconf-query -c xfce4-desktop -p "${BASE}/image-style" -n -t int -s 0 || true
  xfconf-query -c xfce4-desktop -p "${BASE}/color-style" -n -t int -s 0 || true

  xfconf-query -c xfce4-desktop -p "${BASE}/rgba1" -r || true

  # Set background color to #080623
  xfconf-query -c xfce4-desktop -p "${BASE}/rgba1" \
    -n -a \
    -t double -s 0.031373 \
    -t double -s 0.023529 \
    -t double -s 0.137255 \
    -t double -s 1 || true

  xfdesktop --reload || true

  echo "final backdrop settings:"
  xfconf-query -c xfce4-desktop -lv | grep "${BASE}" || true
  xfconf-query -c xfce4-desktop -lv | grep /backdrop/single-workspace-mode || true
} >> "$LOG" 2>&1

#-------------------------------------------------------------------------------
# Desktop customization

# Copy in desktop icons
mkdir -p /config/Desktop
cp /tmp/desktop-defaults/terminal.desktop   /config/Desktop/terminal.desktop
cp /tmp/desktop-defaults/jupyterlab.desktop /config/Desktop/jupyterlab.desktop
chmod +x /config/Desktop/*.desktop

# Show icons from ~/Desktop only
xfconf-query -c xfce4-desktop -p /desktop-icons/style -s 2 || true