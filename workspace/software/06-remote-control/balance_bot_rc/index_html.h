/**
 * HTML web page with thumbstick
 *
 * JavaScript interactive thumbstick that sends commands back to the ESP32 via
 * WebSocket.
 */

static const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
  <title>BalanceBot</title>
  <style>

    /* Remove browser default styling */
    * { box-sizing: border-box; margin: 0; padding: 0; }

    /* Plain background */
    body {
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      background: #fff;
      user-select: none;
    }

    /* Status (connected) styling */
    #status { font-family: monospace; margin-bottom: 1rem; }

    /* Outer circle for thumbstick */
    #stick-zone {
      position: relative;
      width: 220px;
      height: 220px;
      border-radius: 50%;
      border: 2px solid #000;
      background: #544a49;
      touch-action: none;
      cursor: grab;
    }

    /* Define thumbstick area action */
    #stick-zone:active { cursor: grabbing; }

    /* Smaller circle for thumbstick */
    #knob {
      position: absolute;
      width: 72px;
      height: 72px;
      border-radius: 50%;
      border: 2px solid #000;
      background: #b73324;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      pointer-events: none;
    }
  </style>
</head>

<!-- Page is just status text and the thumbstick -->
<body>
<p id="status">not connected</p>
<div id="stick-zone">
  <div id="knob"></div>
</div>

<!-- JavaScript -->
<script>
  // Get elements
  const zone      = document.getElementById('stick-zone');
  const knob      = document.getElementById('knob');
  const statusEl  = document.getElementById('status');

  // Settings
  const ZONE_R = 110;
  const KNOB_R = 36;
  const MAX_R  = ZONE_R - KNOB_R;

  // Globals
  let active = false;
  let cmdVel = 0, cmdYaw = 0;
  let sendInterval = null;
  let ws;

  // Connect WebSocket back to ESP32
  function connect() {
    ws = new WebSocket('ws://' + location.host + '/ws');

    ws.onopen = () => {
      statusEl.textContent = 'connected';
      sendInterval = setInterval(sendCmd, 50);
    };

    ws.onclose = ws.onerror = () => {
      statusEl.textContent = 'not connected';
      clearInterval(sendInterval);
      cmdVel = cmdYaw = 0;
      setTimeout(connect, 1000);
    };
  }

  // Send thumbstick commands over websocket
  function sendCmd() {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ vel: cmdVel, yaw: cmdYaw }));
    }
  }

  // Connect WebSocket
  connect();

  // Get X and Y of thumbstick center
  function zoneCenter() {
    const r = zone.getBoundingClientRect();
    return { x: r.left + ZONE_R, y: r.top + ZONE_R };
  }

  // Draw knob based on touch location
  function updateKnob(clientX, clientY) {
    const c = zoneCenter();
    let dx = clientX - c.x;
    let dy = clientY - c.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist > MAX_R) { dx = dx / dist * MAX_R; dy = dy / dist * MAX_R; }
    knob.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;
    cmdYaw =  parseFloat((dx / MAX_R).toFixed(3));
    cmdVel = -parseFloat((dy / MAX_R).toFixed(3));
  }

  // Reset thumbstick to 0, 0 if released
  function release() {
    if (!active) return;
    active = false;
    knob.style.transform = 'translate(-50%, -50%)';
    cmdVel = cmdYaw = 0;
  }

  // Add listener if mouse clicks on thumbstick
  zone.addEventListener('mousedown', e => { active = true; updateKnob(e.clientX, e.clientY); });
  window.addEventListener('mousemove', e => { if (active) updateKnob(e.clientX, e.clientY); });
  window.addEventListener('mouseup', release);

  // Add listener if user touches thumbstick
  zone.addEventListener('touchstart', e => { e.preventDefault(); active = true; updateKnob(e.touches[0].clientX, e.touches[0].clientY); }, { passive: false });
  window.addEventListener('touchmove', e => { if (active) updateKnob(e.touches[0].clientX, e.touches[0].clientY); }, { passive: false });
  window.addEventListener('touchend', release);
</script>
</body>
</html>
)rawliteral";