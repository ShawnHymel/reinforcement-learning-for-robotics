import mujoco
from pathlib import Path

MJCF_PATH = "/workspace/mechanical/FreeCAD/bala2-fire/bala2-fire-simplified.xml"

model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))

print("qpos components:")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    adr  = model.jnt_qposadr[i]
    typ  = model.jnt_type[i]
    print(f"  Joint {i}: name='{name}', qposadr={adr}, type={typ}")

print("qvel components:")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    adr  = model.jnt_dofadr[i]   # dofadr for qvel, not qposadr
    print(f"  Joint {i}: name='{name}', qveladr={adr}")