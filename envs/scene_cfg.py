"""
Franka manipulation scene configuration.

VALIDATED against Isaac Lab official docs (main, June 2026):
- PushCubeSceneCfg INHERITS from FrankaManipulationSceneCfg (fixes the crash)
  @configclass converts attributes to instance fields — you cannot copy them
  as class attributes. Inheritance is the documented pattern.
- Franka loaded via FRANKA_PANDA_CFG (tries pip path first, falls back to thirdparty)
- RigidObjectCfg for physics-interactive objects
- No 'activate_navigation_corridor' param (does not exist in UsdFileCfg)
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

# ── Franka pre-built config — try pip path first, fall back to thirdparty ─────
try:
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
except ImportError:
    try:
        from thirdparty.Isaaclab.source.isaaclab_assets.isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
    except ImportError:
        # Final fallback: build a minimal Franka config manually
        FRANKA_PANDA_CFG = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path="${ISAAC_ASSETS_PATH}/Robots/Franka/franka_panda.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=10.0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.569,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.810,
                    "panda_joint5": 0.0,
                    "panda_joint6":  2.241,
                    "panda_joint7": 0.785,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                },
                joint_vel={".*": 0.0},
            ),
        )


@configclass
class FrankaManipulationSceneCfg(InteractiveSceneCfg):
    """Base Franka scene: ground, lighting, robot, target cube, wrist camera."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # Franka Panda — pattern from official docs: CFG.replace(prim_path=...)
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # Target object — 4 cm red cube (physics-interactive)
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.1, 0.1), metallic=0.1, roughness=0.5,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.02),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Wrist-mounted TiledCamera — 84x84 RGB
    # TiledCamera is the only tractable rendering mode for pixel RL at scale
    franka_wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link7/tiled_camera",
        update_period=0.0,
        height=84,
        width=84,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 100.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.03),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )


@configclass
class ReachSceneCfg(FrankaManipulationSceneCfg):
    """Reach task: object is a visual-only target, no physics collisions."""

    def __post_init__(self):
        super().__post_init__()
        if hasattr(self.object.spawn, "collision_props"):
            self.object.spawn.collision_props = None


@configclass
class PushCubeSceneCfg(FrankaManipulationSceneCfg):
    """
    PushCube task: adds a table; cube rests on it.

    INHERITS from FrankaManipulationSceneCfg — NOT InteractiveSceneCfg.
    @configclass converts attributes to instance fields so you cannot access them
    as class attributes (FrankaManipulationSceneCfg.ground crashes).
    Inheritance automatically brings ground/light/robot/camera into this scene.
    """

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.8, 0.4),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.3, 0.3)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.2)),
    )

    def __post_init__(self):
        super().__post_init__()
        # Override inherited object with a green pushable cube on the table
        self.object.spawn.size = (0.05, 0.05, 0.05)
        self.object.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.1, 0.8, 0.1)
        )
        self.object.init_state.pos = (0.5, 0.0, 0.425)
