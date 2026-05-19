import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import TiledCameraCfg
from omni.isaac.lab.utils import configclass

@configclass
class FrankaManipulationSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0, 
            color=(0.75, 0.75, 0.75)
        ),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="${ISAAC_NVIDIA_ASSET_DIR}/Robots/Franka/franka_panda.usd",
            activate_navigation_corridor=False,
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
                "panda_joint6": 2.241,
                "panda_joint7": 0.785,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
            joint_vel={".*": 0.0},
        ),
    )

    object: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04), 
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.1, 0.1), 
                metallic=0.1,
                roughness=0.5,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                mass=0.1,
                max_depenetration_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.02), 
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    franka_wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_camera",
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