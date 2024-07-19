"""
This script shows how we evaluated a finetuned Octo model on a real WidowX robot. While the exact specifics may not
be applicable to your use case, this script serves as a didactic example of how to use Octo in a real-world setting.

If you wish, you may reproduce these results by [reproducing the robot setup](https://rail-berkeley.github.io/bridgedata/)
and installing [the robot controller](https://github.com/rail-berkeley/bridge_data_robot)
"""

from datetime import datetime
from functools import partial
import os
import time

from absl import app, flags, logging
import click
import cv2
# from envs.widowx_env import convert_obs, state_to_eep, wait_for_obs, WidowXGym
import imageio
import jax
import jax.numpy as jnp
import numpy as np
# from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, TemporalEnsembleWrapper
from octo.utils.train_callbacks import supply_rng

#============================================================add kuavo lib
from kuavoRobotSDK import kuavo
from dynamic_biped.msg import robotArmInfo
from dynamic_biped.srv import controlEndHand, controlEndHandRequest
from sensor_msgs.msg import JointState
from cam import camera_init,get_rgb

import math
import rospy
from typing import List
import pyrealsense2 as rs
from collections import deque
from dynamic_biped.msg import robotHandPosition        # 机械臂关节的角度和信息
#============================================================
'''

python 04_eval_finetuned_on_kuavo.py  --checkpoint_weights_path=/home/rebot801/hx/experiment_20240716_221638  --checkpoint_step=5000 --im_size=224 --video_save_path=./ 
'''
np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_weights_path", None, "Path to checkpoint", required=True
)
flags.DEFINE_integer("checkpoint_step", None, "Checkpoint step", required=True)

# custom to bridge_data_robot

flags.DEFINE_spaceseplist("goal_eep", [0.3, 0.0, 0.15], "Goal position")
flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.15], "Initial position")
flags.DEFINE_spaceseplist("initial_joint", [2.0486755515625, -1.2131204031047327, -36.75386170704427, -0.4043464679531559, 0.4701775993612256, -8.92914888420224, 1.650684011426176], "Initial joint state")

flags.DEFINE_bool("blocking", False, "Use the blocking controller")


flags.DEFINE_integer("im_size", 256, "Image size", required=False)
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_integer("num_timesteps", 1200, "num timesteps")
flags.DEFINE_integer("window_size", 2, "Observation history length")
flags.DEFINE_integer(
    "action_horizon", 64, "Length of action sequence to execute/ensemble"
)


# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

##############################################################################

STEP_DURATION_MESSAGE = """
Bridge data was collected with non-blocking control and a step duration of 0.2s.
However, we relabel the actions to make it look like the data was collected with
blocking control and we evaluate with blocking control.
Be sure to use a step duration of 0.2 if evaluating with non-blocking control.
"""
STEP_DURATION = 0.033
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[-180, -5, -90, -90, -90, -90, -90, -180, -135, -90, -90, -90, -90, -90], 
                    [30, 135, 90, 0, 90, 90, 90,  30, 5,  90, 0, 90, 90, 90]]
ARM_TOPICS = [{"name": "/robot_arm_q_v_tau"}]
ENV_PARAMS = {
    "robot_arm_q_v_tau": ARM_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}
WINDOW_SIZE=2
##############################################################################
state=None
gripper=None


def rad_to_angle(rad_list: list) -> list:
    """弧度转变为角度"""
    angle_list = [0 for _ in range(len(rad_list))]
    for i, rad in enumerate(rad_list):
        angle_list[i] = rad / math.pi * 180.0
    return angle_list   


def get_gripper_callback(msg):
    global gripper
    left_hand_position = msg.left_hand_position
    right_hand_position = msg.right_hand_position
    hand_position = left_hand_position + right_hand_position
    gripper=str(list(hand_position))
    

def call_control_end_hand_service(left_hand_position: List[float], right_hand_position: List[float]):
        """控制爪子开合的服务"""
        hand_positions = controlEndHandRequest()
        hand_positions.left_hand_position = left_hand_position  # 左手位置
        hand_positions.right_hand_position = right_hand_position  # 右手位置

        try:
            rospy.wait_for_service('/control_end_hand')
            control_end_hand = rospy.ServiceProxy('/control_end_hand', controlEndHand)
            resp = control_end_hand(hand_positions)
            return resp.result

        except rospy.ROSException as e:

            rospy.logerr("Service call failed: %s" % e)
            return False
        
def get_info(states_history,images_history):
    global gripper
    obs=dict()
    # state=rad_to_angle(robot_instance.latest_RobotstatePosition)
    state=np.random.rand(14)
    gripper01=0
    if gripper=="[0, 0, 0, 0, 0, 0, 0, 70, 20, 20, 20, 20]":
        gripper01=0
    elif gripper=="[0, 0, 0, 0, 0, 0, 0, 30, 80, 80, 80, 80]":
        gripper01=1

    l_hand = state[7:14]
    state = np.append(l_hand, gripper01)

    all_proprio = [state] + list(states_history)
    obs['proprio'] = np.stack(all_proprio)

    
    frames = pipeline2.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    current_image=cv2.resize(np.array(color_image),(256,256))

    all_images = [current_image] + list(images_history)
    obs['image_primary'] = np.stack(all_images)


    obs['timestep_pad_mask']=np.full((obs['image_primary'].shape[0]), True, dtype=bool)

    # print(obs.keys())
    return obs

def main(_):
    
    if FLAGS.initial_joint is not None:
        assert isinstance(FLAGS.initial_joint, list)
        initial_joint = [float(e) for e in FLAGS.initial_eep]
        start_state = np.concatenate([initial_joint])
    else:
        start_state = None

    if not FLAGS.blocking:
        assert STEP_DURATION == 0.033, STEP_DURATION_MESSAGE

    # load models
    print("loading model......")
    model = OctoModel.load_pretrained(
        FLAGS.checkpoint_weights_path,
        FLAGS.checkpoint_step,
    )

    # wrap the robot environment
    
    # create policy functions
    def sample_actions(
        pretrained_model: OctoModel,
        observations,
        tasks,
        rng,
    ):
        # add batch dim to observations
        observations = jax.tree_map(lambda x: x[None], observations)
        actions = pretrained_model.sample_actions(
            observations,
            tasks,
            rng=rng,
            unnormalization_statistics=pretrained_model.dataset_statistics["action"],
        )
        # remove batch dim
        return actions[0]

    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
            # argmax=FLAGS.deterministic,
            # temperature=FLAGS.temperature,
        )
    )

    goal_image = jnp.zeros((FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
    goal_instruction = ""
    # goal sampling loop
    while True:
        # modality = click.prompt(
        #     "Language or goal image?", type=click.Choice(["l", "g"])
        # )
        modality="l"
        if modality == "g":
            pass
        #     if click.confirm("Take a new goal?", default=True):
        #         assert isinstance(FLAGS.goal_eep, list)
        #         _eep = [float(e) for e in FLAGS.goal_eep]
        #         goal_eep = state_to_eep(_eep, 0)
        #         widowx_client.move_gripper(1.0)  # open gripper

        #         move_status = None
        #         while move_status != WidowXStatus.SUCCESS:
        #             move_status = widowx_client.move(goal_eep, duration=1.5)

        #         input("Press [Enter] when ready for taking the goal image. ")
        #         obs = wait_for_obs(widowx_client)
        #         obs = convert_obs(obs, FLAGS.im_size)
        #         goal = jax.tree_map(lambda x: x[None], obs)

        #     # Format task for the model
        #     task = model.create_tasks(goals=goal)
        #     # For logging purposes
        #     goal_image = goal["image_primary"][0]
        #     goal_instruction = ""

        elif modality == "l":
            print("Current instruction: ", goal_instruction)
            text="Grab the bottle and put it in the blue box"
            if click.confirm("Take a new instruction?", default=True):
                text = input("Instruction?")
            # Format task for the model
            task = model.create_tasks(texts=[text])
            # For logging purposes
            goal_instruction = text
            goal_image = jnp.zeros_like(goal_image)
        else:
            raise NotImplementedError()

        # input("Press s[Enter] to start.")

        # go to start state
        

        # reset env
        text="Grab the bottle and put it in the blue box"
        task = model.create_tasks(texts=[text])


        # do rollout
        last_tstep = time.time()
        
        ts = 0
        states_history = deque(maxlen=WINDOW_SIZE-1)
        images_history = deque(maxlen=WINDOW_SIZE-1)
        while ts < FLAGS.num_timesteps:
            if time.time() > last_tstep + STEP_DURATION:
                last_tstep = time.time()
                
                obs=get_info(states_history,images_history)
                
                # save images
                states_history.append(obs["proprio"][0])
                images_history.append(obs["image_primary"][0])

                if FLAGS.show_image:
                    cv2.imshow("img_view", obs["image_primary"][0])
                    cv2.waitKey(2)

                # get action
                forward_pass_time = time.time()
                action = np.array(policy_fn(obs, task), dtype=np.float64)
                # print("forward pass time: ", time.time() - forward_pass_time) #0.04s

                select_action=action[10]
                print(select_action)
                all_joint= [0] * 14
                all_joint[7:14]=select_action[:7]

                # perform environment step
                start_time = time.time()
                #=========================================================
                # robot_instance.set_arm_traj_position(all_joint)

                # grap=action[7]
                # if grap>0.5:
                #     call_control_end_hand_service(left_hand_position=[0, 0, 0, 0, 0, 0], right_hand_position=[0, 30, 80, 80, 80, 80])
                # else:
                #     call_control_end_hand_service(left_hand_position=[0, 0, 0, 0, 0, 0], right_hand_position=[0, 70, 20, 20, 20, 20])

                #=========================================================
                print("step time: ", time.time() - start_time)
                time.sleep(0.1)
                ts += 1
            # break
               

        # save video
        # if FLAGS.video_save_path is not None:
        #     os.makedirs(FLAGS.video_save_path, exist_ok=True)
        #     curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #     save_path = os.path.join(
        #         FLAGS.video_save_path,
        #         f"{curr_time}.mp4",
        #     )
        #     video = np.concatenate([np.stack(goals), np.stack(images)], axis=1)
        #     imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)


if __name__ == "__main__":
    # rospy.init_node('demo_test')
    # robot_instance = kuavo("3_7_kuavo")
    # rospy.Subscriber("/robot_hand_position", robotHandPosition, get_gripper_callback)
    pipeline = rs.pipeline()
    pipeline1,pipeline2=camera_init()
    
   
    app.run(main)
