import cv2
import jax
import tensorflow_datasets as tfds
import tqdm
import mediapy
import numpy as np
import os
os.environ['PATH'] = '/usr/local/cuda-12.2/bin:' + os.environ['PATH']
from octo.model.octo_model import OctoModel
import os
current_dir=os.path.dirname(os.path.abspath(__file__))
ckpt_path=os.path.join(current_dir,'../ckpt/octo_finetune_pureBg/pureBg_modifyProrioToker2_20240728_040631')
step=None
model = OctoModel.load_pretrained(ckpt_path,step)
model=model



# create RLDS dataset builder
builder = tfds.builder_from_directory(builder_dir='/home/octo/hx/dataset/rlds/tfds_pure_bg2/example_dataset/1.0.0')
ds = builder.as_dataset(split='train[400:402]')
print(len(ds))
# sample episode + resize to 256x256 (default third-person cam resolution)
episode = next(iter(ds))
steps = list(episode['steps'])
images01 = [cv2.resize(np.array(step['observation']['image01']), (256, 256)) for step in steps]
images02 = [cv2.resize(np.array(step['observation']['image02']), (128, 128)) for step in steps]
states = [np.array(step['observation']['state']) for step in steps]

# extract goal image & language instruction
goal_image = images01[-1]
goal_image02 = images02[-1]
language_instruction = steps[0]['language_instruction'].numpy().decode()

# visualize episode
print(f'Instruction: {language_instruction}')
# mediapy.show_video(images01, fps=20)




WINDOW_SIZE = 2

# create `task` dict
# task = model.create_tasks(goals={"image_primary": goal_image[None]})   # for goal-conditioned
task = model.create_tasks(texts=[language_instruction],goals={"image_primary": goal_image[None],"image_secondary":goal_image02[None]})                  # for language conditioned



# run inference loop, this model only uses 3rd person image observations for bridge
# collect predicted and true actions
pred_actions, true_actions ,true_states= [], [], []
for step in tqdm.trange(len(images01) - (WINDOW_SIZE - 1)):
    input_images01 = np.stack(images01[step:step+WINDOW_SIZE])[None]
    input_images02 = np.stack(images02[step:step+WINDOW_SIZE])[None]
    input_states = np.stack(states[step:step+WINDOW_SIZE])[None]
    observation = {
        'proprio':input_states,
        'image_primary': input_images01,
        'image_secondary': input_images02,
        'timestep_pad_mask': np.full((1, input_images01.shape[1]), True, dtype=bool)
    }
    
    # this returns *normalized* actions --> we need to unnormalize using the dataset statistics
    actions = model.sample_actions(
        observation, 
        task, 
        unnormalization_statistics=model.dataset_statistics["action"], 
        rng=jax.random.PRNGKey(0)
    )
    actions = actions[0] # remove batch dim
    pred_actions.append(actions)
    final_window_step = step + WINDOW_SIZE - 1
    true_actions.append(np.array(steps[final_window_step]["action"]))
    true_states.append(np.array(steps[final_window_step]["observation"]["state"]))
    # break


import matplotlib.pyplot as plt

ACTION_DIM_LABELS = ['1', '2', '3', '4', '5', '6', '7','grasp']

# build image strip to show above actions
img_strip = np.concatenate(np.array(images01[::10]), axis=1)

# set up plt figure
figure_layout = [
    ['image'] * len(ACTION_DIM_LABELS),
    ACTION_DIM_LABELS
]

plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplot_mosaic(figure_layout)
fig.set_size_inches([45, 10])

# plot actions
pred_actions = np.array(pred_actions).squeeze()
true_actions = np.array(true_actions).squeeze()
true_states=np.array(true_states).squeeze()
for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
  # actions have batch, horizon, dim, in this example we just take the first action for simplicity
  axs[action_label].plot(pred_actions[:, 0, action_dim], label='predicted action')
  axs[action_label].plot(true_actions[:, action_dim], label='ground truth')
  # axs[action_label].plot(true_states[:, action_dim], label='ground truth state')
  axs[action_label].set_title(action_label)
  axs[action_label].set_xlabel('Time in one episode')

axs['image'].imshow(img_strip)
axs['image'].set_xlabel('Time in one episode (subsampled)')
plt.legend()