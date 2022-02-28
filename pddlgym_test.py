import pddlgym
import imageio

if __name__ == '__main__':
    env = pddlgym.make("PDDLEnvSokoban-v0")
    obs, debug_info = env.reset()
    img = env.render()
    imageio.imsave("frame1.png", img)
    action = env.action_space.sample(obs)
    obs, reward, done, debug_info = env.step(action)
    img = env.render()
    imageio.imsave("frame2.png", img)