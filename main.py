from UAVEnvironment import UAVEnvironment
from UAVAgent import UAVAgent,ReplayMemory
import rospy
from collections import namedtuple
import torch
from gazebo_msgs.msg import ContactsState
import numpy as np
def collision_callback(msg):
    global collision
    if len(msg.states) > 0:
        print("Çarpışma tespit edildi!")
        collision = True


def main():
    rospy.Subscriber('/gazebo/link_states', ContactsState, collision_callback)

    global collision
    env = UAVEnvironment()

    env.takeoff()
    
    
    plane = UAVAgent()
    total_reward = 0
    episodes = 1000
    for e in range(episodes):
        
        print("Episode: {}/{}, Total Reward: {}".format(e , episodes, total_reward))
        total_reward = 0
        collision = False
        state = env.get_state()
        
        for step in range(env.max_steps):
            action = plane.findAction(state)
            next_state, reward, done = env.step(action)
            # print("State:",state)
            # print("Action",action)
            # print("Episode ",e,"Step ",step)
            plane.remember(state, action, reward, next_state,done)
            plane.replay()
            state = next_state

            total_reward += reward
            if next_state[2] < 0.5 or abs(next_state[0]) > 100 or abs(next_state[1]) > 100 or abs(next_state[2]) > 100:
                break
        env.clear()
            
        #     if done:
        #         break
        
        # if e % 10 == 0:
        #     agent.update_target_model()  # Hedef modeli güncelle
        #     if done:
                
        #         print("Episode: {}/{}, Total Reward: {}".format(e + 1, episodes, total_reward))
        #         break

if __name__ == "__main__":
    rospy.init_node('mc_node', anonymous=True)
    main()