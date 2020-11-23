import matplotlib.pyplot as plt
import sys
sys.path.append(".")
import numpy as np
import pyglet
import math
from scipy.optimize import linear_sum_assignment
from array import array
import random
import energy

class Env(object):
    viewer = None
    region_x = 600
    region_y = 600
    region_xy = np.sqrt(region_x ** 2 + region_y ** 2)
    dt = 1  # refresh rate
    uav_acceleration_x_bound = [-1, 1]
    uav_acceleration_y_bound = [-1, 1]
    uav_speed_x_bound = [-2, 2]
    uav_speed_y_bound = [-2, 2]
    goal_acceleration_x_bound = [-0.1, 0.1]
    goal_acceleration_y_bound = [-0.1, 0.1]
    goal_speed_x_bound = [-0.2, 0.2]
    goal_speed_y_bound = [-0.2, 0.2]

    #wireless_working = 1    #正在为地面提供网络服务的UAV
    #standby_charge = 2      #在充电桩待机的UAV
    #returning_charge = 3    #在返回充电桩路上的UAV
    #startfrom_charge = 4    #从充电桩出发的UAV
    #request_service = 5     #UAV发出请求替代

    def __init__(self, wireless_working_num, standby_charge_num, charging_num, user_goal_num):
        self.uav_num = wireless_working_num + standby_charge_num
        self.uav_infos = np.zeros(self.uav_num, dtype=[('working_state', np.int), ('speed_x', np.float32), ('speed_y', np.float32), ('position_x', np.float32), ('position_y', np.float32), ('acceleration_x', np.float32), ('acceleration_y', np.float32), ('energy', np.float32)])
        self.charging_infos = np.zeros(charging_num, dtype=[('position_x', np.float32), ('position_y', np.float32)])
        self.user_goal_infos = np.zeros(user_goal_num, dtype=[('speed_x', np.float32), ('speed_y', np.float32), ('position_x', np.float32), ('position_y', np.float32)])
        self.wireless_working_num = wireless_working_num
        self.out_of_power = -1000
        self.max_energy_util = 0.001
    
    def obslist_i(self, i):
        return np.array([self.uav_infos[i]['acceleration_x'],self.uav_infos[i]['acceleration_y'],self.uav_infos[i]['speed_x'],self.uav_infos[i]['speed_y'],self.uav_infos[i]['position_x'],self.uav_infos[i]['position_y'],self.uav_infos[i]['position_x']-self.uav_infos[int(abs(i-1))]['position_x'],self.uav_infos[i]['position_y']-self.uav_infos[int(abs(i-1))]['position_y'],self.uav_infos[i]['position_x']-self.user_goal_infos[0]['position_x'],self.uav_infos[i]['position_y']-self.user_goal_infos[0]['position_y'],self.uav_infos[i]['working_state'],self.uav_infos[i]['energy']])

    def reset(self):
        obslist = []
        id_c = 0
        uav_sequence = np.arange(self.uav_num)
        np.random.shuffle(uav_sequence)
        self.pre_action = []
        self.pre_request_service = []
        self.pre_ask_return = []

        for u in range(np.size(self.uav_infos)):
            if u < self.wireless_working_num:
                self.uav_infos[uav_sequence[u]]['speed_x'] = 0.
                self.uav_infos[uav_sequence[u]]['speed_y'] = 0.
                self.uav_infos[uav_sequence[u]]['acceleration_x'] = 0.
                self.uav_infos[uav_sequence[u]]['acceleration_y'] = 0.
                self.uav_infos[uav_sequence[u]]['position_x'] = np.random.uniform(0, self.region_x)
                self.uav_infos[uav_sequence[u]]['position_y'] = np.random.uniform(0, self.region_y)
                self.uav_infos[uav_sequence[u]]['working_state'] = 1
                self.uav_infos[uav_sequence[u]]['energy'] = np.random.uniform(200,600)
            else:
                self.uav_infos[uav_sequence[u]]['speed_x'] = 0.
                self.uav_infos[uav_sequence[u]]['speed_y'] = 0.
                self.uav_infos[uav_sequence[u]]['acceleration_x'] = 0.
                self.uav_infos[uav_sequence[u]]['acceleration_y'] = 0.
                self.uav_infos[uav_sequence[u]]['position_x'] = self.charging_infos[id_c]['position_x'] = np.random.uniform(0, self.region_x)
                self.uav_infos[uav_sequence[u]]['position_y'] = self.charging_infos[id_c]['position_y'] = np.random.uniform(0, self.region_y)
                self.uav_infos[uav_sequence[u]]['working_state'] = 2
                self.uav_infos[uav_sequence[u]]['energy'] = 1000
                id_c = id_c +1

        for g in self.user_goal_infos:
            g['speed_x'] = 0.
            g['speed_y'] = 0.
            g['position_x'] = np.random.uniform(0, self.region_x)
            g['position_y'] = np.random.uniform(0, self.region_y)

        for i in range(np.size(self.uav_infos)):
            obslist.append(self.obslist_i(i))
            self.pre_action.append([self.uav_infos[i]['acceleration_x'], self.uav_infos[i]['acceleration_y'], self.uav_infos[i]['working_state']])            
        return obslist

    def step(self, actions):
        self.goal_new_state()
        game_over = 0
        done = []
        reward = []
        obslist = []
        next_actions = []
        request_service = []
        actions = actions.reshape((int(actions.size/3), 3))

        for i in range(np.size(self.uav_infos)):
            if self.uav_infos[i]['working_state'] == 1 or 3 or 4 or 5:
                v = np.sqrt(self.uav_infos[i]['speed_x']**2+self.uav_infos[i]['speed_y']**2)
                energy_c = energy.energy_consumption(v)
                self.uav_infos[i]['energy'] = self.uav_infos[i]['energy'] - energy_c
                e = energy.energy_distance_estimation(v, self.dt, energy_c)
                if e>self.max_energy_util:
                    self.max_energy_util = e

        #能量耗尽，system崩溃结束。
        for i in range(len(actions)):
            if self.uav_infos[i]['energy']<=0:
                game_over = 1
                break 
        if game_over == 1:
            for i in range(np.size(self.uav_infos)):
                obslist.append(self.obslist_i(i))
                reward.append(self.out_of_power)
                done.append(True)
            return obslist, reward, done
        
        goal_x = self.user_goal_infos[0]['position_x']
        goal_y = self.user_goal_infos[0]['position_y']
        for i in range (len(actions)):
            if self.pre_action[i][2] == 1:
                pre_d = np.sqrt((self.uav_infos[i]['position_x']-goal_x)**2+(self.uav_infos[i]['position_y']-goal_y)**2)
                self.uav_new_position(actions, i)
                d = np.sqrt((self.uav_infos[i]['position_x']-goal_x)**2 + (self.uav_infos[i]['position_y'] -goal_y)**2)
                d_charging = np.sqrt((self.uav_infos[i]['position_x']-self.charging_infos[0]['position_x'])**2 + (self.uav_infos[i]['position_y'] -self.charging_infos[0]['position_y'])**2)
                min_energy = 2*d_charging/self.max_energy_util
                done.append(False)
                if actions[i][2]>=0:    #1->1
                    if min_energy<self.uav_infos[i]['energy']:
                        if min_energy+20<self.uav_infos[i]['energy']:
                            reward.append((pre_d-d)*100)
                        else:
                            reward.append((pre_d-d)*10+(self.uav_infos[i]['energy']-20-min_energy)*100)
                    else:
                        reward.append((pre_d-d)+(self.uav_infos[i]['energy']-min_energy)*5)
                    self.uav_infos[i]['working_state'] = 1
                else:   #1->5
                    if min_energy+200<self.uav_infos[i]['energy']:
                        reward.append(-100)
                        self.uav_infos[i]['working_state'] = 1
                    else:
                        if min_energy+40<self.uav_infos[i]['energy']:
                            reward.append((pre_d-d)*10-0.1*(self.uav_infos[i]['energy']-40-min_energy))
                        else:
                            reward.append((pre_d-d)*10-(self.uav_infos[i]['energy']-40-min_energy))
                        self.uav_infos[i]['working_state'] = 5
                        request_service.append(i)
                
            elif self.pre_action[i][2] == 2:
                self.uav_new_position(actions, i)
                done.append(False)
                if actions[i][2]>=0:    #2->2
                    self.uav_infos[i]['working_state'] = 2
                    if len(self.pre_request_service)>0:
                        reward.append(-20)
                    else:
                        reward.append(20)
                else:   #2->4
                    if len(self.pre_request_service)>0:
                        self.uav_infos[i]['working_state'] = 4
                        reward.append(20)
                        request_service = []
                    else:
                        self.uav_infos[i]['working_state'] = 2
                        reward.append(-100)
                
                a = 1

            elif self.pre_action[i][2] == 3:
                done.append(False)
                pre_d = np.sqrt((self.uav_infos[i]['position_x']-self.charging_infos[0]['position_x'])**2+(self.uav_infos[i]['position_y']-self.charging_infos[0]['position_y'])**2)
                self.uav_new_position(actions, i)
                d = np.sqrt((self.uav_infos[i]['position_x']-self.charging_infos[0]['position_x'])**2+(self.uav_infos[i]['position_y']-self.charging_infos[0]['position_y'])**2)

                if actions[i][2]>=0:    #3->3
                    if d<10:
                        reward.append(-100)
                    else:
                        reward.append((pre_d-d)*100)
                    self.uav_infos[i]['working_state'] = 3
                else:   #3->2
                    if d<10:
                        self.uav_infos[i]['working_state'] = 2
                        if self.uav_infos[i]['energy']>10:
                            reward.append((10-self.uav_infos[i]['energy'])*10)
                        else:
                            reward.append(100)
                        self.uav_infos[i]['energy'] = 1000
                    else:
                        reward.append(-100)
                        self.uav_infos[i]['working_state'] = 3

            elif self.pre_action[i][2] == 4:
                done.append(False)
                pre_d = np.sqrt((self.uav_infos[i]['position_x']-self.user_goal_infos[0]['position_x'])**2+(self.uav_infos[i]['position_y']-self.user_goal_infos[0]['position_y'])**2)
                self.uav_new_position(actions, i)
                d = np.sqrt((self.uav_infos[i]['position_x']-self.user_goal_infos[0]['position_x'])**2+(self.uav_infos[i]['position_y']-self.user_goal_infos[0]['position_y'])**2)
                if actions[i][2]>=0:    #4->4
                    reward.append((pre_d-d)*100)
                    self.uav_infos[i]['working_state'] = 4
                else: #4->1
                    if d < 10:
                        reward.append(100)
                        self.uav_infos[i]['working_state'] = 1
                        self.pre_ask_return.append(i)
                    else: 
                        reward.append(-100)
                        self.uav_infos[i]['working_state'] = 4
            elif self.pre_action[i][2] == 5:
                done.append(False)
                pre_d = np.sqrt((self.uav_infos[i]['position_x']-goal_x)**2+(self.uav_infos[i]['position_y']-goal_y)**2)
                self.uav_new_position(actions, i)
                d = np.sqrt((self.uav_infos[i]['position_x']-goal_x)**2 + (self.uav_infos[i]['position_y'] -goal_y)**2)
                if actions[i][2]>=0: #5->5
                    self.uav_infos[i]['working_state'] = 5
                    request_service.append(i)
                    if len(self.pre_ask_return)>0:
                        reward.append(-100)
                    else:
                        reward.append((pre_d-d)*100)
                else:   #5->3
                    if len(self.pre_ask_return)>0:
                        reward.append(100)
                        self.uav_infos[i]['working_state'] = 3
                        self.pre_ask_return = []
                    else:
                        reward.append(-100)
                        request_service.append(i)
                        self.uav_infos[i]['working_state'] = 5
            
            next_actions.append([self.uav_infos[i]['acceleration_x'], self.uav_infos[i]['acceleration_y'], self.uav_infos[i]['working_state']])
        
        self.pre_action = next_actions   
        self.pre_request_service = request_service
        for i in range (len(actions)):
            obslist.append(self.obslist_i(i))
        return obslist, reward, done

    def goal_new_state(self):
        for g in self.user_goal_infos:
            goal_acceleration_x = np.random.uniform(np.min(self.goal_acceleration_x_bound), np.max(self.goal_acceleration_x_bound))
            goal_acceleration_y = np.random.uniform(np.min(self.goal_acceleration_y_bound), np.max(self.goal_acceleration_y_bound))
            goal_acceleration_x = np.clip(goal_acceleration_x, *self.goal_acceleration_x_bound)
            goal_acceleration_y = np.clip(goal_acceleration_y, *self.goal_acceleration_y_bound)
            (goal_speed_x, goal_speed_y) = (g['speed_x'], g['speed_y'])
            (goal_position_x, goal_position_y) = (g['position_x'], g['position_y'])
            goal_speed_x_ = goal_speed_x + goal_acceleration_x * self.dt
            goal_speed_y_ = goal_speed_y + goal_acceleration_y * self.dt
            goal_speed_x_ = np.clip(goal_speed_x_, *self.goal_speed_x_bound)
            goal_speed_y_ = np.clip(goal_speed_y_, *self.goal_speed_y_bound)
            goal_position_x_ = goal_position_x + goal_speed_x * self.dt + goal_acceleration_x * np.square(self.dt) / 2
            goal_position_y_ = goal_position_y + goal_speed_y * self.dt + goal_acceleration_y * np.square(self.dt) / 2
            if goal_position_x_ < 0:
                goal_position_x_ = 0
                goal_speed_x_ = 0
                goal_speed_y_ = 0
            if goal_position_x_ > self.region_x:
                goal_position_x_ = self.region_x
                goal_speed_x_ = 0
                goal_speed_y_ = 0
            if goal_position_y_ < 0:
                goal_position_y_ = 0
                goal_speed_x_ = 0
                goal_speed_y_ = 0
            if goal_position_y_ > self.region_y:
                goal_position_y_ = self.region_y
                goal_speed_x_ = 0
                goal_speed_y_ = 0
            g['speed_x'] = goal_speed_x_
            g['speed_y'] = goal_speed_y_
            g['position_x'] = goal_position_x_
            g['position_y'] = goal_position_y_
    
    def uav_new_position(self, actions, i):
        if self.uav_infos[i]['working_state'] == 2:
            self.uav_infos[i]['acceleration_x'] = 0
            self.uav_infos[i]['acceleration_y'] = 0
            self.uav_infos[i]['speed_x'] = 0
            self.uav_infos[i]['speed_y'] = 0
            self.uav_infos[i]['position_x'] = self.charging_infos[0]['position_x']
            self.uav_infos[i]['position_y'] = self.charging_infos[0]['position_y']
        else:
            self.uav_infos[i]['acceleration_x'] = actions[i][0]
            self.uav_infos[i]['acceleration_y'] = actions[i][1]
            uav_acceleration_x = np.clip(self.uav_infos[i]['acceleration_x'], *self.uav_acceleration_x_bound)
            uav_acceleration_y = np.clip(self.uav_infos[i]['acceleration_y'], *self.uav_acceleration_y_bound)
            self.uav_infos[i]['acceleration_x'] = uav_acceleration_x
            self.uav_infos[i]['acceleration_y'] = uav_acceleration_y
            (uav_speed_x, uav_speed_y) = (self.uav_infos[i]['speed_x'], self.uav_infos[i]['speed_y'])
            (uav_position_x, uav_position_y) = (self.uav_infos[i]['position_x'], self.uav_infos[i]['position_y'])
            uav_speed_x_ = uav_speed_x + self.uav_infos[i]['acceleration_x'] * self.dt
            uav_speed_y_ = uav_speed_y + self.uav_infos[i]['acceleration_y'] * self.dt
            if uav_speed_x_ > np.max(self.uav_speed_x_bound):
                uav_speed_x_ = np.max(self.uav_speed_x_bound)
            if uav_speed_x_ < np.min(self.uav_speed_x_bound):
                uav_speed_x_ = np.min(self.uav_speed_x_bound)
            if uav_speed_y_ > np.max(self.uav_speed_y_bound):
                uav_speed_y_ = np.max(self.uav_speed_y_bound)
            if uav_speed_y_ < np.min(self.uav_speed_y_bound):
                uav_speed_y_ = np.min(self.uav_speed_y_bound)
            uav_position_x_ = uav_position_x + uav_speed_x * self.dt + self.uav_infos[i]['acceleration_x'] * np.square(self.dt) / 2
            uav_position_y_ = uav_position_y + uav_speed_y * self.dt + self.uav_infos[i]['acceleration_y'] * np.square(self.dt) / 2
            self.uav_infos[i]['speed_x'] = uav_speed_x_
            self.uav_infos[i]['speed_y'] = uav_speed_y_
            self.uav_infos[i]['position_x'] = uav_position_x_
            self.uav_infos[i]['position_y'] = uav_position_y_


    def state_transfer_failed(self):
        done = []
        reward = []
        obslist = []
        done.append(True)
        reward.append(self.state_transfer_failed_reward)
        obslist.append(np.array([self.uav_infos[i]['speed_x'], self.uav_infos[i]['speed_y'], self.uav_infos[i]['acceleration_x'], self.uav_infos[i]['acceleration_y'], self.uav_infos[i]['position_x'], self.uav_infos[i]['position_y'], self.uav_infos[i]['working_state'], self.uav_infos[i]['energy']]))
        return done, reward, obslist


    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.uav_infos, self.charging_infos, self.user_goal_infos, self.region_x, self.region_y)
        self.viewer.render()

    def sample_action(self, uav_num):
        return [[np.random.uniform(-1, 1)] * (2 * uav_num)]
        #return [[np.random.uniform(-1, 1)] * 2 for i in range(uav_num)]


class Viewer(pyglet.window.Window):
    def __init__(self, uav_infos, charging_infos, user_goal_infos, x, y):
        self.uav_infos = uav_infos
        self.charging_infos = charging_infos
        self.user_goal_infos = user_goal_infos
        pyglet.resource.path = ['resources']
        pyglet.resource.reindex()
        uav = pyglet.resource.image('uav.png')
        charging = pyglet.resource.image('charging.png')
        user = pyglet.resource.image('user.png')
        radiation = pyglet.resource.image('radiation.png')
        super(Viewer, self).__init__(width=x, height=y, resizable=False, caption='Multi-Agent', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.batch = pyglet.graphics.Batch()  # display whole batch at once
        self.uavs = []
        self.uavs_radiation = []
        self.chargings = []
        self.user_goals = []
        for u in self.uav_infos:
            uav_sprite = pyglet.sprite.Sprite(img=uav, x=u['position_x'] - 17., y=u['position_y'] - 10.)
            radiation_sprite = pyglet.sprite.Sprite(img=radiation, x=u['position_x'] - 100., y=u['position_y'] - 100.)
            self.uavs.append(uav_sprite)
            self.uavs_radiation.append(radiation_sprite)
        for u in self.charging_infos:
            charging_sprite = pyglet.sprite.Sprite(img=charging, x=u['position_x'] - 20., y=u['position_y'] - 13.)
            self.chargings.append(charging_sprite)
        for u in self.user_goal_infos:
            user_sprite = pyglet.sprite.Sprite(img=user, x=u['position_x'] - 7., y=u['position_y'] - 7.)
            self.user_goals.append(user_sprite)

    def render(self):
        self._update_charging()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        for uav in self.uavs:
            uav.draw()
        for uavs_radiation in self.uavs_radiation:
            uavs_radiation.draw()
        for charging in self.chargings:
            charging.draw()
        for user_goal in self.user_goals:
            user_goal.draw()

    def _update_charging(self):
        for i in range(np.size(self.charging_infos)):
            self.chargings[i].update(x=(self.charging_infos[i])['position_x'] - 20., y=(self.charging_infos[i])['position_y'] - 13.)
        for i in range(np.size(self.uav_infos)):
            self.uavs[i].update(x=(self.uav_infos[i])['position_x'] - 17., y=(self.uav_infos[i])['position_y'] - 10.)
            self.uavs_radiation[i].update(x=(self.uav_infos[i])['position_x'] - 100., y=(self.uav_infos[i])['position_y'] - 100.)
        for i in range(np.size(self.user_goal_infos)):
            self.user_goals[i].update(x=(self.user_goal_infos[i])['position_x'] - 7., y=(self.user_goal_infos[i])['position_y'] - 7.)

if __name__ == '__main__':
    env = Env(2, 2, 2, 4)
    obslist = env.reset()
    actiontest = [0.1,0.1, 0.1,0.1, 0.1,0.1, 0.1,0.1]
    actiontest = np.array(actiontest)
    while True:
        env.render()
        env.step(actiontest) #环境配置OK