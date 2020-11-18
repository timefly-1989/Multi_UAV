import sys
sys.path.append(".")
import numpy as np
import pyglet
import math
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from array import array
import random

class Env(object):
    viewer = None
    region_x = 600
    region_y = 600
    region_xy = np.sqrt(region_x ** 2 + region_y ** 2)
    dt = 1  # refresh rate
    uav_acceleration_x_bound = [-1, 1]
    uav_acceleration_y_bound = [-1, 1]
    uav_speed_x_bound = [-5, 5]
    uav_speed_y_bound = [-5, 5]
    goal_acceleration_x_bound = [-0.1, 0.1]
    goal_acceleration_y_bound = [-0.1, 0.1]
    goal_speed_x_bound = [-0.5, 0.5]
    goal_speed_y_bound = [-0.5, 0.5]

    #wireless_working = 1    #正在为地面提供网络服务的UAV
    #standby_charge = 2      #在充电桩待机的UAV
    #returning_charge = 3    #在返回充电桩路上的UAV
    #startfrom_charge = 4    #从充电桩出发的UAV
    #request_service = 5     #UAV发出请求替代

    def __init__(self, wireless_working_num, standby_charge_num, charging_num, user_goal_num):
        self.uav_num = wireless_working_num + standby_charge_num
        self.uav_infos = np.zeros(self.uav_num, dtype=[('working_state', np.int), ('speed_x', np.float32), ('speed_y', np.float32), ('position_x', np.float32), ('position_y', np.float32), ('acceleration_x', np.float32), ('acceleration_y', np.float32), ('energy', np.float32), ('wireless_for', np.int), ('return_to', np.int), ('replace_which', np.int)])
        self.charging_infos = np.zeros(charging_num, dtype=[('position_x', np.float32), ('position_y', np.float32)])
        self.user_goal_infos = np.zeros(user_goal_num, dtype=[('speed_x', np.float32), ('speed_y', np.float32), ('position_x', np.float32), ('position_y', np.float32)])

        self.wireless_working_num = wireless_working_num

        self.out_of_power = -10
        self.game_over = 0
    
    #task=1:指定无线服务的UAV->地面目标
    #task=2:
    #task=3:指点UAV返回到那个充电柱
    #task=4:
    #task=5:
    def assign_task(self, task, startfrom, target):
        E_matrix = [[0 for col in range(len(target))] for row in range(len(startfrom))]
        i = j = 0
        for u in startfrom:
            for g in target:
                E_matrix[i][j] = np.sqrt((self.uav_infos[u]['position_x']-self.user_goal_infos[g]['position_x'])**2+(self.uav_infos[u]['position_y']-self.user_goal_infos[g]['position_y'])**2)
                j = j+1
            i = i+1
            j = 0
        E_matrix =np.array(E_matrix)
        row_ind,col_ind=linear_sum_assignment(E_matrix)
        if task == 1:
            for u in row_ind:
                self.uav_infos[startfrom[u]]['wireless_for'] = target[u]

    def reset(self):
        obslist = []
        id_c = 0
        uav_sequence = np.arange(self.uav_num)
        np.random.shuffle(uav_sequence)
        self.pre_action = []

        self.pre_wireless_working_info = []
        self.wireless_working_info = []
        self.pre_request_service = []
        self.request_service = []

        #standby_charge_info = []
        #returning_charge_info = []
        #startfrom_charge_info = []
        #request_service_info = []

        for u in range(np.size(self.uav_infos)):
            if u < self.wireless_working_num:
                self.uav_infos[uav_sequence[u]]['speed_x'] = 0.
                self.uav_infos[uav_sequence[u]]['speed_y'] = 0.
                self.uav_infos[uav_sequence[u]]['acceleration_x'] = 0.
                self.uav_infos[uav_sequence[u]]['acceleration_y'] = 0.
                self.uav_infos[uav_sequence[u]]['position_x'] = np.random.uniform(0, self.region_x)
                self.uav_infos[uav_sequence[u]]['position_y'] = np.random.uniform(0, self.region_y)
                self.uav_infos[uav_sequence[u]]['working_state'] = 1
                self.uav_infos[uav_sequence[u]]['energy'] = np.random.uniform(50,100)
                self.uav_infos[uav_sequence[u]]['wireless_for'] = -1
                self.uav_infos[uav_sequence[u]]['return_to'] = -1
                self.uav_infos[uav_sequence[u]]['replace_which'] = -1
                self.pre_wireless_working_info.append(uav_sequence[u])
            else:
                self.uav_infos[uav_sequence[u]]['speed_x'] = 0.
                self.uav_infos[uav_sequence[u]]['speed_y'] = 0.
                self.uav_infos[uav_sequence[u]]['acceleration_x'] = 0.
                self.uav_infos[uav_sequence[u]]['acceleration_y'] = 0.
                self.uav_infos[uav_sequence[u]]['position_x'] = self.charging_infos[id_c]['position_x'] = np.random.uniform(0, self.region_x)
                self.uav_infos[uav_sequence[u]]['position_y'] = self.charging_infos[id_c]['position_y'] = np.random.uniform(0, self.region_y)
                self.uav_infos[uav_sequence[u]]['working_state'] = 2
                self.uav_infos[uav_sequence[u]]['energy'] = 100
                self.uav_infos[uav_sequence[u]]['wireless_for'] = -1
                self.uav_infos[uav_sequence[u]]['return_to'] = -1
                self.uav_infos[uav_sequence[u]]['replace_which'] = -1
                id_c = id_c +1
        i = 0
        goal_infos = []
        for g in self.user_goal_infos:
            g['speed_x'] = 0.
            g['speed_y'] = 0.
            g['position_x'] = np.random.uniform(0, self.region_x)
            g['position_y'] = np.random.uniform(0, self.region_y)
            goal_infos.append(i)
            i = i+1
        self.assign_task(1, self.pre_wireless_working_info, goal_infos)
        has_request = 0
        for u in self.uav_infos:
            obslist.append(np.array([u['speed_x'], u['speed_y'], u['acceleration_x'], u['acceleration_y'], u['position_x'], u['position_y'], u['working_state'], u['energy'],has_request]))
            self.pre_action.append([u['acceleration_x'], u['acceleration_y'], u['working_state']])
        return obslist

    def step(self, actions):
        self.goal_new_state()
        done = []
        reward = []
        obslist = []
        self.next_action = []
        has_request = 0
        actions = actions.reshape((int(actions.size/3), 3))

        self.wireless_working_info = []

        self.wait_to_working = []
        for u in range(len(actions)):
            if self.pre_action[u][2] == 2 and actions[u][2]==4:
                self.wait_to_working.append[u]
        if len(self.wait_to_working)<len(self.request_service):
            has_request = 1
        
        #能量耗尽，system崩溃结束。
        for i in range(len(actions)):
            if self.uav_infos[i]['energy']<=0:
                self.game_over = 1
                break        

        if self.game_over == 1:
            for i in range(np.size(self.uav_infos)):
                obslist.append(np.array([self.uav_infos[i]['speed_x'], self.uav_infos[i]['speed_y'], self.uav_infos[i]['acceleration_x'], self.uav_infos[i]['acceleration_y'], self.uav_infos[i]['position_x'], self.uav_infos[i]['position_y'], self.uav_infos[i]['working_state'], self.uav_infos[i]['energy']]))
                reward.append(self.out_of_power)
                done.append(True)
            return obslist, reward, done

        for i in range (len(actions)):
            if self.pre_action[i][2] == 1:
                goal_x = self.user_goal_infos[self.uav_infos[i]['wireless_for']]['position_x']
                goal_y = self.user_goal_infos[self.uav_infos[i]['wireless_for']]['position_y']
                pre_d = np.sqrt((self.uav_infos[i]['position_x']-goal_x)**2+(self.uav_infos[i]['position_y']-goal_y)**2)
                self.uav_new_position(actions, i)
                d = np.sqrt((self.uav_infos[i]['position_x']-goal_x)**2 + (self.uav_infos[i]['position_y'] -goal_y)**2)
                reward.append(pre_d-d)
                done.append(False)
                obslist.append(np.array([self.uav_infos[i]['speed_x'], self.uav_infos[i]['speed_y'], self.uav_infos[i]['acceleration_x'], self.uav_infos[i]['acceleration_y'], self.uav_infos[i]['position_x']-goal_x, self.uav_infos[i]['position_y']-goal_y, self.uav_infos[i]['working_state'], self.uav_infos[i]['energy']]))

                if actions[i][2]>=0:
                    self.pre_action[i][2] == 1  #1->1
                    self.wireless_working_info.append(i)
                else:
                    self.pre_action[i][2] == 5    #1->5
                    self.request_service.append(i)

            elif self.pre_action[i][2] == 2:
                if actions[i][2]>=0:
                    if len(self.pre_request_service)>0:    #2->2
                        done.append(False)
                        reward.append(-10)
                        min_d = 10*self.region_xy
                        min_u = -1
                        for u in self.pre_request_service:
                            m_d = np.sqrt((self.uav_infos[i]['position_x']-self.uav_infos[u]['position_x'])**2 + (self.uav_infos[i]['position_y']-self.uav_infos[u]['position_u'])**2)
                            if m_d<min_d:
                                min_d = m_d
                                min_u = u
                        obslist.append(np.array([self.uav_infos[i]['speed_x'], self.uav_infos[i]['speed_y'], self.uav_infos[i]['acceleration_x'], self.uav_infos[i]['acceleration_y'], self.uav_infos[i]['position_x']-self.uav_infos[min_u]['position_x'], self.uav_infos[i]['position_y']-self.uav_infos[min_u]['position_y'], self.uav_infos[i]['working_state'], self.uav_infos[i]['energy']]))
                    else:
                        done.append(False)
                        reward.append(10)
                        min_d = 10*self.region_xy
                        min_u = -1
                        for u in self.wireless_working_info:
                            m_d = np.sqrt((self.uav_infos[i]['position_x']-self.user_goal_infos[u]['position_x'])**2 + (self.uav_infos[i]['position_y']-self.uav_infos[u]['position_u'])**2)
                            if m_d<min_d:
                                min_d = m_d
                                min_u = u
                        obslist.append(np.array([self.uav_infos[i]['speed_x'], self.uav_infos[i]['speed_y'], self.uav_infos[i]['acceleration_x'], self.uav_infos[i]['acceleration_y'], self.uav_infos[i]['position_x']-self.uav_infos[min_u]['position_x'], self.uav_infos[i]['position_y']-self.uav_infos[min_u]['position_y'], self.uav_infos[i]['working_state'], self.uav_infos[i]['energy']]))

                else:
                    pass    #2->4
            elif self.pre_action[i][2] == 3:
                if actions[i][2]>=0:
                    pass    #3->3
                else:
                    pass    #3->2
            elif self.pre_action[i][2] == 4:
                if actions[i][2]>=0:
                    pass    #4->4
                else:
                    pass    #4->1
            elif self.pre_action[i][2] == 5:
                if actions[i][2]>=0:
                    pass    #5->5
                else:
                    pass    #5->3



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