from smac.env.starcraft2.maps import get_map_params

import torch as th
import enum
import numpy as np

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

def get_unit_type_from_map_type(map_type):
    match map_type:
        case "marines":
            return ["marine"]
        case "stalkers_and_zealots":
            return ["stalker", "zeolot"]
        case "MMM":
            return ["marine", "marauder", "medivac"]
        case _:
            raise Exception("Do not support map_type:", map_type, "so far")

    """elif map_type == "zeolots":
        return ["zeolot"]
    elif map_type == "stalkers":
        return ["stalker"]
    elif map_type == "bane":
        return ["baneling", "zergling"]
    elif map_type == "colossi_stalkers_zealots":
        return ["stalker", "zeolot", "colossi"]"""
    
class SC2Decomposer:
    def __init__(self, args):
        # Load map params
        self.map_name = args.env_args["map_name"]
        map_params = get_map_params(self.map_name)
        self.n_agents = map_params["n_agents"]
        self.n_enemies = map_params["n_enemies"]
        self.episode_limit = map_params["limit"]
        self.map_type = map_params["map_type"]

        # Observations and state
        # True/False
        self.obs_own_health = args.env_args["obs_own_health"]
        self.obs_all_health = args.env_args["obs_all_health"]
        self.obs_instead_of_state = args.env_args["obs_instead_of_state"]
        self.obs_last_action = args.env_args["obs_last_action"]
        self.obs_pathing_grid = args.env_args["obs_pathing_grid"]
        self.obs_terrain_height = args.env_args["obs_terrain_height"]
        self.obs_timestep_number = args.env_args["obs_timestep_number"]
        self.state_last_action = args.env_args["state_last_action"]
        self.state_timestep_number = args.env_args["state_timestep_number"]
        if self.obs_all_health:
            self.obs_own_health = True
        self.n_obs_pathing = 8
        self.n_obs_height = 9

        # Actions
        self.n_actions_no_attack = 6
        self.n_actions_move = 4
        self.n_actions = self.n_actions_no_attack + self.n_enemies

        # Map info
        self._agent_race = map_params["a_race"]
        self._bot_race = map_params["b_race"]
        self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        self.shield_bits_enemy = 1 if self._bot_race == "P" else 0
        self.unit_type_bits = map_params["unit_type_bits"]
        self.map_type = map_params["map_type"]

        # get the shape of obs' components
        self.move_feats, self.enemy_feats, self.ally_feats, self.own_feats, self.obs_nf_en, self.obs_nf_al = \
            self.get_obs_size()
        self.own_obs_dim = self.move_feats + self.own_feats 
        # own_obs_dim, obs_nf_en, obs_nf_al should be fixed
        self.obs_dim = self.move_feats + self.enemy_feats + self.ally_feats + self.own_feats

        # get the shape of state's components
        self.enemy_state_dim, self.ally_state_dim, self.last_action_state_dim, self.timestep_number_state_dim, self.state_nf_en, self.state_nf_al = \
            self.get_state_size()
        self.state_dim = self.enemy_state_dim + self.ally_state_dim + self.last_action_state_dim + self.timestep_number_state_dim

        self.unit_types = get_unit_type_from_map_type(self.map_type)

    def align_feats_dim(self, aligned_unit_type_bits, aligned_shield_bits_ally, aligned_shield_bits_enemy, map_type_set):
        self.aligned_unit_type_bits = aligned_unit_type_bits
        self.aligned_shield_bits_ally = aligned_shield_bits_ally
        self.aligned_shield_bits_enemy = aligned_shield_bits_enemy

        # for obs
        self.aligned_obs_nf_al = self.obs_nf_al
        self.aligned_obs_nf_en = self.obs_nf_en
        self.aligned_own_feats = self.own_feats
        self.aligned_ally_feats = self.ally_feats
        self.aligned_enemy_feats = self.enemy_feats
        self.aligned_own_obs_dim = self.own_obs_dim
        self.aligned_obs_dim = self.obs_dim
        
        # for state
        self.aligned_state_nf_en = self.state_nf_en
        self.aligned_state_nf_al = self.state_nf_al
        self.aligned_ally_state_dim = self.ally_state_dim
        self.aligned_enemy_state_dim = self.enemy_state_dim
        self.aligned_state_dim = self.state_dim

        # update aligned_dim
        self.pad_shield_ally, self.pad_shield_enemy = False, False
        if self.obs_all_health:
            if self.shield_bits_ally == 0 and self.aligned_shield_bits_ally == 1:
                self.pad_shield_ally = True
                self.aligned_obs_nf_al += 1
                self.aligned_own_feats += 1

                self.aligned_state_nf_al += 1

                self.aligned_ally_state_dim += self.n_agents

            if self.shield_bits_enemy == 0 and self.aligned_shield_bits_enemy == 1:
                self.pad_shield_enemy = True
                self.aligned_obs_nf_en += 1

                self.aligned_state_nf_en += 1

                self.aligned_enemy_state_dim += self.n_enemies
        
        assert self.aligned_unit_type_bits >= self.unit_type_bits
        self.pad_unit_type = False
        if self.aligned_unit_type_bits > self.unit_type_bits:
            self.pad_unit_type = True
            tmp_difference = self.aligned_unit_type_bits - self.unit_type_bits
            self.aligned_obs_nf_al += tmp_difference
            self.aligned_obs_nf_en += tmp_difference
            self.aligned_own_feats += tmp_difference

            self.aligned_state_nf_al += tmp_difference
            self.aligned_state_nf_en += tmp_difference

            self.aligned_ally_state_dim += self.n_agents * tmp_difference
            self.aligned_enemy_state_dim += self.n_enemies * tmp_difference
        
        self.aligned_ally_feats = self.n_agents * self.aligned_ally_state_dim
        self.aligned_enemy_feats = self.n_enemies * self.aligned_enemy_state_dim
        self.aligned_own_obs_dim = self.move_feats + self.aligned_own_feats 
        self.aligned_obs_dim = self.move_feats + self.aligned_enemy_feats + self.aligned_ally_feats + self.aligned_own_feats

        self.aligned_state_dim = self.aligned_enemy_state_dim + self.aligned_ally_state_dim + self.last_action_state_dim + self.timestep_number_state_dim
        
        # now we only allow "marine", "sz" , "MMM"
        self.unit_type2_order = {}
        match [aligned_unit_type_bits, self.map_type]:
            case [3, "marines"]:
                if "marauder" in map_type_set:
                    self.unit_type2_order["marine"] = 1
                else: # marine + sz
                    self.unit_type2_order["marine"] = 0
            case [3, "stalkers_and_zealots"]:
                self.unit_type2_order["stalker"] = 1
                self.unit_type2_order["zeolot"] = 2
            case [3, "MMM"]:
                self.unit_typ2e_order["marauder"] = 0
                self.unit_type2_order["marine"] = 1
                self.unit_type2_order["medivac"] = 2
            case [5, "marines"]:
                self.unit_type2_order["marine"] = 0
            case [5, "MMM"]:
                self.unit_typ2e_order["marauder"] = 0
                self.unit_type2_order["marine"] = 1
                self.unit_type2_order["medivac"] = 2
            case [5, "stalkers_and_zealots"]:
                self.unit_type2_order["stalker"] = 3
                self.unit_type2_order["zeolot"] = 4
            case [_, _]:
                if aligned_unit_type_bits < 3:
                    pass
                else:
                    raise Exception("Not support")
        


    def get_obs_size(self):
        nf_al = 4 + self.unit_type_bits
        nf_en = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally
            nf_en += 1 + self.shield_bits_enemy

        own_feats = self.unit_type_bits
        if self.obs_own_health:
            own_feats += 1 + self.shield_bits_ally
        if self.obs_timestep_number:
            own_feats += 1

        if self.obs_last_action:
            nf_al += self.n_actions

        move_feats = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats += self.n_obs_height

        enemy_feats = self.n_enemies * nf_en
        ally_feats = (self.n_agents - 1) * nf_al
        
        return move_feats, enemy_feats, ally_feats, own_feats, nf_en, nf_al

    def get_state_size(self):
        if self.obs_instead_of_state:
            raise Exception("Not Implemented for obs_instead_of_state")
        
        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits
        
        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al
        
        last_action_state, timestep_number_state = 0, 0
        if self.state_last_action:
            last_action_state = self.n_agents * self.n_actions
        if self.state_timestep_number:
            timestep_number_state = 1
        
        return enemy_state, ally_state, last_action_state, timestep_number_state, nf_en, nf_al

    def decompose_state(self, state_input):
        # state_input = [ally_state, enemy_state, last_action_state, timestep_number_state]
        # assume state_input.shape == [batch_size, seq_len, state]
        
        shape = state_input.shape
        if len(shape) > 2:
            state_input = state_input.reshape(np.prod(shape[:-1]), shape[-1])

        # extract ally_states
        ally_states = [self._align(state_input[:, i * self.state_nf_al:(i + 1) * self.state_nf_al], is_ob=False).reshape(*shape[:-1], -1) for i in range(self.n_agents)]
        #ally_states = [self._align(state_input[:, :, i * self.state_nf_al:(i + 1) * self.state_nf_al], is_ob=False) for i in range(self.n_agents)]
        
        # extract enemy_states
        base = self.n_agents * self.state_nf_al
        enemy_states = [self._align(state_input[:, base + i * self.state_nf_en:base + (i + 1) * self.state_nf_en], is_ob=False, is_ally=False).reshape(*shape[:-1], -1) for i in range(self.n_enemies)]
        #enemy_states = [self._align(state_input[:, :, base + i * self.state_nf_en:base + (i + 1) * self.state_nf_en], is_ob=False, is_ally=False) for i in range(self.n_enemies)]

        # extract last_action_states
        base += self.n_enemies * self.state_nf_en
        last_action_states = [state_input[:, base + i * self.n_actions:base + (i + 1) * self.n_actions].reshape(*shape[:-1], -1) for i in range(self.n_agents)]
        # extract timestep_number_state
        base += self.n_agents * self.n_actions
        timestep_number_state = state_input[:, base:base+self.timestep_number_state_dim].reshape(*shape[:-1], -1)        

        return ally_states, enemy_states, last_action_states, timestep_number_state

    def decompose_obs(self, obs_input):
        """
        obs_input: env_obs + last_action + agent_id
        env_obs = [move_feats, enemy_feats, ally_feats, own_feats]
        """
        
        # extract move feats
        # obs_input.shape == (bs * n_agents, obs_shape)
        move_feats = obs_input[:, :self.move_feats]
        # extract enemy_feats
        base = self.move_feats
        #print("enemy feats")
        enemy_feats = [self._align(obs_input[:, base + i * self.obs_nf_en:base + (i + 1) * self.obs_nf_en], is_ally=False) for i in range(self.n_enemies)]

        # extract ally_feats
        base += self.obs_nf_en * self.n_enemies # should not be aligned here cause obs_input is not aligned
        #print("ally feats")
        ally_feats = [self._align(obs_input[:, base + i * self.obs_nf_al:base + (i + 1) * self.obs_nf_al]) for i in range(self.n_agents - 1)]
    
        # extract own feats
        base += self.obs_nf_al * (self.n_agents - 1)
        own_feats = self._align(obs_input[:, base:base + self.own_feats], is_own=True)
      
        # own
        own_obs = th.cat([move_feats, own_feats], dim=-1)
        #print(self.map_name, own_obs.shape, enemy_feats[0].shape, ally_feats[0].shape)
        return own_obs, enemy_feats, ally_feats

    def decompose_action_info(self, action_info):
        """
        action_info: shape [(bs), n_agent, n_action], can be last_action_input
        """
        shape = action_info.shape
        
        if len(shape) > 2:
            action_info = action_info.reshape(np.prod(shape[:-1]), shape[-1])
        no_attack_action_info = action_info[:, :self.n_actions_no_attack]
        attack_action_info = action_info[:, self.n_actions_no_attack:self.n_actions_no_attack + self.n_enemies]
    
        # recover shape
        no_attack_action_info = no_attack_action_info.reshape(*shape[:-1], self.n_actions_no_attack)    
        attack_action_info = attack_action_info.reshape(*shape[:-1], self.n_enemies)
        # get compact action
        bin_attack_info = th.sum(attack_action_info, dim=-1).unsqueeze(-1)
        compact_action_info = th.cat([no_attack_action_info, bin_attack_info], dim=-1)
        return no_attack_action_info, attack_action_info, compact_action_info
    
    def decompose_joint_action_info(self, joint_action_info):
        """
        action_info: shape [bs*T(*n_agent), n_agents * n_action], can be last_action_input
        """
        shape = joint_action_info.shape
        gap = shape[-1] // self.n_agents
        if len(shape) > 2:
            joint_action_info = joint_action_info.reshape(np.prod(shape[:-1]), shape[-1])
        joint_no_attack_action_info = []
        joint_attack_action_info = []
        for i in range(self.n_agents):
            joint_no_attack_action_info.append(joint_action_info[:, \
                                        i * gap: i * gap + self.n_actions_no_attack])
            joint_attack_action_info.append(joint_action_info[:, \
                                        i * gap + self.n_actions_no_attack: i * gap + self.n_actions_no_attack + self.n_enemies])
        joint_no_attack_action_info = th.cat(joint_no_attack_action_info, dim=-1)
        joint_attack_action_info = th.cat(joint_attack_action_info, dim=-1)
        #joint_bin_attack_info = th.sum(joint_attack_action_info, dim=-1).unsqueeze(-1)
        #joint_compact_attack_info = th.cat([joint_no_attack_action_info, joint_bin_attack_info], dim=-1)
        return joint_no_attack_action_info, joint_attack_action_info #, joint_compact_attack_info


    def _is_align(self, feats, is_ob=True, is_ally=True, is_own=False):
        feats_dim = feats.shape[-1]
        if is_ob:
            if is_ally:
                if is_own:
                    return feats_dim == self.aligned_own_feats
                else:
                    return feats_dim == self.aligned_obs_nf_al
            else:
                return feats_dim == self.aligned_obs_nf_en
        else:
            if is_ally:
                return feats_dim == self.aligned_state_nf_al
            else:
                return feats_dim == self.aligned_state_nf_en
    
    def _align(self, feats, is_ob=True, is_ally=True, is_own=False):
        # feats (bs*n_agents, x), should check shape first
        bs, feats_dim = feats.shape
        copy_dim = feats_dim - self.unit_type_bits
        insight_type = "all" # default
        if self._is_align(feats, is_ob, is_ally, is_own):
            return feats
        if is_ob:
            if is_ally:
                start_pad_unit_dim = copy_dim + self.aligned_shield_bits_ally - self.shield_bits_ally 
                if is_own:
                    target_dim = self.aligned_own_feats
                else:
                    target_dim = self.aligned_obs_nf_al
                    insight_type = "ally"
            else:
                start_pad_unit_dim = copy_dim + self.aligned_shield_bits_enemy - self.shield_bits_enemy
                target_dim = self.aligned_obs_nf_en
                insight_type = "enemy"
        else:
            if is_ally:
                start_pad_unit_dim = copy_dim + self.aligned_shield_bits_ally - self.shield_bits_ally
                target_dim = self.aligned_state_nf_al
            else:
                start_pad_unit_dim = copy_dim + self.aligned_shield_bits_enemy - self.shield_bits_enemy
                target_dim = self.aligned_state_nf_en

        ret_features = th.zeros(bs, target_dim, device=feats.device)
        ret_features[:, :copy_dim] = feats[:, :copy_dim]
        ret_features = self._pad_unit_type(feats, ret_features, copy_dim, start_pad_unit_dim, insight_type)
        return ret_features
    
    def _align_backup(self, feats, is_ob=True, is_ally=True, is_own=False):
        # feats (bs*n_agents, x), should check shape first
        bs, feats_dim = feats.shape
        if self._is_align(feats, is_ob, is_ally, is_own):
            return feats
        else:
            if is_ob:
                if is_ally:
                    if is_own:
                        target_dim = self.aligned_own_feats
                        insight_type = "all"
                    else:
                        target_dim = self.aligned_obs_nf_al
                        insight_type = "ally"
                    ret_features = th.zeros(bs, target_dim, device=feats.device)
                    copy_dim = feats_dim - self.unit_type_bits
                    ret_features[:, :copy_dim] = feats[:, :copy_dim]
                    start_pad_unit_dim = copy_dim + self.aligned_shield_bits_ally - self.shield_bits_ally 
                    # do not need to process shield (already set 0)
                    ret_features = self._pad_unit_type(feats, ret_features, copy_dim, start_pad_unit_dim=start_pad_unit_dim, insight_type=insight_type)
                else: # enemy
                    ret_features = th.zeros(bs, self.aligned_obs_nf_en, device=feats.device)
                    copy_dim = feats_dim - self.unit_type_bits
                    ret_features[:, :copy_dim] = feats[:, :copy_dim]
                    start_pad_unit_dim = copy_dim + self.aligned_shield_bits_enemy - self.shield_bits_enemy
                    ret_features = self._pad_unit_type(feats, ret_features, copy_dim, start_pad_unit_dim=start_pad_unit_dim, insight_type="enemy")
            else:
                if is_ally:
                    ret_features = th.zeros(bs, self.aligned_state_nf_al, device=feats.device)
                    copy_dim = feats_dim - self.unit_type_bits
                    ret_features[:, :copy_dim] = feats[:, :copy_dim]
                    start_pad_unit_dim = copy_dim + self.aligned_shield_bits_ally - self.shield_bits_ally
                    ret_features = self._pad_unit_type(feats, ret_features, copy_dim, start_pad_unit_dim=start_pad_unit_dim)
                else:
                    ret_features = th.zeros(bs, self.aligned_state_nf_en, device=feats.device)
                    copy_dim = feats_dim - self.unit_type_bits
                    ret_features[:, :copy_dim] = feats[:, :copy_dim]
                    start_pad_unit_dim = copy_dim + self.aligned_shield_bits_enemy - self.shield_bits_enemy
                    ret_features = self._pad_unit_type(feats, ret_features, copy_dim, start_pad_unit_dim=start_pad_unit_dim)
                        
            return ret_features

    def _pad_unit_type(self, features, ret_features, copy_dim, start_pad_unit_dim, insight_type="all"):
        match self.map_type:
            case "marines":
                if self.unit_type_bits == 0:
                    match insight_type:
                        case "all":
                            insight_slice = slice(None)
                        case "ally":
                            insight_slice = th.argwhere(features[:, 0] > 0).squeeze(-1)
                        case "enemy":
                            insight_slice = th.argwhere(features[:, 1] > 0).squeeze(-1) # check distance/sight
                        case _:
                            raise ValueError("insight_type should be in ['all', 'ally', 'enemy']")

                    ret_features[insight_slice, start_pad_unit_dim+self.unit_type2_order["marine"]] = 1
                else: #  we may set unit_type=1 in smac_maps, 
                    insight_slice = th.argwhere(features[:, copy_dim]==1)[0].squeeze(-1)
                    ret_features[insight_slice, start_pad_unit_dim+self.unit_type2_order["marine"]] = 1
            # below will not use insight_slice because if not insight, wont occur in \cup x_idx due to all 0 in unit_type_bits
            case "stalkers_and_zealots":
                #print(features.shape, copy_dim)
                assert features.shape[-1] == copy_dim+2 # just in case
                # actually, in original smac, enemy obs stalker and zeolot are in reverse
                # we dont distinguish here as our query/key take enemy and ally obs/state as input separately
                # modification has been made in my fork and request has been pulled to oxwhirl/smac in #111
                stalker_idx = th.argwhere(features[:, copy_dim] == 1).squeeze(-1)
                zeolot_idx = th.argwhere(features[:, copy_dim+1] == 1).squeeze(-1)
                ret_features[stalker_idx, start_pad_unit_dim+self.unit_type2_order["stalker"]] = 1
                ret_features[zeolot_idx, start_pad_unit_dim+self.unit_type2_order["zeolot"]] = 1 
            case "MMM":
                marauder_idx = th.argwhere(features[:, copy_dim] == 1).squeeze(-1) # no shield
                marine_idx = th.argwhere(features[:, copy_dim+1] == 1).squeeze(-1)
                medivac_idx = th.argwhere(features[:, copy_dim+1] == 1).squeeze(-1)
                ret_features[marauder_idx, start_pad_unit_dim+self.unit_type2_order["marauder"]] = 1
                ret_features[marine_idx, start_pad_unit_dim+self.unit_type2_order["marine"]] = 1
                ret_features[medivac_idx, start_pad_unit_dim+self.unit_type2_order["medivac"]] = 1
            case _:
                raise ValueError("map_type should be in ['marines', 'stalkers_and_zealots', 'MMM']")
        return ret_features

    def _print_info(self):
        print(self.map_name)
        print("unit_type_bits", self.unit_type_bits)
        print("shield_bits_ally", self.shield_bits_ally)
        print("shield_bits_enemy", self.shield_bits_enemy)
        print("obs_all_health", self.obs_all_health)
        print("obs_own_health", self.obs_own_health)
        print("obs_last_action", self.obs_last_action)
        print("obs_pathing_grid", self.obs_pathing_grid)
        print("obs_terrain_height", self.obs_terrain_height)

        #print("obs_size", self.get_obs_size())
        #print("state_size", self.get_state_size())

        print("obs_nf_al", self.obs_nf_al)
        print("obs_nf_en", self.obs_nf_en)
        print("own_feats", self.own_feats)
        print("ally_feats", self.ally_feats)
        print("enemy_feats", self.enemy_feats)
        print("own_obs_dim", self.own_obs_dim)
        print("obs_dim", self.obs_dim)
        print("state_nf_en", self.state_nf_en)
        print("state_nf_en", self.state_nf_en)
        print("ally_state_dim", self.ally_state_dim)
        print("state_nf_en", self.state_nf_en)
        print("state_nf_en", self.state_nf_en)

        print("aligned_obs_nf_al", self.aligned_obs_nf_al)
        print("aligned_obs_nf_en", self.aligned_obs_nf_en)
        print("aligned_own_feats", self.aligned_own_feats)
        print("aligned_ally_feats", self.aligned_ally_feats)
        print("aligned_enemy_feats", self.aligned_enemy_feats)
        print("aligned_own_obs_dim", self.aligned_own_obs_dim)
        print("aligned_obs_dim", self.aligned_obs_dim)
        print("aligned_state_nf_en", self.aligned_state_nf_en)
        print("aligned_state_nf_en", self.aligned_state_nf_en)
        print("aligned_ally_state_dim", self.aligned_ally_state_dim)
        print("aligned_state_nf_en", self.aligned_state_nf_en)
        print("aligned_state_nf_en", self.aligned_state_nf_en)

        """
        self.aligned_obs_nf_al = self.obs_nf_al
        self.aligned_obs_nf_en = self.obs_nf_en
        self.aligned_own_feats = self.own_feats
        self.aligned_ally_feats = self.ally_feats
        self.aligned_enemy_feats = self.enemy_feats
        self.aligned_own_obs_dim = self.own_obs_dim
        self.aligned_obs_dim = self.obs_dim
        
        # for state
        self.aligned_state_nf_en = self.state_nf_en
        self.aligned_state_nf_al = self.state_nf_al
        self.aligned_ally_state_dim = self.ally_state_dim
        self.aligned_enemy_state_dim = self.enemy_state_dim
        self.aligned_state_dim = self.state_dim"""